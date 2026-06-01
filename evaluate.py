"""
SmartSignal(DQN) vs 고정신호 — 시나리오별 평가 (M1).

핵심: 평가용 동결 시나리오 5종(scenarios/eval/*.rou.xml)을 순회하며,
시나리오마다 Fixed 와 SmartSignal 을 **동일 seed(동일 트래픽)** 로 비교 → 차이는 신호 전략 때문만.

제어 규약은 smart_signal.py 와 동일:
- state 29차원 (앞 num_green_phases = phase one-hot)
- DQN 출력 0=Keep / 1=Next → green phase 사이클 전환
- 고정신호 baseline: green phase를 일정 간격으로 순환

⚠️ TODO (차터 M2): Webster 최적 고정주기 baseline 추가, throughput(도착차량) 메트릭.
실행:
    python evaluate.py                 # 기본: 시나리오당 10ep
    python evaluate.py --episodes 30   # 통계 강화
"""
import os
import sys
import csv
import argparse
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from scenarios import EVAL_SCENARIOS, freeze_eval_scenarios


def set_sumo_home():
    if "SUMO_HOME" in os.environ:
        return
    for c in ["C:/Program Files (x86)/Eclipse/Sumo", "/usr/share/sumo", "/opt/homebrew/share/sumo"]:
        if c and os.path.isdir(c):
            os.environ["SUMO_HOME"] = c
            return
    print("[ERROR] SUMO_HOME을 찾을 수 없습니다.")
    sys.exit(1)


set_sumo_home()
import sumo_rl

parser = argparse.ArgumentParser()
parser.add_argument("--model",         default="results/smart_signal.pth")
parser.add_argument("--net",           default="network/intersection.net.xml")
parser.add_argument("--scenarios_dir", default="scenarios/eval")
parser.add_argument("--episodes",      type=int, default=10, help="시나리오당 반복 수")
parser.add_argument("--seconds",       type=int, default=3600)
parser.add_argument("--seed_base",     type=int, default=1000)
parser.add_argument("--fixed_hold",    type=int, default=6, help="고정신호: green phase 유지 결정스텝")
args = parser.parse_args()

MIN_GREEN, MAX_GREEN = 15, 60
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_size, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, action_size),
        )

    def forward(self, x):
        return self.net(x)


# ── 모델 로드 (없으면 Fixed-only) ──
policy_net = None
NUM_GREEN_PHASES = 4
if os.path.isfile(args.model):
    ckpt = torch.load(args.model, map_location=device)
    if isinstance(ckpt, dict) and "policy_net" in ckpt:
        state_size = ckpt.get("state_size", 29)
        action_size = ckpt.get("action_size", 2)
        NUM_GREEN_PHASES = ckpt.get("num_green_phases", 4)
        weights = ckpt["policy_net"]
    else:
        weights = ckpt
        state_size = weights["net.0.weight"].shape[1]
        action_size = weights[max(k for k in weights if k.endswith(".weight"))].shape[0]
    policy_net = DQN(state_size, action_size).to(device)
    policy_net.load_state_dict(weights)
    policy_net.eval()
    MODES = ["fixed", "rl"]
    print(f"[INFO] 모델 로드: {args.model} "
          f"(state={state_size}, action={action_size}, green_phases={NUM_GREEN_PHASES})")
else:
    MODES = ["fixed"]
    print(f"[WARN] 모델 없음({args.model}) → Fixed 신호만 평가 (하베스 점검 모드).")
    print("       SmartSignal 비교는 'python smart_signal.py' 학습 후 다시 실행하세요.")

# ── 시나리오 파일 확보 (없으면 자동 생성) ──
scenario_paths = {}
for name in EVAL_SCENARIOS:
    p = os.path.join(args.scenarios_dir, f"{name}.rou.xml")
    if not os.path.isfile(p):
        print("[INFO] 시나리오 파일이 없어 생성합니다...")
        freeze_eval_scenarios(args.scenarios_dir)
    scenario_paths[name] = p


def run_episode(mode, route_file, seed):
    """한 에피소드 실행 → 메트릭 dict."""
    env = sumo_rl.SumoEnvironment(
        net_file=args.net, route_file=route_file, use_gui=False,
        num_seconds=args.seconds, min_green=MIN_GREEN, max_green=MAX_GREEN,
        single_agent=True, sumo_warnings=False, sumo_seed=seed,
    )
    obs, _ = env.reset()
    total_reward, done, step = 0.0, False, 0
    waits, queues = [], []
    while not done:
        cur_phase = int(np.array(obs[:NUM_GREEN_PHASES]).argmax())
        if mode == "fixed":
            action = (step // args.fixed_hold) % NUM_GREEN_PHASES
        else:
            with torch.no_grad():
                q = policy_net(torch.FloatTensor(np.array(obs, dtype=np.float32)).to(device))
                dqn_action = int(q.argmax().item())
            action = cur_phase if dqn_action == 0 else (cur_phase + 1) % NUM_GREEN_PHASES
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_reward += reward
        step += 1
        if isinstance(info, dict):
            if info.get('system_total_waiting_time') is not None:
                waits.append(info['system_total_waiting_time'])
            if info.get('system_total_stopped') is not None:
                queues.append(info['system_total_stopped'])
    env.close()
    return {
        'total_reward': total_reward,
        'avg_waiting_time': float(np.mean(waits)) if waits else 0.0,
        'avg_queue': float(np.mean(queues)) if queues else 0.0,
        'max_queue': float(max(queues)) if queues else 0.0,
    }


# ── 평가 루프 (시나리오 × 모드 × 에피소드, 페어드 seed) ──
records = []   # 개별 에피소드 결과
print(f"\n평가 시작: {len(EVAL_SCENARIOS)} 시나리오 × {args.episodes}ep × {MODES}\n")
for name in EVAL_SCENARIOS:
    route = scenario_paths[name]
    for ep in range(args.episodes):
        seed = args.seed_base + ep   # Fixed/RL 동일 seed → 동일 트래픽
        for mode in MODES:
            m = run_episode(mode, route, seed)
            records.append({'scenario': name, 'mode': mode, 'episode': ep, 'seed': seed, **m})
    # 시나리오 요약 출력
    line = f"[{name:11s}]"
    for mode in MODES:
        w = [r['avg_waiting_time'] for r in records if r['scenario'] == name and r['mode'] == mode]
        q = [r['avg_queue'] for r in records if r['scenario'] == name and r['mode'] == mode]
        label = 'Fixed' if mode == 'fixed' else 'Smart'
        line += f"  {label}: wait={np.mean(w):7.1f} queue={np.mean(q):5.1f}"
    print(line)


# ── 종합 표 ──
def agg(scenario, mode, key):
    vals = [r[key] for r in records if r['scenario'] == scenario and r['mode'] == mode]
    return (np.mean(vals), np.std(vals)) if vals else (0.0, 0.0)


print("\n" + "=" * 78)
print(f"{'시나리오':12s} {'메트릭':16s} {'Fixed':>16s} " +
      (f"{'SmartSignal':>16s} {'개선':>7s}" if 'rl' in MODES else ""))
print("=" * 78)
for name in EVAL_SCENARIOS:
    for key, label in [('avg_waiting_time', 'Avg Wait'),
                       ('avg_queue', 'Avg Queue'),
                       ('max_queue', 'Max Queue')]:
        fm, fs = agg(name, 'fixed', key)
        row = f"{name:12s} {label:16s} {fm:9.2f}±{fs:5.2f}"
        if 'rl' in MODES:
            rm, rs = agg(name, 'rl', key)
            imp = (fm - rm) / max(fm, 1e-9) * 100
            row += f" {rm:9.2f}±{rs:5.2f} {imp:+6.1f}%"
        print(row)
    print("-" * 78)


# ── CSV 저장 ──
os.makedirs("results", exist_ok=True)
out_csv = "results/evaluation.csv"
with open(out_csv, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=records[0].keys())
    w.writeheader()
    w.writerows(records)
print(f"[INFO] CSV 저장: {out_csv}")


# ── 차트 (시나리오별 그룹 막대: avg_waiting_time, avg_queue) ──
scen = list(EVAL_SCENARIOS.keys())
x = np.arange(len(scen))
width = 0.38
fig, axes = plt.subplots(1, 2, figsize=(15, 6))
fig.patch.set_facecolor('#0D1B2A')
fig.suptitle('Fixed vs SmartSignal — 시나리오별 비교', color='white',
             fontsize=15, fontweight='bold')

for ax, (key, title) in zip(axes, [('avg_waiting_time', 'Avg Waiting Time (lower better)'),
                                    ('avg_queue', 'Avg Queue Length (lower better)')]):
    ax.set_facecolor('#0D1B2A')
    f_mean = [agg(s, 'fixed', key)[0] for s in scen]
    f_std = [agg(s, 'fixed', key)[1] for s in scen]
    if 'rl' in MODES:
        ax.bar(x - width / 2, f_mean, width, yerr=f_std, label='Fixed',
               color='#546E7A', capsize=5, edgecolor='white', linewidth=0.5)
        r_mean = [agg(s, 'rl', key)[0] for s in scen]
        r_std = [agg(s, 'rl', key)[1] for s in scen]
        ax.bar(x + width / 2, r_mean, width, yerr=r_std, label='SmartSignal',
               color='#00897B', capsize=5, edgecolor='white', linewidth=0.5)
    else:
        ax.bar(x, f_mean, width * 1.5, yerr=f_std, label='Fixed',
               color='#546E7A', capsize=5, edgecolor='white', linewidth=0.5)
    ax.set_title(title, color='white', fontsize=11, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(scen, color='white', rotation=15)
    ax.tick_params(colors='white')
    ax.spines[:].set_color('#00897B')
    ax.grid(axis='y', alpha=0.15, color='white')
    ax.legend(facecolor='#1a2d40', labelcolor='white')

plt.tight_layout()
out_png = "results/evaluation_by_scenario.png"
plt.savefig(out_png, dpi=150, bbox_inches='tight', facecolor='#0D1B2A')
print(f"[INFO] 그래프 저장: {out_png}")
