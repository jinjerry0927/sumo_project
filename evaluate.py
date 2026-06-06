"""
Fixed / Webster / SmartSignal(DQN) 3자 비교 — 시나리오별 평가 (M2).

핵심: 평가용 동결 시나리오 5종(scenarios/eval/*.rou.xml)을 순회하며,
시나리오마다 3개 제어기를 **동일 seed(동일 트래픽)** 로 비교 → 차이는 신호 전략 때문만.

세 제어기 모두 동일 안전 봉투(min_green=15 / max_green=60) 하에서 동작하며,
green phase 를 어떻게 배분하느냐에서만 차이 난다:
- fixed   : green phase 를 균등 간격으로 순환 (단순 고정신호)
- webster : 수요 기반 최적 green split (baselines.webster_schedule, 최적 고정신호)
- rl      : DQN 출력 0=Keep / 1=Next → 적응형 사이클 전환 (state 29차원)

teleport: sumo_rl 기본은 교착차량 순간이동 off(-1) → 빡빡한 고정주기가 교차로 박스를
영구 교착시키는 시뮬 아티팩트가 생김. SUMO 기본값(300s)을 복원해 세 모드 동일 조건에서
교착을 해소한다(run_episode 참고).

메트릭: avg_waiting_time / avg_queue / max_queue(낮을수록 좋음) +
throughput(도착차량 수, 높을수록 좋음 — TraCI getArrivedNumber 누적, M3 추가).
실행:
    python evaluate.py                 # 기본: 시나리오당 10ep
    python evaluate.py --episodes 30   # 통계 강화
"""
import os
import sys
import csv
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from scenarios import EVAL_SCENARIOS, freeze_eval_scenarios
from baselines import webster_schedule


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


# ── 모델 로드 (global / e2 각각, 있으면 모드 추가) ──
def _load_dqn(path):
    if not os.path.isfile(path):
        return None, None
    ckpt = torch.load(path, map_location=device)
    weights = ckpt["policy_net"] if isinstance(ckpt, dict) and "policy_net" in ckpt else ckpt
    ss = ckpt.get("state_size", weights["net.0.weight"].shape[1]) if isinstance(ckpt, dict) else weights["net.0.weight"].shape[1]
    a_s = ckpt.get("action_size", weights[max(k for k in weights if k.endswith(".weight"))].shape[0]) if isinstance(ckpt, dict) else weights[max(k for k in weights if k.endswith(".weight"))].shape[0]
    ng = ckpt.get("num_green_phases", 4) if isinstance(ckpt, dict) else 4
    net = DQN(ss, a_s).to(device); net.load_state_dict(weights); net.eval()
    return net, ng

NETS = {}
net_g, ng_g = _load_dqn("results/smart_signal.pth")
if net_g is not None:
    NETS["rl_global"] = net_g
net_e, ng_e = _load_dqn("results/smart_signal_e2.pth")
if net_e is not None:
    NETS["rl_e2"] = net_e
NUM_GREEN_PHASES = ng_g or ng_e or 4
MODES = ["fixed", "webster"] + list(NETS.keys())
print(f"[INFO] 평가 모드: {MODES}")

LABELS = {"fixed": "Fixed", "webster": "Webster",
          "rl_global": "SmartSignal(god-view)", "rl_e2": "SmartSignal(E2)"}

# ── 시나리오 파일 확보 (없으면 자동 생성) ──
scenario_paths = {}
for name in EVAL_SCENARIOS:
    p = os.path.join(args.scenarios_dir, f"{name}.rou.xml")
    if not os.path.isfile(p):
        print("[INFO] 시나리오 파일이 없어 생성합니다...")
        freeze_eval_scenarios(args.scenarios_dir)
    scenario_paths[name] = p


def run_episode(mode, route_file, seed, demand=None):
    """한 에피소드 실행 → 메트릭 dict.

    mode == 'webster' 인 경우 demand(방향별 veh/h)로 green split 스케줄을 계산해 순환한다.
    mode == 'rl_e2' 인 경우 E2 검지기 관측(53D)을 사용한다.
    """
    extra = "--no-step-log"
    obs_class = None
    if mode == "rl_e2":
        from e2_observation import E2ObservationFunction
        obs_class = E2ObservationFunction
        extra = "-a network/e2.add.xml --no-step-log"
    env = sumo_rl.SumoEnvironment(
        net_file=args.net, route_file=route_file, use_gui=False,
        num_seconds=args.seconds, min_green=MIN_GREEN, max_green=MAX_GREEN,
        single_agent=True, sumo_warnings=False, sumo_seed=seed,
        # sumo_rl 기본은 teleport off(-1) → 과포화/빡빡한 고정주기에서 교차로 박스가
        # 영구 교착(흡수상태)되는 시뮬 아티팩트 발생. SUMO 기본값(300s)을 복원해
        # 세 제어기 동일 조건에서 교착을 해소(공정). 자세한 진단은 커밋 메시지 참고.
        time_to_teleport=300,
        additional_sumo_cmd=extra,   # 콘솔의 'Step #...' 스팸 억제 (+E2면 검지기 add)
        **({"observation_class": obs_class} if obs_class else {}),
    )
    obs, _ = env.reset()
    # throughput(도착차량 누적): env.step 은 내부에서 simulationStep 을 delta_time(황색 포함)
    # 만큼 여러 번 돌리므로, env.step 뒤 getArrivedNumber()를 1회만 읽으면 대부분의 도착을
    # 놓친다. _sumo_step 을 감싸 매 시뮬레이션 스텝의 도착수를 누적한다(reset 이후 부착).
    arrived = [0]
    _orig_sumo_step = env._sumo_step

    def _counting_sumo_step():
        _orig_sumo_step()
        arrived[0] += env.sumo.simulation.getArrivedNumber()

    env._sumo_step = _counting_sumo_step
    # Webster: env 의 결정 주기(delta_time)에 맞춘 phase 스케줄 1회 계산
    webster_sched = None
    if mode == "webster":
        webster_sched, _ = webster_schedule(
            demand, delta_time=getattr(env, "delta_time", 5),
            min_green=MIN_GREEN, max_green=MAX_GREEN)
    total_reward, done, step = 0.0, False, 0
    waits, queues = [], []
    while not done:
        cur_phase = int(np.array(obs[:NUM_GREEN_PHASES]).argmax())
        if mode == "fixed":
            action = (step // args.fixed_hold) % NUM_GREEN_PHASES
        elif mode == "webster":
            action = webster_sched[step % len(webster_sched)]
        else:  # rl_global / rl_e2
            net = NETS[mode]
            with torch.no_grad():
                q = net(torch.FloatTensor(np.array(obs, dtype=np.float32)).to(device))
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
        'throughput': int(arrived[0]),   # 에피소드 동안 목적지 도착한 차량 수 (높을수록 좋음)
    }


# ── CSV 저장 헬퍼 (시나리오마다 호출 → 중간 끊겨도 완료분 보존) ──
os.makedirs("results", exist_ok=True)
OUT_CSV = "results/evaluation_e2.csv"


def flush_csv(recs):
    if not recs:
        return
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=recs[0].keys())
        w.writeheader()
        w.writerows(recs)


# ── 평가 루프 (시나리오 × 모드 × 에피소드, 페어드 seed) ──
records = []   # 개별 에피소드 결과
n_scen = len(EVAL_SCENARIOS)
total_eps = n_scen * args.episodes * len(MODES)
t_start = time.time()
done_eps = 0
print(f"\n평가 시작: {n_scen} 시나리오 × {args.episodes}ep × {MODES}  (총 {total_eps} 에피소드)\n")
for si, name in enumerate(EVAL_SCENARIOS, 1):
    route = scenario_paths[name]
    demand = EVAL_SCENARIOS[name]
    t_scen = time.time()
    print(f"[{si}/{n_scen}] {name} 시작...")
    for ep in range(args.episodes):
        seed = args.seed_base + ep   # 모든 모드 동일 seed → 동일 트래픽
        for mode in MODES:
            m = run_episode(mode, route, seed, demand=demand)
            records.append({'scenario': name, 'mode': mode, 'episode': ep, 'seed': seed, **m})
            done_eps += 1
        # 에피소드 단위 진행률 (모든 모드 끝날 때마다)
        elapsed = time.time() - t_start
        eta = elapsed / done_eps * (total_eps - done_eps)
        print(f"    {name:11s} ep {ep+1:2d}/{args.episodes}  "
              f"({done_eps}/{total_eps}, 경과 {elapsed/60:4.1f}분, 남은 ~{eta/60:4.1f}분)")
    # 시나리오 요약 + 중간저장
    line = f"[{name:11s} 완료 {(time.time()-t_scen)/60:.1f}분]"
    for mode in MODES:
        w = [r['avg_waiting_time'] for r in records if r['scenario'] == name and r['mode'] == mode]
        q = [r['avg_queue'] for r in records if r['scenario'] == name and r['mode'] == mode]
        line += f"  {LABELS[mode]}: wait={np.mean(w):7.1f} queue={np.mean(q):5.1f}"
    print(line)
    flush_csv(records)   # 시나리오 단위 중간저장 (크래시 보험)
    print(f"    [중간저장] {OUT_CSV} ({len(records)}행)\n")


# ── 종합 표 ──
def agg(scenario, mode, key):
    vals = [r[key] for r in records if r['scenario'] == scenario and r['mode'] == mode]
    return (np.mean(vals), np.std(vals)) if vals else (0.0, 0.0)


# fixed 를 기준선으로, 나머지 모드는 mean±std + (fixed 대비 개선율) 표시
other_modes = [m for m in MODES if m != 'fixed']
colw = 24
header_w = 78 + colw * len(other_modes)
print("\n" + "=" * header_w)
hdr = f"{'시나리오':12s} {'메트릭':12s} {'Fixed':>16s}"
for m in other_modes:
    hdr += f" {LABELS[m] + ' (개선)':>{colw}s}"
print(hdr)
print("=" * header_w)
# (key, 표시명, 방향): 'lower'=낮을수록 좋음, 'higher'=높을수록 좋음(throughput)
TABLE_METRICS = [('avg_waiting_time', 'Avg Wait', 'lower'),
                 ('avg_queue', 'Avg Queue', 'lower'),
                 ('max_queue', 'Max Queue', 'lower'),
                 ('throughput', 'Throughput', 'higher')]
for name in EVAL_SCENARIOS:
    for key, label, direction in TABLE_METRICS:
        fm, fs = agg(name, 'fixed', key)
        row = f"{name:12s} {label:12s} {fm:9.2f}±{fs:5.2f}"
        for m in other_modes:
            mm, ms = agg(name, m, key)
            # 개선율: lower 메트릭은 (fixed-mode), higher(throughput)는 (mode-fixed)
            imp = ((fm - mm) if direction == 'lower' else (mm - fm)) / max(fm, 1e-9) * 100
            row += f" {mm:9.2f}±{ms:5.2f}({imp:+5.1f}%)"
        print(row)
    print("-" * header_w)


# ── CSV 최종 저장 (시나리오마다 중간저장했지만 마지막으로 한 번 더 확정) ──
flush_csv(records)
print(f"[INFO] CSV 저장: {OUT_CSV}")


# ── 차트 (시나리오별 그룹 막대, N개 모드 일반화) ──
STYLE = {'fixed': ('#546E7A', 'Fixed'),
         'webster': ('#8E7CC3', 'Webster'),
         'rl_global': ('#00897B', 'SmartSignal(god-view)'),
         'rl_e2': ('#FF7043', 'SmartSignal(E2)')}
scen = list(EVAL_SCENARIOS.keys())
x = np.arange(len(scen))
n = len(MODES)
width = 0.8 / n
fig, axes = plt.subplots(1, 3, figsize=(21, 6))
fig.patch.set_facecolor('#0D1B2A')
title_modes = ' vs '.join(STYLE.get(m, ('', m))[1] for m in MODES)
fig.suptitle(f'{title_modes} — Per-Scenario Comparison', color='white',
             fontsize=15, fontweight='bold')

for ax, (key, title) in zip(axes, [('avg_waiting_time', 'Avg Waiting Time (lower better)'),
                                    ('avg_queue', 'Avg Queue Length (lower better)'),
                                    ('throughput', 'Throughput (higher better)')]):
    ax.set_facecolor('#0D1B2A')
    for i, m in enumerate(MODES):
        means = [agg(s, m, key)[0] for s in scen]
        stds = [agg(s, m, key)[1] for s in scen]
        color, label = STYLE.get(m, ('#888888', m))
        offset = (i - (n - 1) / 2) * width
        ax.bar(x + offset, means, width, yerr=stds, label=label, color=color,
               capsize=4, edgecolor='white', linewidth=0.5)
    ax.set_title(title, color='white', fontsize=11, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(scen, color='white', rotation=15)
    ax.tick_params(colors='white')
    ax.spines[:].set_color('#00897B')
    ax.grid(axis='y', alpha=0.15, color='white')
    ax.legend(facecolor='#1a2d40', labelcolor='white')

plt.tight_layout()
out_png = "results/evaluation_e2_by_scenario.png"
plt.savefig(out_png, dpi=150, bbox_inches='tight', facecolor='#0D1B2A')
print(f"[INFO] 그래프 저장: {out_png}")
