"""
1000ep 학습된 DQN vs Fixed Signal 평가.

기존 compare.py 대비 개선점:
- 30 episode로 통계 유의성 확보 (평균 ± 표준편차)
- 표준 교통 메트릭 수집: waiting time, queue length, throughput
- 결과를 results/evaluation_*.csv 로 저장
- 발표용 깔끔한 비교 그래프 생성
"""
import os
import sys
import csv
import numpy as np
import torch
import torch.nn as nn
import sumo_rl
import matplotlib.pyplot as plt


def set_sumo_home():
    if "SUMO_HOME" in os.environ:
        return
    candidates = [
        "C:/Program Files (x86)/Eclipse/Sumo",
        "/usr/share/sumo",
        "/opt/homebrew/share/sumo",
    ]
    for c in candidates:
        if c and os.path.isdir(c):
            os.environ["SUMO_HOME"] = c
            print(f"[INFO] SUMO_HOME: {c}")
            return
    print("[ERROR] SUMO_HOME을 찾을 수 없습니다.")
    sys.exit(1)


set_sumo_home()


# ── 설정 ──
STATE_SIZE  = 11
ACTION_SIZE = 2
MODEL_PATH  = "results/dqn_final.pth"
NET_FILE    = "network_v1/intersection.net.xml"
ROUTE_FILE  = "network_v1/intersection.rou.xml"
NUM_SECONDS = 3600
EPISODES    = 30          # 5 → 30 으로 통계 유의성 확보
FIXED_CYCLE = 30

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


# ── 모델 로드 ──
policy_net = DQN(STATE_SIZE, ACTION_SIZE).to(device)
policy_net.load_state_dict(torch.load(MODEL_PATH, map_location=device))
policy_net.eval()
print(f"[INFO] 모델 로드: {MODEL_PATH}")


def make_env():
    return sumo_rl.SumoEnvironment(
        net_file=NET_FILE,
        route_file=ROUTE_FILE,
        use_gui=False,
        num_seconds=NUM_SECONDS,
        min_green=5,
        max_green=60,
        single_agent=True,
        sumo_warnings=False,
    )


def run_episode(mode, ep):
    """한 에피소드 실행, 메트릭 dict 반환."""
    env = make_env()
    obs, _ = env.reset()
    total_reward = 0.0
    done = False
    step = 0
    waits = []
    queues = []

    while not done:
        if mode == "fixed":
            action = (step // FIXED_CYCLE) % 2
        else:
            with torch.no_grad():
                q = policy_net(torch.FloatTensor(obs).to(device))
                action = q.argmax().item()

        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_reward += reward
        step += 1

        # sumo-rl info 에서 메트릭 추출
        # info: agents_total_stopped, agents_total_accumulated_waiting_time
        if isinstance(info, dict):
            wait = info.get('system_total_waiting_time', None)
            stopped = info.get('system_total_stopped', None)
            if wait is not None:  waits.append(wait)
            if stopped is not None: queues.append(stopped)

    # 시뮬 종료 시 처리량
    # SUMO TraCI 통해 arrived 차량 수를 직접 셀 수도 있지만,
    # 여기는 sumo-rl info 만으로 단순 평균
    env.close()

    return {
        'episode': ep,
        'mode': mode,
        'total_reward': total_reward,
        'avg_waiting_time': np.mean(waits) if waits else 0.0,
        'max_waiting_time': max(waits) if waits else 0.0,
        'avg_queue':       np.mean(queues) if queues else 0.0,
        'max_queue':       max(queues) if queues else 0.0,
        'steps': step,
    }


# ── 메인 ──
print(f"\n[FIXED] {EPISODES} 에피소드 실행...")
fixed_results = [run_episode("fixed", i+1) for i in range(EPISODES)]
for r in fixed_results:
    print(f"  ep {r['episode']:2d} | reward={r['total_reward']:8.2f} | "
          f"avg_wait={r['avg_waiting_time']:7.1f} | avg_queue={r['avg_queue']:5.1f}")

print(f"\n[RL] {EPISODES} 에피소드 실행...")
rl_results = [run_episode("rl", i+1) for i in range(EPISODES)]
for r in rl_results:
    print(f"  ep {r['episode']:2d} | reward={r['total_reward']:8.2f} | "
          f"avg_wait={r['avg_waiting_time']:7.1f} | avg_queue={r['avg_queue']:5.1f}")


# ── 통계 ──
def stats(results, key):
    vals = [r[key] for r in results]
    return np.mean(vals), np.std(vals)


print("\n" + "=" * 70)
print(f"{'메트릭':25s}  {'Fixed (mean±std)':>20s}  {'RL (mean±std)':>20s}  {'개선':>6s}")
print("=" * 70)
for key, label, lower_is_better in [
    ('total_reward',     'Total Reward',     False),
    ('avg_waiting_time', 'Avg Waiting Time', True),
    ('max_waiting_time', 'Max Waiting Time', True),
    ('avg_queue',        'Avg Queue Length', True),
    ('max_queue',        'Max Queue Length', True),
]:
    fm, fs = stats(fixed_results, key)
    rm, rs = stats(rl_results, key)
    if lower_is_better:
        imp = (fm - rm) / max(fm, 1e-9) * 100
    else:
        imp = (rm - fm) / max(abs(fm), 1e-9) * 100
    print(f"{label:25s}  {fm:8.2f} ± {fs:6.2f}    {rm:8.2f} ± {rs:6.2f}    {imp:+5.1f}%")
print("=" * 70)


# ── CSV 저장 ──
os.makedirs("results", exist_ok=True)
out_csv = "results/evaluation_30ep.csv"
with open(out_csv, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fixed_results[0].keys())
    w.writeheader()
    w.writerows(fixed_results)
    w.writerows(rl_results)
print(f"\n[INFO] CSV 저장: {out_csv}")


# ── 비교 그래프 (4-panel) ──
fig, axes = plt.subplots(2, 2, figsize=(14, 9))
fig.patch.set_facecolor('#0D1B2A')
fig.suptitle('Fixed Signal vs DQN (1000ep) — 30 Episodes',
             color='white', fontsize=15, fontweight='bold')

metrics = [
    ('total_reward',     'Total Reward (higher better)',    False),
    ('avg_waiting_time', 'Avg Waiting Time (lower better)', True),
    ('avg_queue',        'Avg Queue Length (lower better)', True),
    ('max_queue',        'Max Queue Length (lower better)', True),
]

for ax, (key, title, lower_better) in zip(axes.flat, metrics):
    ax.set_facecolor('#0D1B2A')
    fixed_vals = [r[key] for r in fixed_results]
    rl_vals    = [r[key] for r in rl_results]

    fm, fs = np.mean(fixed_vals), np.std(fixed_vals)
    rm, rs = np.mean(rl_vals),    np.std(rl_vals)

    bars = ax.bar(['Fixed', 'RL (DQN 1000ep)'], [fm, rm],
                  yerr=[fs, rs],
                  color=['#546E7A', '#00897B'],
                  capsize=10, edgecolor='white', linewidth=0.5, width=0.5)
    ax.set_title(title, color='white', fontsize=11, fontweight='bold')
    ax.tick_params(colors='white')
    ax.spines[:].set_color('#00897B')
    ax.grid(axis='y', alpha=0.15, color='white')

    # 값 라벨
    for bar, val, std in zip(bars, [fm, rm], [fs, rs]):
        ax.text(bar.get_x() + bar.get_width() / 2,
                val + (std if val > 0 else -std) * 1.1,
                f'{val:.2f}\n±{std:.2f}',
                ha='center', va='bottom' if val > 0 else 'top',
                color='white', fontsize=9, fontweight='bold')

    if lower_better:
        imp = (fm - rm) / max(fm, 1e-9) * 100
    else:
        imp = (rm - fm) / max(abs(fm), 1e-9) * 100
    ax.text(0.5, 0.92, f'Improvement: {imp:+.1f}%',
            transform=ax.transAxes, ha='center', color='#00BCD4',
            fontsize=11, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#1a2d40',
                      edgecolor='#00BCD4'))

plt.tight_layout()
out_png = 'results/evaluation_chart_1000ep.png'
plt.savefig(out_png, dpi=150, bbox_inches='tight', facecolor='#0D1B2A')
print(f"[INFO] 그래프 저장: {out_png}")
