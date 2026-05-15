import os
import sys
import numpy as np
import torch
import torch.nn as nn
import sumo_rl
import matplotlib.pyplot as plt
import csv

# ── SUMO_HOME 설정 ─────────────────────────────────────────
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

# ── DQN 모델 정의 (dqn_agent.py와 동일 구조) ───────────────
class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_size, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_size)
        )
    def forward(self, x):
        return self.net(x)

STATE_SIZE  = 11
ACTION_SIZE = 2
MODEL_PATH  = "dqn_final.pth"
NET_FILE    = "intersection.net.xml"
ROUTE_FILE  = "intersection.rou.xml"
NUM_SECONDS = 3600
EPISODES    = 5   # 각 모드 몇 번 돌려서 평균낼지

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── 모델 로드 ─────────────────────────────────────────────
policy_net = DQN(STATE_SIZE, ACTION_SIZE).to(device)
policy_net.load_state_dict(torch.load(MODEL_PATH, map_location=device))
policy_net.eval()
print(f"[INFO] 모델 로드 완료: {MODEL_PATH}")

# ── 환경 생성 함수 ────────────────────────────────────────
def make_env():
    return sumo_rl.SumoEnvironment(
        net_file=NET_FILE,
        route_file=ROUTE_FILE,
        use_gui=False,
        num_seconds=NUM_SECONDS,
        min_green=5,
        max_green=60,
        single_agent=True,
    )

# ── 고정 신호 실행 (30스텝마다 0↔1 교대) ─────────────────
FIXED_CYCLE = 30  # 스텝 단위 고정 주기

def run_fixed(episodes=EPISODES):
    rewards = []
    print(f"\n[고정 신호] {episodes} 에피소드 실행 중...")
    for ep in range(1, episodes + 1):
        env = make_env()
        obs, _ = env.reset()
        total_reward = 0.0
        done = False
        step = 0
        while not done:
            action = (step // FIXED_CYCLE) % 2  # 30스텝마다 0↔1 교대
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward
            step += 1
        env.close()
        rewards.append(total_reward)
        print(f"  Fixed ep {ep}/{episodes} | Reward: {total_reward:.2f}")
    return rewards

# ── RL 신호 실행 (학습된 모델 사용) ───────────────────────
def run_rl(episodes=EPISODES):
    rewards = []
    print(f"\n[RL 신호] {episodes} 에피소드 실행 중...")
    for ep in range(1, episodes + 1):
        env = make_env()
        obs, _ = env.reset()
        total_reward = 0.0
        done = False
        while not done:
            with torch.no_grad():
                q = policy_net(torch.FloatTensor(obs).to(device))
                action = q.argmax().item()
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward
        env.close()
        rewards.append(total_reward)
        print(f"  RL ep {ep}/{episodes} | Reward: {total_reward:.2f}")
    return rewards

# ── 실행 ─────────────────────────────────────────────────
fixed_rewards = run_fixed()
rl_rewards    = run_rl()

fixed_mean = np.mean(fixed_rewards)
rl_mean    = np.mean(rl_rewards)
improvement = (rl_mean - fixed_mean) / abs(fixed_mean) * 100

print(f"\n{'='*45}")
print(f"  고정 신호 평균 Reward : {fixed_mean:.2f}")
print(f"  RL 신호   평균 Reward : {rl_mean:.2f}")
print(f"  개선율               : {improvement:+.1f}%")
print(f"{'='*45}\n")

# ── CSV 저장 ──────────────────────────────────────────────
os.makedirs("results", exist_ok=True)
with open("results/comparison_log.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["episode", "fixed_reward", "rl_reward"])
    for i, (fr, rr) in enumerate(zip(fixed_rewards, rl_rewards), 1):
        w.writerow([i, round(fr, 2), round(rr, 2)])

# ── 비교 그래프 ───────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
fig.patch.set_facecolor('#0D1B2A')

# 막대 그래프: 평균 비교
ax1.set_facecolor('#0D1B2A')
bars = ax1.bar(['Fixed Signal', 'RL Signal'],
               [fixed_mean, rl_mean],
               color=['#546E7A', '#00897B'], width=0.45, edgecolor='white', linewidth=0.5)
ax1.set_ylabel('Average Total Reward', color='white', fontsize=11)
ax1.set_title('Fixed vs RL Signal\n(Average Reward)', color='white', fontsize=13, fontweight='bold')
ax1.tick_params(colors='white')
ax1.spines[:].set_color('#00897B')
ax1.grid(axis='y', alpha=0.15, color='white')
for bar, val in zip(bars, [fixed_mean, rl_mean]):
    ax1.text(bar.get_x() + bar.get_width()/2, val - abs(val)*0.05,
             f'{val:.1f}', ha='center', va='top', color='white', fontsize=13, fontweight='bold')
ax1.text(0.5, 0.95, f'Improvement: {improvement:+.1f}%',
         transform=ax1.transAxes, ha='center', color='#00BCD4',
         fontsize=12, fontweight='bold',
         bbox=dict(boxstyle='round,pad=0.3', facecolor='#1a2d40', edgecolor='#00BCD4'))

# 에피소드별 비교
eps = list(range(1, EPISODES + 1))
ax2.set_facecolor('#0D1B2A')
ax2.plot(eps, fixed_rewards, 'o-', color='#546E7A', linewidth=2, markersize=6, label='Fixed Signal')
ax2.plot(eps, rl_rewards,    's-', color='#00897B', linewidth=2, markersize=6, label='RL Signal')
ax2.set_xlabel('Episode', color='white', fontsize=11)
ax2.set_ylabel('Total Reward', color='white', fontsize=11)
ax2.set_title('Episode-wise Comparison', color='white', fontsize=13, fontweight='bold')
ax2.tick_params(colors='white')
ax2.spines[:].set_color('#00897B')
ax2.legend(facecolor='#1a2d40', labelcolor='white', fontsize=10)
ax2.grid(True, alpha=0.15, color='white')

plt.tight_layout(pad=2.5)
plt.savefig('results/comparison_chart.png', dpi=150, bbox_inches='tight', facecolor='#0D1B2A')
print("[INFO] 그래프 저장: results/comparison_chart.png")
print("[INFO] CSV 저장:   results/comparison_log.csv")
