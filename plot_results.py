"""
1000ep 학습 결과 시각화 + 구간별 통계 그래프 생성.
results/reward_convergence_1000.png 로 저장.
"""
import csv
import numpy as np
import matplotlib.pyplot as plt

with open('results/training_log.csv') as f:
    rows = list(csv.DictReader(f))

eps     = [int(r['episode']) for r in rows]
rewards = [float(r['total_reward']) for r in rows]
epsilons = [float(r['epsilon']) for r in rows]

# 10 ep moving average
window = 10
ma = []
for i in range(len(rewards)):
    s = max(0, i - window + 1)
    ma.append(np.mean(rewards[s:i+1]))

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7))
fig.patch.set_facecolor('#0D1B2A')

# 위: Reward
ax1.set_facecolor('#0D1B2A')
ax1.plot(eps, rewards, color='#00BCD4', alpha=0.25, linewidth=0.8, label='Episode Reward')
ax1.plot(eps, ma, color='#00897B', linewidth=2.0, label=f'{window}-ep Moving Avg')
ax1.set_title('DQN Reward Convergence (1000 Episodes)',
              color='white', fontsize=14, fontweight='bold')
ax1.set_xlabel('Episode', color='white')
ax1.set_ylabel('Total Reward', color='white')
ax1.tick_params(colors='white')
ax1.spines[:].set_color('#00897B')
ax1.grid(True, alpha=0.15, color='white')
ax1.legend(facecolor='#1a2d40', labelcolor='white', loc='lower right')

# 핵심 구간 마커
markers = [
    (50, np.mean(rewards[:50]), 'Initial', '#FF5252'),
    (200, np.mean(rewards[150:200]), 'ep 200', '#FFA726'),
    (500, np.mean(rewards[450:500]), 'ep 500', '#FFEE58'),
    (1000, np.mean(rewards[950:1000]), 'Final', '#69F0AE'),
]
for x, y, label, color in markers:
    ax1.scatter([x], [y], color=color, s=80, zorder=5, edgecolor='white', linewidth=1.5)
    ax1.annotate(f'{label}\n{y:.1f}', xy=(x, y), xytext=(0, 12),
                 textcoords='offset points', ha='center', color=color,
                 fontsize=9, fontweight='bold')

# 아래: Epsilon decay
ax2.set_facecolor('#0D1B2A')
ax2.plot(eps, epsilons, color='#FFB74D', linewidth=2.0)
ax2.axhline(y=0.05, color='#FF5252', linestyle='--', alpha=0.7, label='ε_min = 0.05')
ax2.set_title('Epsilon Decay (Exploration → Exploitation)',
              color='white', fontsize=12, fontweight='bold')
ax2.set_xlabel('Episode', color='white')
ax2.set_ylabel('Epsilon', color='white')
ax2.tick_params(colors='white')
ax2.spines[:].set_color('#FFB74D')
ax2.grid(True, alpha=0.15, color='white')
ax2.legend(facecolor='#1a2d40', labelcolor='white', loc='upper right')

plt.tight_layout(pad=2.0)
out = 'results/reward_convergence_1000.png'
plt.savefig(out, dpi=150, bbox_inches='tight', facecolor='#0D1B2A')
print(f'[INFO] 저장: {out}')
