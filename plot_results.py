"""
SmartSignal 학습 곡선 시각화 — results/training_log.csv 를 읽어
reward 수렴(10-ep 이동평균) + epsilon decay 그래프 생성.

주의: 매 에피소드 트래픽이 랜덤(300~1200 veh/h)이라 raw reward는 출렁인다.
정책 성능의 정량 비교는 evaluate.py(고정 조건 30ep)로 판단한다.
"""
import csv
import numpy as np
import matplotlib.pyplot as plt

LOG = 'results/training_log.csv'
with open(LOG) as f:
    rows = list(csv.DictReader(f))

eps      = [int(r['episode']) for r in rows]
rewards  = [float(r['total_reward']) for r in rows]
epsilons = [float(r['epsilon']) for r in rows]
N = len(rewards)

# 10 ep 이동평균
window = 10
ma = [np.mean(rewards[max(0, i - window + 1):i + 1]) for i in range(N)]

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7))
fig.patch.set_facecolor('#0D1B2A')

# 위: Reward
ax1.set_facecolor('#0D1B2A')
ax1.plot(eps, rewards, color='#00BCD4', alpha=0.25, linewidth=0.8, label='Episode Reward')
ax1.plot(eps, ma, color='#00897B', linewidth=2.0, label=f'{window}-ep Moving Avg')
ax1.set_title(f'SmartSignal Reward Convergence ({N} Episodes)',
              color='white', fontsize=14, fontweight='bold')
ax1.set_xlabel('Episode', color='white')
ax1.set_ylabel('Total Reward', color='white')
ax1.tick_params(colors='white')
ax1.spines[:].set_color('#00897B')
ax1.grid(True, alpha=0.15, color='white')
ax1.legend(facecolor='#1a2d40', labelcolor='white', loc='lower right')

# 핵심 구간 마커 (에피소드 수에 맞춰 동적 배치)
frac_marks = [(0.05, 'Initial', '#FF5252'), (0.25, '', '#FFA726'),
              (0.5, '', '#FFEE58'), (1.0, 'Final', '#69F0AE')]
for frac, label, color in frac_marks:
    idx = min(N - 1, max(0, int(N * frac) - 1))
    x, y = eps[idx], ma[idx]
    tag = label or f'ep {x}'
    ax1.scatter([x], [y], color=color, s=80, zorder=5, edgecolor='white', linewidth=1.5)
    ax1.annotate(f'{tag}\n{y:.1f}', xy=(x, y), xytext=(0, 12),
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
out = 'results/reward_convergence.png'
plt.savefig(out, dpi=150, bbox_inches='tight', facecolor='#0D1B2A')
print(f'[INFO] 저장: {out}')
