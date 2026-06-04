"""
9차 발표용: 시연 영상 대체 시각화.
30 에피소드 평가 결과를 산점도 + 박스플롯으로 표현해
'모든 에피소드에서 일관되게 RL이 우수함'을 한 그림에 보여줌.
"""
import csv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams['font.family'] = 'Malgun Gothic'
matplotlib.rcParams['axes.unicode_minus'] = False

with open('results/evaluation_30ep.csv') as f:
    rows = list(csv.DictReader(f))

fixed = [r for r in rows if r['mode'] == 'fixed']
rl    = [r for r in rows if r['mode'] == 'rl']

def col(rows, key): return np.array([float(r[key]) for r in rows])

# 색상 시스템 (navy/teal/accent 고정)
NAVY  = '#0D1B2A'
TEAL  = '#00897B'
CYAN  = '#00BCD4'
GRAY  = '#546E7A'

fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
fig.patch.set_facecolor(NAVY)
fig.suptitle('30 에피소드 전구간 일관성  —  Fixed vs DQN (1000ep)',
             color='white', fontsize=14, fontweight='bold')

# ── (좌) Episode별 Total Reward 산점도 ────────────────────
ax = axes[0]
ax.set_facecolor(NAVY)
eps = np.arange(1, 31)
ax.scatter(eps, col(fixed, 'total_reward'), color=GRAY,
           s=70, edgecolor='white', linewidth=0.5, label='Fixed', zorder=3)
ax.scatter(eps, col(rl, 'total_reward'),    color=TEAL,
           s=70, edgecolor='white', linewidth=0.5, label='RL (DQN 1000ep)', zorder=3)

ax.axhline(np.mean(col(fixed, 'total_reward')), color=GRAY,
           linestyle='--', alpha=0.6, linewidth=1)
ax.axhline(np.mean(col(rl, 'total_reward')),    color=TEAL,
           linestyle='--', alpha=0.6, linewidth=1)

ax.set_title('Total Reward — 30 episodes',
             color='white', fontsize=12, fontweight='bold')
ax.set_xlabel('Episode', color='white')
ax.set_ylabel('Total Reward', color='white')
ax.tick_params(colors='white')
ax.spines[:].set_color(TEAL)
ax.grid(True, alpha=0.15, color='white')
ax.legend(facecolor='#1a2d40', labelcolor='white', loc='center right')

# 간극 강조
gap = np.mean(col(fixed,'total_reward')) - np.mean(col(rl,'total_reward'))
ax.text(0.02, 0.5, f'Gap\n{abs(gap):.0f}',
        transform=ax.transAxes, ha='left', va='center', color=CYAN,
        fontsize=14, fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.4', facecolor='#1a2d40',
                  edgecolor=CYAN))

# ── (우) Avg Waiting Time 박스플롯 ────────────────────────
ax = axes[1]
ax.set_facecolor(NAVY)
data = [col(fixed, 'avg_waiting_time'), col(rl, 'avg_waiting_time')]
bp = ax.boxplot(data, labels=['Fixed', 'RL (DQN 1000ep)'],
                patch_artist=True, widths=0.5,
                medianprops=dict(color='white', linewidth=2),
                whiskerprops=dict(color='white'),
                capprops=dict(color='white'),
                flierprops=dict(markeredgecolor='white'))
bp['boxes'][0].set_facecolor(GRAY)
bp['boxes'][1].set_facecolor(TEAL)
for box in bp['boxes']:
    box.set_edgecolor('white')

ax.set_title('Avg Waiting Time — 분포 비교',
             color='white', fontsize=12, fontweight='bold')
ax.set_ylabel('Avg Waiting Time (s)', color='white')
ax.tick_params(colors='white')
ax.spines[:].set_color(TEAL)
ax.grid(True, alpha=0.15, color='white', axis='y')

# 평균값 라벨 (박스 위쪽, 가독성 위해 컬러 배경박스)
ymax = max(np.max(d) for d in data)
for i, d in enumerate(data, 1):
    m = np.mean(d)
    color = GRAY if i == 1 else TEAL
    ax.annotate(f'평균 {m:.1f}s',
                xy=(i, m), xytext=(i, ymax * 1.08),
                ha='center', va='bottom', color='white',
                fontsize=10, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.35', facecolor=color,
                          edgecolor='white', linewidth=0.8),
                arrowprops=dict(arrowstyle='->', color='white', lw=0.8))
ax.set_ylim(top=ymax * 1.25)

plt.tight_layout(pad=2.0)
out = 'results/distribution_30ep.png'
plt.savefig(out, dpi=150, bbox_inches='tight', facecolor=NAVY)
print(f'[INFO] 저장: {out}')
