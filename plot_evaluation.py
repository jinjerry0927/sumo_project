"""
results/evaluation.csv → 시나리오별 비교 차트 재생성 (재시뮬 없이 스타일/언어만 수정).

evaluate.py 가 만든 CSV를 읽어 차트만 다시 그린다.
mode(fixed/webster/rl)는 CSV에 있는 대로 자동 감지 → M2 Webster 추가 시에도 그대로 동작.

실행:  python plot_evaluation.py
"""
import csv
import numpy as np
import matplotlib.pyplot as plt
from scenarios import EVAL_SCENARIOS

CSV = 'results/evaluation.csv'
rows = list(csv.DictReader(open(CSV)))

# 시나리오/모드 순서 (CSV에 존재하는 것만)
scen = [s for s in EVAL_SCENARIOS if any(r['scenario'] == s for r in rows)]
modes = []
for r in rows:
    if r['mode'] not in modes:
        modes.append(r['mode'])

# 모드별 (색, 표시명)
STYLE = {
    'fixed':   ('#546E7A', 'Fixed'),
    'webster': ('#8E7CC3', 'Webster'),
    'rl':      ('#00897B', 'SmartSignal'),
}


def agg(s, m, k):
    v = [float(r[k]) for r in rows if r['scenario'] == s and r['mode'] == m]
    return (np.mean(v), np.std(v)) if v else (0.0, 0.0)


x = np.arange(len(scen))
n = len(modes)
width = 0.8 / n

fig, axes = plt.subplots(1, 2, figsize=(15, 6))
fig.patch.set_facecolor('#0D1B2A')
title_modes = ' vs '.join(STYLE.get(m, ('', m))[1] for m in modes)
fig.suptitle(f'{title_modes} — Per-Scenario Comparison',
             color='white', fontsize=15, fontweight='bold')

for ax, (key, title) in zip(axes, [('avg_waiting_time', 'Avg Waiting Time (lower better)'),
                                    ('avg_queue', 'Avg Queue Length (lower better)')]):
    ax.set_facecolor('#0D1B2A')
    for i, m in enumerate(modes):
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
out = 'results/evaluation_by_scenario.png'
plt.savefig(out, dpi=150, bbox_inches='tight', facecolor='#0D1B2A')
print(f'[INFO] 저장: {out}  (modes: {modes})')
