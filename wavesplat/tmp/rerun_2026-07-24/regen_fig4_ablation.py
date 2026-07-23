"""Regenerate paper/figures/fig4_ablation.pdf from the 2026-07-24 finemesh re-run.

Differences from the original figure: 5 seeds instead of 3, medians instead of means (per
the binding reporting convention), the Nyquist-adequate dataset, and a fourth bar for the
initializer change -- which is the only one of the three interventions that clears the seed
spread, and hiding it would misrepresent the comparison.
"""
import json
import os
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..'))
RUNS = os.path.join(HERE, 'ws_runs.jsonl')
OUT = os.path.join(REPO, 'paper', 'figures', 'fig4_ablation.pdf')

CONFIGS = ['Full', 'A1_noDisp', 'A2_noConf', 'v2+disp']
LABELS = ['Full model', 'w/o Dispersion', 'w/o Confidence', 'v2 initializer']
COLORS = ['#4CAF50', '#FF5722', '#9C27B0', '#1E88E5']

ORDER = [('episode_BF3_1comp_seed0_slope.h5', '1comp\nS'),
         ('episode_BF3_1comp_seed0_uniform.h5', '1comp\nU'),
         ('episode_BF3_2comp_seed0_slope.h5', '2comp\nS'),
         ('episode_BF3_2comp_seed0_uniform.h5', '2comp\nU'),
         ('episode_BF3_5comp_seed0_slope.h5', '5comp\nS'),
         ('episode_BF3_5comp_seed0_uniform.h5', '5comp\nU')]

by = defaultdict(list)
with open(RUNS) as f:
    for line in f:
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if 'error' not in r:
            by[(r['episode'], r['cfg_name'])].append(r['avg_nRMSE'])

x = np.arange(len(ORDER))
width = 0.2
fig, ax = plt.subplots(figsize=(11, 5))
for i, (cfg, label, color) in enumerate(zip(CONFIGS, LABELS, COLORS)):
    med = [float(np.median(by[(ep, cfg)])) if by[(ep, cfg)] else 0.0 for ep, _ in ORDER]
    sd = [float(np.std(by[(ep, cfg)], ddof=1)) if len(by[(ep, cfg)]) > 1 else 0.0
          for ep, _ in ORDER]
    ax.bar(x + (i - 1.5) * width, med, width, yerr=sd, label=label, color=color,
           edgecolor='white', capsize=3, alpha=0.88)

ax.set_ylabel('nRMSE (median of 5 seeds, error bars $\\pm$1 sd)', fontsize=11)
ax.set_title('Ablation study on the Nyquist-adequate dataset', fontsize=13)
ax.set_xticks(x)
ax.set_xticklabels([lab for _, lab in ORDER])
ax.legend(fontsize=10)
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
os.makedirs(os.path.dirname(OUT), exist_ok=True)
fig.savefig(OUT, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f'saved {OUT}')
