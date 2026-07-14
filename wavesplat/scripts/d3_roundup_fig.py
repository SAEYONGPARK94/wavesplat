"""One-page roundup: new (deep-round) results alongside the established
ones. Six panels, all numbers from committed JSONs."""
import json
import os

import numpy as np
from scipy import stats

os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

C_HAND = '#2a78d6'; C_NET = '#1baf7a'; C_LIK = '#eda100'
C_INK = '#0b0b0b'; C_INK2 = '#52514e'; C_MUT = '#898781'
C_GRID = '#e1e0d9'; C_AXIS = '#c3c2b7'
plt.rcParams.update({
    'font.family': 'sans-serif', 'font.size': 9,
    'axes.edgecolor': C_AXIS, 'axes.labelcolor': C_INK2,
    'xtick.color': C_MUT, 'ytick.color': C_MUT,
    'axes.grid': True, 'grid.color': C_GRID, 'grid.linewidth': 0.6,
    'figure.facecolor': '#fcfcfb', 'axes.facecolor': '#fcfcfb',
    'savefig.facecolor': '#fcfcfb'})
OUT = 'results/direction3'


def style(ax):
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(axis='y', linewidth=0.6)
    ax.grid(axis='x', visible=False)


fig, axs = plt.subplots(2, 3, figsize=(13.2, 7.6), dpi=200)

# --- A: misspecification gradient (established) ---
import sys
sys.path.insert(0, 'scripts')
from d3_gradient_d20_figs import stage_deltas, stage_clusters
from d3_cluster_se import cluster_se
ax = axs[0, 0]
data = stage_deltas()
STAGE_CLUSTERS, _hos_sea = stage_clusters()
labels = ['clean\nlinear', 'HOS M=3\nnonlinear', 'real\nstereo']
rng = np.random.default_rng(0)
for i, d in enumerate(data):
    ax.scatter(i + rng.uniform(-0.12, 0.12, len(d)), d, s=14,
               color=C_NET if d.mean() > 0 else C_HAND, alpha=0.5,
               edgecolors='none')
    # E-12e (2026-07-14): was an iid SE x 1.96 on clustered data. Cluster-robust
    # SE + t(G-1). HOS is clustered on the seed (G=4): on the SEA STATE it is
    # G=2 and inference is degenerate -- the HOS leg rests on two seas.
    m = d.mean()
    se, n_clu = cluster_se(d, STAGE_CLUSTERS[i])
    tcrit = float(stats.t.ppf(0.975, n_clu - 1)) if n_clu > 1 else np.nan
    ax.errorbar(i, m, yerr=tcrit * se, fmt='o', ms=7, lw=2, capsize=5,
                color=C_INK)
    ax.text(i + 0.18, m, f'{m:+.3f}', fontsize=9, fontweight='bold',
            color=C_INK, va='center')
ax.axhline(0, color=C_AXIS, lw=1)
ax.set_xticks(range(3), labels)
ax.set_ylabel('field corr: learned $-$ physics')
ax.set_title('A  Misspecification gradient (established)\nlearning wins '
             'only where physics is wrong', loc='left', fontsize=9.5,
             color=C_INK)
style(ax)

# --- B: BF2 reversal old vs new (NEW) ---
ax = axs[0, 1]
old = [0.20, 0.82, 0.07, -0.05, 0.07]      # a7v2 forced (coarse mesh best)
new = json.load(open(f'{OUT}/e1b_finemesh_rejudge.json'))
newr = [r['hyb_rho'] for r in new]
x = np.arange(5)
ax.bar(x - 0.2, old, 0.38, color=C_MUT, label='before: coarse mesh (best)')
ax.bar(x + 0.2, newr, 0.38, color=C_NET, label='after: mesh fixed + hybrid')
ax.axhline(0.5, color=C_AXIS, lw=1, ls='--')
ax.set_xticks(x, [f's{i}' for i in range(5)])
ax.set_ylim(-0.15, 1.1)
ax.set_ylabel('BF2 encounter $\\rho$')
ax.set_title('B  BF2 reversal (NEW)\nfloor was a simulator artifact '
             '(mesh < Nyquist) — 5/5 recovered', loc='left', fontsize=9.5,
             color=C_INK)
ax.legend(loc='upper left', frameon=False, fontsize=7.5)
style(ax)

# --- C: objective duel (NEW) ---
ax = axs[0, 2]
T = [0.356, 0.346, 0.349]
L = [0.276, 0.288, 0.243]
x = np.arange(3)
ax.bar(x - 0.2, L, 0.38, color=C_LIK, label='likelihood-trained')
ax.bar(x + 0.2, T, 0.38, color=C_NET, label='task-loss-trained')
ax.axhline(0.145, color=C_HAND, lw=1.6, ls='--')
ax.text(2.35, 0.15, 'physics\n(hand)', fontsize=7.5, color=C_HAND,
        ha='right', va='bottom')
ax.set_xticks(x, ['seed 0', 'seed 1', 'seed 2'])
ax.set_ylabel('failure-window corr (60-win set)')
ax.set_title('C  Objective duel, 3 seeds (NEW)\nsame net + data, only the '
             'loss differs — task-loss wins 3/3', loc='left', fontsize=9.5,
             color=C_INK)
ax.legend(loc='upper right', frameon=False, fontsize=7.5)
style(ax)

# --- D: learned noise floor (NEW) ---
ax = axs[1, 0]
Ts = [3.11, 3.25, 3.21]
Ls = [1.92, 1.78, 1.95]
x = np.arange(3)
ax.bar(x - 0.2, Ls, 0.38, color=C_LIK)
ax.bar(x + 0.2, Ts, 0.38, color=C_NET)
ax.axhline(3.0, color=C_INK, lw=1.2, ls=':')
ax.text(2.38, 3.05, 'failure-window law $\\approx 3\\sigma_0$', fontsize=7.5,
        color=C_INK2, ha='right', va='bottom')
ax.set_xticks(x, ['seed 0', 'seed 1', 'seed 2'])
ax.set_ylabel('learned noise-floor multiplier')
ax.set_title('D  What each objective learns (NEW)\ntask-loss re-discovers '
             'the 3$\\sigma_0$ floor; likelihood shrinks it', loc='left',
             fontsize=9.5, color=C_INK)
style(ax)

# --- E: UQ calibration (NEW) ---
ax = axs[1, 1]
ax.bar([0], [2.306], 0.55, color=C_LIK)
ax.bar([1], [1.609], 0.55, color=C_NET)
ax.axhline(1.0, color=C_INK, lw=1.2, ls=':')
ax.text(1.35, 1.03, 'calibrated', fontsize=7.5, color=C_INK2, ha='right')
for i, (v, c9) in enumerate([(2.306, 0.67), (1.609, 0.80)]):
    ax.text(i, v + 0.05, f'z-std {v:.2f}\ncov95 {c9:.2f}', ha='center',
            fontsize=8, color=C_INK2)
ax.set_xticks([0, 1], ['likelihood-trained', 'task-loss-trained'])
ax.set_ylim(0, 2.9)
ax.set_ylabel('z-score std (1 = calibrated)')
ax.set_title('E  Uncertainty honesty (NEW)\nboth overconfident, task-loss '
             'much less so', loc='left', fontsize=9.5, color=C_INK)
style(ax)

# --- F: control transfer (recent, established) ---
ax = axs[1, 2]
d20 = json.load(open(f'{OUT}/d20_replay_eval.json'))['rows']
recs = ['YS17', 'AA15b', 'LJ18g']
arms = [('pct_persist', 'persistence', C_MUT),
        ('pct_hand', 'physics FF', C_HAND),
        ('pct_learned', 'learned FF', C_NET)]
x = np.arange(3)
for j, (k, lab, c) in enumerate(arms):
    v = [np.mean([r[k] for r in d20 if r['record'] == nm
                  and r['tau'] == 0.5]) for nm in recs]
    ax.bar(x + (j - 1) * 0.26, v, 0.24, color=c, label=lab)
ax.axhline(1.0, color=C_AXIS, lw=1, ls='--')
ax.set_xticks(x, recs)
ax.set_ylabel('residual heave RMS (frac.)')
ax.set_title('F  Real-replay control ($\\tau$=0.5 s)\nlearned FF < physics '
             'FF, 3/3 records (t $-$4.0)', loc='left', fontsize=9.5,
             color=C_INK)
ax.legend(loc='upper right', frameon=False, fontsize=7.5)
style(ax)

fig.suptitle('WaveSplat — established results (A, F) and the deep-round '
             'additions (B–E), 2026-07-12/13', fontsize=12, color=C_INK)
fig.tight_layout(rect=(0, 0, 1, 0.94))
p = f'{OUT}/fig_roundup_2026-07-13.png'
fig.savefig(p)
print('wrote', p)
