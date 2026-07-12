"""Two centerpiece figures from committed JSONs (no recompute).

fig_gradient : the misspecification gradient — d(learned-hand) whole-field
  corr across three validation stages (clean linear synthetic / HOS M=3
  nonlinear / real stereo). The thesis sentence as one plot.
fig_d20_control : real-replay closed-loop residual motion by record x arm
  x actuator delay (D-20).
"""
import json
import os
import sys

import numpy as np

os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

C_HAND = '#2a78d6'
C_NET = '#1baf7a'
C_INK = '#0b0b0b'
C_INK2 = '#52514e'
C_MUT = '#898781'
C_GRID = '#e1e0d9'
C_AXIS = '#c3c2b7'
plt.rcParams.update({
    'font.family': 'sans-serif', 'font.size': 10,
    'axes.edgecolor': C_AXIS, 'axes.labelcolor': C_INK2,
    'xtick.color': C_MUT, 'ytick.color': C_MUT,
    'axes.grid': True, 'grid.color': C_GRID, 'grid.linewidth': 0.6,
    'figure.facecolor': '#fcfcfb', 'axes.facecolor': '#fcfcfb',
    'savefig.facecolor': '#fcfcfb',
})
OUT = 'results/direction3'


def style(ax):
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(axis='y', linewidth=0.6)
    ax.grid(axis='x', visible=False)


def stage_deltas():
    """per-unit d(learned-hand), whole-field corr, three stages."""
    # clean linear synthetic: appendix sheet (5 non-BF2) + dir180 (30)
    clean = []
    s = json.load(open(f'{OUT}/appendix_synth_eval.json'))
    for e in s['episodes']:
        if 'BF2' in e['episode']:
            continue
        clean.append(np.mean(e['corr_learned']) - np.mean(e['corr_hand']))
    a2 = json.load(open(f'{OUT}/a2_dir180_eval.json'))
    for e in a2['episodes']:
        clean.append(np.mean(e['corr_learned']) - np.mean(e['corr_hand']))
    hos = [r['learned'] - r['hand'] for r in
           json.load(open(f'{OUT}/a1_field_hos.json'))]
    real = [r['field_learned'] - r['field_hand'] for r in
            json.load(open(f'{OUT}/field_wide_eval.json'))]
    return [np.array(clean), np.array(hos), np.array(real)]


def fig_gradient():
    data = stage_deltas()
    labels = ['clean linear\nsynthetic\n(35 episodes)',
              'HOS M=3\nnonlinear\n(8 cells)',
              'real stereo\nseas\n(60 windows)']
    sub = ['no misspecification', 'bound waves,\nnonlinear evolution',
           'broadband, directional\nstructure, recon noise']
    fig, ax = plt.subplots(figsize=(7.6, 5.2), dpi=200)
    rng = np.random.default_rng(0)
    for i, d in enumerate(data):
        x = i + rng.uniform(-0.13, 0.13, len(d))
        ax.scatter(x, d, s=26, color=C_NET if d.mean() > 0 else C_HAND,
                   alpha=0.55, edgecolors='none', zorder=2)
        m = d.mean()
        se = d.std(ddof=1) / np.sqrt(len(d))
        ax.errorbar(i, m, yerr=1.96 * se, fmt='o', ms=9, lw=2.2,
                    capsize=6, color=C_INK, zorder=4)
        ax.text(i + 0.2, m, f'{m:+.3f}', fontsize=11, color=C_INK,
                va='center', fontweight='bold')
        ax.text(i, ax.get_ylim()[0], '', fontsize=8)
        print(f'stage {i}: n={len(d)} mean {m:+.4f} se {se:.4f}')
    ax.axhline(0, color=C_AXIS, lw=1.2, zorder=1)
    ax.text(-0.48, 0.006, 'learning\nwins', fontsize=8.5, color=C_NET,
            va='bottom', ha='left')
    ax.text(-0.48, -0.012, 'physics prior\nwins', fontsize=8.5,
            color=C_HAND, va='top', ha='left')
    ax.set_xlim(-0.55, 2.55)
    ax.set_xticks(range(3), labels)
    for i, t in enumerate(sub):
        ax.annotate(t, (i, 1), xycoords=('data', 'axes fraction'),
                    xytext=(0, -14), textcoords='offset points',
                    ha='center', fontsize=8, color=C_MUT)
    ax.set_ylabel('whole-field corr:  learned $-$ physics prior')
    ax.set_title('Learning wins exactly where physics is misspecified\n'
                 '(same estimator pair, three validation stages; '
                 'dots = units, bars = 95% CI)',
                 fontsize=11, color=C_INK, loc='left')
    style(ax)
    fig.tight_layout()
    p = f'{OUT}/fig_gradient_2026-07-12.png'
    fig.savefig(p)
    plt.close(fig)
    print('wrote', p)


def fig_d20():
    d = json.load(open(f'{OUT}/d20_replay_eval.json'))
    rows = d['rows']
    recs = ['YS17', 'AA15b', 'LJ18g']
    arms = [('pct_persist', 'persistence', C_MUT),
            ('pct_hand', 'physics-prior FF', C_HAND),
            ('pct_learned', 'learned FF', C_NET)]
    fig, axs = plt.subplots(1, 2, figsize=(10.6, 4.6), dpi=200,
                            sharey=True)
    for ax, tau in zip(axs, (0.5, 1.0)):
        x = np.arange(len(recs))
        for j, (key, lab, col) in enumerate(arms):
            v = [np.mean([r[key] for r in rows
                          if r['record'] == nm and r['tau'] == tau])
                 for nm in recs]
            ax.bar(x + (j - 1) * 0.26, v, 0.24, color=col,
                   label=lab if tau == 0.5 else None)
            for xi, vi in zip(x, v):
                ax.text(xi + (j - 1) * 0.26, vi + 0.02, f'{vi:.2f}',
                        ha='center', fontsize=7.5, color=C_INK2)
        ax.axhline(1.0, color=C_AXIS, lw=1.2, ls='--')
        ax.text(2.45, 1.02, 'uncompensated', fontsize=8, color=C_MUT,
                ha='right')
        ax.set_xticks(x, recs)
        ax.set_title(f'actuator delay $\\tau$ = {tau} s', fontsize=10,
                     loc='left', color=C_INK)
        style(ax)
    axs[0].set_ylabel('residual heave RMS\n(fraction of uncompensated)')
    axs[0].legend(loc='upper left', frameon=False, fontsize=9)
    g2 = d['gate_D20_2']
    fig.suptitle('Real-sea replay closed loop: the learned estimator\'s '
                 'gain transfers to control\n(learned FF vs physics-prior '
                 f'FF: paired $\\Delta$ {g2["pooled_paired_mean"]:+.3f}, '
                 't $-$4.0, 3/3 records; streaming re-estimation, '
                 'RAO$\\approx$1 heave proxy)',
                 fontsize=11, color=C_INK)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    p = f'{OUT}/fig_d20_control_2026-07-12.png'
    fig.savefig(p)
    plt.close(fig)
    print('wrote', p)


if __name__ == '__main__':
    fig_gradient()
    fig_d20()
