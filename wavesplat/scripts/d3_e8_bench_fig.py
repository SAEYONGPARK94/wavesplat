"""E-8 figure: the benchmark, re-scored on a mesh that can render it, and
re-quoted at the sparsity the thesis actually assumes.

Two things the reader has to be able to check at a glance:
  - fixing the renderer moved BF2 and almost nothing else, so the benchmark
    headline was never in danger (only its worst row was);
  - the sparse budget (600 obs/window = 50 returns/frame, the original
    assumption) costs field accuracy (ann -0.051, fan -0.029, both t > 5) but
    leaves the encounter forecast -- the metric control actually consumes --
    statistically unchanged (+0.030 +- 0.029, t 1.0, 23/55 episodes improved).
"""
import json
import os

import numpy as np

os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import matplotlib                       # noqa: E402
matplotlib.use('Agg')
import matplotlib.pyplot as plt         # noqa: E402

C_COARSE = '#898781'
C_FINE = '#1baf7a'
C_SPARSE = '#2a78d6'
C_INK = '#0b0b0b'
C_INK2 = '#52514e'
C_GRID = '#e1e0d9'
C_AXIS = '#c3c2b7'
plt.rcParams.update({
    'font.family': 'sans-serif', 'font.size': 9.5,
    'axes.edgecolor': C_AXIS, 'axes.labelcolor': C_INK2,
    'xtick.color': '#898781', 'ytick.color': '#898781',
    'figure.facecolor': '#fcfcfb', 'axes.facecolor': '#fcfcfb',
    'savefig.facecolor': '#fcfcfb',
})
OUT = 'results/direction3'
ARMS = [('coarse', 'shipped mesh 6.2 m\n2000 obs', C_COARSE),
        ('fine', 'adequate mesh 1.34 m\n2000 obs', C_FINE),
        ('fine_sparse', 'adequate mesh\n600 obs (sparse)', C_SPARSE)]


def main():
    d = json.load(open(f'{OUT}/e8_bench55_rescore.json'))
    seas = ['BF2', 'BF3', 'BF4', 'BF5']

    fig, axes = plt.subplots(1, 3, figsize=(13.6, 4.6), dpi=190)
    fig.subplots_adjust(left=0.06, right=0.99, top=0.76, bottom=0.16, wspace=0.24)

    for ax, met, title in zip(
            axes, ['fan', 'enc'],
            ['whole-field ρ (fan, 15–200 m)', 'encounter forecast ρ']):
        x = np.arange(len(seas))
        for k, (arm, lbl, col) in enumerate(ARMS):
            v = [d['by_sea'][s][arm][met] for s in seas]
            ax.bar(x + (k - 1) * 0.27, v, 0.25, color=col,
                   label=lbl if met == 'fan' else None)
        ax.set_xticks(x)
        ax.set_xticklabels([f"{s}\n(n={d['by_sea'][s]['coarse']['n']})"
                            for s in seas], fontsize=8.5)
        if met == 'fan':
            ax.axhline(0.7, color=C_INK2, ls='--', lw=0.9)
            ax.text(-0.45, 0.72, 'operability 0.7', fontsize=7, color=C_INK2)
        else:
            ax.text(-0.45, 0.05, 'dir0 = anti-causal for encounter:\n'
                    'machinery check, not operational skill', fontsize=6.5,
                    color=C_INK2)
        ax.set_ylim(0, 1.02)
        ax.set_ylabel('ρ', fontsize=10)
        ax.set_title(title, fontsize=10.5, color=C_INK, fontweight='bold',
                     loc='left')
        ax.grid(axis='y', color=C_GRID, lw=0.6)
        ax.set_axisbelow(True)
        ax.spines[['top', 'right']].set_visible(False)
    axes[0].legend(frameon=False, fontsize=7.5, loc='lower right', ncols=1)

    # overall summary panel
    ax = axes[2]
    ax.clear()
    o = d['overall']
    mets = ['ann', 'fan', 'enc']
    x = np.arange(3)
    for k, (arm, lbl, col) in enumerate(ARMS):
        ax.bar(x + (k - 1) * 0.27, [o[arm][m] for m in mets], 0.25, color=col)
        for i, m in enumerate(mets):
            ax.text(i + (k - 1) * 0.27, o[arm][m] + 0.015,
                    f'{o[arm][m]:.3f}', ha='center', fontsize=6.6,
                    color=C_INK2, rotation=90)
    ax.set_xticks(x)
    ax.set_xticklabels(['near field\n(ann)', 'wide fan', 'encounter'],
                       fontsize=8.5)
    ax.axhline(0.7, color=C_INK2, ls='--', lw=0.9)
    ax.set_ylim(0, 1.02)
    ax.set_ylabel('ρ', fontsize=10)
    ax.set_title('all 55 episodes\nsparse costs the field (t>5); enc change '
                 'not detectable (t 1.0)',
                 fontsize=10.5, color=C_INK, fontweight='bold', loc='left')
    ax.grid(axis='y', color=C_GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.spines[['top', 'right']].set_visible(False)

    fig.suptitle('Benchmark v2, re-scored on a mesh that can render it — and '
                 're-quoted at the sparsity the thesis assumes\n'
                 'BF2 is the only row whose verdict changed; at 50 returns/frame '
                 'the field degrades and the encounter channel shows no '
                 'detectable change',
                 fontsize=12.5, color=C_INK, fontweight='bold', y=0.99)
    p = f'{OUT}/fig_bench55_2026-07-13.png'
    fig.savefig(p, bbox_inches='tight')
    print('wrote', p)


if __name__ == '__main__':
    main()
