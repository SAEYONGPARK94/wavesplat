"""results/gifs/encounter_rescore.png — predictable-range profile (W9-c).

Small multiples: columns = cells, rows = leads. Two series (analytic / learned).
Palette pinned by the brief and validated (light mode: L band PASS, chroma PASS,
normal dE 33.6, min(protan,deutan) dE 24.7, contrast 4.30 / 3.12 vs #fcfcfb).
"""
import os, sys, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT); os.chdir(ROOT)

WS, COND = '#2a78d6', '#eb6834'
INK, INK2, MUTED, SURF, GRID = '#1a1a19', '#5c5c5a', '#8a8a87', '#fcfcfb', '#e3e3e0'
CELL_TITLE = {
    'c':    ('cell C — CONTINUUM', '465 atoms · $D_{eff}$ 245 m'),
    'cbig': ('cell C-big — CONTINUUM', '465 atoms · $D_{eff}$ 475 m'),
    'a':    ('cell A — discrete', '15 atoms · $D_{eff}$ 245 m'),
    'abig': ('cell A-big — discrete', '15 atoms · $D_{eff}$ 475 m'),
}
ORDER = ['c', 'cbig', 'a', 'abig']


def fmt_r(v):
    return 'NONE' if v is None else f'{v:g} m'


def main(res='tmp/encounter_rescore_2026-07-26/results.json',
         out='results/gifs/encounter_rescore.png',
         verdict='(verdict pending)'):
    R = json.load(open(res))
    dists, leads = R['dists'], R['leads']
    cells = [c for c in ORDER if c in R['cells'] and R['cells'][c]]
    x = np.arange(len(dists))
    i22 = dists.index(22.0)

    fig, axes = plt.subplots(len(leads), len(cells),
                             figsize=(3.55 * len(cells), 1.95 * len(leads) + 2.9),
                             sharex=True, sharey=True, squeeze=False)
    fig.patch.set_facecolor(SURF)

    for ci, cell in enumerate(cells):
        d = R['cells'][cell]
        for li, lead in enumerate(leads):
            ax = axes[li, ci]
            ax.set_facecolor(SURF)
            for sp in ('top', 'right'):
                ax.spines[sp].set_visible(False)
            for sp in ('left', 'bottom'):
                ax.spines[sp].set_color(GRID)
            ax.tick_params(colors=INK2, labelsize=8.5, length=3, color=GRID)
            ax.grid(True, axis='y', color=GRID, lw=0.8, zorder=0)
            ax.set_axisbelow(True)
            ax.axhspan(0.5, 1.15, color='#2a78d6', alpha=0.045, lw=0, zorder=0)
            ax.set_xlim(-0.35, len(dists) - 0.65)
            ax.axhline(0.5, color=MUTED, lw=1.0, zorder=1)
            ax.axhline(0.3, color=MUTED, lw=1.0, ls=(0, (4, 3)), zorder=1)
            ax.axhline(0.0, color=INK2, lw=1.0, zorder=1)
            ax.axvline(i22, color=MUTED, lw=1.0, ls=(0, (1, 2.5)), zorder=1)
            for arm, col, lab in (('ws', WS, 'analytic'), ('cond', COND, 'learned')):
                if arm not in d:
                    continue
                s = np.array(d[arm]['skill'])[li]
                ax.plot(x, s, color=col, lw=2.0, marker='o', ms=6.5,
                        mfc=col, mec=SURF, mew=1.6, zorder=4 if arm == 'ws' else 3,
                        label=lab if (li == 0 and ci == 0) else None)
            if li == 0:
                t1, t2 = CELL_TITLE[cell]
                ax.set_title(t1, color=INK, fontsize=11, fontweight='bold', pad=17)
                ax.text(0.5, 1.045, t2, transform=ax.transAxes, ha='center',
                        color=INK2, fontsize=8.5)
            if ci == 0:
                ax.set_ylabel(f'lead {lead:g} s\nskill', color=INK, fontsize=9.5)
            if li == len(leads) - 1:
                ax.set_xticks(x)
                ax.set_xticklabels([f'{v:g}' for v in dists])
                for k, tl in enumerate(ax.get_xticklabels()):
                    if k == i22:
                        tl.set_color(INK); tl.set_fontweight('bold')
                ax.set_xlabel('distance ahead of footprint edge  [m]\n'
                              '(22 m = 2 s @ 22 kt)', color=INK2, fontsize=9)
        # per-cell R* + global-map anchor strip under the column
        lines = []
        for arm, lab in (('ws', 'analytic'), ('cond', 'learned ')):
            if arm in d:
                lines.append(f"{lab} R*.5 {fmt_r(d[arm]['rstar50']):>5s}  "
                             f"R*.3 {fmt_r(d[arm]['rstar30']):>5s}")
        gm = '  '.join(f"{lab} {d[arm]['map_skill']:+.3f}".replace('-', '\u2212')
                       for arm, lab in (('ws', 'ws'), ('cond', 'lrn')) if arm in d)
        lines.append(f"global map  {gm}")
        axes[-1, ci].text(0.5, -0.60, '\n'.join(lines), transform=axes[-1, ci].transAxes,
                          ha='center', va='top', fontsize=7.8, color=INK2,
                          family='DejaVu Sans Mono')

    axes[0, 0].set_ylim(-0.82, 1.02)
    axes[0, 0].set_yticks([-0.5, 0.0, 0.3, 0.5, 1.0])

    fig.suptitle('Predictable range of a per-window (still-cut, no-tracking) encounter forecast',
                 color=INK, fontsize=15, fontweight='bold', x=0.5, y=0.985)
    fig.text(0.5, 0.955,
             'skill = 1 − nRMSE / zero-predictor,  at points down-wave of the LiDAR footprint  ·  '
             '4 origins × 10 episodes/cell, median-of-5 seeds  ·  prereg '
             'tmp/encounter_rescore_2026-07-26/PREREG.md',
             ha='center', color=INK2, fontsize=9)
    h, l = axes[0, 0].get_legend_handles_labels()
    leg = fig.legend(h, l, loc='upper right', bbox_to_anchor=(0.995, 0.945),
                     frameon=False, fontsize=10, labelcolor=INK, ncol=2,
                     handlelength=1.6, columnspacing=1.4)
    fig.text(0.5, 0.038, verdict, ha='center', va='center', color=INK,
             fontsize=11.0, fontweight='bold', linespacing=1.5,
             bbox=dict(boxstyle='round,pad=0.6', fc='#f2f2ef', ec=GRID, lw=1.2))
    fig.subplots_adjust(left=0.062, right=0.994, top=0.885, bottom=0.215,
                        wspace=0.11, hspace=0.17)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=150, facecolor=SURF)
    print('saved', out)


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--verdict', default='(verdict pending)')
    a = ap.parse_args()
    main(verdict=a.verdict)
