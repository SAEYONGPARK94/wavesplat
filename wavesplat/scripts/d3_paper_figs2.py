"""
d3_paper_figs2.py — remaining paper figures (2026-07-06, OE target):
  fig_concept  schematic: vessel, forward fan, encounter track, horizon
  fig_sites    real-sea five-site skill with per-window heterogeneity
  fig_e2e      heave-acceleration reduction bars (tuned controller)
PNG + PDF into results/figs/.

Usage: python3 scripts/d3_paper_figs2.py
"""
import os
import sys
import json

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import patches

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
os.chdir(os.path.join(os.path.dirname(__file__), '..'))

FD = 'results/figs'
C1, C2, C3 = '#5B7EC9', '#D08770', '#7CA982'


def save(fig, name):
    fig.savefig(f'{FD}/{name}.png', dpi=160)
    fig.savefig(f'{FD}/{name}.pdf')
    print(f'saved {FD}/{name}')


def fig_concept():
    fig, ax = plt.subplots(figsize=(7.0, 3.6), dpi=110)
    # waves (crest lines moving toward ship from the right... ship moves +x)
    for xc in np.arange(20, 190, 18):
        yy = np.linspace(-55, 55, 100)
        ax.plot(xc + 4 * np.sin(yy / 18), yy, color='#B8C9E8', lw=2.2,
                zorder=1)
    # forward fan (20-100 m ahead)
    wedge = patches.Wedge((0, 0), 100, -30, 30, width=80,
                          facecolor=C1, alpha=0.28, edgecolor=C1, lw=1.2,
                          zorder=2)
    ax.add_patch(wedge)
    ax.annotate('forward fan\n20–100 m, ±60°(FOV)/±30°(shown)\n'
                '~48 returns @10 Hz, 2 cm', xy=(58, 34), fontsize=8,
                color='#2A4A8A', ha='center', zorder=5)
    # ship
    ax.plot([0], [0], marker=(3, 0, -90), ms=22, color='0.15', zorder=6)
    ax.annotate('vessel, U = 22 kt', xy=(-4, -13), fontsize=9,
                ha='center', color='0.15')
    # encounter track + horizon
    ax.annotate('', xy=(96, 0), xytext=(8, 0),
                arrowprops=dict(arrowstyle='-|>', color=C2, lw=2.2,
                                ls='--'))
    ax.annotate('encounter track (FF preview 0–2 s)', xy=(40, -8),
                fontsize=8.5, color=C2)
    xh = 100.0 / (11.3 + 5.5) * 11.3   # ship-frame distance at lead*
    ax.axvline(xh, color='0.35', lw=1.2, ls=':')
    ax.annotate(r'information horizon $\ell^*\!=\!R_{max}/(U+c_g)$'
                '\n≈ 6 s of lead', xy=(xh + 3, -42), fontsize=8.5,
                color='0.3')
    # wave arrow
    ax.annotate('', xy=(120, 47), xytext=(165, 47),
                arrowprops=dict(arrowstyle='-|>', color='#7A8BAA', lw=2))
    ax.annotate('waves (head seas)', xy=(142, 51), fontsize=8.5,
                ha='center', color='#57678A')
    ax.set_xlim(-25, 195)
    ax.set_ylim(-60, 62)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('Given-sensor encounter forecasting: geometry and '
                 'information horizon', fontsize=10)
    fig.tight_layout()
    save(fig, 'fig_concept')


def fig_sites():
    sites = [('YS17', 'Yellow Sea', 20.0), ('AA15a', 'Adriatic 15a', 20.0),
             ('LJ18g', 'La Jument swell', 10.0),
             ('AA15b', 'Adriatic 15b', 40.0), ('AA14', 'Adriatic 14', 10.0)]
    fig, ax = plt.subplots(figsize=(6.6, 4.0), dpi=110)
    xs, pooled = [], []
    for i, (rec, label, Wb) in enumerate(sites):
        d = json.load(open(
            f'results/direction3/falcon_real_operator_{rec}.json'))
        ll = np.array(d['leads'])
        m = ll <= 10.0
        # best-W pooled + per-window points at that W
        best = (-9, None)
        for W in d['w_mems']:
            preds = np.array([w['by_W'][str(W)]['pred']
                              for w in d['windows']])
            gts = np.array([w['gt'] for w in d['windows']])
            p, g = preds[:, m].ravel(), gts[:, m].ravel()
            v = np.isfinite(g)
            c = float(np.corrcoef(p[v], g[v])[0, 1])
            if c > best[0]:
                best = (c, W)
        pooled.append(best[0])
        xs.append(i)
        pw = []
        for w in d['windows']:
            p = np.array(w['by_W'][str(best[1])]['pred'])[m]
            g = np.array(w['gt'])[m]
            v = np.isfinite(g)
            if v.sum() > 3:
                pw.append(float(np.corrcoef(p[v], g[v])[0, 1]))
        ax.scatter([i + np.random.default_rng(i).uniform(-0.08, 0.08)
                    for _ in pw], pw, s=22, color=C1, alpha=0.55,
                   zorder=3, edgecolors='none')
    ax.bar(xs, pooled, width=0.55, color=C1, alpha=0.25, zorder=2)
    for i, p_ in zip(xs, pooled):
        ax.annotate(f'{p_:.2f}', xy=(i, p_ + 0.02), ha='center',
                    fontsize=9, color='#2A4A8A')
    ax.axhline(0, color='0.8', lw=0.8)
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{l}\n(best W={int(w)} s)"
                        for _, l, w in sites], fontsize=8)
    ax.set_ylabel('encounter corr (leads 0–10 s)', fontsize=10)
    ax.set_title('Five real-sea sites, zero-shot: pooled skill (bars) '
                 'and per-window heterogeneity (dots)', fontsize=10)
    ax.set_ylim(-0.6, 1.02)
    for s_ in ('top', 'right'):
        ax.spines[s_].set_visible(False)
    ax.grid(axis='y', color='0.93', lw=0.7)
    ax.tick_params(labelsize=9)
    fig.tight_layout()
    save(fig, 'fig_sites')


def fig_e2e():
    d = json.load(open('results/direction3/e2e_ctrl_tune.json'))
    rows = d['stage2']
    seas = sorted(set(r['sea'] for r in rows))
    fig, ax = plt.subplots(figsize=(6.0, 3.8), dpi=110)
    width = 0.35
    for i, sea in enumerate(seas):
        rs = [r for r in rows if r['sea'] == sea]
        gt = 100 * np.mean([r['red_gt'] for r in rs])
        st = 100 * np.mean([r['red_st'] for r in rs])
        ax.bar(i - width / 2, gt, width, color=C3,
               label='perfect preview (GT-FF)' if i == 0 else None)
        ax.bar(i + width / 2, st, width, color=C2,
               label='streaming forecast (Stream-FF)' if i == 0 else None)
        ax.annotate(f'−{gt:.0f}%', xy=(i - width / 2, gt + 0.4),
                    ha='center', fontsize=9)
        ax.annotate(f'−{st:.0f}%', xy=(i + width / 2, st + 0.4),
                    ha='center', fontsize=9)
    ax.set_xticks(range(len(seas)))
    ax.set_xticklabels([f'{s} head seas, 22 kt (tuned preview 0.2 s)'
                        for s in seas], fontsize=9)
    ax.set_ylabel('heave-accel. RMS reduction [%]', fontsize=10)
    ax.set_title('Control closure: feedforward benefit (2 seeds/sea)',
                 fontsize=10)
    ax.legend(fontsize=8, frameon=False)
    for s_ in ('top', 'right'):
        ax.spines[s_].set_visible(False)
    ax.grid(axis='y', color='0.93', lw=0.7)
    ax.tick_params(labelsize=9)
    fig.tight_layout()
    save(fig, 'fig_e2e')


if __name__ == '__main__':
    os.makedirs(FD, exist_ok=True)
    fig_concept()
    fig_sites()
    fig_e2e()
