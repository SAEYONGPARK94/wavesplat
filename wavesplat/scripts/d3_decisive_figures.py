"""Figures for the 2026-07-15 decisive-experiment session (E-16/E-17, #2, D-20).

Static PNGs, matplotlib. Colorblind-safe Okabe-Ito palette. dataviz principles:
form chosen by the data's job, one y-axis per panel, thin marks, recessive grid,
direct labels, legend for >=2 series, no dual axis, no rainbow.

  python3 scripts/d3_decisive_figures.py
"""
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
os.chdir(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, 'scripts')
sys.path.insert(0, 'tmp')

R = 'results/direction3'
OUT = f'{R}/figures'
os.makedirs(OUT, exist_ok=True)

# Okabe-Ito (CVD-safe)
BLUE, ORANGE, GREEN, VERM = '#0072B2', '#E69F00', '#009E73', '#D55E00'
SKY, PURPLE, GRAY, INK = '#56B4E9', '#CC79A7', '#9AA0A6', '#1a1a1a'
MUTED = '#5f6368'

plt.rcParams.update({
    'figure.dpi': 150, 'savefig.dpi': 150, 'font.size': 11,
    'axes.edgecolor': '#c8ccd0', 'axes.linewidth': 0.8,
    'axes.grid': True, 'grid.color': '#e6e8eb', 'grid.linewidth': 0.8,
    'axes.axisbelow': True, 'axes.spines.top': False, 'axes.spines.right': False,
    'xtick.color': MUTED, 'ytick.color': MUTED, 'axes.labelcolor': INK,
    'text.color': INK, 'font.family': 'DejaVu Sans', 'figure.facecolor': 'white',
})
G = 9.81


def _lam(Tp, depth=50.0):
    om = 2 * np.pi / Tp
    k = om ** 2 / G
    for _ in range(80):
        k = om ** 2 / (G * np.tanh(k * depth))
    return float(2 * np.pi / k)


def fig_e16():
    """Two-regime headline: Delta(linear) & Delta(M3) vs lam, and dd vs lam."""
    d = json.load(open(f'{R}/e16_physics_operator_corrected.json'))
    e15 = json.load(open(f'{R}/e15_hos_multidraw.json'))['summary']['sea_means']
    seas = list(d['sea_M1'])
    Tp = np.array([float(s.split('_')[0][2:]) for s in seas])
    lam = np.array([_lam(t) for t in Tp])
    dlin = np.array([d['sea_M1'][s] for s in seas])
    dm3 = np.array([e15[s] for s in seas])
    dd = dm3 - dlin
    short = lam < 90                       # the well-determined regime (Tp6/7)

    fig, (a, b) = plt.subplots(1, 2, figsize=(11.6, 4.7))

    # -- panel A: the two physics arms vs wavelength ----------------------
    a.axhline(0, color='#b8bcc0', lw=1.0, zorder=1)
    a.scatter(lam, dlin, s=42, c=BLUE, edgecolor='white', lw=0.6, zorder=3,
              label='linear physics (correct)')
    a.scatter(lam, dm3, s=42, c=ORANGE, edgecolor='white', lw=0.6, zorder=3,
              label='HOS M3 (nonlinear)')
    for x, y0, y1 in zip(lam, dlin, dm3):        # connect the paired arms
        a.plot([x, x], [y0, y1], color='#d7dadd', lw=0.8, zorder=2)
    # trend lines
    for y, c in ((dlin, BLUE), (dm3, ORANGE)):
        o = np.argsort(lam)
        z = np.polyfit(lam, y, 1)
        a.plot(lam[o], np.polyval(z, lam[o]), color=c, lw=1.6, alpha=0.5, zorder=2)
    a.axvspan(lam.min() - 5, 90, color='#f2f4f6', zorder=0)
    a.text(71, -0.145, 'well-determined\n(short $\\lambda$)',
           fontsize=8.5, color=MUTED, ha='center', va='bottom')
    a.set_xlabel('peak wavelength  $\\lambda_p$  (m)')
    a.set_ylabel('learning gain  $\\Delta$ = net $-$ hand')
    a.set_title('A   both arms rise with $\\lambda$ (aperture);\n'
                'linear is negative only at short $\\lambda$', fontsize=11, loc='left')
    a.legend(frameon=False, fontsize=9.5, loc='upper left')
    a.set_ylim(-0.16, 0.27)

    # -- panel B: the paired physics effect dd = M3 - linear --------------
    b.axhline(0, color='#b8bcc0', lw=1.0, zorder=1)
    b.scatter(lam[~short], dd[~short], s=42, c=GRAY, edgecolor='white', lw=0.6,
              zorder=3, label='long $\\lambda$  (dd $\\approx$ 0: aperture)')
    b.scatter(lam[short], dd[short], s=64, c=GREEN, edgecolor='white', lw=0.7,
              marker='D', zorder=4, label='short $\\lambda$  (dd $>$ 0: physics)')
    b.axvspan(lam.min() - 5, 90, color='#eef6f2', zorder=0)
    b.text(71, 0.185, 'bound-wave\nmisspecification', fontsize=9, color=GREEN,
           ha='center', va='bottom', weight='bold')
    b.text(224, 0.012, 'overall dd = +0.003 (p 0.85)', fontsize=9, color=MUTED,
           ha='right', va='bottom', style='italic')
    b.set_xlabel('peak wavelength  $\\lambda_p$  (m)')
    b.set_ylabel('paired physics effect  dd = $\\Delta_{M3} - \\Delta_{lin}$')
    b.set_title('B   nonlinearity adds gain only where\n'
                'aperture cannot (short $\\lambda$)', fontsize=11, loc='left')
    b.legend(frameon=False, fontsize=9.5, loc='lower left')
    b.set_ylim(-0.27, 0.27)

    fig.suptitle('E-16  decisive experiment: aperture sets the gain magnitude; '
                 'physics misspecification is the sole driver where the field is '
                 'well-determined', fontsize=11.5, y=1.02, x=0.01, ha='left',
                 color=INK, weight='bold')
    fig.tight_layout()
    fig.savefig(f'{OUT}/F_e16_two_regime.png', bbox_inches='tight')
    plt.close(fig)
    print(f'  wrote {OUT}/F_e16_two_regime.png')


def fig_e17():
    """Phase replication: short-lam dd > 0 across independent phase seeds."""
    d = json.load(open(f'{R}/e17_phase_replication.json'))
    rows = [r for r in d['rows'] if r['regime'] == 'short']
    seas = sorted({r['sea'] for r in rows})
    seeds = sorted({r['seed'] for r in rows})
    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    ax.axhline(0, color='#b8bcc0', lw=1.0, zorder=1)
    x = np.arange(len(seas))
    for j, sd in enumerate(seeds):
        y = [next(r['dd'] for r in rows if r['sea'] == s and r['seed'] == sd)
             for s in seas]
        fl = [next(r['flip'] for r in rows if r['sea'] == s and r['seed'] == sd)
              for s in seas]
        ax.scatter(x + (j - 1.5) * 0.13, y, s=54,
                   c=[GREEN if f else SKY for f in fl],
                   edgecolor='white', lw=0.6, zorder=3)
    # per-sea mean line
    for i, s in enumerate(seas):
        ys = [r['dd'] for r in rows if r['sea'] == s]
        ax.plot([x[i] - 0.28, x[i] + 0.28], [np.mean(ys)] * 2, color=INK,
                lw=1.6, zorder=4)
    ax.set_xticks(x)
    ax.set_xticklabels([s.replace('Tp', 'Tp ').replace('_Hs', ' Hs') for s in seas])
    ax.set_ylabel('paired physics effect  dd')
    ax.set_xlabel('short-$\\lambda$ sea')
    ax.set_title('E-17  the short-$\\lambda$ sign flip holds across 4 independent '
                 'phase realizations\n(dd $>$ 0 in 12/12 cells; per-seed t(3) '
                 'p = 0.0024)', fontsize=10.5, loc='left')
    leg = [Line2D([], [], marker='o', ls='', mfc=GREEN, mec='white',
                  label='full sign flip ($\\Delta_{lin}<0<\\Delta_{M3}$)'),
           Line2D([], [], marker='o', ls='', mfc=SKY, mec='white',
                  label='dd $>$ 0 but $\\Delta_{lin}\\geq 0$'),
           Line2D([], [], color=INK, lw=1.6, label='per-sea mean')]
    ax.legend(handles=leg, frameon=False, fontsize=9, loc='upper right')
    ax.set_ylim(-0.02, 0.22)
    fig.tight_layout()
    fig.savefig(f'{OUT}/F_e17_phase_replication.png', bbox_inches='tight')
    plt.close(fig)
    print(f'  wrote {OUT}/F_e17_phase_replication.png')


def fig_mmse_relax():
    """#2: net-MMSE (sparse & dense kp) vs RELAX per record."""
    d = json.load(open(f'{R}/mmse_relax_sparsekp_inference.json'))
    pr = d['per_record']
    recs = list(pr)
    ns = np.array([pr[r]['net_sparse'] for r in recs])
    nd = np.array([pr[r]['net_dense'] for r in recs])
    rx = np.array([pr[r]['relax'] for r in recs])
    o = np.argsort(ns - rx)                    # order by the effect
    recs = [recs[i] for i in o]; ns, nd, rx = ns[o], nd[o], rx[o]
    y = np.arange(len(recs))
    fig, ax = plt.subplots(figsize=(7.8, 4.8))
    h = 0.26
    ax.barh(y + h, rx, height=h, color=GRAY, label='RELAX (no kp)', zorder=3)
    ax.barh(y, nd, height=h, color=SKY, label='net-MMSE, dense kp', zorder=3)
    ax.barh(y - h, ns, height=h, color=BLUE, label='net-MMSE, sparse kp (fair)',
            zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels(recs)
    ax.set_xlabel('condition-A field skill  (corr, common unobserved)')
    ax.set_title('#2  "MMSE > RELAX" survives the sparse-kp fix\n'
                 'net-sparse 0.545 $\\approx$ dense 0.538  |  '
                 'net $-$ RELAX +0.144 (p 0.014, 6/8)', fontsize=10.5, loc='left')
    ax.legend(frameon=False, fontsize=9, loc='lower right')
    ax.set_xlim(0, 0.72)
    ax.grid(axis='y', alpha=0)
    fig.tight_layout()
    fig.savefig(f'{OUT}/F_mmse_vs_relax.png', bbox_inches='tight')
    plt.close(fig)
    print(f'  wrote {OUT}/F_mmse_vs_relax.png')


def fig_d20():
    """D-20: learned-FF vs hand-FF residual per record (5 records)."""
    d = json.load(open(f'{R}/d20_replay_eval_clusterrobust.json'))
    rp = d['rec_paired_mean']
    recs = list(rp)
    dv = np.array([rp[r] for r in recs])
    o = np.argsort(dv)
    recs = [recs[i] for i in o]; dv = dv[o]
    y = np.arange(len(recs))
    fig, ax = plt.subplots(figsize=(7.2, 3.9))
    ax.axvline(0, color='#b8bcc0', lw=1.0, zorder=1)
    ax.barh(y, dv, height=0.5, color=[GREEN if v < 0 else VERM for v in dv],
            zorder=3, edgecolor='white', lw=0.6)
    for yi, v in zip(y, dv):
        ax.text(v - 0.006 if v < 0 else v + 0.006, yi, f'{v:+.3f}',
                va='center', ha='right' if v < 0 else 'left', fontsize=9,
                color=MUTED)
    ax.set_yticks(y)
    ax.set_yticklabels(recs)
    ax.set_xlabel('learned-FF $-$ hand-FF   residual motion (negative = learning helps)')
    ax.set_title('D-20  control transfer, 5 records: learned beats hand 5/5\n'
                 'record-mean $-$0.123  (wild cluster bootstrap p = 0.025)',
                 fontsize=10.5, loc='left')
    ax.set_xlim(-0.34, 0.06)
    ax.grid(axis='y', alpha=0)
    fig.tight_layout()
    fig.savefig(f'{OUT}/F_d20_control_transfer.png', bbox_inches='tight')
    plt.close(fig)
    print(f'  wrote {OUT}/F_d20_control_transfer.png')


if __name__ == '__main__':
    print('generating decisive-session figures ->', OUT)
    fig_e16()
    fig_e17()
    fig_mmse_relax()
    fig_d20()
    print('done.')
