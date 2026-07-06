"""
d3_paper_figs3.py — the four remaining paper figures (2026-07-06):
  fig_ladder       limit-ladder lollipop (delta-skill per intervention)
  fig_architecture two-timescale streaming architecture block diagram
  fig_gate_ledger  gate effect across exploratory + 2 confirmations
  fig_realexample  static real-sea example (LJ18g storm swell, best win)
PNG+PDF into results/figs/ (and copied to the submission package by
the caller).

Usage: python3 scripts/d3_paper_figs3.py [--skip_real]
"""
import os
import sys
import json
import argparse

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import patches

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
os.chdir(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, 'scripts')

FD = 'results/figs'
C1, C2, C3 = '#5B7EC9', '#D08770', '#7CA982'


def save(fig, name):
    fig.savefig(f'{FD}/{name}.png', dpi=160)
    fig.savefig(f'{FD}/{name}.pdf')
    print(f'saved {FD}/{name}')


def fig_ladder():
    rungs = [
        ('true-spectrum oracle', -0.004),
        ('4x observation budget (real)', 0.0),
        ('12x learning scale (4k→50k)', 0.0),
        ('2x bank resolution', 0.0),
        ('2nd-order bound-wave corr.', 0.0),
        ('pair-preserving subsample', -0.201),
        ('idealized adjacent clusters', -0.140),
    ]
    fig, ax = plt.subplots(figsize=(6.4, 3.6), dpi=110)
    y = np.arange(len(rungs))[::-1]
    for yi, (name, d) in zip(y, rungs):
        ax.plot([0, d], [yi, yi], color='0.8', lw=2)
        ax.plot([d], [yi], 'o', ms=8,
                color=C2 if d < -0.05 else C1)
        ax.annotate(f'{d:+.2f}' if abs(d) > 0.002 else '≈0',
                    xy=(d, yi), xytext=(d - 0.012 if d < -0.05
                                        else d + 0.012, yi),
                    va='center',
                    ha='right' if d < -0.05 else 'left', fontsize=9)
    ax.axvline(0, color='0.3', lw=1.2)
    ax.set_yticks(y)
    ax.set_yticklabels([r[0] for r in rungs], fontsize=9)
    ax.set_xlabel('Δ slope skill vs. production estimator '
                  '(clean testbed)', fontsize=10)
    ax.set_title('The limit ladder: no examined intervention improves '
                 'the estimator', fontsize=10)
    ax.set_xlim(-0.28, 0.10)
    for s_ in ('top', 'right', 'left'):
        ax.spines[s_].set_visible(False)
    ax.tick_params(left=False, labelsize=9)
    fig.tight_layout()
    save(fig, 'fig_ladder')


def fig_architecture():
    fig, ax = plt.subplots(figsize=(7.2, 3.4), dpi=110)
    ax.axis('off')

    def box(x, y, w, h, text, fc, fs=8.5):
        ax.add_patch(patches.FancyBboxPatch(
            (x, y), w, h, boxstyle='round,pad=0.02', fc=fc, ec='0.4',
            lw=0.8))
        ax.annotate(text, xy=(x + w / 2, y + h / 2), ha='center',
                    va='center', fontsize=fs)

    def arrow(x0, y0, x1, y1, text='', c='0.3'):
        ax.annotate('', xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle='-|>', color=c, lw=1.4))
        if text:
            ax.annotate(text, xy=((x0 + x1) / 2, (y0 + y1) / 2 + 0.035),
                        ha='center', fontsize=7.5, color=c)

    box(0.01, 0.55, 0.16, 0.3, 'LiDAR frames\n10 Hz\n48–51 returns',
        '#E8EEF9')
    box(0.26, 0.52, 0.30, 0.36,
        'FAST LOOP (per frame, 18 ms)\nRLS ≡ batch MMSE\n'
        'AR(1)-toward-prior memory\n+ NIS health', '#DCE7F7')
    box(0.66, 0.55, 0.32, 0.30,
        'encounter forecast\nη at ship future positions\n'
        '(FF band 0–2 s → controller)', '#F8E7DC')
    box(0.26, 0.06, 0.30, 0.30,
        'SLOW LOOP (~10 s, async, 0.8 s)\nγ refresh: beamform / learned '
        'prior\nrebuild (w,P) from buffer\n+ residual gate (preliminary)',
        '#E4F0E4')
    arrow(0.17, 0.70, 0.26, 0.70)
    arrow(0.56, 0.70, 0.66, 0.70)
    arrow(0.37, 0.52, 0.37, 0.36)
    ax.annotate('trailing\nbuffer', xy=(0.345, 0.44), ha='right',
                fontsize=7.5, color='0.3')
    arrow(0.49, 0.36, 0.49, 0.52)
    ax.annotate('γ, (w,P)\ntransfer', xy=(0.515, 0.44), ha='left',
                fontsize=7.5, color='0.3')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title('Two-timescale streaming architecture', fontsize=10)
    fig.tight_layout()
    save(fig, 'fig_architecture')


def fig_gate_ledger():
    rows = [('exploratory\n(3 dev records, 36 cells)', -0.38, 0.34, C1),
            ('confirmatory #1\n(4 unseen small-patch, 48)', -0.11, 0.17,
             C2),
            ('confirmatory #2\n(mid/large patch, 45)', -0.21, 0.084, C2)]
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.2), dpi=110)
    x = np.arange(len(rows))
    axes[0].bar(x, [r[1] for r in rows], width=0.5,
                color=[r[3] for r in rows])
    axes[0].axhline(0, color='0.3', lw=1)
    axes[0].set_ylabel('Spearman (risk vs. skill)', fontsize=9)
    axes[0].set_title('rank correlation', fontsize=9)
    axes[1].bar(x, [r[2] for r in rows], width=0.5,
                color=[r[3] for r in rows])
    axes[1].axhline(0.10, color='0.4', lw=1, ls=':')
    axes[1].annotate('pre-registered bar (+0.10)', xy=(0.02, 0.105),
                     fontsize=7.5, color='0.35')
    axes[1].set_ylabel('confident−risky split Δcorr', fontsize=9)
    axes[1].set_title('effect size', fontsize=9)
    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels([r[0] for r in rows], fontsize=7.2)
        for s_ in ('top', 'right'):
            ax.spines[s_].set_visible(False)
        ax.tick_params(labelsize=8)
    fig.suptitle('Reliability gate across three independent evaluations: '
                 'direction consistent, pre-set bars not met', fontsize=9.5)
    fig.tight_layout()
    save(fig, 'fig_gate_ledger')


def fig_realexample():
    # recompute LJ18g best window (win 1 of 6, W=10) and render one lead
    from realdata_analysis import directional_spread
    from d3_falcon_real_operator import (RECORDS, bilinear,
                                         resolve_direction)
    from d3_sa_multirecord import MRecord
    from d3_sa_nomp import mmse_bank_estimate
    G = 9.81
    cfg = RECORDS['LJ18g']
    rec = MRecord(cfg['path'], stride=cfg['stride'])
    depth, U, W = cfg['depth'], cfg['U'], 10.0
    W_max = 40.0
    nfw = int(round(W_max / rec.dt))
    n_lead = int(round(cfg['lead_max'] / rec.dt))
    cx = 0.5 * (rec.gx.min() + rec.gx.max())
    cy = 0.5 * (rec.gy.min() + rec.gy.max())
    Xg, Yg = np.meshgrid(rec.gx, rec.gy)
    i0 = int(np.linspace(nfw + 1, rec.T_all - n_lead - 2, 6)[1])
    rng = np.random.default_rng(701)
    Zmem = rec.window(i0 - nfw + 1, nfw)
    Zfut = rec.window(i0, n_lead + 1)
    t0 = float(rec.t[i0])
    Hs = 4.0 * float(np.nanstd(Zmem))
    spread, lam_p, pdir_ax, kp = directional_spread(
        np.nan_to_num(Zmem[-1]), rec.dx, rec.dy)
    om_p = np.sqrt(G * kp * np.tanh(kp * depth))
    pdir, _ = resolve_direction(Zmem, rec, kp, pdir_ax, om_p / kp)
    h = -np.array([np.cos(np.radians(pdir)), np.sin(np.radians(pdir))])
    T_tr = W_max + cfg['lead_max']

    def ship_pos(t):
        return np.array([cx, cy]) + U * (t - (t0 - W_max) - T_tr / 2) * h

    r0, r1 = cfg['r_fan']
    P_all, z_all = [], []
    for j in range(nfw):
        t_j = float(rec.t[i0 - nfw + 1 + j])
        sp = ship_pos(t_j)
        RX, RY = Xg - sp[0], Yg - sp[1]
        R = np.hypot(RX, RY)
        ca = (RX * h[0] + RY * h[1]) / np.maximum(R, 1e-9)
        m = ((R >= r0) & (R <= r1) & (ca >= 0.5) & np.isfinite(Zmem[j]))
        idx = np.flatnonzero(m)
        if len(idx) == 0:
            continue
        pick = rng.choice(idx, min(12, len(idx)), replace=False)
        iy, ix = np.unravel_index(pick, Zmem[j].shape)
        P_all.append(np.column_stack([rec.gx[ix], rec.gy[iy],
                                      np.full(len(pick), t_j)]))
        z_all.append(Zmem[j][iy, ix] + rng.normal(0, 0.02, len(pick)))
    P = np.vstack(P_all)
    z = np.concatenate(z_all)
    m = P[:, 2] >= t0 - W
    P, z = P[m], z[m]
    if len(z) > 600:
        ii = np.random.default_rng(0).choice(len(z), 600, False)
        P, z = P[ii], z[ii]
    leads = np.arange(0.0, cfg['lead_max'] + 1e-9, 1.0)
    sq = max(1, int(round(4.0 / rec.dx)))
    gxq, gyq = rec.gx[::sq], rec.gy[::sq]
    Xq, Yq = np.meshgrid(gxq, gyq)
    lead_show = 10.0
    Pq_map = np.column_stack([Xq.ravel(), Yq.ravel(),
                              np.full(Xq.size, t0 + lead_show)])
    Pq_enc = np.array([[*ship_pos(t0 + l_), t0 + l_] for l_ in leads])
    fq = mmse_bank_estimate(P, z, depth, np.vstack([Pq_map, Pq_enc]),
                            sigma0=0.05 * Hs, kp_hint=kp, seed=i0)
    fmap = fq[:Xq.size].reshape(Xq.shape)
    fenc = fq[Xq.size:]
    gt_enc = np.array([bilinear(Zfut[int(round(l_ / rec.dt))],
                                rec.gx, rec.gy, px, py)
                       for (px, py, _t), l_ in zip(Pq_enc, leads)])
    gt_map = Zfut[int(round(lead_show / rec.dt))][::sq, ::sq]

    fig = plt.figure(figsize=(7.2, 5.2), dpi=110)
    gs = fig.add_gridspec(2, 2, height_ratios=[2.1, 1.0], hspace=0.42,
                          wspace=0.12, left=0.08, right=0.97, top=0.90,
                          bottom=0.10)
    axL = fig.add_subplot(gs[0, 0])
    axR = fig.add_subplot(gs[0, 1])
    axT = fig.add_subplot(gs[1, :])
    vmax = 1.3 * np.nanstd(gt_map) * 2
    ext = [rec.gx.min(), rec.gx.max(), rec.gy.min(), rec.gy.max()]
    axL.imshow(gt_map, origin='lower', extent=ext, cmap='RdBu_r',
               vmin=-vmax, vmax=vmax, interpolation='bilinear')
    axR.imshow(fmap, origin='lower', extent=ext, cmap='RdBu_r',
               vmin=-vmax, vmax=vmax, interpolation='bilinear')
    axL.set_title('measured (stereo), t₀+10 s', fontsize=9)
    axR.set_title('forecast from fan returns only', fontsize=9)
    for ax in (axL, axR):
        sp = ship_pos(t0 + lead_show)
        ax.plot([sp[0]], [sp[1]], marker=(3, 0, pdir + 90), ms=10,
                color='0.1')
        ax.tick_params(labelsize=7)
    axR.set_yticklabels([])
    v = np.isfinite(gt_enc)
    axT.plot(leads[v], gt_enc[v], '-', color=C1, lw=2, label='measured')
    axT.plot(leads, fenc, '-', color=C2, lw=2, label='forecast')
    cc = np.corrcoef(fenc[v], gt_enc[v])[0, 1]
    axT.set_xlabel('lead [s]', fontsize=9)
    axT.set_ylabel('η at ship [m]', fontsize=9)
    axT.legend(fontsize=8, frameon=False, ncol=2)
    axT.set_title(f'encounter series (corr {cc:+.2f})', fontsize=9)
    for s_ in ('top', 'right'):
        axT.spines[s_].set_visible(False)
    axT.grid(axis='y', color='0.93', lw=0.7)
    axT.tick_params(labelsize=8)
    fig.suptitle('Real-sea example: La Jument storm swell '
                 '(λ≈300 m ≫ fan, Hs≈8 m), 10 s memory', fontsize=10)
    save(fig, 'fig_realexample')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--skip_real', action='store_true')
    a = ap.parse_args()
    os.makedirs(FD, exist_ok=True)
    fig_ladder()
    fig_architecture()
    fig_gate_ledger()
    if not a.skip_real:
        fig_realexample()
