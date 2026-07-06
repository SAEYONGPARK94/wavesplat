"""
d3_falcon_real_gif.py — GIF: REAL sea (YS17 stereo) vs Falcon-fan forecast.

Reproduces one window of d3_falcon_real_operator.py (same seeds/protocol),
then renders lead 0..20 s: left = measured stereo surface, right = field
predicted from the causal forward-fan LiDAR observations only (memory W),
bottom = encounter series at the ship's future positions (GT vs predicted).
Output: results/gifs/falcon_real_YS17_forecast.gif

Usage: python3 scripts/d3_falcon_real_gif.py [--record YS17] [--win 3]
       [--W 40]
"""
import os
import sys
import argparse

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import animation

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
os.chdir(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, 'scripts')

from realdata_analysis import directional_spread
from d3_falcon_real_operator import RECORDS, bilinear, resolve_direction
from d3_sa_multirecord import MRecord
from d3_sa_nomp import mmse_bank_estimate

G = 9.81
OUT = 'results/gifs/falcon_real_{}_forecast.gif'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--record', default='YS17')
    ap.add_argument('--win', type=int, default=3)
    ap.add_argument('--W', type=float, default=40.0)
    ap.add_argument('--suffix', default='')
    ap.add_argument('--n_win', type=int, default=6)
    ap.add_argument('--m_scan', type=int, default=12)
    ap.add_argument('--m_cap', type=int, default=600)
    ap.add_argument('--noise', type=float, default=0.02)
    a = ap.parse_args()
    cfg = RECORDS[a.record]
    U = cfg['U']
    rec = MRecord(cfg['path'], stride=cfg['stride'])
    depth = cfg['depth']
    lead_max = cfg['lead_max']
    W_max = 40.0                                   # match the run's w_mems max
    nfw = int(round(W_max / rec.dt))
    n_lead = int(round(lead_max / rec.dt))
    cx = 0.5 * (rec.gx.min() + rec.gx.max())
    cy = 0.5 * (rec.gy.min() + rec.gy.max())
    Xg, Yg = np.meshgrid(rec.gx, rec.gy)
    lo, hi = nfw + 1, rec.T_all - n_lead - 2
    i0 = int(np.linspace(lo, hi, a.n_win).astype(int)[a.win])
    wi = a.win

    rng = np.random.default_rng(700 + wi)          # same stream as the run
    Zmem = rec.window(i0 - nfw + 1, nfw)
    Zfut = rec.window(i0, n_lead + 1)
    t0 = float(rec.t[i0])
    Hs = 4.0 * float(np.nanstd(Zmem))
    spread, lam_p, pdir_ax, kp = directional_spread(
        np.nan_to_num(Zmem[-1]), rec.dx, rec.dy)
    om_p = np.sqrt(G * kp * np.tanh(kp * depth))
    cp = om_p / kp
    pdir, _ = resolve_direction(Zmem, rec, kp, pdir_ax, cp)
    w = np.array([np.cos(np.radians(pdir)), np.sin(np.radians(pdir))])
    h = -w
    T_tr = W_max + lead_max

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
        pick = rng.choice(idx, min(a.m_scan, len(idx)), replace=False)
        iy, ix = np.unravel_index(pick, Zmem[j].shape)
        P_all.append(np.column_stack([rec.gx[ix], rec.gy[iy],
                                      np.full(len(pick), t_j)]))
        z_all.append(Zmem[j][iy, ix] + rng.normal(0, a.noise, len(pick)))
    P_all = np.vstack(P_all)
    z_all = np.concatenate(z_all)
    m = P_all[:, 2] >= t0 - a.W
    P, z = P_all[m], z_all[m]
    if len(z) > a.m_cap:
        ii = np.random.default_rng(0).choice(len(z), a.m_cap, replace=False)
        P, z = P[ii], z[ii]

    # ---- queries: coarse map grid x 21 lead frames + encounter series
    leads = np.arange(0.0, lead_max + 1e-9, 1.0)
    sq = max(1, int(round(4.0 / rec.dx)))          # ~4 m query spacing
    gxq, gyq = rec.gx[::sq], rec.gy[::sq]
    Xq, Yq = np.meshgrid(gxq, gyq)
    npx = Xq.size
    Pq = []
    for l_ in leads:
        Pq.append(np.column_stack([Xq.ravel(), Yq.ravel(),
                                   np.full(npx, t0 + l_)]))
    Pq_enc = np.array([[*ship_pos(t0 + l_), t0 + l_] for l_ in leads])
    Pq = np.vstack(Pq + [Pq_enc])
    print(f"[win {wi}] t0={t0:.0f}s Hs={Hs:.2f} lam={lam_p:.0f}m "
          f"pdir={pdir:.0f}deg | solving MMSE ({len(z)} obs, "
          f"{len(Pq)} queries)...", flush=True)
    fq = mmse_bank_estimate(P, z, depth, Pq, sigma0=0.05 * Hs,
                            kp_hint=kp, seed=i0)
    maps = fq[:npx * len(leads)].reshape(len(leads), *Xq.shape)
    enc_pred = fq[npx * len(leads):]
    enc_gt = np.array([bilinear(Zfut[int(round(l_ / rec.dt))],
                                rec.gx, rec.gy, px, py)
                       for (px, py, _t), l_ in zip(Pq_enc, leads)])

    # ---- figure (diverging field maps + 2-series encounter panel)
    C_GT, C_PRED = '#5B7EC9', '#D08770'            # series: measured / predicted
    vmax = 1.4 * np.nanstd(Zfut[0]) * 2
    fig = plt.figure(figsize=(9.6, 6.4), dpi=85)
    gs = fig.add_gridspec(2, 2, height_ratios=[2.4, 1.0], hspace=0.34,
                          wspace=0.14, left=0.07, right=0.97, top=0.90,
                          bottom=0.09)
    axL = fig.add_subplot(gs[0, 0])
    axR = fig.add_subplot(gs[0, 1])
    axT = fig.add_subplot(gs[1, :])
    ext = [rec.gx.min(), rec.gx.max(), rec.gy.min(), rec.gy.max()]
    imL = axL.imshow(Zfut[0], origin='lower', extent=ext, cmap='RdBu_r',
                     vmin=-vmax, vmax=vmax, interpolation='bilinear')
    imR = axR.imshow(maps[0], origin='lower', extent=ext, cmap='RdBu_r',
                     vmin=-vmax, vmax=vmax, interpolation='bilinear')
    axL.set_title('Measured (stereo cameras)', fontsize=11)
    axR.set_title('Predicted (forward-fan LiDAR only)', fontsize=11)
    for ax in (axL, axR):
        ax.set_xlabel('x [m]', fontsize=8)
        ax.tick_params(labelsize=7)
    axL.set_ylabel('y [m]', fontsize=8)
    axR.set_yticklabels([])
    cb = fig.colorbar(imR, ax=(axL, axR), fraction=0.03, pad=0.015)
    cb.set_label('surface elevation η [m]', fontsize=8)
    cb.ax.tick_params(labelsize=7)
    # ship + future track overlays
    trk = np.array([ship_pos(t0 + l_) for l_ in leads])
    arts = []
    for ax in (axL, axR):
        ax.plot(trk[:, 0], trk[:, 1], '--', color='0.25', lw=1.0, alpha=0.8)
        arts.append(ax.plot([], [], marker=(3, 0, pdir + 90), ms=11,
                            color='0.1', ls='none')[0])
    # encounter panel: two series -> legend, no dual axis
    v = np.isfinite(enc_gt)
    axT.plot(leads[v], enc_gt[v], '-', color=C_GT, lw=2.0, label='measured')
    axT.plot(leads, enc_pred, '-', color=C_PRED, lw=2.0, label='predicted')
    axT.axhline(0, color='0.85', lw=0.8, zorder=0)
    cursor = axT.axvline(0, color='0.3', lw=1.0)
    cc = np.corrcoef(enc_pred[v], enc_gt[v])[0, 1]
    axT.set_xlabel('lead time [s]  (ship reaches this point at t0+lead)', fontsize=9)
    axT.set_ylabel('η [m]', fontsize=9)
    axT.set_title(f'Encounter forecast: eta at the ship future positions '
                  f'(corr {cc:+.2f})', fontsize=10)
    axT.legend(loc='upper right', fontsize=8, frameon=False, ncol=2)
    axT.tick_params(labelsize=8)
    for s_ in ('top', 'right'):
        axT.spines[s_].set_visible(False)
    axT.grid(axis='y', color='0.92', lw=0.7)
    sup = fig.suptitle('', fontsize=12)

    def update(fi):
        l_ = leads[fi]
        zt = Zfut[int(round(l_ / rec.dt))]
        imL.set_data(zt)
        imR.set_data(maps[fi])
        sp = ship_pos(t0 + l_)
        for art in arts:
            art.set_data([sp[0]], [sp[1]])
        cursor.set_xdata([l_])
        mc_v = np.isfinite(zt[::sq, ::sq])
        mc = np.corrcoef(maps[fi][mc_v], zt[::sq, ::sq][mc_v])[0, 1]
        sup.set_text(f'REAL sea ({a.record}) — causal forward LiDAR, {a.W:.0f}s memory'
                     f' | lead +{l_:.0f}s | map corr {mc:+.2f}')
        return [imL, imR, cursor, sup] + arts

    ani = animation.FuncAnimation(fig, update, frames=len(leads))
    os.makedirs('results/gifs', exist_ok=True)
    out = OUT.format(a.record + a.suffix)
    ani.save(out, writer=animation.PillowWriter(fps=2))
    print(f"saved {out} ({os.path.getsize(out) / 1e6:.1f} MB, "
          f"enc corr {cc:+.2f})")


if __name__ == '__main__':
    main()
