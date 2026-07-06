"""
d3_falcon_sim_gif.py — GIFs of the SIMULATOR-based forward-fan encounter
validation, same layout as the real-data GIF (left GT / right predicted
from fan LiDAR only / bottom encounter series).

Cases (a few, per user request):
  linear_U3      linear superposition sea (200-comp JONSWAP), U=3 m/s
  hos_U3         HOS M=3 nonlinear sea, U=3 m/s
  hos_design22kt HOS M=3, Candela design point (U=11.3 m/s, BF4)

Usage: python3 scripts/d3_falcon_sim_gif.py --case hos_design22kt
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

from wavesim.wave_generator import jonswap_components
from wavesim.hos_simulator import HOSSimulator, HOSConfig
from d3_sa_fim_identifiability import omega_of
from d3_sa_nomp import mmse_bank_estimate

G = 9.81
OUT = 'results/gifs/falcon_sim_{}.gif'

CASES = {
    'linear_U3': dict(kind='linear', U=3.0, Hs=2.0, Tp=8.0, W=20.0,
                      lead_max=20.0, lead_step=1.0, label='Linear sea'),
    'hos_U3': dict(kind='hos', U=3.0, Hs=2.0, Tp=8.0, W=20.0,
                   lead_max=20.0, lead_step=1.0, Lx=500.0, Nx=128,
                   label='HOS M=3 nonlinear sea'),
    'hos_design22kt': dict(kind='hos', U=11.3, Hs=1.2, Tp=7.0, W=10.0,
                           lead_max=10.0, lead_step=0.5, Lx=800.0, Nx=160,
                           label='HOS M=3, Candela design pt (22 kt, BF4)'),
}
SCAN_DT = 0.5
M_PER_SCAN = 12
NOISE = 0.02
DEPTH = 50.0
SPINUP = 10.0
DT_HOS = 0.1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--case', default='hos_U3', choices=list(CASES))
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()
    c = CASES[a.case]
    U, Hs, Tp, W = c['U'], c['Hs'], c['Tp'], c['W']
    leads = np.arange(0.0, c['lead_max'] + 1e-9, c['lead_step'])
    kp = (2 * np.pi / Tp) ** 2 / G
    rng = np.random.default_rng(400 + a.seed)

    if c['kind'] == 'linear':
        comps = jonswap_components(Hs=Hs, Tp=Tp, n_components=200,
                                   depth=DEPTH, direction_spread_deg=25.0,
                                   seed=9000 + a.seed)
        kx = np.array([x.kx for x in comps])
        ky = np.array([x.ky for x in comps])
        A = np.array([x.A for x in comps])
        ph = np.array([x.phi for x in comps])
        om = omega_of(kx, ky, DEPTH)

        def eta_at(x, y, t):
            ps = kx[None] * np.atleast_1d(x)[:, None] \
                + ky[None] * np.atleast_1d(y)[:, None] \
                - om[None] * np.atleast_1d(t)[:, None] + ph[None]
            return (A[None] * np.cos(ps)).sum(1)
        t0 = W
        cx, cy = 0.0, 0.0
        wrap = None
    else:
        cfg = HOSConfig(Lx=c['Lx'], Ly=400.0, Nx=c['Nx'], Ny=128,
                        depth=DEPTH, M=3)
        sim = HOSSimulator(cfg)
        sim.initialize_jonswap(Hs=Hs, Tp=Tp, direction_deg=0.0,
                               direction_spread_deg=25.0, seed=9000 + a.seed)
        t0 = SPINUP + W
        cx, cy = c['Lx'] / 2, 200.0
        wrap = (c['Lx'], 400.0)

        def eta_at(x, y, t):          # requires sim.t == t (stepped outside)
            assert abs(sim.t - t) < 1e-6, (sim.t, t)
            return sim.get_eta_at(np.mod(np.atleast_1d(x), wrap[0]),
                                  np.mod(np.atleast_1d(y), wrap[1]))

    def ship_pos(t):                   # heading -x into +x-propagating waves
        return np.array([cx - U * (t - t0), cy])

    # ---- causal fan scans (step HOS as we go)
    scan_ts = np.arange(t0 - W, t0 + 1e-9, SCAN_DT)
    P_all, z_all = [], []
    for t_s in scan_ts:
        if c['kind'] == 'hos':
            while sim.t < t_s - 1e-9:
                sim.step(DT_HOS)
        sp = ship_pos(t_s)
        rr = rng.uniform(20.0, 100.0, M_PER_SCAN)
        aa = rng.uniform(-np.pi / 3, np.pi / 3, M_PER_SCAN)
        xs = sp[0] - rr * np.cos(aa)
        ys = sp[1] + rr * np.sin(aa)
        z = eta_at(xs, ys, sim.t if c['kind'] == 'hos' else t_s)
        P_all.append(np.column_stack([xs, ys, np.full(M_PER_SCAN, t_s)]))
        z_all.append(z + rng.normal(0, NOISE, M_PER_SCAN))
    P = np.vstack(P_all)
    z = np.concatenate(z_all)

    # ---- query set: map box x leads + encounter points
    x_lo = ship_pos(t0 + c['lead_max'])[0] - 30.0
    x_hi = ship_pos(t0)[0] + 110.0
    gxq = np.arange(x_lo, x_hi, 4.0)
    gyq = np.arange(cy - 90.0, cy + 90.0, 4.0)
    Xq, Yq = np.meshgrid(gxq, gyq)
    npx = Xq.size
    Pq = [np.column_stack([Xq.ravel(), Yq.ravel(),
                           np.full(npx, t0 + l_)]) for l_ in leads]
    Pq_enc = np.array([[*ship_pos(t0 + l_), t0 + l_] for l_ in leads])
    Pq = np.vstack(Pq + [Pq_enc])
    print(f"[{a.case}] solving MMSE ({len(z)} obs, {len(Pq)} queries)...",
          flush=True)
    fq = mmse_bank_estimate(P, z, DEPTH, Pq, sigma0=0.05 * Hs,
                            kp_hint=kp, seed=a.seed)
    maps_pred = fq[:npx * len(leads)].reshape(len(leads), *Xq.shape)
    enc_pred = fq[npx * len(leads):]

    # ---- GT maps + encounter (step HOS forward through the leads)
    maps_gt, enc_gt = [], []
    for l_ in leads:
        t_ = t0 + l_
        if c['kind'] == 'hos':
            while sim.t < t_ - 1e-9:
                sim.step(DT_HOS)
        tt = sim.t if c['kind'] == 'hos' else t_
        maps_gt.append(eta_at(Xq.ravel(), Yq.ravel(), tt).reshape(Xq.shape))
        sp = ship_pos(t_)
        enc_gt.append(float(eta_at(sp[:1], sp[1:2], tt)[0]))
    maps_gt = np.array(maps_gt)
    enc_gt = np.array(enc_gt)

    # ---- figure (same system as the real-data GIF)
    C_GT, C_PRED = '#5B7EC9', '#D08770'
    vmax = 1.4 * float(np.std(maps_gt[0])) * 2
    fig = plt.figure(figsize=(9.6, 6.4), dpi=85)
    gs = fig.add_gridspec(2, 2, height_ratios=[2.4, 1.0], hspace=0.34,
                          wspace=0.14, left=0.07, right=0.97, top=0.90,
                          bottom=0.09)
    axL = fig.add_subplot(gs[0, 0])
    axR = fig.add_subplot(gs[0, 1])
    axT = fig.add_subplot(gs[1, :])
    ext = [gxq.min(), gxq.max(), gyq.min(), gyq.max()]
    imL = axL.imshow(maps_gt[0], origin='lower', extent=ext, cmap='RdBu_r',
                     vmin=-vmax, vmax=vmax, interpolation='bilinear')
    imR = axR.imshow(maps_pred[0], origin='lower', extent=ext, cmap='RdBu_r',
                     vmin=-vmax, vmax=vmax, interpolation='bilinear')
    axL.set_title('Simulator truth', fontsize=11)
    axR.set_title('Predicted (forward-fan LiDAR only)', fontsize=11)
    for ax in (axL, axR):
        ax.set_xlabel('x [m]', fontsize=8)
        ax.tick_params(labelsize=7)
    axL.set_ylabel('y [m]', fontsize=8)
    axR.set_yticklabels([])
    cb = fig.colorbar(imR, ax=(axL, axR), fraction=0.03, pad=0.015)
    cb.set_label('surface elevation η [m]', fontsize=8)
    cb.ax.tick_params(labelsize=7)
    trk = np.array([ship_pos(t0 + l_) for l_ in leads])
    arts = []
    for ax in (axL, axR):
        ax.plot(trk[:, 0], trk[:, 1], '--', color='0.25', lw=1.0, alpha=0.8)
        arts.append(ax.plot([], [], marker=(3, 0, 90), ms=11,
                            color='0.1', ls='none')[0])
    axT.plot(leads, enc_gt, '-', color=C_GT, lw=2.0, label='truth')
    axT.plot(leads, enc_pred, '-', color=C_PRED, lw=2.0, label='predicted')
    axT.axhline(0, color='0.85', lw=0.8, zorder=0)
    cursor = axT.axvline(0, color='0.3', lw=1.0)
    cc = float(np.corrcoef(enc_pred, enc_gt)[0, 1])
    axT.set_xlabel('lead time [s]  (ship reaches this point at t0+lead)',
                   fontsize=9)
    axT.set_ylabel('η [m]', fontsize=9)
    axT.set_title(f'Encounter forecast at ship future positions '
                  f'(corr {cc:+.2f})', fontsize=10)
    axT.legend(loc='upper right', fontsize=8, frameon=False, ncol=2)
    axT.tick_params(labelsize=8)
    for s_ in ('top', 'right'):
        axT.spines[s_].set_visible(False)
    axT.grid(axis='y', color='0.92', lw=0.7)
    sup = fig.suptitle('', fontsize=12)

    def update(fi):
        l_ = leads[fi]
        imL.set_data(maps_gt[fi])
        imR.set_data(maps_pred[fi])
        sp = ship_pos(t0 + l_)
        for art in arts:
            art.set_data([sp[0]], [sp[1]])
        cursor.set_xdata([l_])
        mc = float(np.corrcoef(maps_pred[fi].ravel(),
                               maps_gt[fi].ravel())[0, 1])
        sup.set_text(f"{c['label']} — U={U} m/s, {W:.0f}s memory | "
                     f"lead +{l_:.1f}s | map corr {mc:+.2f}")
        return [imL, imR, cursor, sup] + arts

    ani = animation.FuncAnimation(fig, update, frames=len(leads))
    os.makedirs('results/gifs', exist_ok=True)
    out = OUT.format(a.case)
    ani.save(out, writer=animation.PillowWriter(fps=2))
    print(f"saved {out} ({os.path.getsize(out) / 1e6:.1f} MB, "
          f"enc corr {cc:+.2f})")


if __name__ == '__main__':
    main()
