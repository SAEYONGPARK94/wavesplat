"""
d3_stream_demo_gif.py — GIF: the real-time streaming pipeline running at
the Candela design point (HOS M=3, 22 kt, BF4). Ship-centered moving
window: left = sea truth, right = the streaming filter's CURRENT field
estimate (updated every frame from the fan returns only), bottom = the
verified 2 s-ahead encounter forecast trace (prediction made 2 s ago vs
what the ship actually meets now).

Usage: python3 scripts/d3_stream_demo_gif.py
"""
import os
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import animation

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
os.chdir(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, 'scripts')

from wavesim.hos_simulator import HOSSimulator, HOSConfig
from d3_stream_seqmmse import make_bank, gamma_beamform, StreamMMSE

G = 9.81
OUT = 'results/gifs/falcon_stream_demo_22kt.gif'
U, Hs, Tp, DEPTH = 11.3, 1.2, 7.0, 50.0
DT, M = 0.1, 12
LEAD = 2.0
T_DEMO = 12.0                       # seconds of streaming shown
FRAME_EVERY = 3                     # GIF frame every 0.3 s


def main():
    kp = (2 * np.pi / Tp) ** 2 / G
    cfg = HOSConfig(Lx=800.0, Ly=400.0, Nx=160, Ny=128, depth=DEPTH, M=3)
    sim = HOSSimulator(cfg)
    sim.initialize_jonswap(Hs=Hs, Tp=Tp, direction_deg=0.0,
                           direction_spread_deg=25.0, seed=9100)
    rng = np.random.default_rng(500)
    t0 = 20.0

    def ship_pos(t):
        return np.array([540.0 - U * (t - t0), 200.0])
    h = np.array([-1.0, 0.0])

    def fan(sp):
        rr = rng.uniform(20.0, 100.0, M)
        aa = rng.uniform(-np.pi / 3, np.pi / 3, M)
        return sp[0] - rr * np.cos(aa), sp[1] + rr * np.sin(aa)

    while sim.t < 10.0 - 1e-9:
        sim.step(DT)
    Pw, zw = [], []
    t = 10.0
    while t < t0 - 1e-9:
        sp = ship_pos(t)
        xs, ys = fan(sp)
        z = sim.get_eta_at(np.mod(xs, cfg.Lx), np.mod(ys, cfg.Ly)) \
            + rng.normal(0, 0.02, M)
        Pw.append(np.column_stack([xs, ys, np.full(M, t)]))
        zw.append(z)
        sim.step(DT)
        t += DT
    Pw, zw = np.vstack(Pw), np.concatenate(zw)
    ii = np.random.default_rng(1).choice(len(zw), 600, False)
    kxb, kyb, om = make_bank(kp, DEPTH)
    gam, s2 = gamma_beamform(kxb, kyb, om, Pw[ii], zw[ii], 0.05 * Hs,
                             DEPTH)
    st = StreamMMSE(kxb, kyb, om, gam, s2, tau=5.0, dt=DT, prune=400)
    for k in range(len(zw) // M):
        st.step(Pw[k * M:(k + 1) * M], zw[k * M:(k + 1) * M])

    # box grid (ship-relative): x in [-40, +140] ahead, y +-80
    bx = np.arange(-40.0, 141.0, 5.0)
    by = np.arange(-80.0, 81.0, 5.0)
    BX, BY = np.meshgrid(bx, by)
    frames_data = []
    trace_t, trace_pred, trace_gt = [], [], []
    pending = {}
    k = 0
    while t < t0 + T_DEMO - 1e-9:
        sp = ship_pos(t)
        xs, ys = fan(sp)
        z = sim.get_eta_at(np.mod(xs, cfg.Lx), np.mod(ys, cfg.Ly)) \
            + rng.normal(0, 0.02, M)
        st.step(np.column_stack([xs, ys, np.full(M, t)]), z)
        # 2 s-ahead encounter prediction
        te = t + LEAD
        se = ship_pos(te)
        fpred, _ = st.predict(np.array([[se[0], se[1], te]]))
        pending[round(te, 3)] = float(fpred[0])
        key = round(t, 3)
        if key in pending:
            gt_now = float(sim.get_eta_at(np.mod(sp[:1], cfg.Lx),
                                          np.mod(sp[1:2], cfg.Ly))[0])
            trace_t.append(t - t0)
            trace_pred.append(pending.pop(key))
            trace_gt.append(gt_now)
        if k % FRAME_EVERY == 0:
            # ship-relative box, ahead = -x world = +ahead axis
            Xw = sp[0] - BX
            Yw = sp[1] + BY
            truth = sim.get_eta_at(np.mod(Xw.ravel(), cfg.Lx),
                                   np.mod(Yw.ravel(), cfg.Ly)
                                   ).reshape(BX.shape)
            Q = np.column_stack([Xw.ravel(), Yw.ravel(),
                                 np.full(BX.size, t)])
            est, _ = st.predict(Q)
            frames_data.append(dict(t=t - t0, truth=truth,
                                    est=est.reshape(BX.shape),
                                    trace=(list(trace_t),
                                           list(trace_pred),
                                           list(trace_gt))))
        sim.step(DT)
        t += DT
        k += 1

    # ---- render
    C_GT, C_PRED = '#5B7EC9', '#D08770'
    vmax = 1.3 * float(np.std(frames_data[0]['truth'])) * 2
    fig = plt.figure(figsize=(9.6, 6.4), dpi=85)
    gs = fig.add_gridspec(2, 2, height_ratios=[2.2, 1.0], hspace=0.36,
                          wspace=0.12, left=0.08, right=0.97, top=0.89,
                          bottom=0.09)
    axL = fig.add_subplot(gs[0, 0])
    axR = fig.add_subplot(gs[0, 1])
    axT = fig.add_subplot(gs[1, :])
    ext = [bx.min(), bx.max(), by.min(), by.max()]
    imL = axL.imshow(frames_data[0]['truth'], origin='lower', extent=ext,
                     cmap='RdBu_r', vmin=-vmax, vmax=vmax,
                     interpolation='bilinear')
    imR = axR.imshow(frames_data[0]['est'], origin='lower', extent=ext,
                     cmap='RdBu_r', vmin=-vmax, vmax=vmax,
                     interpolation='bilinear')
    axL.set_title('Sea truth (ship frame)', fontsize=11)
    axR.set_title('Streaming estimate (fan returns only)', fontsize=11)
    for ax in (axL, axR):
        ax.plot([0], [0], marker=(3, 0, -90), ms=12, color='0.1')
        ax.axvline(20, color='0.4', lw=0.7, ls=':')
        ax.axvline(100, color='0.4', lw=0.7, ls=':')
        ax.set_xlabel('ahead of ship [m]', fontsize=8)
        ax.tick_params(labelsize=7)
    axL.set_ylabel('across [m]', fontsize=8)
    axR.set_yticklabels([])
    cb = fig.colorbar(imR, ax=(axL, axR), fraction=0.03, pad=0.015)
    cb.set_label('η [m]', fontsize=8)
    cb.ax.tick_params(labelsize=7)
    lp, = axT.plot([], [], '-', color=C_PRED, lw=2.0,
                   label='predicted 2 s earlier')
    lg, = axT.plot([], [], '-', color=C_GT, lw=2.0,
                   label='actually encountered')
    axT.set_xlim(0, T_DEMO)
    axT.set_ylim(-1.1 * Hs, 1.1 * Hs)
    axT.axhline(0, color='0.85', lw=0.8, zorder=0)
    axT.set_xlabel('time [s]', fontsize=9)
    axT.set_ylabel('η at ship [m]', fontsize=9)
    axT.set_title('Verified encounter forecast (lead 2 s)', fontsize=10)
    axT.legend(loc='upper right', fontsize=8, frameon=False, ncol=2)
    axT.tick_params(labelsize=8)
    for s_ in ('top', 'right'):
        axT.spines[s_].set_visible(False)
    axT.grid(axis='y', color='0.92', lw=0.7)
    sup = fig.suptitle('', fontsize=12)

    def update(fi):
        d = frames_data[fi]
        imL.set_data(d['truth'])
        imR.set_data(d['est'])
        tt, tp, tg = d['trace']
        lp.set_data(tt, tp)
        lg.set_data(tt, tg)
        mc = float(np.corrcoef(d['est'].ravel(), d['truth'].ravel())[0, 1])
        sup.set_text(f'Real-time streaming @ 22 kt (HOS M=3, BF4) — '
                     f't = {d["t"]:.1f} s | ~47 ms/frame | '
                     f'map corr {mc:+.2f}')
        return [imL, imR, lp, lg, sup]

    ani = animation.FuncAnimation(fig, update, frames=len(frames_data))
    os.makedirs('results/gifs', exist_ok=True)
    ani.save(OUT, writer=animation.PillowWriter(fps=3))
    tt, tp, tg = frames_data[-1]['trace']
    cc = np.corrcoef(tp, tg)[0, 1]
    print(f"saved {OUT} ({os.path.getsize(OUT) / 1e6:.1f} MB, "
          f"verified 2s-lead corr {cc:+.2f}, {len(frames_data)} frames)")


if __name__ == '__main__':
    main()
