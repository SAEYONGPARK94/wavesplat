"""
d3_stream_design_pipeline.py — the full real-time pipeline, assembled
(2026-07-06 evening, user-directed follow-up #3): per-frame streaming
filter (d3_stream_seqmmse) x the given-sensor forward-fan operator, at
(a) the Candela design point (HOS M=3, 22 kt, BF4) and (b) YS17 real
stereo. Every 0.1 s frame: ingest the fan returns (RLS/Kalman update,
latency measured) and emit the encounter forecast at the ship's future
positions (FF band 0.5-4 s). Scored against sim truth / stereo surface.

This is the control-facing demo number: "obs stream -> <=100 ms update
-> FF-band encounter forecast corr X", streaming (no per-window batch
re-solve).

Usage: python3 scripts/d3_stream_design_pipeline.py hos    [--n_seas 2]
       python3 scripts/d3_stream_design_pipeline.py ys17
"""
import os
import sys
import json
import time
import argparse

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
os.chdir(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, 'scripts')

from wavesim.hos_simulator import HOSSimulator, HOSConfig
from realdata_analysis import directional_spread
from d3_falcon_real_operator import RECORDS, bilinear, resolve_direction
from d3_sa_multirecord import MRecord
from d3_stream_seqmmse import make_bank, gamma_beamform, StreamMMSE, design

G = 9.81
OUT = 'results/direction3/stream_design_pipeline_{}.json'
LEADS = np.array([0.5, 1.0, 2.0, 4.0])
M_FRAME = 12                        # overridden by --m_frame
NOISE = 0.02


def fan_sample_synth(rng, sp, h):
    rr = rng.uniform(20.0, 100.0, M_FRAME)
    aa = rng.uniform(-np.pi / 3, np.pi / 3, M_FRAME)
    xs = sp[0] + rr * (h[0] * np.cos(aa) - h[1] * np.sin(aa))
    ys = sp[1] + rr * (h[1] * np.cos(aa) + h[0] * np.sin(aa))
    return xs, ys


def score(preds, tag):
    print(f"\n===== {tag}: streaming FF-band encounter forecast =====")
    res = {}
    for l_ in LEADS:
        pp = np.array([p for (lead, p, g) in preds if lead == l_])
        gg = np.array([g for (lead, p, g) in preds if lead == l_])
        v = np.isfinite(gg)
        cc = float(np.corrcoef(pp[v], gg[v])[0, 1])
        res[f'{l_:g}'] = dict(corr=cc, n=int(v.sum()))
        print(f"  lead {l_:.1f}s: corr {cc:+.3f} (n={v.sum()})")
    return res


def run_hos(a):
    U, Hs, Tp = 11.3, 1.2, 7.0                    # design point
    kp = (2 * np.pi / Tp) ** 2 / G
    dt = 0.1
    preds, lat = [], []
    for si in range(a.n_seas):
        cfg = HOSConfig(Lx=800.0, Ly=400.0, Nx=160, Ny=128, depth=50.0, M=3)
        sim = HOSSimulator(cfg)
        sim.initialize_jonswap(Hs=Hs, Tp=Tp, direction_deg=0.0,
                               direction_spread_deg=25.0, seed=9100 + si)
        rng = np.random.default_rng(500 + si)
        t0 = 20.0                                  # scoring starts
        t_end = 40.0

        def ship_pos(t):
            return np.array([500.0 - U * (t - t0), 200.0])
        h = np.array([-1.0, 0.0])

        # warmup 10-20 s: collect obs for the slow-loop prior
        Pw, zw = [], []
        while sim.t < 10.0 - 1e-9:
            sim.step(dt)
        t = 10.0
        while t < t0 - 1e-9:
            sp = ship_pos(t)
            xs, ys = fan_sample_synth(rng, sp, h)
            z = sim.get_eta_at(np.mod(xs, cfg.Lx), np.mod(ys, cfg.Ly)) \
                + rng.normal(0, NOISE, M_FRAME)
            Pw.append(np.column_stack([xs, ys, np.full(M_FRAME, t)]))
            zw.append(z)
            sim.step(dt)
            t += dt
        Pw, zw = np.vstack(Pw), np.concatenate(zw)
        ii = np.random.default_rng(1).choice(len(zw), min(600, len(zw)), False)
        kxb, kyb, om = make_bank(kp, 50.0)
        gam, s2 = gamma_beamform(kxb, kyb, om, Pw[ii], zw[ii],
                                 0.05 * Hs, 50.0)
        st = StreamMMSE(kxb, kyb, om, gam, s2, tau=5.0, dt=dt, prune=400)
        for Pf, zf in ((Pw[k * M_FRAME:(k + 1) * M_FRAME],
                        zw[k * M_FRAME:(k + 1) * M_FRAME])
                       for k in range(len(zw) // M_FRAME)):
            st.step(Pf, zf)

        # streaming loop with per-frame forecast
        pending = []                              # (t_target, x, y, pred)
        while t < t_end - 1e-9:
            sp = ship_pos(t)
            xs, ys = fan_sample_synth(rng, sp, h)
            z = sim.get_eta_at(np.mod(xs, cfg.Lx), np.mod(ys, cfg.Ly)) \
                + rng.normal(0, NOISE, M_FRAME)
            t1 = time.perf_counter()
            st.step(np.column_stack([xs, ys, np.full(M_FRAME, t)]), z)
            tq = t + LEADS
            spq = np.array([ship_pos(tt) for tt in tq])
            Q = np.column_stack([spq, tq])
            f, _ = st.predict(Q)
            lat.append(time.perf_counter() - t1)
            for l_, tt, (xq, yq), fv in zip(LEADS, tq, spq, f):
                pending.append([tt, xq, yq, l_, fv])
            sim.step(dt)
            t += dt
            done = [p for p in pending if p[0] <= t + 1e-9]
            pending = [p for p in pending if p[0] > t + 1e-9]
            for tt, xq, yq, l_, fv in done:
                gt = float(sim.get_eta_at(np.mod([xq], cfg.Lx),
                                          np.mod([yq], cfg.Ly))[0])
                preds.append((l_, fv, gt))
        print(f"sea {si} done ({len(preds)} scored preds)", flush=True)
    lat = np.array(lat) * 1e3
    res = score(preds, 'HOS design 22kt')
    print(f"latency/frame (update+forecast): median {np.median(lat):.1f} ms"
          f", p95 {np.percentile(lat, 95):.1f} ms")
    json.dump(dict(mode='hos_design22kt', U=U, Hs=Hs, Tp=Tp,
                   m_frame=M_FRAME,
                   leads=LEADS.tolist(), by_lead=res,
                   lat_ms_median=float(np.median(lat)),
                   lat_ms_p95=float(np.percentile(lat, 95))),
              open(OUT.format('hos' + a.tag2), 'w'), indent=1)
    print(f"saved {OUT.format('hos' + a.tag2)}")


def run_ys17(a):
    refresh = a.gamma_refresh
    cfg = RECORDS['YS17']
    rec = MRecord(cfg['path'], stride=cfg['stride'])
    depth, U = cfg['depth'], cfg['U']
    W_max = 40.0
    nfw = int(round(W_max / rec.dt))
    lead_pad = int(round(max(LEADS) / rec.dt)) + 1
    cx = 0.5 * (rec.gx.min() + rec.gx.max())
    cy = 0.5 * (rec.gy.min() + rec.gy.max())
    Xg, Yg = np.meshgrid(rec.gx, rec.gy)
    i0s = np.linspace(nfw + 1, rec.T_all - lead_pad - 202, 6).astype(int)
    preds, lat, win_stats = [], [], []
    for wi, i0 in enumerate(i0s):
        i0 = int(i0)
        rng = np.random.default_rng(700 + wi)
        # stream = memory window + 20 s of live scoring frames
        n_live = 200
        Zall = rec.window(i0 - nfw + 1, nfw + n_live + lead_pad)
        t0 = float(rec.t[i0])
        Hs = 4.0 * float(np.nanstd(Zall[:nfw]))
        spread, lam_p, pdir_ax, kp = directional_spread(
            np.nan_to_num(Zall[nfw - 1]), rec.dx, rec.dy)
        om_p = np.sqrt(G * kp * np.tanh(kp * depth))
        pdir, _ = resolve_direction(Zall[:nfw], rec, kp, pdir_ax, om_p / kp)
        h = -np.array([np.cos(np.radians(pdir)), np.sin(np.radians(pdir))])
        T_tr = W_max + 20.0 + max(LEADS)

        def ship_pos(t):
            return (np.array([cx, cy])
                    + U * (t - (t0 - W_max) - T_tr / 2) * h)

        def fan_obs(j, t_j):
            sp = ship_pos(t_j)
            RX, RY = Xg - sp[0], Yg - sp[1]
            R = np.hypot(RX, RY)
            ca = (RX * h[0] + RY * h[1]) / np.maximum(R, 1e-9)
            m = ((R >= cfg['r_fan'][0]) & (R <= cfg['r_fan'][1])
                 & (ca >= 0.5) & np.isfinite(Zall[j]))
            idx = np.flatnonzero(m)
            if len(idx) == 0:
                return None, None
            pick = rng.choice(idx, min(M_FRAME, len(idx)), replace=False)
            iy, ix = np.unravel_index(pick, Zall[j].shape)
            P = np.column_stack([rec.gx[ix], rec.gy[iy],
                                 np.full(len(pick), t_j)])
            return P, Zall[j][iy, ix] + rng.normal(0, NOISE, len(pick))

        # slow-loop prior from the memory window
        Pw, zw = [], []
        for j in range(nfw):
            P, z = fan_obs(j, float(rec.t[i0 - nfw + 1 + j]))
            if P is not None:
                Pw.append(P)
                zw.append(z)
        Pw, zw = np.vstack(Pw), np.concatenate(zw)
        ii = np.random.default_rng(1).choice(len(zw), min(600, len(zw)),
                                             False)
        kxb, kyb, om = make_bank(kp, depth)
        gam, s2 = gamma_beamform(kxb, kyb, om, Pw[ii], zw[ii],
                                 0.05 * Hs, depth)
        st = StreamMMSE(kxb, kyb, om, gam, s2, tau=20.0, dt=rec.dt,
                        prune=400)
        kx_f, ky_f, om_f = kxb, kyb, om            # full bank (for rebuild)
        k0 = 0
        for j in range(nfw):                       # replay memory as stream
            n_j = min(M_FRAME, len(zw) - k0)
            st.step(Pw[k0:k0 + n_j], zw[k0:k0 + n_j])
            k0 += n_j
        # live streaming + forecasting (+ online gamma + NIS gate)
        buf_P, buf_z = [Pw], [zw]
        nis_win, w_preds = [], []
        t_last_ref = t0
        gate_vals, gate_hist, t_last_gate, gate_ms = [], [], t0, []
        for j in range(nfw, nfw + n_live):
            t_j = float(rec.t[i0 - nfw + 1 + j])
            P, z = fan_obs(j, t_j)
            if P is None:
                continue
            t1 = time.perf_counter()
            nis = st.step(P, z)
            tq = t_j + LEADS
            spq = np.array([ship_pos(tt) for tt in tq])
            f, _ = st.predict(np.column_stack([spq, tq]))
            lat.append(time.perf_counter() - t1)
            nis_win.append(nis)
            buf_P.append(P)
            buf_z.append(z)
            if a.gate and t_j - t_last_gate >= 5.0:
                tg0 = time.perf_counter()
                Pb = np.vstack(buf_P)
                zb = np.concatenate(buf_z)
                mrec = Pb[:, 2] >= t_j - 20.0
                Pb, zb = Pb[mrec], zb[mrec]
                if len(zb) > 480:
                    ib = np.random.default_rng(4).choice(len(zb), 480,
                                                         False)
                    Pb, zb = Pb[ib], zb[ib]
                nv = max(8, len(zb) // 4)
                iv = np.random.default_rng(5).choice(len(zb), nv, False)
                vm = np.zeros(len(zb), bool)
                vm[iv] = True
                Phi_tr = design(st.kxb, st.kyb, st.om, Pb[~vm])
                Phi_v = design(st.kxb, st.kyb, st.om, Pb[vm])
                Kb = (Phi_tr * st.gam[None]) @ Phi_tr.T \
                    + st.s2 * np.eye((~vm).sum())
                fv = (Phi_v * st.gam[None]) @ (
                    Phi_tr.T @ np.linalg.solve(Kb, zb[~vm]))
                vr = float(np.sqrt(np.mean((fv - zb[vm]) ** 2))
                           / (np.std(zb[vm]) + 1e-12))
                gate_hist.append(vr)
                med_g = float(np.median(gate_hist))
                mad_g = float(np.median(np.abs(np.array(gate_hist)
                                               - med_g))) + 1e-9
                gate_vals.append((vr - med_g) / mad_g
                                 if len(gate_hist) >= 4 else 0.0)
                gate_ms.append((time.perf_counter() - tg0) * 1e3)
                t_last_gate = t_j
            if refresh > 0 and t_j - t_last_ref >= refresh:
                Pb = np.vstack(buf_P)
                zb = np.concatenate(buf_z)
                mrec = Pb[:, 2] >= t_j - 20.0      # trailing 20 s buffer
                Pb, zb = Pb[mrec], zb[mrec]
                if len(zb) > 600:
                    ib = np.random.default_rng(3).choice(len(zb), 600,
                                                         False)
                    Pb, zb = Pb[ib], zb[ib]
                gam_n, s2_n = gamma_beamform(kx_f, ky_f, om_f, Pb, zb,
                                             0.05 * Hs, depth)
                st.s2 = s2_n
                st.rebuild(Pb, zb, gam_n, kx_f, ky_f, om_f, prune=400)
                t_last_ref = t_j
            for l_, tt, (xq, yq), fv in zip(LEADS, tq, spq, f):
                jj = j + int(round(l_ / rec.dt))
                gt = bilinear(Zall[jj], rec.gx, rec.gy, xq, yq)
                preds.append((l_, fv, gt))
                w_preds.append((l_, fv, gt))
        pp = np.array([p_ for (_l, p_, g_) in w_preds])
        gg = np.array([g_ for (_l, p_, g_) in w_preds])
        vv = np.isfinite(gg)
        c_w = float(np.corrcoef(pp[vv], gg[vv])[0, 1])
        gz = float(np.mean(gate_vals)) if gate_vals else None
        win_stats.append(dict(wi=wi, corr=c_w,
                              nis=float(np.mean(nis_win)), gate=gz,
                              gate_ms=(float(np.median(gate_ms))
                                       if gate_ms else None)))
        print(f"win {wi} done (corr={c_w:+.2f}, "
              f"mean NIS={np.mean(nis_win):.2f}"
              + (f", gate_z={gz:+.2f}, gate {np.median(gate_ms):.0f}ms"
                 if gz is not None else "") + ")", flush=True)
    lat = np.array(lat) * 1e3
    res = score(preds, 'YS17 real streaming')
    print(f"latency/frame (update+forecast): median {np.median(lat):.1f} ms"
          f", p95 {np.percentile(lat, 95):.1f} ms")
    if len(win_stats) > 2:
        from scipy.stats import spearmanr
        rho, pv = spearmanr([w_['nis'] for w_ in win_stats],
                            [w_['corr'] for w_ in win_stats])
        print(f"NIS gate: spearman(mean NIS, window corr) = {rho:+.2f} "
              f"(p={pv:.3f}, n={len(win_stats)})")
        if win_stats[0].get('gate') is not None:
            rho2, pv2 = spearmanr([w_['gate'] for w_ in win_stats],
                                  [w_['corr'] for w_ in win_stats])
            print(f"INLINE val-res gate: spearman = {rho2:+.2f} "
                  f"(p={pv2:.3f}, n={len(win_stats)})")
    tag = ('ys17' if refresh <= 0 else f'ys17_ref{refresh:g}') + a.tag2
    json.dump(dict(mode=tag, U=U, leads=LEADS.tolist(), by_lead=res,
                   gamma_refresh=refresh, win_stats=win_stats,
                   lat_ms_median=float(np.median(lat)),
                   lat_ms_p95=float(np.percentile(lat, 95))),
              open(OUT.format(tag), 'w'), indent=1)
    print(f"saved {OUT.format(tag)}")


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('mode', choices=['hos', 'ys17'])
    ap.add_argument('--n_seas', type=int, default=2)
    ap.add_argument('--gamma_refresh', type=float, default=0.0,
                    help='slow-loop gamma refresh period [s]; 0=off')
    ap.add_argument('--m_frame', type=int, default=12,
                    help='fan returns per frame (episodes: median 48)')
    ap.add_argument('--gate', action='store_true',
                    help='inline rolling val-residual gate (5 s period, '
                         'causal running median/MAD normalization)')
    ap.add_argument('--tag2', default='',
                    help='output filename suffix')
    a = ap.parse_args()
    globals()['M_FRAME'] = a.m_frame
    run_hos(a) if a.mode == 'hos' else run_ys17(a)
