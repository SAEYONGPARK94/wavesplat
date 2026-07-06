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
from d3_stream_seqmmse import make_bank, gamma_beamform, StreamMMSE

G = 9.81
OUT = 'results/direction3/stream_design_pipeline_{}.json'
LEADS = np.array([0.5, 1.0, 2.0, 4.0])
M_FRAME = 12
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
                   leads=LEADS.tolist(), by_lead=res,
                   lat_ms_median=float(np.median(lat)),
                   lat_ms_p95=float(np.percentile(lat, 95))),
              open(OUT.format('hos'), 'w'), indent=1)
    print(f"saved {OUT.format('hos')}")


def run_ys17(a):
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
    preds, lat = [], []
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
        k0 = 0
        for j in range(nfw):                       # replay memory as stream
            n_j = min(M_FRAME, len(zw) - k0)
            st.step(Pw[k0:k0 + n_j], zw[k0:k0 + n_j])
            k0 += n_j
        # live streaming + forecasting
        for j in range(nfw, nfw + n_live):
            t_j = float(rec.t[i0 - nfw + 1 + j])
            P, z = fan_obs(j, t_j)
            if P is None:
                continue
            t1 = time.perf_counter()
            st.step(P, z)
            tq = t_j + LEADS
            spq = np.array([ship_pos(tt) for tt in tq])
            f, _ = st.predict(np.column_stack([spq, tq]))
            lat.append(time.perf_counter() - t1)
            for l_, tt, (xq, yq), fv in zip(LEADS, tq, spq, f):
                jj = j + int(round(l_ / rec.dt))
                gt = bilinear(Zall[jj], rec.gx, rec.gy, xq, yq)
                preds.append((l_, fv, gt))
        print(f"win {wi} done ({len(preds)} preds)", flush=True)
    lat = np.array(lat) * 1e3
    res = score(preds, 'YS17 real streaming')
    print(f"latency/frame (update+forecast): median {np.median(lat):.1f} ms"
          f", p95 {np.percentile(lat, 95):.1f} ms")
    json.dump(dict(mode='ys17', U=U, leads=LEADS.tolist(), by_lead=res,
                   lat_ms_median=float(np.median(lat)),
                   lat_ms_p95=float(np.percentile(lat, 95))),
              open(OUT.format('ys17'), 'w'), indent=1)
    print(f"saved {OUT.format('ys17')}")


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('mode', choices=['hos', 'ys17'])
    ap.add_argument('--n_seas', type=int, default=2)
    a = ap.parse_args()
    run_hos(a) if a.mode == 'hos' else run_ys17(a)
