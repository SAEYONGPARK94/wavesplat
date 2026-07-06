"""
d3_stokes2_probe.py — the last untested Q2 card (2026-07-06): bound-wave
(2nd-order Stokes) correction. Bound harmonics are phase-locked to the
free waves and carry high-k (=slope) content WITHOUT new observations:
predict eta_lin from the linear bank posterior, then add the self-
interaction Stokes term  eta2 = sum_b (k_b A_b^2 / 2) cos(2 psi_b)
built from the ESTIMATED components (A_b, phi_b from the posterior
coefficients — note MMSE shrinkage enters squared, the honest handicap).
Score slope/eta/vel at the encounter stencil against HOS M=3 truth
(which contains the real bound waves), design point, 4 seas.

Usage: python3 scripts/d3_stokes2_probe.py [--n_seas 4]
"""
import os
import sys
import json
import argparse

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
os.chdir(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, 'scripts')

from wavesim.hos_simulator import HOSSimulator, HOSConfig
from d3_stream_seqmmse import make_bank, gamma_beamform, design

G = 9.81
OUT = 'results/direction3/stokes2_probe.json'
U, HS, TP, DEPTH = 11.3, 1.2, 7.0, 50.0
W = 10.0
LEADS = np.arange(0.0, 6.1, 0.5)
SCAN_DT = 0.5
M_SCAN = 48
NOISE = 0.02
SPINUP = 10.0
DT = 0.1
DX, DT_FD = 0.5, 0.05


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n_seas', type=int, default=4)
    a = ap.parse_args()
    kp = (2 * np.pi / TP) ** 2 / G
    acc = {m: {k: ([], []) for k in ('eta', 'slope', 'vel')}
           for m in ('linear', 'stokes2')}
    for si in range(a.n_seas):
        cfg = HOSConfig(Lx=800.0, Ly=400.0, Nx=160, Ny=128,
                        depth=DEPTH, M=3)
        sim = HOSSimulator(cfg)
        sim.initialize_jonswap(Hs=HS, Tp=TP, direction_deg=0.0,
                               direction_spread_deg=25.0, seed=9000 + si)
        rng = np.random.default_rng(400 + si)
        t0 = SPINUP + W
        cx, cy = 500.0, 200.0

        def ship_pos(t):
            return np.array([cx - U * (t - t0), cy])

        Pall, zall = [], []
        for t_s in np.arange(t0 - W, t0 + 1e-9, SCAN_DT):
            while sim.t < t_s - 1e-9:
                sim.step(DT)
            sp = ship_pos(t_s)
            rr = rng.uniform(20.0, 100.0, M_SCAN)
            aa = rng.uniform(-np.pi / 3, np.pi / 3, M_SCAN)
            xs = sp[0] - rr * np.cos(aa)
            ys = sp[1] + rr * np.sin(aa)
            z = sim.get_eta_at(np.mod(xs, cfg.Lx), np.mod(ys, cfg.Ly))
            Pall.append(np.column_stack([xs, ys, np.full(M_SCAN, t_s)]))
            zall.append(z + rng.normal(0, NOISE, M_SCAN))
        P = np.vstack(Pall)
        z = np.concatenate(zall)

        # posterior coefficients on the bank (val-selected gamma, batch)
        kxb, kyb, om = make_bank(kp, DEPTH)
        ii = np.random.default_rng(1).choice(len(z), min(600, len(z)),
                                             False)
        gam, s2 = gamma_beamform(kxb, kyb, om, P[ii], z[ii], 0.05 * HS,
                                 DEPTH)
        Phi = design(kxb, kyb, om, P)
        K = (Phi * gam[None]) @ Phi.T + s2 * np.eye(len(z))
        w = gam * (Phi.T @ np.linalg.solve(K, z))
        B = len(kxb)
        a_c, c_c = w[:B], w[B:]
        Ab = np.hypot(a_c, c_c)
        ph = np.arctan2(-c_c, a_c)
        kb = np.hypot(kxb, kyb)

        def pred(Q, stokes):
            e = design(kxb, kyb, om, Q) @ w
            if stokes:
                ps = (kxb[None] * Q[:, 0:1] + kyb[None] * Q[:, 1:2]
                      - om[None] * Q[:, 2:3] + ph[None])
                e = e + ((kb * Ab ** 2 / 2.0)[None]
                         * np.cos(2 * ps)).sum(1)
            return e

        cen = np.array([[*ship_pos(t0 + l_), t0 + l_] for l_ in LEADS])
        Pq = []
        for c in cen:
            Pq += [c, c + [DX, 0, 0], c - [DX, 0, 0],
                   c + [0, 0, DT_FD], c - [0, 0, DT_FD]]
        Pq = np.array(Pq)
        gt = np.empty((len(cen), 5))
        for i, (c, l_) in enumerate(zip(cen, LEADS)):
            while sim.t < t0 + l_ - 1e-9:
                sim.step(DT)
            xs = np.array([c[0], c[0] + DX, c[0] - DX, c[0], c[0]])
            ys = np.full(5, c[1])
            gt[i, :3] = sim.get_eta_at(np.mod(xs[:3], cfg.Lx),
                                       np.mod(ys[:3], cfg.Ly))
            e0 = gt[i, 0]
            sim.step(DT_FD)
            ep = float(sim.get_eta_at(np.mod(xs[:1], cfg.Lx),
                                      np.mod(ys[:1], cfg.Ly))[0])
            gt[i, 3] = ep
            gt[i, 4] = 2 * e0 - ep
        for mth, st_ in (('linear', False), ('stokes2', True)):
            fq = pred(Pq, st_).reshape(-1, 5)
            pe = fq[:, 0]
            ps_ = (fq[:, 1] - fq[:, 2]) / (2 * DX)
            pv = (fq[:, 3] - fq[:, 4]) / (2 * DT_FD)
            ge = gt[:, 0]
            gs = (gt[:, 1] - gt[:, 2]) / (2 * DX)
            gv = (gt[:, 3] - gt[:, 4]) / (2 * DT_FD)
            for k, p_, g_ in (('eta', pe, ge), ('slope', ps_, gs),
                              ('vel', pv, gv)):
                acc[mth][k][0].append(p_)
                acc[mth][k][1].append(g_)
        print(f"sea {si} done", flush=True)

    res = {}
    for mth in acc:
        res[mth] = {}
        line = f"{mth:8s}:"
        for k in ('eta', 'slope', 'vel'):
            p_ = np.concatenate(acc[mth][k][0])
            g_ = np.concatenate(acc[mth][k][1])
            cc = float(np.corrcoef(p_, g_)[0, 1])
            amp = float(np.std(p_) / (np.std(g_) + 1e-12))
            res[mth][k] = dict(corr=cc, amp_ratio=amp)
            line += f"  {k} {cc:+.3f}({amp:.2f})"
        print(line)
    json.dump(res, open(OUT, 'w'), indent=1)
    print(f"saved {OUT}")


if __name__ == '__main__':
    main()
