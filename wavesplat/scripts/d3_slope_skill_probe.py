"""
d3_slope_skill_probe.py — lever #4 step 1 (2026-07-06): the E2E FF gap
suspect is that the controller consumes SLOPE (d_eta/dx) and vertical
velocity (d_eta/dt) at the preview point, not elevation. Measure all
three skills of the current estimator at the design point (HOS M=3,
22 kt, BF4), at the ship's future positions (FF band 0-2 s + map lead
0-6 s), via central finite differences of both truth and prediction
(no estimator change needed).

If slope/velocity corr tracks elevation corr, the E2E bottleneck is
elsewhere (controller phase sensitivity); if it is much lower, a
slope-aware gamma tilt is the remedy (step 2).

Usage: python3 scripts/d3_slope_skill_probe.py [--n_seas 4]
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
from d3_sa_nomp import mmse_bank_estimate

G = 9.81
OUT = 'results/direction3/slope_skill_probe.json'
U, HS, TP, DEPTH = 11.3, 1.2, 7.0, 50.0
W = 10.0
LEADS = np.arange(0.0, 6.1, 0.5)
SCAN_DT = 0.5
M_SCAN = 12
NOISE = 0.02
SPINUP = 10.0
DT = 0.1
DX, DT_FD = 0.5, 0.05               # finite-difference steps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n_seas', type=int, default=4)
    ap.add_argument('--tilt', type=float, default=0.0)
    a = ap.parse_args()
    kp = (2 * np.pi / TP) ** 2 / G
    acc = {k: ([], []) for k in ('eta', 'slope', 'vel')}
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
            Pall.append(np.column_stack([xs, ys,
                                         np.full(M_SCAN, t_s)]))
            zall.append(z + rng.normal(0, NOISE, M_SCAN))
        P = np.vstack(Pall)
        z = np.concatenate(zall)

        # query stencil: center, x+-dx, t+-dt at each encounter point
        cen = np.array([[*ship_pos(t0 + l_), t0 + l_] for l_ in LEADS])
        Pq = []
        for c in cen:
            Pq += [c,
                   c + [DX, 0, 0], c - [DX, 0, 0],
                   c + [0, 0, DT_FD], c - [0, 0, DT_FD]]
        Pq = np.array(Pq)
        fq = mmse_bank_estimate(P, z, DEPTH, Pq, sigma0=0.05 * HS,
                                kp_hint=kp, seed=si,
                                gamma_tilt=a.tilt).reshape(-1, 5)
        # truth stencil (step sim to each lead time)
        gt = np.empty_like(fq)
        for i, (c, l_) in enumerate(zip(cen, LEADS)):
            while sim.t < t0 + l_ - 1e-9:
                sim.step(DT)
            xs = np.array([c[0], c[0] + DX, c[0] - DX, c[0], c[0]])
            ys = np.full(5, c[1])
            gt[i, :3] = sim.get_eta_at(np.mod(xs[:3], cfg.Lx),
                                       np.mod(ys[:3], cfg.Ly))
            # time stencil: evaluate at t +- DT_FD around this lead
            # (approximate: step DT_FD forward once, use backward pair)
            e0 = gt[i, 0]
            sim.step(DT_FD)
            ep = float(sim.get_eta_at(np.mod(xs[:1], cfg.Lx),
                                      np.mod(ys[:1], cfg.Ly))[0])
            gt[i, 3] = ep
            gt[i, 4] = 2 * e0 - ep          # backward mirror approx
        pe = fq[:, 0]
        ps = (fq[:, 1] - fq[:, 2]) / (2 * DX)
        pv = (fq[:, 3] - fq[:, 4]) / (2 * DT_FD)
        ge = gt[:, 0]
        gs = (gt[:, 1] - gt[:, 2]) / (2 * DX)
        gv = (gt[:, 3] - gt[:, 4]) / (2 * DT_FD)
        for k, p_, g_ in (('eta', pe, ge), ('slope', ps, gs),
                          ('vel', pv, gv)):
            acc[k][0].append(p_)
            acc[k][1].append(g_)
        print(f"sea {si} done", flush=True)

    res = {}
    print(f"\nskill at encounter points (leads 0-6 s, design pt):")
    for k in acc:
        p_ = np.concatenate(acc[k][0])
        g_ = np.concatenate(acc[k][1])
        cc = float(np.corrcoef(p_, g_)[0, 1])
        amp = float(np.std(p_) / (np.std(g_) + 1e-12))
        res[k] = dict(corr=cc, amp_ratio=amp)
        print(f"  {k:6s}: corr {cc:+.3f}  amp ratio {amp:.2f}")
    out_p = OUT.replace('.json', f'_tilt{a.tilt:g}.json') if a.tilt else OUT
    json.dump(res, open(out_p, 'w'), indent=1)
    print(f"saved {out_p if a.tilt else OUT}")


if __name__ == '__main__':
    main()
