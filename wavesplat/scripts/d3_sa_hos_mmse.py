"""
d3_sa_hos_mmse.py — HOS nonlinear validation of the 2026-07-05 winner
(spectrum-weighted linear MMSE) under the SA moving-platform protocol.

Standing directive: any headline must reproduce on nonlinear (HOS) seas.
The MMSE estimator is a LINEAR-model method (dispersion-constrained free
modes) — bound phases/harmonics of a 3rd-order sea violate the dispersion
constraint, so quantify the degradation vs RELAX (also linear-model, so
the comparison isolates estimator structure, not model class).

Protocol mirrors d3_sa_achievability_gap: patch 2 lam x 2 lam, T = 10 Tp,
400 obs, fp = 0.18 lam, fixed vs ping-pong track at {0.43, 1.0} c_p,
scoring corr on common-unobserved pixels at t_star. Sea = HOSSimulator
(M=3 by default; --M 1 gives the linear control), 20 s spin-up.

Usage: python3 scripts/d3_sa_hos_mmse.py [--n_seas 6] [--M 3]
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
from estimator_improved import iterative_relax_cv
from d3_sa_nomp import mmse_bank_estimate
from d3_sa_achievability_gap import (dispersion_lam, sample_obs,
                                     masks_common_unobs, corr, hashabs,
                                     N_OBS, NOISE_FRAC, TP, HS, DEPTH, G)

OUT = 'results/direction3/sa_hos_mmse.json'
SPINUP = 20.0
DT = 0.15


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n_seas', type=int, default=6)
    ap.add_argument('--M', type=int, default=3)
    a = ap.parse_args()

    lam = dispersion_lam(TP, DEPTH)
    cp = lam / TP
    T = 10 * TP
    fp = 0.18 * lam
    r_fp = fp / 2
    side = 2.0 * lam
    # HOS periodic world 4 lam x 4 lam; scoring patch = central 2 lam
    L_world = 4.0 * lam
    off = (L_world - side) / 2
    gx = np.linspace(0, side, 96)
    gy = np.linspace(0, side, 96)
    Xg, Yg = np.meshgrid(gx, gy)
    mu = masks_common_unobs(gx, gy, r_fp, 0.6 * fp)
    Xs, Ys = Xg[mu], Yg[mu]
    if len(Xs) > 1500:
        take = np.random.default_rng(0).choice(len(Xs), 1500, replace=False)
        Xs, Ys = Xs[take], Ys[take]
    t_star = T
    Pq = np.column_stack([Xs, Ys, np.full(len(Xs), t_star)])
    print(f"HOS M={a.M} world {L_world:.0f} m, patch {side:.0f} m, "
          f"T={T:.0f} s, score px {len(Xs)}")

    conds = [('fixed', None), ('track043', 0.43), ('track100', 1.0)]
    res = {c: {} for c, _ in conds}
    for si in range(a.n_seas):
        tic = time.time()
        sim = HOSSimulator(HOSConfig(Lx=L_world, Ly=L_world, Nx=128, Ny=128,
                                     depth=DEPTH, M=a.M))
        sim.initialize_jonswap(Hs=HS, Tp=TP, seed=4000 + si)
        sim.run_to(SPINUP, DT)
        # per-condition obs geometries (patch frame), then one time sweep
        Ps = {}
        for cname, ur in conds:
            rng = np.random.default_rng(9000 + 17 * si + hashabs(cname))
            U = (ur or 0) * cp
            P = sample_obs('fixed' if cname == 'fixed' else 'moving', U,
                           lam, cp, T, gx, gy, r_fp, rng)
            Ps[cname] = (P, rng)
        # advance once through time, sampling every condition's point
        order = np.argsort(Ps['fixed'][0][:, 2])   # same tt for all conds
        zs = {c: np.empty(N_OBS) for c, _ in conds}
        for i in order:
            t_i = Ps['fixed'][0][i, 2]
            sim.run_to(SPINUP + t_i, DT)
            for cname, _ in conds:
                p = Ps[cname][0][i]
                zs[cname][i] = float(sim.get_eta_at(
                    np.array([p[0] + off]), np.array([p[1] + off]))[0])
        sim.run_to(SPINUP + t_star, DT)
        gt_u = sim.get_eta_at(Xs + off, Ys + off)
        gt_grid = sim.get_eta_at((Xg + off).ravel(),
                                 (Yg + off).ravel()).reshape(Xg.shape)
        Hs_w = 4 * gt_grid.std()
        sigma = NOISE_FRAC * HS
        target = {'gx': gx, 'gy': gy, 't': t_star, 'gt': gt_grid,
                  'Hs': Hs_w}
        for cname, ur in conds:
            P, rng = Ps[cname]
            z = zs[cname] + rng.normal(0, sigma, N_OBS)
            P32, z32 = P.astype(np.float32), z.astype(np.float32)
            row = res[cname].setdefault(si, {})
            fld, Nr, _ = iterative_relax_cv(P32.copy(), z32.copy(), DEPTH,
                                            target, n_outer=2, seed=si)
            ix = np.clip(np.round((Xs - gx[0]) / (gx[1] - gx[0])).astype(int),
                         0, len(gx) - 1)
            iy = np.clip(np.round((Ys - gy[0]) / (gy[1] - gy[0])).astype(int),
                         0, len(gy) - 1)
            row['relax'] = corr(np.asarray(fld)[iy, ix], gt_u)
            row['relax_N'] = Nr
            fq = mmse_bank_estimate(P, z, DEPTH, Pq, sigma0=sigma, seed=si)
            row['mmse'] = corr(fq, gt_u)
        print(f"sea {si} ({time.time()-tic:.0f}s): " + " | ".join(
            f"{cn} rlx {res[cn][si]['relax']:+.2f} "
            f"mmse {res[cn][si]['mmse']:+.2f}" for cn, _ in conds),
            flush=True)

    print(f"\n===== HOS M={a.M} (mean corr, common unobs, "
          f"{a.n_seas} seas) =====")
    summ = {}
    for cname, _ in conds:
        summ[cname] = {
            e: float(np.mean([res[cname][si][e] for si in res[cname]]))
            for e in ('relax', 'mmse')}
        print(f"{cname:10s} relax {summ[cname]['relax']:+.3f}  "
              f"mmse {summ[cname]['mmse']:+.3f}")
    path = OUT.replace('.json', f'_M{a.M}.json')
    json.dump(dict(summary=summ, rows=res, M=a.M, n_seas=a.n_seas),
              open(path, 'w'), indent=1, default=float)
    print(f"saved {path}")


if __name__ == '__main__':
    main()
