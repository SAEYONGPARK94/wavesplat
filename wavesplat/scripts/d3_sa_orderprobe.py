"""
d3_sa_orderprobe.py — Where does the continuous-sea prize live: MODEL ORDER
or k-PRECISION? oracle_ls2 restricted to the top-N true modes, N sweep.
If top-8/12 true-k ridge already reaches ~oracle_ls2(32), the estimators'
N~10 is enough and the gap is k precision; if corr grows steeply with N,
the gap is model order — greedy CV stopping (N~10) can never cash the prize
and the fix is "many components + heavy ridge" rather than better greedy.
Same seas/obs as sa_achievability_gap_cont.json.
"""
import os
import sys
import json

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
os.chdir(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, 'scripts')

from wavesim.wave_generator import jonswap_components
from wavesplat.model.tnse import fit_amplitudes_phases
from d3_sa_fim_identifiability import omega_of, render_modes
from d3_sa_achievability_gap import (dispersion_lam, sample_obs,
                                     masks_common_unobs, true_field,
                                     hashabs, corr, N_OBS, NOISE_FRAC,
                                     TP, HS, DEPTH, G)

NS = [4, 8, 12, 16, 24, 32, 48, 64]
OUT = 'results/direction3/sa_orderprobe.json'


def main():
    n_seas = 10
    lam = dispersion_lam(TP, DEPTH)
    cp = lam / TP
    T = 10 * TP
    fp = 0.18 * lam
    r_fp = fp / 2
    side = 2.0 * lam
    gx = np.linspace(0, side, 96)
    gy = np.linspace(0, side, 96)
    Xg, Yg = np.meshgrid(gx, gy)
    mu = masks_common_unobs(gx, gy, r_fp, 0.6 * fp)
    Xs, Ys = Xg[mu], Yg[mu]
    if len(Xs) > 1500:
        take = np.random.default_rng(0).choice(len(Xs), 1500, replace=False)
        Xs, Ys = Xs[take], Ys[take]
    t_star = T
    conds = [('fixed', None), ('track043', 0.43), ('track100', 1.0)]
    res = {c: {n: [] for n in NS} for c, _ in conds}
    for si in range(n_seas):
        c = jonswap_components(Hs=HS, Tp=TP, n_components=200, depth=DEPTH,
                               seed=4000 + si)
        th = (np.array([x.kx for x in c]), np.array([x.ky for x in c]),
              np.array([x.A for x in c]), np.array([x.phi for x in c]))
        order = np.argsort(th[2])[::-1]
        om = omega_of(th[0], th[1], DEPTH)
        gt_u = render_modes(*th, DEPTH, Xs, Ys, t_star)
        sigma = NOISE_FRAC * HS
        for cname, ur in conds:
            rng = np.random.default_rng(9000 + 17 * si + hashabs(cname))
            U = (ur or 0) * cp
            P = sample_obs('fixed' if cname == 'fixed' else 'moving', U, lam,
                           cp, T, gx, gy, r_fp, rng)
            psi = (th[0][None] * P[:, 0:1] + th[1][None] * P[:, 1:2]
                   - om[None] * P[:, 2:3] + th[3][None])
            z = (th[2][None] * np.cos(psi)).sum(1) \
                + rng.normal(0, sigma, N_OBS)
            obs = np.column_stack([P, z]).astype(np.float32)
            for N in NS:
                sel = order[:N]
                A_, p_ = fit_amplitudes_phases(
                    th[0][sel].astype(np.float32),
                    th[1][sel].astype(np.float32), obs, DEPTH,
                    lambda_reg=1.0)
                f = render_modes(th[0][sel], th[1][sel],
                                 np.asarray(A_, float), np.asarray(p_, float),
                                 DEPTH, Xs, Ys, t_star)
                res[cname][N].append(corr(f, gt_u))
        print(f"sea {si} done", flush=True)

    print(f"\n{'cond':10s}" + "".join(f"  N={n:<4d}" for n in NS))
    summ = {}
    for cname, _ in conds:
        summ[cname] = {int(n): float(np.mean(res[cname][n])) for n in NS}
        print(f"{cname:10s}" + "".join(
            f"  {summ[cname][n]:.3f}" for n in NS))
    json.dump(dict(summary=summ, NS=NS), open(OUT, 'w'), indent=1,
              default=float)
    print(f"saved {OUT}")


if __name__ == '__main__':
    main()
