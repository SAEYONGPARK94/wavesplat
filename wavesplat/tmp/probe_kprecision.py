"""How precise must k be at N=48 for the order-prize to survive, and how
precise IS the greedy detection? (repo tmp probe, 2026-07-05)"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
os.chdir(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, 'scripts')

from wavesim.wave_generator import jonswap_components
from wavesplat.model.tnse import fit_amplitudes_phases
from d3_sa_fim_identifiability import omega_of, render_modes
from d3_sa_achievability_gap import (dispersion_lam, sample_obs,
                                     masks_common_unobs, hashabs, corr,
                                     N_OBS, NOISE_FRAC, TP, HS, DEPTH)
from d3_sa_nomp import nomp_grow_estimate, predict, fit_coef_cv

DELTAS = [0.0, 0.001, 0.002, 0.005, 0.01]
NSEAS = 5
NCOMP = 48

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
take = np.random.default_rng(0).choice(len(Xs), 1500, replace=False)
Xs, Ys = Xs[take], Ys[take]
t_star = T
Pq = np.column_stack([Xs, Ys, np.full(len(Xs), t_star)])

res = {c: {d: [] for d in DELTAS} for c in ('fixed', 'track043', 'track100')}
kerr = {c: [] for c in res}
for si in range(NSEAS):
    c = jonswap_components(Hs=HS, Tp=TP, n_components=200, depth=DEPTH,
                           seed=4000 + si)
    th = (np.array([x.kx for x in c]), np.array([x.ky for x in c]),
          np.array([x.A for x in c]), np.array([x.phi for x in c]))
    order = np.argsort(th[2])[::-1]
    sel = order[:NCOMP]
    om = omega_of(th[0], th[1], DEPTH)
    gt_u = render_modes(*th, DEPTH, Xs, Ys, t_star)
    sigma = NOISE_FRAC * HS
    for cname in res:
        rng = np.random.default_rng(9000 + 17 * si + hashabs(cname))
        U = (0.43 if cname == 'track043' else 1.0 if cname == 'track100'
             else 0) * cp
        P = sample_obs('fixed' if cname == 'fixed' else 'moving', U, lam,
                       cp, T, gx, gy, r_fp, rng)
        psi = (th[0][None] * P[:, 0:1] + th[1][None] * P[:, 1:2]
               - om[None] * P[:, 2:3] + th[3][None])
        z = (th[2][None] * np.cos(psi)).sum(1) + rng.normal(0, sigma, N_OBS)
        obs = np.column_stack([P, z]).astype(np.float32)
        prng = np.random.default_rng(77 + si)
        for d in DELTAS:
            ang = prng.uniform(0, 2 * np.pi, NCOMP)
            kxp = th[0][sel] + d * np.cos(ang)
            kyp = th[1][sel] + d * np.sin(ang)
            A_, p_ = fit_amplitudes_phases(kxp.astype(np.float32),
                                           kyp.astype(np.float32), obs,
                                           DEPTH, lambda_reg=1.0)
            f = render_modes(kxp, kyp, np.asarray(A_, float),
                             np.asarray(p_, float), DEPTH, Xs, Ys, t_star)
            res[cname][d].append(corr(f, gt_u))
        # greedy-detected k error vs nearest true mode (all 200)
        gkx, gky, gcoef, _ = nomp_grow_estimate(P, z, DEPTH, n_comp=NCOMP,
                                                seed=si)
        dd = np.sqrt((gkx[:, None] - th[0][None]) ** 2
                     + (gky[:, None] - th[1][None]) ** 2).min(1)
        kerr[cname].append(np.median(dd))
    print(f"sea {si} done", flush=True)

print(f"\nperturbed true-k ridge (N={NCOMP}), mean corr over {NSEAS} seas:")
print(f"{'cond':10s}" + "".join(f"  dk={d:<6g}" for d in DELTAS))
for cname in res:
    print(f"{cname:10s}" + "".join(
        f"  {np.mean(res[cname][d]):.3f}   " for d in DELTAS))
print("\nmedian detected-k error to nearest true mode:")
for cname in kerr:
    print(f"{cname:10s} {np.mean(kerr[cname]):.4f} rad/m "
          f"(per-sea {[round(v, 4) for v in kerr[cname]]})")
