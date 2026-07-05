"""Protocol-family ablation: does the net prior (trained across Tp 5-11 on
the 2-lam/10-Tp protocol) hold at other (Tp, patch) cells, and does the
hand-MMSE too? Cells: baseline (7, 2.0), (5, 2.0), (9, 2.0), (7, 1.3),
(7, 3.0). 6 seas x {fixed, track100} x {hand, net}. All geometry scales
with lam(Tp) by construction; this VERIFIES the scale covariance.
(repo tmp probe)"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
os.chdir(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, 'scripts')

from wavesim.wave_generator import jonswap_components
from d3_sa_fim_identifiability import omega_of
from d3_sa_nomp import mmse_bank_estimate
from d3_prior_net_eval import get_net, net_gamma_mmse
from d3_sa_achievability_gap import (dispersion_lam, sample_obs,
                                     masks_common_unobs, hashabs, corr,
                                     N_OBS, NOISE_FRAC, HS, DEPTH)

NSEAS = 6
net = get_net()
CELLS = [(7.0, 2.0), (5.0, 2.0), (9.0, 2.0), (7.0, 1.3), (7.0, 3.0)]

for Tp, side_lam in CELLS:
    lam = dispersion_lam(Tp, DEPTH)
    cp = lam / Tp
    T = 10 * Tp
    fp = 0.18 * lam
    r_fp = fp / 2
    side = side_lam * lam
    gx = np.linspace(0, side, 96)
    gy = np.linspace(0, side, 96)
    Xg, Yg = np.meshgrid(gx, gy)
    mu = masks_common_unobs(gx, gy, r_fp, 0.6 * fp)
    Xs, Ys = Xg[mu], Yg[mu]
    if len(Xs) > 800:
        take = np.random.default_rng(0).choice(len(Xs), 800, replace=False)
        Xs, Ys = Xs[take], Ys[take]
    t_star = T
    Pq = np.column_stack([Xs, Ys, np.full(len(Xs), t_star)])
    res = {c: {'hand': [], 'net': []} for c in ('fixed', 'track100')}
    for si in range(NSEAS):
        c = jonswap_components(Hs=HS, Tp=Tp, n_components=200, depth=DEPTH,
                               seed=7500 + si)
        th = (np.array([x.kx for x in c]), np.array([x.ky for x in c]),
              np.array([x.A for x in c]), np.array([x.phi for x in c]))
        om_t = omega_of(th[0], th[1], DEPTH)
        psi_q = (th[0][None] * Xs[:, None] + th[1][None] * Ys[:, None]
                 - om_t[None] * t_star + th[3][None])
        gt_u = (th[2][None] * np.cos(psi_q)).sum(1)
        sigma = NOISE_FRAC * HS
        for cname in res:
            rng = np.random.default_rng(9000 + 17 * si + hashabs(cname))
            U = (1.0 if cname == 'track100' else 0) * cp
            P = sample_obs('fixed' if cname == 'fixed' else 'moving', U,
                           lam, cp, T, gx, gy, r_fp, rng)
            psi = (th[0][None] * P[:, 0:1] + th[1][None] * P[:, 1:2]
                   - om_t[None] * P[:, 2:3] + th[3][None])
            z = (th[2][None] * np.cos(psi)).sum(1) \
                + rng.normal(0, sigma, N_OBS)
            res[cname]['hand'].append(corr(
                mmse_bank_estimate(P, z, DEPTH, Pq, sigma0=sigma, seed=si),
                gt_u))
            res[cname]['net'].append(corr(
                net_gamma_mmse(net, P, z, DEPTH, Pq, sigma, seed=si,
                               use_em=False), gt_u))
    line = f"Tp={Tp:g} side={side_lam:g}lam: "
    for est in ('hand', 'net'):
        f0 = np.mean(res['fixed'][est])
        f1 = np.mean(res['track100'][est])
        line += f"{est} fixed {f0:+.3f} track {f1:+.3f} | "
    print(line, flush=True)
