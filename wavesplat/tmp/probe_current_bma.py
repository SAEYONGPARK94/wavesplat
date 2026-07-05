"""BMA-over-Uc final probe: does evidence-weighted averaging beat the
plain (Uc=0) design under current, without the spurious-argmax failure?
8 seas, track100 (the vulnerable case) + fixed, |Uc| in {0, 0.3}.
(repo tmp probe — last card of the current arc)"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
os.chdir(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, 'scripts')
sys.path.insert(0, 'tmp')

from wavesim.wave_generator import jonswap_components
from d3_sa_fim_identifiability import omega_of
from d3_sa_achievability_gap import (dispersion_lam, sample_obs,
                                     masks_common_unobs, hashabs, corr,
                                     N_OBS, NOISE_FRAC, TP, HS, DEPTH)
from probe_current_fit_lib import mmse_current_lib, mmse_current_bma

NSEAS = 8
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
take = np.random.default_rng(0).choice(len(Xs), 800, replace=False)
Xs, Ys = Xs[take], Ys[take]
t_star = T
Pq = np.column_stack([Xs, Ys, np.full(len(Xs), t_star)])
kp_true = 2 * np.pi / lam

for uc_mag in (0.0, 0.3):
    res = {c: {'plain': [], 'bma': [], 'oracle': []}
           for c in ('fixed', 'track100')}
    esss = []
    for si in range(NSEAS):
        c = jonswap_components(Hs=HS, Tp=TP, n_components=200, depth=DEPTH,
                               seed=4600 + si)
        th = (np.array([x.kx for x in c]), np.array([x.ky for x in c]),
              np.array([x.A for x in c]), np.array([x.phi for x in c]))
        om_t = omega_of(th[0], th[1], DEPTH)
        crng = np.random.default_rng(500 + si)
        ang = crng.uniform(0, 2 * np.pi)
        ucx_t, ucy_t = uc_mag * np.cos(ang), uc_mag * np.sin(ang)
        om_true = om_t + th[0] * ucx_t + th[1] * ucy_t
        psi_q = (th[0][None] * Xs[:, None] + th[1][None] * Ys[:, None]
                 - om_true[None] * t_star + th[3][None])
        gt_u = (th[2][None] * np.cos(psi_q)).sum(1)
        sigma = NOISE_FRAC * HS
        for cname in res:
            rng = np.random.default_rng(9000 + 17 * si + hashabs(cname))
            U = (1.0 if cname == 'track100' else 0) * cp
            P = sample_obs('fixed' if cname == 'fixed' else 'moving', U,
                           lam, cp, T, gx, gy, r_fp, rng)
            psi = (th[0][None] * P[:, 0:1] + th[1][None] * P[:, 1:2]
                   - om_true[None] * P[:, 2:3] + th[3][None])
            z = (th[2][None] * np.cos(psi)).sum(1) \
                + rng.normal(0, sigma, N_OBS)
            fpl, _ = mmse_current_lib(P, z, DEPTH, Pq, sigma,
                                      uc_known=(0.0, 0.0), seed=si,
                                      kp_known=kp_true)
            res[cname]['plain'].append(corr(fpl, gt_u))
            fb, ess = mmse_current_bma(P, z, DEPTH, Pq, sigma, seed=si,
                                       kp_known=kp_true)
            res[cname]['bma'].append(corr(fb, gt_u))
            if cname == 'track100':
                esss.append(ess)
            fo, _ = mmse_current_lib(P, z, DEPTH, Pq, sigma,
                                     uc_known=(ucx_t, ucy_t), seed=si,
                                     kp_known=kp_true)
            res[cname]['oracle'].append(corr(fo, gt_u))
        print(f"|Uc|={uc_mag} sea {si} done", flush=True)
    print(f"\n== |Uc|={uc_mag} (ESS of BMA weights, track: "
          f"{np.mean(esss):.1f}/25) ==")
    for cname in res:
        m = {k: np.mean(v) for k, v in res[cname].items()}
        print(f"{cname:9s} plain {m['plain']:+.3f}  bma {m['bma']:+.3f}  "
              f"oracle {m['oracle']:+.3f}", flush=True)
