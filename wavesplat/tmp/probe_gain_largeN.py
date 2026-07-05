"""Large-N reassessment of the SYNTHETIC motion gain for the shipped
estimators (hand mmse_bank_estimate, net v3) — the 16-sea evidence-variant
probe showed ~0 gain, flagging seed/estimator sensitivity of the earlier
+0.08~0.13 (8-10 seas). 32 seas, default spectrum, no current.
(repo tmp probe)"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
os.chdir(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, 'scripts')

from wavesim.wave_generator import jonswap_components
from d3_sa_fim_identifiability import omega_of, render_modes
from d3_sa_nomp import mmse_bank_estimate
from d3_prior_net_eval import get_net, net_gamma_mmse
from d3_sa_achievability_gap import (dispersion_lam, sample_obs,
                                     masks_common_unobs, hashabs, corr,
                                     N_OBS, NOISE_FRAC, TP, HS, DEPTH)

NSEAS = 32
net = get_net()
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
take = np.random.default_rng(0).choice(len(Xs), 1000, replace=False)
Xs, Ys = Xs[take], Ys[take]
t_star = T
Pq = np.column_stack([Xs, Ys, np.full(len(Xs), t_star)])

conds = ('fixed', 'track043', 'track100')
res = {c: {'hand': [], 'net': []} for c in conds}
for si in range(NSEAS):
    c = jonswap_components(Hs=HS, Tp=TP, n_components=200, depth=DEPTH,
                           seed=7000 + si)
    th = (np.array([x.kx for x in c]), np.array([x.ky for x in c]),
          np.array([x.A for x in c]), np.array([x.phi for x in c]))
    om_t = omega_of(th[0], th[1], DEPTH)
    gt_u = render_modes(*th, DEPTH, Xs, Ys, t_star)
    sigma = NOISE_FRAC * HS
    for cname in conds:
        rng = np.random.default_rng(9000 + 17 * si + hashabs(cname))
        U = (0.43 if cname == 'track043' else 1.0 if cname == 'track100'
             else 0) * cp
        P = sample_obs('fixed' if cname == 'fixed' else 'moving', U, lam,
                       cp, T, gx, gy, r_fp, rng)
        psi = (th[0][None] * P[:, 0:1] + th[1][None] * P[:, 1:2]
               - om_t[None] * P[:, 2:3] + th[3][None])
        z = (th[2][None] * np.cos(psi)).sum(1) + rng.normal(0, sigma, N_OBS)
        res[cname]['hand'].append(corr(
            mmse_bank_estimate(P, z, DEPTH, Pq, sigma0=sigma, seed=si),
            gt_u))
        res[cname]['net'].append(corr(
            net_gamma_mmse(net, P, z, DEPTH, Pq, sigma, seed=si,
                           use_em=False), gt_u))
    if (si + 1) % 8 == 0:
        print(f"sea {si + 1}/{NSEAS} done", flush=True)

print(f"\n=== large-N (n={NSEAS}, fresh seeds 7000+) ===")
for est in ('hand', 'net'):
    f0 = np.array(res['fixed'][est])
    print(f"{est:5s} fixed {f0.mean():+.3f} (sd {f0.std():.3f})")
    for cname in ('track043', 'track100'):
        f1 = np.array(res[cname][est])
        d = f1 - f0
        se = d.std() / np.sqrt(len(d))
        print(f"      {cname:9s} {f1.mean():+.3f}  gain {d.mean():+.3f} "
              f"(se {se:.3f}, win+ {int((d>0).sum())}/{len(d)})")
