"""Why does the net-MMSE motion gain exist on the synthetic harness (+0.08)
but vanish on real records? H2 = spectral bandwidth: a broadband sea feeds
a FIXED footprint enough temporal information over 10 Tp that there is no
starvation for motion to relieve. Sweep (gamma_J, spread) from broadband
(1.0, 60 deg) through the harness default (3.3, 30) to narrowband (6, 15),
fixed vs track100, hand-MMSE and net-MMSE. Prediction under H2: motion
gain grows with narrowness. (repo tmp probe, 2026-07-05 밤)"""
import os
import sys

import numpy as np
import torch

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

NSEAS = 8
BANDS = [('broad', 1.0, 60.0), ('default', 3.3, 30.0),
         ('narrow', 6.0, 15.0)]

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
net = get_net()

res = {b: {c: {'hand': [], 'net': []} for c in ('fixed', 'track100')}
       for b, _, _ in BANDS}
for bname, gj, spr in BANDS:
    for si in range(NSEAS):
        c = jonswap_components(Hs=HS, Tp=TP, n_components=200, depth=DEPTH,
                               gamma=gj, direction_spread_deg=spr,
                               seed=4600 + si)
        th = (np.array([x.kx for x in c]), np.array([x.ky for x in c]),
              np.array([x.A for x in c]), np.array([x.phi for x in c]))
        om_t = omega_of(th[0], th[1], DEPTH)
        gt_u = render_modes(*th, DEPTH, Xs, Ys, t_star)
        sigma = NOISE_FRAC * HS
        for cname in ('fixed', 'track100'):
            rng = np.random.default_rng(9000 + 17 * si + hashabs(cname))
            U = (1.0 if cname == 'track100' else 0) * cp
            P = sample_obs('fixed' if cname == 'fixed' else 'moving', U,
                           lam, cp, T, gx, gy, r_fp, rng)
            psi = (th[0][None] * P[:, 0:1] + th[1][None] * P[:, 1:2]
                   - om_t[None] * P[:, 2:3] + th[3][None])
            z = (th[2][None] * np.cos(psi)).sum(1) \
                + rng.normal(0, sigma, N_OBS)
            fh = mmse_bank_estimate(P, z, DEPTH, Pq, sigma0=sigma, seed=si)
            res[bname][cname]['hand'].append(corr(fh, gt_u))
            fn = net_gamma_mmse(net, P, z, DEPTH, Pq, sigma, seed=si,
                                use_em=False)
            res[bname][cname]['net'].append(corr(fn, gt_u))
    print(f"{bname} done", flush=True)

print(f"\n{'band':8s} {'est':5s} {'fixed':>8s} {'track100':>9s} "
      f"{'motion gain':>12s}")
for bname, gj, spr in BANDS:
    for est in ('hand', 'net'):
        f0 = np.mean(res[bname]['fixed'][est])
        f1 = np.mean(res[bname]['track100'][est])
        d = np.array(res[bname]['track100'][est]) \
            - np.array(res[bname]['fixed'][est])
        print(f"{bname:8s} {est:5s} {f0:+8.3f} {f1:+9.3f} {d.mean():+8.3f} "
              f"(win+ {int((d>0).sum())}/{len(d)})")
