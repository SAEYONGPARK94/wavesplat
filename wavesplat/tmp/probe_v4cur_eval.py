"""v4(current-augmented) vs v3 prior net: no-current regression check +
recovery under Uc=0.3. 8 seas, fixed & track100. (repo tmp probe)"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
os.chdir(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, 'scripts')

from wavesim.wave_generator import jonswap_components
from d3_sa_fim_identifiability import omega_of
from d3_prior_net_eval import get_net, net_gamma_mmse
from d3_sa_achievability_gap import (dispersion_lam, sample_obs,
                                     masks_common_unobs, hashabs, corr,
                                     N_OBS, NOISE_FRAC, TP, HS, DEPTH)

NSEAS = 8
net3 = get_net('results/direction3/prior_net_v3.pt')
net4 = get_net('results/direction3/prior_net_v4cur.pt')
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

for uc_mag in (0.0, 0.3):
    res = {c: {'v3': [], 'v4': []} for c in ('fixed', 'track100')}
    for si in range(NSEAS):
        c = jonswap_components(Hs=HS, Tp=TP, n_components=200, depth=DEPTH,
                               seed=4600 + si)
        th = (np.array([x.kx for x in c]), np.array([x.ky for x in c]),
              np.array([x.A for x in c]), np.array([x.phi for x in c]))
        om_t = omega_of(th[0], th[1], DEPTH)
        crng = np.random.default_rng(500 + si)
        ang = crng.uniform(0, 2 * np.pi)
        om_true = om_t + th[0] * uc_mag * np.cos(ang) \
            + th[1] * uc_mag * np.sin(ang)
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
            for tag, nn in (('v3', net3), ('v4', net4)):
                f = net_gamma_mmse(nn, P, z, DEPTH, Pq, sigma, seed=si,
                                   use_em=False)
                res[cname][tag].append(corr(f, gt_u))
    print(f"\n== |Uc|={uc_mag} ==")
    for cname in res:
        v3 = np.mean(res[cname]['v3'])
        v4 = np.mean(res[cname]['v4'])
        d = np.array(res[cname]['v4']) - np.array(res[cname]['v3'])
        print(f"{cname:9s} v3 {v3:+.3f}  v4 {v4:+.3f}  d {d.mean():+.3f} "
              f"(win+ {int((d>0).sum())}/{len(d)})", flush=True)
