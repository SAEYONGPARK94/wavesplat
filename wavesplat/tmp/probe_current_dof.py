"""Is the current-induced moving-gain loss INFORMATION (design degeneracy)
or still estimator/probe artifact? With TRUE Uc in the design (oracle),
any remaining loss must be structural. Diagnostics on the exact obs
geometries: effective DOF of the current-corrected design
  eff_dof = trace(Phi (Phi'Phi + I)^-1 Phi')
restricted to the energetic band, plus 16-sea oracle scores.
(repo tmp probe, 2026-07-06 재개 세션)"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
os.chdir(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, 'scripts')

from wavesim.wave_generator import jonswap_components
from d18_relax_estimator import omega
from d3_sa_fim_identifiability import omega_of
from d3_sa_achievability_gap import (dispersion_lam, sample_obs,
                                     masks_common_unobs, hashabs, corr,
                                     N_OBS, NOISE_FRAC, TP, HS, DEPTH)
sys.path.insert(0, 'tmp')
from probe_current_fit_lib import mmse_current_lib

NSEAS = 16
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

# fixed reference bank (true kp) for the DOF diagnostic
kp_true = 2 * np.pi / lam
dk_c = kp_true / 10
ax = np.arange(-2.5 * kp_true, 2.5 * kp_true + dk_c / 2, dk_c)
KX, KY = np.meshgrid(ax, ax)
kk = np.hypot(KX, KY)
sel = (kk > 0.35 * kp_true) & (kk < 2.5 * kp_true)
kxb, kyb = KX[sel].astype(float), KY[sel].astype(float)
om0 = omega(np.hypot(kxb, kyb) + 1e-12, DEPTH)


def eff_dof(P, ucx, ucy):
    omc = om0 + kxb * ucx + kyb * ucy
    psi = kxb[None] * P[:, 0:1] + kyb[None] * P[:, 1:2] \
        - omc[None] * P[:, 2:3]
    Phi = np.concatenate([np.cos(psi), np.sin(psi)], axis=1)
    G = Phi.T @ Phi
    tr = np.trace(np.linalg.solve(G + np.eye(G.shape[0]), G))
    return float(tr)


for uc_mag in (0.0, 0.3):
    sc = {c: [] for c in ('fixed', 'track100')}
    dof = {c: [] for c in ('fixed', 'track100')}
    for si in range(NSEAS):
        c = jonswap_components(Hs=HS, Tp=TP, n_components=200, depth=DEPTH,
                               seed=4600 + si)
        th = (np.array([x.kx for x in c]), np.array([x.ky for x in c]),
              np.array([x.A for x in c]), np.array([x.phi for x in c]))
        om_t = omega_of(th[0], th[1], DEPTH)
        crng = np.random.default_rng(500 + si)
        ang = crng.uniform(0, 2 * np.pi)
        ucx, ucy = uc_mag * np.cos(ang), uc_mag * np.sin(ang)
        om_true = om_t + th[0] * ucx + th[1] * ucy
        psi_q = (th[0][None] * Xs[:, None] + th[1][None] * Ys[:, None]
                 - om_true[None] * t_star + th[3][None])
        gt_u = (th[2][None] * np.cos(psi_q)).sum(1)
        sigma = NOISE_FRAC * HS
        for cname in sc:
            rng = np.random.default_rng(9000 + 17 * si + hashabs(cname))
            U = (1.0 if cname == 'track100' else 0) * cp
            P = sample_obs('fixed' if cname == 'fixed' else 'moving', U,
                           lam, cp, T, gx, gy, r_fp, rng)
            psi = (th[0][None] * P[:, 0:1] + th[1][None] * P[:, 1:2]
                   - om_true[None] * P[:, 2:3] + th[3][None])
            z = (th[2][None] * np.cos(psi)).sum(1) \
                + rng.normal(0, sigma, N_OBS)
            fo, _ = mmse_current_lib(P, z, DEPTH, Pq, sigma,
                                     uc_known=(ucx, ucy), seed=si,
                                     kp_known=kp_true)
            sc[cname].append(corr(fo, gt_u))
            if si < 6:
                dof[cname].append(eff_dof(P, ucx, ucy))
    print(f"\n== |Uc|={uc_mag} (oracle-Uc design, {NSEAS} seas) ==")
    for cname in sc:
        print(f"{cname:9s} corr {np.mean(sc[cname]):+.3f} "
              f"(sd {np.std(sc[cname]):.3f}) "
              f"eff_dof {np.mean(dof[cname]):.0f}", flush=True)
    d = np.array(sc['track100']) - np.array(sc['fixed'])
    print(f"motion gain {d.mean():+.3f} "
          f"(win+ {int((d>0).sum())}/{len(d)})")
