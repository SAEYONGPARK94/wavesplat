"""UQ calibration probe: the MMSE posterior variance is analytic —
var_q = phi_q' (G - G Phi' K^-1 Phi G) phi_q + 0 (noise-free field).
Check z-score calibration (std of err/sigma, coverage at 68/95%) for the
hand prior and the v3 net prior on the 10-sea harness, common-unobs px.
(repo tmp probe, 2026-07-05 저녁)"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
os.chdir(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, 'scripts')

from wavesim.wave_generator import jonswap_components
from d18_relax_estimator import beamform_top, omega
from d3_sa_fim_identifiability import omega_of, render_modes
from d3_sa_nomp import beamform_power, split_masks
from d3_prior_net_eval import get_net, DEVICE
from d3_prior_net_train import band_mask, NGRID, KREL
from d3_sa_achievability_gap import (dispersion_lam, sample_obs,
                                     masks_common_unobs, hashabs, corr,
                                     N_OBS, NOISE_FRAC, TP, HS, DEPTH)

NSEAS = 10


def mmse_with_var(P, z, depth, Pq, sigma0, gamma_mode, seed=0):
    """Returns (pred_q, var_q). gamma_mode: 'hand' (power^p val-select,
    no EM for simplicity) or 'net' (v3)."""
    rng = np.random.default_rng(seed)
    n = len(z)
    vi = rng.choice(n, max(8, n // 4), replace=False)
    vmask = np.zeros(n, bool)
    vmask[vi] = True
    Ptr, ztr = P[~vmask], z[~vmask]
    Pv, zv = P[vmask], z[vmask]
    bx, by = beamform_top(P.astype(np.float32), z.astype(np.float32),
                          depth, klim=2.0, n=120)
    kp = max(np.hypot(bx, by), 1e-3)
    bx, by = beamform_top(P.astype(np.float32), z.astype(np.float32),
                          depth, klim=3.5 * kp, n=120)
    kp = max(np.hypot(bx, by), 1e-3)
    ax = np.linspace(-KREL * kp, KREL * kp, NGRID)
    KX, KY = np.meshgrid(ax, ax)
    kk = np.hypot(KX, KY)
    sel = (kk > 0.35 * kp) & (kk < KREL * kp)
    kxb, kyb = KX[sel].astype(float), KY[sel].astype(float)
    B = len(kxb)
    om = omega(np.hypot(kxb, kyb) + 1e-12, depth)

    def design(Q):
        psi = kxb[None] * Q[:, 0:1] + kyb[None] * Q[:, 1:2] \
            - om[None] * Q[:, 2:3]
        return np.concatenate([np.cos(psi), np.sin(psi)], axis=1)

    m1, m2 = split_masks(P, mode='legs')
    _, _, p1 = beamform_power(P[m1], z[m1], depth, KREL * kp, NGRID)
    _, _, p2 = beamform_power(P[m2], z[m2], depth, KREL * kp, NGRID)
    p1 = (p1 / max(p1.max(), 1e-12)).reshape(NGRID, NGRID)
    p2 = (p2 / max(p2.max(), 1e-12)).reshape(NGRID, NGRID)
    pm = np.minimum(p1, p2)
    if gamma_mode == 'net':
        X = np.stack([p1, p2, pm,
                      np.log10(p1 + 1e-6) / 6 + 1,
                      np.log10(p2 + 1e-6) / 6 + 1,
                      np.log10(pm + 1e-6) / 6 + 1]).astype(np.float32)
        with torch.no_grad():
            Ylog = get_net()(torch.tensor(X[None],
                                          device=DEVICE))[0].cpu().numpy()
        g = np.maximum(10.0 ** Ylog[sel], 1e-8)
        g = g / g.sum()
        gams = [np.concatenate([g, g]) * np.var(z)]
        smults = (1.0, 3.0, 10.0)
    else:
        pw = pm.reshape(-1)[sel.reshape(-1)]
        gams = []
        for pexp in (0.5, 1.0, 2.0):
            gg = np.maximum(pw, 1e-12) ** pexp
            gg = gg / gg.sum()
            gams.append(np.concatenate([gg, gg]) * np.var(z))
        smults = (1.0, 3.0, 10.0)
    Phi_tr, Phi_v = design(Ptr), design(Pv)
    best = None
    for gam in gams:
        for smult in smults:
            s2 = (smult * sigma0) ** 2
            K = (Phi_tr * gam[None]) @ Phi_tr.T + s2 * np.eye(len(ztr))
            fv = (Phi_v * gam[None]) @ (Phi_tr.T @ np.linalg.solve(K, ztr))
            vr = float(np.sqrt(np.mean((fv - zv) ** 2)))
            if best is None or vr < best[0]:
                best = (vr, gam, smult)
    _, gam, smult = best
    s2 = (smult * sigma0) ** 2
    Phi = design(P)
    K = (Phi * gam[None]) @ Phi.T + s2 * np.eye(n)
    Ki = np.linalg.inv(K)
    Dq = design(Pq)
    pred = (Dq * gam[None]) @ (Phi.T @ (Ki @ z))
    # posterior variance per query px
    v_prior = (Dq ** 2 * gam[None]).sum(1)
    A = (Dq * gam[None]) @ Phi.T            # (nq, n)
    v_red = np.einsum('ij,jk,ik->i', A, Ki, A)
    var_q = np.maximum(v_prior - v_red, 1e-12)
    return pred, var_q


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

zs = {m: {c: [] for c in ('fixed', 'track043', 'track100')}
      for m in ('hand', 'net')}
for si in range(NSEAS):
    c = jonswap_components(Hs=HS, Tp=TP, n_components=200, depth=DEPTH,
                           seed=4000 + si)
    th = (np.array([x.kx for x in c]), np.array([x.ky for x in c]),
          np.array([x.A for x in c]), np.array([x.phi for x in c]))
    om_t = omega_of(th[0], th[1], DEPTH)
    gt_u = render_modes(*th, DEPTH, Xs, Ys, t_star)
    sigma = NOISE_FRAC * HS
    for cname in zs['hand']:
        rng = np.random.default_rng(9000 + 17 * si + hashabs(cname))
        U = (0.43 if cname == 'track043' else 1.0 if cname == 'track100'
             else 0) * cp
        P = sample_obs('fixed' if cname == 'fixed' else 'moving', U, lam,
                       cp, T, gx, gy, r_fp, rng)
        psi = (th[0][None] * P[:, 0:1] + th[1][None] * P[:, 1:2]
               - om_t[None] * P[:, 2:3] + th[3][None])
        z = (th[2][None] * np.cos(psi)).sum(1) + rng.normal(0, sigma, N_OBS)
        for mmode in zs:
            pred, var_q = mmse_with_var(P, z, DEPTH, Pq, sigma, mmode,
                                        seed=si)
            zz = (pred - gt_u) / np.sqrt(var_q)
            zs[mmode][cname].append(zz)
    print(f"sea {si} done", flush=True)

print("\nUQ calibration (z = err/sigma_post; ideal: std 1.0, "
      "cov68 0.68, cov95 0.95):")
for mmode in zs:
    for cname in zs[mmode]:
        zz = np.concatenate(zs[mmode][cname])
        print(f"{mmode:5s} {cname:10s} z-std {zz.std():5.2f}  "
              f"cov68 {np.mean(np.abs(zz) < 1.0):.2f}  "
              f"cov95 {np.mean(np.abs(zz) < 1.96):.2f}")
