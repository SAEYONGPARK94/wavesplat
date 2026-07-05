"""Spatial (extrapolation) validation for the MMSE estimator — does
replacing the on-track RANDOM val split with a trajectory END-SLICE holdout
(train on central 75%, validate on the outer 2 x 12.5% along the principal
axis) align hyperparameter/EM gating better with off-track generalization?
Compares against the random-val baseline (0.272/0.239/0.241 on 10 seas).
(repo tmp probe, 2026-07-05 afternoon)"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
os.chdir(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, 'scripts')

from wavesim.wave_generator import jonswap_components
from d18_relax_estimator import beamform_top, omega
from d3_sa_fim_identifiability import omega_of, render_modes
from d3_sa_achievability_gap import (dispersion_lam, sample_obs,
                                     masks_common_unobs, hashabs, corr,
                                     N_OBS, NOISE_FRAC, TP, HS, DEPTH)

NSEAS = 10


def mmse_core(P, z, depth, Pq, sigma0, vmask, kp=None):
    """One MMSE fit with a GIVEN val mask (rest mirrors mmse_bank_estimate)."""
    Ptr, ztr = P[~vmask], z[~vmask]
    Pv, zv = P[vmask], z[vmask]
    if kp is None:
        bx, by = beamform_top(P.astype(np.float32), z.astype(np.float32),
                              depth, klim=2.0, n=120)
        kp = max(np.hypot(bx, by), 1e-3)
        bx, by = beamform_top(P.astype(np.float32), z.astype(np.float32),
                              depth, klim=3.5 * kp, n=120)
        kp = max(np.hypot(bx, by), 1e-3)
    dk_c = kp / 10.0
    ax = np.arange(-2.5 * kp, 2.5 * kp + dk_c / 2, dk_c)
    KX, KY = np.meshgrid(ax, ax)
    kk = np.hypot(KX, KY)
    sel = (kk > 0.35 * kp) & (kk < 2.5 * kp)
    kxb, kyb = KX[sel].astype(float), KY[sel].astype(float)
    B = len(kxb)
    om = omega(np.hypot(kxb, kyb) + 1e-12, depth)

    def design(Q):
        psi = kxb[None] * Q[:, 0:1] + kyb[None] * Q[:, 1:2] \
            - om[None] * Q[:, 2:3]
        return np.concatenate([np.cos(psi), np.sin(psi)], axis=1)

    Phi_tr, Phi_v = design(Ptr), design(Pv)
    pw = np.empty(B)
    for i in range(0, B, 512):
        psi = (kxb[i:i + 512, None] * P[None, :, 0]
               + kyb[i:i + 512, None] * P[None, :, 1]
               - om[i:i + 512, None] * P[None, :, 2])
        pw[i:i + 512] = (z[None] * np.cos(psi)).sum(1) ** 2 \
            + (z[None] * np.sin(psi)).sum(1) ** 2
    pw = pw / max(pw.max(), 1e-12)
    m, n = len(ztr), len(z)
    best = None
    for pexp in (0.5, 1.0, 2.0):
        g = pw ** pexp
        g = g / g.sum()
        gam0 = np.concatenate([g, g]) * np.var(z)
        for smult in (1.0, 3.0, 10.0):
            s2 = (smult * sigma0) ** 2
            K = (Phi_tr * gam0[None]) @ Phi_tr.T + s2 * np.eye(m)
            fv = (Phi_v * gam0[None]) @ (Phi_tr.T @ np.linalg.solve(K, ztr))
            vr = float(np.sqrt(np.mean((fv - zv) ** 2)))
            if best is None or vr < best[0]:
                best = (vr, pexp, smult, gam0)
    vr0, pexp, smult, gam = best
    s2 = (smult * sigma0) ** 2
    best_em = (vr0, gam.copy())
    for _ in range(6):
        K = (Phi_tr * gam[None]) @ Phi_tr.T + s2 * np.eye(m)
        Ki_Phi = np.linalg.solve(K, Phi_tr)
        mu = gam * (Phi_tr.T @ np.linalg.solve(K, ztr))
        sig = gam - gam ** 2 * np.einsum('ij,ij->j', Phi_tr, Ki_Phi)
        gb = 0.5 * (mu[:B] ** 2 + mu[B:] ** 2
                    + np.maximum(sig[:B], 0) + np.maximum(sig[B:], 0))
        gam = np.concatenate([gb, gb])
        fv = (Phi_v * gam[None]) @ (Phi_tr.T @ np.linalg.solve(
            (Phi_tr * gam[None]) @ Phi_tr.T + s2 * np.eye(m), ztr))
        vr = float(np.sqrt(np.mean((fv - zv) ** 2)))
        if vr < best_em[0] - 1e-5:
            best_em = (vr, gam.copy())
        else:
            break
    gam = best_em[1]
    Phi_all = design(P)
    K = (Phi_all * gam[None]) @ Phi_all.T + s2 * np.eye(n)
    fq = np.empty(len(Pq))
    w_pred = Phi_all.T @ np.linalg.solve(K, z)
    for i in range(0, len(Pq), 4096):
        fq[i:i + 4096] = (design(Pq[i:i + 4096]) * gam[None]) @ w_pred
    return fq


def vmask_random(P, seed):
    rng = np.random.default_rng(seed)
    n = len(P)
    vi = rng.choice(n, max(8, n // 4), replace=False)
    m = np.zeros(n, bool)
    m[vi] = True
    return m


def vmask_extrap(P):
    xy = P[:, :2] - P[:, :2].mean(0)
    _, _, Vt = np.linalg.svd(xy, full_matrices=False)
    s = xy @ Vt[0]
    lo, hi = np.quantile(s, 0.125), np.quantile(s, 0.875)
    return (s < lo) | (s > hi)


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

res = {v: {c: [] for c in ('fixed', 'track043', 'track100')}
       for v in ('random', 'extrap')}
for si in range(NSEAS):
    c = jonswap_components(Hs=HS, Tp=TP, n_components=200, depth=DEPTH,
                           seed=4000 + si)
    th = (np.array([x.kx for x in c]), np.array([x.ky for x in c]),
          np.array([x.A for x in c]), np.array([x.phi for x in c]))
    om_t = omega_of(th[0], th[1], DEPTH)
    gt_u = render_modes(*th, DEPTH, Xs, Ys, t_star)
    sigma = NOISE_FRAC * HS
    for cname in res['random']:
        rng = np.random.default_rng(9000 + 17 * si + hashabs(cname))
        U = (0.43 if cname == 'track043' else 1.0 if cname == 'track100'
             else 0) * cp
        P = sample_obs('fixed' if cname == 'fixed' else 'moving', U, lam,
                       cp, T, gx, gy, r_fp, rng)
        psi = (th[0][None] * P[:, 0:1] + th[1][None] * P[:, 1:2]
               - om_t[None] * P[:, 2:3] + th[3][None])
        z = (th[2][None] * np.cos(psi)).sum(1) + rng.normal(0, sigma, N_OBS)
        for v in res:
            vm = vmask_random(P, si) if v == 'random' else vmask_extrap(P)
            fq = mmse_core(P, z, DEPTH, Pq, NOISE_FRAC * HS, vm)
            res[v][cname].append(corr(fq, gt_u))
    print(f"sea {si}: " + " | ".join(
        f"{cn} rnd {res['random'][cn][-1]:+.2f} ext "
        f"{res['extrap'][cn][-1]:+.2f}" for cn in res['random']),
        flush=True)

print("\nmean corr (common unobs), random vs extrap val:")
for cn in res['random']:
    a = np.mean(res['random'][cn])
    b = np.mean(res['extrap'][cn])
    d = np.array(res['extrap'][cn]) - np.array(res['random'][cn])
    print(f"{cn:10s} random {a:+.3f}  extrap {b:+.3f}  "
          f"paired d {d.mean():+.3f} (win+ {int((d>0).sum())}/{len(d)})")
