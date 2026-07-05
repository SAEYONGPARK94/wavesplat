"""Spectrum-weighted linear MMSE over the fine continuous bank — no
identification at all. Prior variance per candidate from the (PSF-smeared)
beamform power; hyperparams (power exponent, noise mult) val-selected.
If this approaches oracle_ls2 (0.28/0.41/0.46), the moving-platform answer
is 'estimated-spectrum MMSE', not sparse identification. (repo tmp probe)"""
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
                                     N_OBS, NOISE_FRAC, TP, HS, DEPTH, G)
from d3_sa_nomp import beamform_power

NSEAS = 10
DK_C = 0.015


def mmse_estimate(P, z, depth, Pq, seed=0, val_frac=0.25):
    rng = np.random.default_rng(seed)
    n = len(z)
    vi = rng.choice(n, max(8, int(val_frac * n)), replace=False)
    vmask = np.zeros(n, bool)
    vmask[vi] = True
    Ptr, ztr = P[~vmask], z[~vmask]
    Pv, zv = P[vmask], z[vmask]
    bx, by = beamform_top(P.astype(np.float32), z.astype(np.float32),
                          depth, klim=2.0, n=120)
    kp = max(np.hypot(bx, by), 1e-3)
    ax = np.arange(-2.5 * kp, 2.5 * kp + DK_C / 2, DK_C)
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

    Phi_tr = design(Ptr)
    Phi_v = design(Pv)
    Phi_q = design(Pq)
    # prior shape from beamform power on ALL obs (PSF-smeared spectrum)
    klim_bank = 2.5 * kp
    # evaluate power exactly at the bank points via chunked numpy
    pw = np.empty(B)
    CH = 512
    for i in range(0, B, CH):
        psi = (kxb[i:i + CH, None] * P[None, :, 0]
               + kyb[i:i + CH, None] * P[None, :, 1]
               - om[i:i + CH, None] * P[None, :, 2])
        pw[i:i + CH] = (z[None] * np.cos(psi)).sum(1) ** 2 \
            + (z[None] * np.sin(psi)).sum(1) ** 2
    pw = pw / pw.max()
    m = len(ztr)
    best = None
    for pexp in (0.5, 1.0, 2.0):
        g = pw ** pexp
        g = g / g.sum()                     # unit total prior variance shape
        gam = np.concatenate([g, g]) / 2.0 * np.var(z) * 2.0
        for smult in (1.0, 3.0, 10.0):
            s2 = (smult * NOISE_FRAC * HS) ** 2
            K = (Phi_tr * gam[None]) @ Phi_tr.T + s2 * np.eye(m)
            alpha = np.linalg.solve(K, ztr)
            fv = (Phi_v * gam[None]) @ (Phi_tr.T @ alpha)
            vr = float(np.sqrt(np.mean((fv - zv) ** 2)))
            if best is None or vr < best[0]:
                best = (vr, pexp, smult)
    _, pexp, smult = best
    g = pw ** pexp
    g = g / g.sum()
    gam = np.concatenate([g, g]) / 2.0 * np.var(z) * 2.0
    s2 = (smult * NOISE_FRAC * HS) ** 2
    # val-gated SBL-EM refinement of the prior shape (empirical Bayes):
    # gamma_b <- (mu_c^2 + mu_s^2 + Sig_cc + Sig_ss)/2, stop when val RMSE
    # stops improving
    n_em = 0
    best_em = (best[0], gam.copy())
    for it in range(6):
        K = (Phi_tr * gam[None]) @ Phi_tr.T + s2 * np.eye(m)
        Ki_Phi = np.linalg.solve(K, Phi_tr)
        mu = gam * (Phi_tr.T @ np.linalg.solve(K, ztr))
        sig = gam - gam ** 2 * np.einsum('ij,ij->j', Phi_tr, Ki_Phi)
        gb = 0.5 * (mu[:B] ** 2 + mu[B:] ** 2
                    + np.maximum(sig[:B], 0) + np.maximum(sig[B:], 0))
        gam_new = np.concatenate([gb, gb])
        fv = (Phi_v * gam_new[None]) @ (Phi_tr.T @ np.linalg.solve(
            (Phi_tr * gam_new[None]) @ Phi_tr.T + s2 * np.eye(m), ztr))
        vr = float(np.sqrt(np.mean((fv - zv) ** 2)))
        gam = gam_new
        if vr < best_em[0] - 1e-5:
            best_em = (vr, gam.copy())
            n_em = it + 1
        else:
            break
    gam = best_em[1]
    Phi_all = design(P)
    K = (Phi_all * gam[None]) @ Phi_all.T + s2 * np.eye(n)
    alpha = np.linalg.solve(K, z)
    fq = (Phi_q * gam[None]) @ (Phi_all.T @ alpha)
    return fq, pexp, smult, n_em


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

res = {c: [] for c in ('fixed', 'track043', 'track100')}
for si in range(NSEAS):
    c = jonswap_components(Hs=HS, Tp=TP, n_components=200, depth=DEPTH,
                           seed=4000 + si)
    th = (np.array([x.kx for x in c]), np.array([x.ky for x in c]),
          np.array([x.A for x in c]), np.array([x.phi for x in c]))
    om_t = omega_of(th[0], th[1], DEPTH)
    gt_u = render_modes(*th, DEPTH, Xs, Ys, t_star)
    sigma = NOISE_FRAC * HS
    for cname in res:
        rng = np.random.default_rng(9000 + 17 * si + hashabs(cname))
        U = (0.43 if cname == 'track043' else 1.0 if cname == 'track100'
             else 0) * cp
        P = sample_obs('fixed' if cname == 'fixed' else 'moving', U, lam,
                       cp, T, gx, gy, r_fp, rng)
        psi = (th[0][None] * P[:, 0:1] + th[1][None] * P[:, 1:2]
               - om_t[None] * P[:, 2:3] + th[3][None])
        z = (th[2][None] * np.cos(psi)).sum(1) + rng.normal(0, sigma, N_OBS)
        fq, pexp, smult, n_em = mmse_estimate(P, z, DEPTH, Pq, seed=si)
        cc = corr(fq, gt_u)
        res[cname].append((cc, pexp, smult))
    print(f"sea {si}: " + " ".join(
        f"{cn} {res[cn][-1][0]:+.3f}(p{res[cn][-1][1]:g},s{res[cn][-1][2]:g},"
        f"em{n_em})" for cn in res), flush=True)

print("\nmean corr (common unobs):")
for cn in res:
    v = np.array([r[0] for r in res[cn]])
    print(f"{cn:10s} {v.mean():+.3f}  per-sea {[round(x,2) for x in v]}")
