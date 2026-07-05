"""② Multi-window JOINT current estimation — the cure for single-window
weak identifiability (3 objectives picked spurious currents). The current
is shared across a record's windows while the sea phases decorrelate, so
joint evidence  score(Uc) = sum_w log p(z_w | Uc)  should localize Uc.

Synthetic validation: 6 windows (independent seas, seeds 7100+), SHARED
true Uc; obs = track100 per window (the operationally relevant case).
Report |Uc_hat - Uc_true| for joint vs single-window selection, and the
per-window field corr with joint-Uc-corrected MMSE vs plain vs oracle-Uc.
4 trials x {|Uc| = 0, 0.3}. (repo tmp probe)"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
os.chdir(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, 'scripts')
sys.path.insert(0, 'tmp')

from wavesim.wave_generator import jonswap_components
from d18_relax_estimator import beamform_top, omega
from d3_sa_fim_identifiability import omega_of
from d3_sa_achievability_gap import (dispersion_lam, sample_obs,
                                     masks_common_unobs, hashabs, corr,
                                     N_OBS, NOISE_FRAC, TP, HS, DEPTH)
from probe_current_fit_lib import mmse_current_lib

NWIN = 6
NTRIAL = 4
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
dk_c = kp_true / 10
ax = np.arange(-2.5 * kp_true, 2.5 * kp_true + dk_c / 2, dk_c)
KXb, KYb = np.meshgrid(ax, ax)
kkb = np.hypot(KXb, KYb)
selb = (kkb > 0.35 * kp_true) & (kkb < 2.5 * kp_true)
kxb, kyb = KXb[selb].astype(float), KYb[selb].astype(float)
om0 = omega(np.hypot(kxb, kyb) + 1e-12, DEPTH)


def win_evidence(P, z, sigma0, ucx, ucy):
    omc = om0 + kxb * ucx + kyb * ucy
    psi = kxb[None] * P[:, 0:1] + kyb[None] * P[:, 1:2] \
        - omc[None] * P[:, 2:3]
    Phi = np.concatenate([np.cos(psi), np.sin(psi)], axis=1)
    pw = np.empty(len(kxb))
    for i in range(0, len(kxb), 512):
        ps = (kxb[i:i + 512, None] * P[None, :, 0]
              + kyb[i:i + 512, None] * P[None, :, 1]
              - omc[i:i + 512, None] * P[None, :, 2])
        pw[i:i + 512] = (z[None] * np.cos(ps)).sum(1) ** 2 \
            + (z[None] * np.sin(ps)).sum(1) ** 2
    pw = pw / max(pw.max(), 1e-12)
    best = None
    for pexp in (1.0, 2.0):
        g = pw ** pexp
        g = g / g.sum()
        gam = np.concatenate([g, g]) * np.var(z)
        for smult in (1.0, 3.0):
            s2 = (smult * sigma0) ** 2
            K = (Phi * gam[None]) @ Phi.T + s2 * np.eye(len(z))
            L = np.linalg.cholesky(K)
            alpha = np.linalg.solve(L.T, np.linalg.solve(L, z))
            ll = -0.5 * float(z @ alpha) - float(np.log(np.diag(L)).sum())
            if best is None or ll > best:
                best = ll
    return best


for uc_mag in (0.0, 0.3):
    errs_j, errs_s = [], []
    cor_plain, cor_joint, cor_orac = [], [], []
    for tr in range(NTRIAL):
        crng = np.random.default_rng(900 + tr)
        angc = crng.uniform(0, 2 * np.pi)
        ucx_t, ucy_t = uc_mag * np.cos(angc), uc_mag * np.sin(angc)
        wins = []
        for w in range(NWIN):
            c = jonswap_components(Hs=HS, Tp=TP, n_components=200,
                                   depth=DEPTH, seed=7100 + 31 * tr + w)
            th = (np.array([x.kx for x in c]),
                  np.array([x.ky for x in c]),
                  np.array([x.A for x in c]),
                  np.array([x.phi for x in c]))
            om_t = omega_of(th[0], th[1], DEPTH)
            om_true = om_t + th[0] * ucx_t + th[1] * ucy_t
            rng = np.random.default_rng(9000 + 17 * (31 * tr + w)
                                        + hashabs('track100'))
            P = sample_obs('moving', cp, lam, cp, T, gx, gy, r_fp, rng)
            psi = (th[0][None] * P[:, 0:1] + th[1][None] * P[:, 1:2]
                   - om_true[None] * P[:, 2:3] + th[3][None])
            sigma = NOISE_FRAC * HS
            z = (th[2][None] * np.cos(psi)).sum(1) \
                + rng.normal(0, sigma, N_OBS)
            psi_q = (th[0][None] * Xs[:, None] + th[1][None] * Ys[:, None]
                     - om_true[None] * t_star + th[3][None])
            gt_u = (th[2][None] * np.cos(psi_q)).sum(1)
            wins.append((P, z, gt_u, sigma))
        # joint + single selection on a coarse->fine grid
        cand = [(ux, uy) for ux in np.linspace(-0.3, 0.3, 5)
                for uy in np.linspace(-0.3, 0.3, 5)]
        ev = np.array([[win_evidence(P, z, s, ux, uy)
                        for (ux, uy) in cand]
                       for (P, z, _, s) in wins])
        j0 = int(np.argmax(ev.sum(0)))
        ux0, uy0 = cand[j0]
        cand2 = [(ux0 + dx, uy0 + dy)
                 for dx in (-0.075, 0, 0.075) for dy in (-0.075, 0, 0.075)]
        ev2 = np.array([[win_evidence(P, z, s, ux, uy)
                         for (ux, uy) in cand2]
                        for (P, z, _, s) in wins])
        jx, jy = cand2[int(np.argmax(ev2.sum(0)))]
        errs_j.append(np.hypot(jx - ucx_t, jy - ucy_t))
        for w in range(NWIN):
            sx, sy = cand[int(np.argmax(ev[w]))]
            errs_s.append(np.hypot(sx - ucx_t, sy - ucy_t))
        for (P, z, gt_u, s) in wins:
            fo, _ = mmse_current_lib(P, z, DEPTH, Pq, s,
                                     uc_known=(0.0, 0.0), seed=0,
                                     kp_known=kp_true)
            cor_plain.append(corr(fo, gt_u))
            fj, _ = mmse_current_lib(P, z, DEPTH, Pq, s,
                                     uc_known=(jx, jy), seed=0,
                                     kp_known=kp_true)
            cor_joint.append(corr(fj, gt_u))
            fr, _ = mmse_current_lib(P, z, DEPTH, Pq, s,
                                     uc_known=(ucx_t, ucy_t), seed=0,
                                     kp_known=kp_true)
            cor_orac.append(corr(fr, gt_u))
        print(f"|Uc|={uc_mag} trial {tr}: joint err {errs_j[-1]:.3f}",
              flush=True)
    print(f"\n== |Uc|={uc_mag}: |Uc_hat-Uc| joint {np.mean(errs_j):.3f} "
          f"vs single {np.mean(errs_s):.3f} (median {np.median(errs_s):.3f})")
    print(f"   corr plain(Uc=0) {np.mean(cor_plain):+.3f}  "
          f"joint-Uc {np.mean(cor_joint):+.3f}  "
          f"oracle-Uc {np.mean(cor_orac):+.3f}")
