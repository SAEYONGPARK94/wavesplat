"""H6 처방 검증: val-selected CURRENT FITTING inside the MMSE.
The ambient current is a global 2-parameter dispersion shift
omega -> omega + kx*Ucx + ky*Ucy; grid-search (coarse 5x5 over +-0.3 m/s,
then 3x3 refine at half step) by val RMSE, hand prior (power^p) fixed at
the winning hyperparams per candidate. If this restores the motion gain
under synthetic current, the real-data gap has both cause and cure.
(repo tmp probe, 2026-07-05 밤)"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
os.chdir(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, 'scripts')

from wavesim.wave_generator import jonswap_components
from d18_relax_estimator import beamform_top, omega
from d3_sa_fim_identifiability import omega_of
from d3_sa_nomp import mmse_bank_estimate
from d3_sa_achievability_gap import (dispersion_lam, sample_obs,
                                     masks_common_unobs, hashabs, corr,
                                     N_OBS, NOISE_FRAC, TP, HS, DEPTH)

NSEAS = 8


def mmse_current(P, z, depth, Pq, sigma0, seed=0, uc_known=None):
    """hand-prior MMSE with global current: selected by GP evidence, or
    given (uc_known) — the oracle-Uc decisive test."""
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
    dk_c = kp / 10.0
    ax = np.arange(-2.5 * kp, 2.5 * kp + dk_c / 2, dk_c)
    KX, KY = np.meshgrid(ax, ax)
    kk = np.hypot(KX, KY)
    sel = (kk > 0.35 * kp) & (kk < 2.5 * kp)
    kxb, kyb = KX[sel].astype(float), KY[sel].astype(float)
    om0 = omega(np.hypot(kxb, kyb) + 1e-12, depth)

    def design(Q, ucx, ucy):
        omc = om0 + kxb * ucx + kyb * ucy
        psi = kxb[None] * Q[:, 0:1] + kyb[None] * Q[:, 1:2] \
            - omc[None] * Q[:, 2:3]
        return np.concatenate([np.cos(psi), np.sin(psi)], axis=1)

    def pw_of(ucx, ucy):
        """Prior-shape beamform power with CURRENT-CORRECTED dispersion —
        the uncorrected map smears under current (found via the oracle-Uc
        non-restoration bug)."""
        omc = om0 + kxb * ucx + kyb * ucy
        pw = np.empty(len(kxb))
        for i in range(0, len(kxb), 512):
            psi = (kxb[i:i + 512, None] * P[None, :, 0]
                   + kyb[i:i + 512, None] * P[None, :, 1]
                   - omc[i:i + 512, None] * P[None, :, 2])
            pw[i:i + 512] = (z[None] * np.cos(psi)).sum(1) ** 2 \
                + (z[None] * np.sin(psi)).sum(1) ** 2
        return pw / max(pw.max(), 1e-12)

    m = len(ztr)

    def fit_val(ucx, ucy):
        Phi_tr = design(Ptr, ucx, ucy)
        Phi_v = design(Pv, ucx, ucy)
        best = None
        for pexp in (1.0, 2.0):
            g = pw ** pexp
            g = g / g.sum()
            gam = np.concatenate([g, g]) * np.var(z)
            for smult in (1.0, 3.0):
                s2 = (smult * sigma0) ** 2
                K = (Phi_tr * gam[None]) @ Phi_tr.T + s2 * np.eye(m)
                fv = (Phi_v * gam[None]) @ (Phi_tr.T
                                            @ np.linalg.solve(K, ztr))
                vr = float(np.sqrt(np.mean((fv - zv) ** 2)))
                if best is None or vr < best[0]:
                    best = (vr, gam, s2)
        return best

    # v3: current selection by GP EVIDENCE (log marginal likelihood) on ALL
    # obs — principled, no val split, log|K| penalizes spurious currents
    # (v1 on-track val and v2 shell-concentration both picked |Uc|~0.3-0.5
    # at truth 0 on sparse moving samples)
    def evidence(ucx, ucy):
        Phi_all = design(P, ucx, ucy)
        pw = pw_of(ucx, ucy)
        best_e = None
        for pexp in (1.0, 2.0):
            g = pw ** pexp
            g = g / g.sum()
            gam = np.concatenate([g, g]) * np.var(z)
            for smult in (1.0, 3.0):
                s2 = (smult * sigma0) ** 2
                K = (Phi_all * gam[None]) @ Phi_all.T + s2 * np.eye(n)
                L = np.linalg.cholesky(K)
                alpha = np.linalg.solve(L.T, np.linalg.solve(L, z))
                ll = -0.5 * float(z @ alpha) \
                    - float(np.log(np.diag(L)).sum())
                if best_e is None or ll > best_e[0]:
                    best_e = (ll, gam, s2)
        return best_e

    if uc_known is not None:
        ucx, ucy = uc_known
        e = evidence(ucx, ucy)
        gam, s2 = e[1], e[2]
    else:
        best_u = None
        for ucx in np.linspace(-0.3, 0.3, 5):
            for ucy in np.linspace(-0.3, 0.3, 5):
                e = evidence(ucx, ucy)
                if best_u is None or e[0] > best_u[0]:
                    best_u = (e[0], ucx, ucy, e[1], e[2])
        _, ux0, uy0, _, _ = best_u
        for ucx in ux0 + np.array([-0.075, 0, 0.075]):
            for ucy in uy0 + np.array([-0.075, 0, 0.075]):
                e = evidence(ucx, ucy)
                if e[0] > best_u[0]:
                    best_u = (e[0], ucx, ucy, e[1], e[2])
        _, ucx, ucy, gam, s2 = best_u
    Phi_all = design(P, ucx, ucy)
    K = (Phi_all * gam[None]) @ Phi_all.T + s2 * np.eye(n)
    w = Phi_all.T @ np.linalg.solve(K, z)
    out = np.empty(len(Pq))
    for i in range(0, len(Pq), 4096):
        out[i:i + 4096] = (design(Pq[i:i + 4096], ucx, ucy) * gam[None]) @ w
    return out, np.hypot(ucx, ucy)


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
    res = {c: {'plain': [], 'curfit': []} for c in ('fixed', 'track100')}
    uhat = []
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
        for cname in res:
            rng = np.random.default_rng(9000 + 17 * si + hashabs(cname))
            U = (1.0 if cname == 'track100' else 0) * cp
            P = sample_obs('fixed' if cname == 'fixed' else 'moving', U,
                           lam, cp, T, gx, gy, r_fp, rng)
            psi = (th[0][None] * P[:, 0:1] + th[1][None] * P[:, 1:2]
                   - om_true[None] * P[:, 2:3] + th[3][None])
            z = (th[2][None] * np.cos(psi)).sum(1) \
                + rng.normal(0, sigma, N_OBS)
            fp_, _ = mmse_bank_estimate(P, z, DEPTH, Pq, sigma0=sigma,
                                        seed=si), None
            res[cname]['plain'].append(corr(fp_[0] if isinstance(
                fp_, tuple) else fp_, gt_u))
            fc, uh = mmse_current(P, z, DEPTH, Pq, sigma, seed=si)
            res[cname]['curfit'].append(corr(fc, gt_u))
            fo, _ = mmse_current(P, z, DEPTH, Pq, sigma, seed=si,
                                 uc_known=(ucx, ucy))
            res[cname].setdefault('curORACLE', []).append(corr(fo, gt_u))
            if cname == 'track100':
                uhat.append(uh)
    print(f"\n== true |Uc| = {uc_mag} m/s ==")
    for est in ('plain', 'curfit', 'curORACLE'):
        f0 = np.mean(res['fixed'][est])
        f1 = np.mean(res['track100'][est])
        d = np.array(res['track100'][est]) - np.array(res['fixed'][est])
        print(f"{est:7s} fixed {f0:+.3f} track {f1:+.3f} gain {d.mean():+.3f}"
              f" (win+ {int((d>0).sum())}/{len(d)})", flush=True)
    print(f"curfit |Uc|-hat (track): {np.mean(uhat):.3f} m/s")
