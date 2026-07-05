"""Importable current-fit MMSE (extracted from probe_current_fit.py)."""
import os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))
from d18_relax_estimator import beamform_top, omega

def mmse_current_lib(P, z, depth, Pq, sigma0, seed=0, uc_known=None,
                     kp_known=None):
    """hand-prior MMSE with global current: selected by GP evidence, or
    given (uc_known) — the oracle-Uc decisive test."""
    rng = np.random.default_rng(seed)
    n = len(z)
    vi = rng.choice(n, max(8, n // 4), replace=False)
    vmask = np.zeros(n, bool)
    vmask[vi] = True
    Ptr, ztr = P[~vmask], z[~vmask]
    Pv, zv = P[vmask], z[vmask]
    if kp_known is not None:
        kp = float(kp_known)
    else:
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


