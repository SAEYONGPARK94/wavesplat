"""
bank_mmse.py — operational port of the research bank-MMSE estimator.

This is the estimator behind every published real-sea field number
(``scripts/d3_sa_nomp.mmse_bank_estimate``, the 2026-07-05 winner), ported into
the package so the REAL-DOMAIN GATE can route to it instead of a single
``v2_konly`` run.  Measured on the frozen GIF-ledger windows (2026-07-27,
tmp/confirm2_2026-07-27/real_gif_info.json) the deployed ``v2_konly`` path sat
at nowcast nRMSE 0.858 / 0.927 / 1.048 (AA15a w5 / AA15b w3 / YS17 w0) while
this estimator, under the OBSERVATION-ONLY parametrization implemented here,
reaches 0.498 / 0.666 / 0.718 (median-of-3 field, tmp/bankmmse_port_2026-07-29/
m3_field.json) — the dominant real-sea gap was the missing port, not the method.

Two deliberate differences from the research call, both required for a GT-free
deployment and both measured to preserve the skill
(tmp/bankmmse_port_2026-07-29/blind_vs_frozen.json):

  * ``sigma0 = 0.05 * Hs_obs`` with ``Hs_obs = 4*std(z)`` from the observations
    themselves (the research harness used the stereo ground-truth Hs; the
    val-selected noise multiplier grid {1,3,10} absorbs the few-% mismatch);
  * ``kp_hint = None``: the peak wavenumber comes from the estimator's own
    two-stage beamform zoom on the observations (the research harness passed
    the GT-derived kp).

The fit/render split is new but cannot change values: the estimator is linear
in z and renders every query point independently (``out = (Phi_q*gam) @ w_pred``
per chunk), which the GIF harness asserted, not assumed.  The math below is a
verbatim transcription of the ``mmse_bank_estimate`` default path (no
dual-band, no bound-wave, no gamma-tilt, no Doppler — none of which the frozen
real-sea protocol used); ``test_bank_mmse_port.py`` pins bit-exactness against
the research function.
"""

import numpy as np
import torch

G = 9.81

# Same device policy as the research implementation (d18_relax_estimator.DEV):
# the beamform zoom is the only torch stage; everything else is float64 numpy.
_DEV = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def _omega(k, d):
    """Deep/finite-depth dispersion, verbatim from d18_relax_estimator.omega."""
    return np.sqrt(G * k * np.tanh(k * d) + 1e-12)


def _beamform_top(P, r, depth, klim=0.6, n=80):
    """Strongest dispersion-constrained beamform peak of residual r.

    Verbatim from d18_relax_estimator.beamform_top (same device policy, same
    float32 grid, same 2048-chunking), so the two-stage kp zoom of the port is
    bit-identical to the research estimator's.
    """
    x = torch.tensor(P[:, 0], dtype=torch.float32, device=_DEV)
    y = torch.tensor(P[:, 1], dtype=torch.float32, device=_DEV)
    t = torch.tensor(P[:, 2], dtype=torch.float32, device=_DEV)
    rt = torch.tensor(r, dtype=torch.float32, device=_DEV)
    ax = torch.linspace(-klim, klim, n, device=_DEV)
    KX, KY = torch.meshgrid(ax, ax, indexing='xy')
    kx = KX.reshape(-1)
    ky = KY.reshape(-1)
    k = torch.sqrt(kx ** 2 + ky ** 2 + 1e-9)
    om = torch.sqrt(G * k * torch.tanh(k * depth) + 1e-9)
    pw = torch.empty(kx.shape[0], device=_DEV)
    B = 2048
    for i in range(0, kx.shape[0], B):
        ps = kx[i:i + B, None] * x[None] + ky[i:i + B, None] * y[None] \
            - om[i:i + B, None] * t[None]
        pw[i:i + B] = (rt[None] * torch.cos(ps)).sum(1) ** 2 \
            + (rt[None] * torch.sin(ps)).sum(1) ** 2
    j = int(torch.argmax(pw))
    return float(kx[j]), float(ky[j])


class BankMMSEFit:
    """One fitted spectrum-weighted linear-MMSE bank.

    Holds exactly the state the render needs: the bank (kxb, kyb, om), the
    val/EM-selected prior variances gam (2B,), and the dual weights
    w_pred = Phi_all.T @ K^{-1} z.  ``render`` is pure linear-in-z evaluation;
    calling it never changes the fit.
    """

    def __init__(self, kxb, kyb, om, gam, w_pred, *, kp, dk_c, seed, sigma0,
                 smult, pexp, vr, vr_rel, n_obs):
        self.kxb = kxb
        self.kyb = kyb
        self.om = om
        self.gam = gam
        self.w_pred = w_pred
        self.kp = float(kp)
        self.dk_c = float(dk_c)
        self.seed = int(seed)
        self.sigma0 = float(sigma0)
        self.smult = float(smult)
        self.pexp = float(pexp)
        # held-out predictive misfit of the selected prior (GT-free fit signal)
        self.vr = float(vr)
        self.vr_rel = float(vr_rel)
        self.n_obs = int(n_obs)
        self.n_candidates = int(len(kxb))

    def _design(self, Q):
        psi = self.kxb[None] * Q[:, 0:1] + self.kyb[None] * Q[:, 1:2] \
            - self.om[None] * Q[:, 2:3]
        return np.concatenate([np.cos(psi), np.sin(psi)], axis=1)

    def render(self, Pq):
        """Field at query points Pq (m, 3) = (x, y, t).  Chunked as in the
        research renderer (real grids can be ~1e4 px)."""
        Pq = np.asarray(Pq, float)
        out = np.empty(len(Pq))
        for i in range(0, len(Pq), 4096):
            Phi_q = self._design(Pq[i:i + 4096])
            out[i:i + 4096] = (Phi_q * self.gam[None]) @ self.w_pred
        return out


def fit_bank_mmse(P, z, depth, sigma0, kp_hint=None, dk_c=None, val_frac=0.25,
                  seed=0, max_em=6, band=(0.35, 2.5)):
    """Fit the spectrum-weighted linear MMSE over the fine continuous bank.

    Verbatim transcription of scripts/d3_sa_nomp.mmse_bank_estimate (default
    path) up to its query render, returning a BankMMSEFit instead of rendering:
    NO identification; per-candidate prior variance from the (PSF-smeared)
    beamform power; shape hyperparams (power exponent, noise multiplier)
    val-selected; then val-gated SBL-EM refinement of the prior.

    sigma0: nominal obs noise std (deployment: 0.05 * (4*std(z))); the
    val-selected multiplier grid {1,3,10} absorbs misspecification.
    kp_hint: peak wavenumber if known; None (deployment) = two-stage beamform
    zoom on the observations.  dk_c: bank spacing, default kp/10.
    """
    P = np.asarray(P, float)
    z = np.asarray(z, float)
    rng = np.random.default_rng(seed)
    n = len(z)
    vi = rng.choice(n, max(8, int(val_frac * n)), replace=False)
    vmask = np.zeros(n, bool)
    vmask[vi] = True
    Ptr, ztr = P[~vmask], z[~vmask]
    Pv, zv = P[vmask], z[vmask]
    if kp_hint is None:
        bx, by = _beamform_top(P.astype(np.float32), z.astype(np.float32),
                               depth, klim=2.0, n=120)
        kp = max(np.hypot(bx, by), 1e-3)
        bx, by = _beamform_top(P.astype(np.float32), z.astype(np.float32),
                               depth, klim=3.5 * kp, n=120)
        kp = max(np.hypot(bx, by), 1e-3)
    else:
        kp = float(kp_hint)
    if dk_c is None:
        dk_c = kp / 10.0
    b_lo, b_hi = band
    ax = np.arange(-b_hi * kp, b_hi * kp + dk_c / 2, dk_c)
    KX, KY = np.meshgrid(ax, ax)
    kk = np.hypot(KX, KY)
    sel = (kk > b_lo * kp) & (kk < b_hi * kp)
    kxb, kyb = KX[sel].astype(float), KY[sel].astype(float)
    om = _omega(np.hypot(kxb, kyb) + 1e-12, depth)
    B = len(kxb)

    def design(Q):
        psi = kxb[None] * Q[:, 0:1] + kyb[None] * Q[:, 1:2] \
            - om[None] * Q[:, 2:3]
        return np.concatenate([np.cos(psi), np.sin(psi)], axis=1)

    Phi_tr = design(Ptr)
    Phi_v = design(Pv)
    # prior shape: beamform power at the bank points (all obs)
    pw = np.empty(B)
    CH = 512
    for i in range(0, B, CH):
        psi = (kxb[i:i + CH, None] * P[None, :, 0]
               + kyb[i:i + CH, None] * P[None, :, 1]
               - om[i:i + CH, None] * P[None, :, 2])
        pw[i:i + CH] = (z[None] * np.cos(psi)).sum(1) ** 2 \
            + (z[None] * np.sin(psi)).sum(1) ** 2
    pw = pw / max(pw.max(), 1e-12)
    m = len(ztr)
    best = None
    for pexp in (0.5, 1.0, 2.0):
        g = pw ** pexp
        g = g / g.sum()
        gam = np.concatenate([g, g]) * np.var(z)
        for smult in (1.0, 3.0, 10.0):
            s2 = (smult * sigma0) ** 2
            K = (Phi_tr * gam[None]) @ Phi_tr.T + s2 * np.eye(m)
            fv = (Phi_v * gam[None]) @ (Phi_tr.T @ np.linalg.solve(K, ztr))
            vr = float(np.sqrt(np.mean((fv - zv) ** 2)))
            if best is None or vr < best[0]:
                best = (vr, pexp, smult, gam)
    vr_best, pexp, smult, gam = best
    s2 = (smult * sigma0) ** 2
    # val-gated SBL-EM refinement of the prior shape (empirical Bayes)
    best_em = (vr_best, gam.copy())
    for _ in range(max_em):
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
    alpha = np.linalg.solve(K, z)
    w_pred = Phi_all.T @ alpha
    return BankMMSEFit(kxb, kyb, om, gam, w_pred,
                       kp=kp, dk_c=dk_c, seed=seed, sigma0=sigma0,
                       smult=smult, pexp=pexp, vr=best_em[0],
                       vr_rel=best_em[0] / (np.std(zv) + 1e-12), n_obs=n)
