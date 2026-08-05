"""Per-record instrument response: push a KNOWN cos^{2s} spread through the exact
same grid / mask / window length / estimator that the record used.

Longuet-Higgins cos^{2s}(theta/2):  r1 = s/(s+1)  =>  sigma_1 = sqrt(2/(s+1)) rad.
All components share one omega, so the field is rank-2 in time:
    eta(t) = cos(wt) Re{Psi} + sin(wt) Im{Psi}
"""
import numpy as np
from spread_est import (analyse_block, moments_from_E, band_summary,
                        spread_from_moments)

NDIR = 360


def sigma_to_s(sig_deg):
    if sig_deg <= 0:
        return np.inf
    return max(2.0 / np.radians(sig_deg) ** 2 - 1.0, 1e-3)


def synth_field(nx, ny, dx, dy, nt, dt, lam, fpeak, sig_deg, theta0, rng,
                mask=None):
    k = 2 * np.pi / lam
    w = 2 * np.pi * fpeak
    x = np.arange(nx) * dx
    y = np.arange(ny) * dy
    th = np.linspace(-np.pi, np.pi, NDIR, endpoint=False)
    s = sigma_to_s(sig_deg)
    if np.isinf(s):
        D = np.zeros(NDIR); D[np.argmin(abs(th - theta0))] = 1.0
    else:
        D = np.power(np.clip(np.cos(0.5 * (th - theta0)) ** 2, 0, None), s)
    D = D / D.sum()
    A = np.sqrt(D)
    ph = rng.uniform(0, 2 * np.pi, NDIR)
    Psi = np.zeros((nx, ny), complex)
    kx, ky = k * np.cos(th), k * np.sin(th)
    for j in range(NDIR):
        if A[j] <= 0:
            continue
        Psi += A[j] * np.exp(1j * (kx[j] * x[:, None] + ky[j] * y[None, :] + ph[j]))
    tt = np.arange(nt) * dt
    eta = (np.cos(w * tt)[:, None, None] * Psi.real[None].astype(np.float32) +
           np.sin(w * tt)[:, None, None] * Psi.imag[None].astype(np.float32)).astype(np.float32)
    if mask is not None:
        eta *= mask[None].astype(np.float32)
    a1 = float((D * np.cos(th)).sum())
    b1 = float((D * np.sin(th)).sum())
    return eta, float(spread_from_moments(a1, b1))


def response(nx, ny, dx, dy, nt, dt, lam, fpeak, mask=None,
             sigs=(0, 10, 20, 30, 40, 50), n_real=4, seed=20260805,
             depth=None, Lmin=None):
    """Mean instrument response over n_real random-phase realizations.

    n_real matters: over an aperture of ~1 wavelength a random-phase sea gives
    very few independent 'looks', so a single realization is not the response,
    it is one draw from a wide sampling distribution.  The per-realization
    scatter is returned too -- it IS the sampling uncertainty on the record's
    own number.
    """
    out = []
    thetas = (0.31, 0.31 + np.pi / 4)   # two aperture orientations, averaged AFTER
    for sg in sigs:
        tots, bands, per, st = [], [], [], None
        for it, th0 in enumerate(thetas):
            Eacc = None
            for m in range(n_real):
                rng = np.random.default_rng(seed + 1009 * m + 77 * it + int(round(sg)))
                eta, st = synth_field(nx, ny, dx, dy, nt, dt, lam, fpeak, sg, th0,
                                      rng, mask=mask)
                f, kx, ky, E = analyse_block(eta, dt, dx, dy)
                a1, b1, S, _, _ = moments_from_E(f, kx, ky, E, use_shell=True,
                                                 depth=depth, Lmin=Lmin)
                per.append(float(band_summary(f, a1, b1, S)['spread_total_deg']))
                Eacc = E if Eacc is None else Eacc + E
                del eta, E
            Eacc /= n_real
            a1, b1, S, _, _ = moments_from_E(f, kx, ky, Eacc, use_shell=True,
                                             depth=depth, Lmin=Lmin)
            bs = band_summary(f, a1, b1, S)
            tots.append(bs['spread_total_deg']); bands.append(bs['spread_band_deg'])
            del Eacc
        out.append(dict(sigma_true=st,
                        sigma_meas_band=float(np.mean(bands)),
                        sigma_meas_total=float(np.mean(tots)),
                        by_theta0=[float(v) for v in tots],
                        per_real_total=per,
                        real_sd=float(np.std(per))))
    return out


def invert(resp, measured, key='sigma_meas_total'):
    """Invert the response curve. -> (sigma_true_est, status)."""
    xs = np.array([r['sigma_true'] for r in resp], float)
    ys = np.array([r[key] for r in resp], float)
    if not np.all(np.isfinite(ys)):
        return np.nan, 'nonfinite'
    if not np.all(np.diff(ys) > 0):
        return np.nan, 'non_monotone_response'   # estimator carries no directional info
    if measured <= ys[0]:
        return 0.0, 'at_or_below_floor'
    if measured >= ys[-1]:
        return float(xs[-1]), 'above_top_of_curve'
    return float(np.interp(measured, ys, xs)), 'ok'
