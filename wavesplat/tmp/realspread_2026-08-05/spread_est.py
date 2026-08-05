"""Directional spread of WASS space-time surfaces, using CDIP's OWN estimator.

ARM IDENTITY (the whole point):
    sigma_1(f) = sqrt(2 * (1 - sqrt(a1(f)^2 + b1(f)^2)))          [rad]
    a1(f) = INT E(f,theta) cos(theta) dtheta / INT E(f,theta) dtheta
    b1(f) = INT E(f,theta) sin(theta) dtheta / INT E(f,theta) dtheta
which is *exactly* what a heave-pitch-roll buoy reports.  CDIP measures a1/b1
directly; here we measure the full E(f,kx,ky) and integrate it down to the same
two numbers.  Both are moments of a CONTINUOUS directional distribution --
unlike our synthetic generator, whose a1/b1 come from a DISCRETE component set.

Two band summaries, both mirrored from tmp/cdip_support_2026-08-03/cdip_support.py:
    spread_band_deg  : energy-weighted mean of sigma_1(f) over 0.04-0.50 Hz
    spread_total_deg : sigma_1 of the energy-weighted (A1,B1) over the band

Route: 3-D FFT (t,x,y) -> E(omega>0, kx, ky).  Positive-omega half resolves the
180 deg ambiguity that a single-frame 2-D FFT cannot.  Within each omega slice
the energy is confined to a dispersion ridge |k| = k(omega); the ridge radius is
found EMPIRICALLY (depth- and current-agnostic, so Doppler shift is absorbed)
and an annulus around it rejects off-shell reconstruction noise.

sigma_1 is rotation invariant (r1 = |a1 + i b1|), so the unknown alignment of the
WASS local frame to true north does NOT affect the number.
"""
import numpy as np
from scipy import fft as sfft

G = 9.80665
FMIN, FMAX = 0.04, 0.50          # identical to cdip_support.py
ANNULUS = (0.65, 1.45)           # multiples of the empirical ridge radius
RIDGE_LO, RIDGE_HI = 0.45, 2.60  # allowed ridge radius / deep-water k


def spread_from_moments(a1, b1):
    """sigma_1 in DEGREES.  Byte-for-byte the CDIP-side formula."""
    a1 = np.asarray(a1, float)
    b1 = np.asarray(b1, float)
    r1 = np.sqrt(np.clip(a1 ** 2 + b1 ** 2, 0, 1))
    return np.degrees(np.sqrt(np.clip(2.0 * (1.0 - r1), 0, None)))


import os
SPATIAL_TAPER = os.environ.get('SPATIAL_TAPER', 'hann')   # hann | tukey0.5 | tukey0.25 | rect


def tukey(n, alpha):
    if alpha <= 0:
        return np.ones(n)
    if alpha >= 1:
        return np.hanning(n + 2)[1:-1]
    w = np.ones(n)
    m = int(np.floor(alpha * (n - 1) / 2)) + 1
    t = np.arange(m)
    ramp = 0.5 * (1 + np.cos(np.pi * (2 * t / (alpha * (n - 1)) - 1)))
    w[:m] = ramp
    w[n - m:] = ramp[::-1]
    return w


def spatial_win(n, kind=None):
    kind = kind or SPATIAL_TAPER
    if kind == 'rect':
        return np.ones(n)
    if kind.startswith('tukey'):
        return tukey(n, float(kind[5:]))
    return np.hanning(n + 2)[1:-1]


def hann3(nt, nx, ny):
    return (np.hanning(nt + 2)[1:-1][:, None, None] *
            spatial_win(nx)[None, :, None] *
            spatial_win(ny)[None, None, :]).astype(np.float32)


def wavenumber_freq_spectrum(z, dt, dx, dy):
    """z (nt,nx,ny) float32, already de-meaned.  -> f (nf,), kx, ky, E(nf,nx,ny)."""
    nt, nx, ny = z.shape
    w = hann3(nt, nx, ny)
    F = sfft.rfft((z * w).astype(np.float32), axis=0, workers=1)
    F = sfft.fft2(F, axes=(1, 2), workers=1)
    E = (F.real.astype(np.float64) ** 2 + F.imag.astype(np.float64) ** 2)
    del F
    f = np.fft.rfftfreq(nt, dt)
    kx = 2 * np.pi * np.fft.fftfreq(nx, dx)
    ky = 2 * np.pi * np.fft.fftfreq(ny, dy)
    return f, kx, ky, E


def k_of_f(f, depth):
    """Linear dispersion  w^2 = g k tanh(k d), solved by fixed-point iteration."""
    w2 = (2 * np.pi * np.asarray(f, float)) ** 2
    k = w2 / G                                   # deep-water start
    if depth is None or not np.isfinite(depth) or depth <= 0:
        return k
    for _ in range(60):
        k = np.where(k * depth > 25, w2 / G, w2 / (G * np.tanh(np.clip(k * depth, 1e-9, 25))))
    return k


def moments_from_E(f, kx, ky, E, use_shell=True, depth=None, Lmin=None):
    """-> per-frequency a1,b1,S(f), shell radius, on-shell energy fraction.

    The shell is the THEORETICAL dispersion radius k(f,d), not an empirical
    ridge: when the peak wave is longer than the aperture (L/lambda < 2, which
    is most of this dataset) there are too few DFT bins for a ridge to exist,
    and the 'ridge' one finds is a binning artefact.

    Annulus half-width = max(0.35 k, dk_res) with dk_res = 2 pi / L: when the
    aperture cannot separate on- from off-shell energy, the annulus admits
    everything rather than pretending to reject noise.
    """
    KX = kx[:, None]
    KY = ky[None, :]
    K = np.hypot(KX, KY)
    TH = np.arctan2(KY, KX)
    cosT, sinT = np.cos(TH), np.sin(TH)
    kny = min(abs(kx).max(), abs(ky).max())

    nf = len(f)
    a1 = np.full(nf, np.nan)
    b1 = np.full(nf, np.nan)
    S = np.zeros(nf)
    ridge = np.full(nf, np.nan)
    onshell = np.full(nf, np.nan)

    Kf = K.ravel()

    for i in range(nf):
        if not (FMIN <= f[i] <= FMAX):
            continue
        Ei = E[i].copy()
        Ei[0, 0] = 0.0                       # kill the k=0 (spatial mean) bin
        tot = Ei.sum()
        S[i] = tot
        if tot <= 0:
            continue
        kr = float(k_of_f(f[i], depth))      # theoretical dispersion shell
        if kr > 0.9 * kny:                   # wave shorter than the sampling
            continue
        ridge[i] = kr
        if use_shell:
            half = max(0.35 * kr, (2 * np.pi / Lmin) if Lmin else 0.0)
            sel = (K >= kr - half) & (K <= kr + half)
        else:
            sel = K > 0
        w = np.where(sel, Ei, 0.0)
        wt = w.sum()
        onshell[i] = wt / tot
        if wt <= 0:
            continue
        a1[i] = (w * cosT).sum() / wt
        b1[i] = (w * sinT).sum() / wt
        if use_shell:
            S[i] = wt                        # band weighting uses on-shell energy
    return a1, b1, S, ridge, onshell


def band_summary(f, a1, b1, S):
    """CDIP's two band collapses, same code path as cdip_support.py."""
    m = (f >= FMIN) & (f <= FMAX) & np.isfinite(a1) & np.isfinite(b1) & (S > 0)
    if not m.any():
        return dict(spread_band_deg=np.nan, spread_total_deg=np.nan,
                    Tp=np.nan, n_bins=0)
    w = S[m]
    sig = spread_from_moments(a1[m], b1[m])
    band = float((w * sig).sum() / w.sum())
    A1 = float((w * a1[m]).sum() / w.sum())
    B1 = float((w * b1[m]).sum() / w.sum())
    total = float(spread_from_moments(A1, B1))
    fp = float(f[m][np.argmax(w)])
    return dict(spread_band_deg=band, spread_total_deg=total,
                Tp=1.0 / fp if fp > 0 else np.nan, fp=fp, n_bins=int(m.sum()),
                A1=A1, B1=B1)


def analyse_block(z, dt, dx, dy, use_shell=True):
    """z (nt,nx,ny) -> accumulate one window's spectrum."""
    z = z - z.mean(axis=0, keepdims=True)
    return wavenumber_freq_spectrum(z, dt, dx, dy)
