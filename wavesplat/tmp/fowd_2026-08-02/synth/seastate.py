"""
Sea-state parameters from a point elevation time series.

Independent implementation (dual-verification partner for the FOWD toolkit).
Definitions follow the standard spectral-moment formulation used in FOWD
(Haefner et al. 2021) and Goda (2010); every formula is written out so the
FOWD cross-check can be made term by term.
"""

import numpy as np


def spectral_moments(f, S, orders=(-2, -1, 0, 1, 2, 3, 4)):
    """Spectral moments m_n = int f^n S(f) df over the resolved band."""
    out = {}
    for n in orders:
        out[n] = float(np.trapezoid(S * f ** n, f))
    return out


def welch_psd(eta, fs, nperseg=None, detrend=True):
    """Welch PSD with Hann window, 50% overlap. Returns (f, S) one-sided."""
    eta = np.asarray(eta, float)
    if detrend:
        eta = eta - eta.mean()
    n = len(eta)
    if nperseg is None:
        nperseg = min(n, 256 * 4)
    step = nperseg // 2
    win = np.hanning(nperseg)
    norm = (win ** 2).sum() * fs
    segs = []
    for start in range(0, n - nperseg + 1, step):
        seg = eta[start:start + nperseg]
        seg = seg - seg.mean()
        F = np.fft.rfft(seg * win)
        segs.append((np.abs(F) ** 2) / norm)
    P = np.mean(segs, axis=0)
    P[1:-1] *= 2.0
    f = np.fft.rfftfreq(nperseg, 1.0 / fs)
    return f, P


def zero_crossing_waves(eta, fs):
    """Down-crossing wave heights/periods (FOWD uses zero down-crossings)."""
    eta = np.asarray(eta, float)
    eta = eta - eta.mean()
    t = np.arange(len(eta)) / fs
    sign = np.sign(eta)
    # indices where sign goes + -> -
    idx = np.where((sign[:-1] > 0) & (sign[1:] <= 0))[0]
    if len(idx) < 2:
        return np.array([]), np.array([])
    # linear interpolation of crossing times
    tc = t[idx] + (0 - eta[idx]) / (eta[idx + 1] - eta[idx]) / fs
    heights, periods = [], []
    for a, b in zip(idx[:-1], idx[1:]):
        seg = eta[a:b + 1]
        heights.append(seg.max() - seg.min())
        periods.append(tc[list(idx).index(b)] - tc[list(idx).index(a)])
    return np.array(heights), np.array(np.diff(tc))


def sea_state_params(eta, fs, depth=50.0, band=(0.02, 0.6)):
    """
    Full parameter set for one elevation record.

    Returns dict with spectral + time-domain parameters.
    """
    eta = np.asarray(eta, float)
    eta = eta - eta.mean()
    f, S = welch_psd(eta, fs)
    sel = (f >= band[0]) & (f <= band[1])
    f_b, S_b = f[sel], S[sel]

    m = spectral_moments(f_b, S_b)
    m0, m1, m2, m4 = m[0], m[1], m[2], m[4]

    Hs = 4.0 * np.sqrt(m0)
    Tm01 = m0 / m1 if m1 > 0 else np.nan
    Tm02 = np.sqrt(m0 / m2) if m2 > 0 else np.nan

    # peak frequency (parabolic refinement around the max bin)
    ip = int(np.argmax(S_b))
    if 0 < ip < len(S_b) - 1:
        y0, y1, y2 = S_b[ip - 1], S_b[ip], S_b[ip + 1]
        denom = (y0 - 2 * y1 + y2)
        delta = 0.5 * (y0 - y2) / denom if denom != 0 else 0.0
        fp = f_b[ip] + delta * (f_b[1] - f_b[0])
    else:
        fp = f_b[ip]
    Tp = 1.0 / fp if fp > 0 else np.nan

    # --- bandwidth measures ---
    # narrowness nu (Longuet-Higgins 1975): sqrt(m0 m2 / m1^2 - 1)
    nu = np.sqrt(max(m0 * m2 / m1 ** 2 - 1.0, 0.0)) if m1 > 0 else np.nan
    # broadness eps (Cartwright & Longuet-Higgins 1956)
    eps = np.sqrt(max(1.0 - m2 ** 2 / (m0 * m4), 0.0)) if m4 > 0 else np.nan
    # Goda peakedness
    Qp = (2.0 / m0 ** 2) * np.trapezoid(f_b * S_b ** 2, f_b) if m0 > 0 else np.nan

    # --- wavenumber at peak (full dispersion) ---
    kp = _wavenumber(2 * np.pi * fp, depth) if fp > 0 else np.nan
    steepness = 0.5 * kp * Hs
    Ursell = (Hs / 2.0) / (kp ** 2 * depth ** 3) if depth > 0 else np.nan

    # Benjamin-Feir index (FOWD / Serio et al. 2005 form):
    # BFI = steepness / (nu) * sqrt(...) -- we use the common narrow-band form
    # BFI = sqrt(2) * steepness / bandwidth, with bandwidth = nu
    BFI = np.sqrt(2.0) * steepness / nu if nu and nu > 0 else np.nan

    # --- time domain ---
    sk = float(np.mean(eta ** 3) / np.mean(eta ** 2) ** 1.5)
    ku = float(np.mean(eta ** 4) / np.mean(eta ** 2) ** 2)

    return dict(
        Hs=float(Hs), Tm01=float(Tm01), Tm02=float(Tm02), Tp=float(Tp),
        fp=float(fp), nu=float(nu), eps=float(eps), Qp=float(Qp),
        kp=float(kp), steepness=float(steepness), Ursell=float(Ursell),
        BFI=float(BFI), skewness=sk, excess_kurtosis=ku - 3.0,
        m0=m0, m1=m1, m2=m2, m4=m4,
    )


def _wavenumber(omega, depth, g=9.81):
    """Solve omega^2 = g k tanh(k d) by Newton iteration."""
    k = omega ** 2 / g
    for _ in range(80):
        th = np.tanh(k * depth)
        fk = g * k * th - omega ** 2
        dfk = g * th + g * k * depth * (1 - th ** 2)
        k = k - fk / dfk
        if abs(fk) < 1e-14:
            break
    return float(k)


def analytic_excess_kurtosis(amps):
    """
    Excess kurtosis of eta = sum_j a_j cos(theta_j), theta_j iid U(0, 2pi),
    frequencies incommensurate.

      var  = sum a_j^2 / 2
      E[eta^4] = (3/8) sum a_j^4 + (3/4) * 2 * sum_{i<j} a_i^2 a_j^2 * ... ->
      exact result: excess kurtosis = -1.5 * sum a^4 / (sum a^2)^2
    """
    a2 = np.sum(np.asarray(amps, float) ** 2)
    a4 = np.sum(np.asarray(amps, float) ** 4)
    return -1.5 * a4 / a2 ** 2
