"""E-12a — a CAUSAL peak-wavenumber estimator, and its validation.

The pipeline takes kp from the spatial spectrum of ONE frame (Zmem[-1]), which
is floored by the patch (k >= 2*pi/L).  But Zmem is already a 40 s causal buffer
(~400-500 frames) on the full grid, and the code only uses its last frame.

kp_causal: multi-pixel Welch PSD over the buffer's TIME axis -> peak frequency
(parabolic-interpolated on a zero-padded spectrum) -> linear dispersion -> kp.
Uses only data already in the causal buffer: no future frames, no full record.

GATE (pre-registered, docs/E12_KP_FLOOR_AUDIT_prereg_2026-07-14.md):
    on all 5 fan records, over all 12 frozen windows,
    0.8 <= kp_causal / kp_true <= 1.25
where kp_true is the full-record temporal estimate (tmp/record_wavelength.json).
FAIL -> do not run E-12b; fall back to the E-11 sparse beamforming hint.
"""
import sys
import os
import json
import numpy as np
from scipy.signal import welch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
os.chdir(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, 'scripts')

from d3_falcon_real_operator import RECORDS          # noqa: E402
from d3_sa_multirecord import MRecord                # noqa: E402
from d3_sa_gonogo import directional_spread          # noqa: E402

G = 9.81
RECS = ['YS17', 'AA15a', 'AA14', 'AA15b', 'LJ18g']
TRUE = {r['name']: r for r in json.load(open('tmp/record_wavelength.json'))}


def k_from_omega(w, d):
    k = w ** 2 / G
    for _ in range(80):
        t = np.tanh(k * d)
        k -= (G * k * t - w ** 2) / (G * (t + k * d * (1 - t ** 2)))
        k = max(k, 1e-6)
    return float(k)


def kp_causal(Zmem, dt, depth, n_px=400, f_lo=0.05, f_hi=0.8, pad=8, seed=0):
    """Peak wavenumber from the causal buffer's time axis.

    Zmem : (nt, ny, nx), NaN outside the valid mask, per-pixel temporal mean
           already removed by MRecord.window().
    """
    nt = Zmem.shape[0]
    ok = np.all(np.isfinite(Zmem), axis=0)
    iy, ix = np.nonzero(ok)
    if len(iy) < 20:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    sel = rng.choice(len(iy), size=min(n_px, len(iy)), replace=False)
    ts = Zmem[:, iy[sel], ix[sel]]                     # (nt, n_px)
    ts = ts - ts.mean(axis=0, keepdims=True)

    # one segment = the whole buffer (df = 1/T_buf); the variance reduction
    # comes from averaging over hundreds of pixels, not over time segments.
    # zero-pad to interpolate the peak finer than the bin spacing.
    f, P = welch(ts, fs=1.0 / dt, nperseg=nt, nfft=pad * nt,
                 window='hann', axis=0)
    Pm = P.mean(axis=1)
    band = (f >= f_lo) & (f <= f_hi)
    fb, Pb = f[band], Pm[band]
    j = int(np.argmax(Pb))
    # parabolic refinement in log-power
    if 0 < j < len(Pb) - 1:
        y0, y1, y2 = np.log(Pb[j - 1]), np.log(Pb[j]), np.log(Pb[j + 1])
        denom = y0 - 2 * y1 + y2
        d_ = 0.5 * (y0 - y2) / denom if abs(denom) > 1e-12 else 0.0
        fp = float(fb[j] + np.clip(d_, -0.5, 0.5) * (fb[1] - fb[0]))
    else:
        fp = float(fb[j])
    kp = k_from_omega(2 * np.pi * fp, depth)
    return kp, fp



if __name__ == '__main__':
    raise SystemExit("run tmp/e12a_kp_causal.py for the validation")
