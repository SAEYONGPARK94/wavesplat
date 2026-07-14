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


rows = []
print(f"{'rec':<7}{'win':>4}{'kp_spatial':>12}{'kp_causal':>11}{'kp_true':>9}"
      f"{'causal/true':>13}{'spatial/true':>14}")
print('-' * 72)
for name in RECS:
    cfg = RECORDS[name]
    rec = MRecord(cfg['path'], stride=cfg['stride'])
    nfw = int(round(40.0 / rec.dt))
    n_lead = int(round(cfg['lead_max'] / rec.dt))
    i0s = np.linspace(nfw + 1, rec.T_all - n_lead - 2, 12).astype(int)
    kp_true = 2 * np.pi / TRUE[name]['lam_temporal']
    for wi, i0 in enumerate(i0s):
        Zmem = rec.window(int(i0) - nfw + 1, nfw)
        _, _, _, kp_sp = directional_spread(np.nan_to_num(Zmem[-1]),
                                            rec.dx, rec.dy)
        kp_c, fp_c = kp_causal(Zmem, rec.dt, cfg['depth'])
        rows.append(dict(record=name, wi=wi, kp_spatial=float(kp_sp),
                         kp_causal=float(kp_c), kp_true=float(kp_true),
                         r_causal=float(kp_c / kp_true),
                         r_spatial=float(kp_sp / kp_true),
                         lam_causal=float(2 * np.pi / kp_c)))
        r = rows[-1]
        print(f"{name:<7}{wi:>4}{kp_sp:12.3f}{kp_c:11.3f}{kp_true:9.3f}"
              f"{r['r_causal']:13.2f}{r['r_spatial']:14.2f}")
    rec.f.close()

json.dump(rows, open('tmp/e12a_kp_causal.json', 'w'), indent=1)

print('\n=== E-12a GATE: 0.80 <= kp_causal/kp_true <= 1.25 on all 5 records ===')
allpass = True
print(f"{'rec':<8}{'median ratio':>14}{'min':>7}{'max':>7}{'within 25%':>12}"
      f"{'  (spatial median)':>18}")
for name in RECS:
    rr = np.array([r['r_causal'] for r in rows if r['record'] == name])
    rs = np.array([r['r_spatial'] for r in rows if r['record'] == name])
    ok = np.mean((rr >= 0.8) & (rr <= 1.25))
    med = float(np.median(rr))
    rec_pass = 0.8 <= med <= 1.25
    allpass &= rec_pass
    print(f"{name:<8}{med:14.2f}{rr.min():7.2f}{rr.max():7.2f}"
          f"{100*ok:11.0f}%{float(np.median(rs)):18.2f}")
print(f"\nE-12a: {'PASS' if allpass else 'FAIL'} "
      f"(median ratio within [0.80, 1.25] on {'all 5' if allpass else 'NOT all'} records)")
