"""Validate the temporal lam_p estimator + inspect the BS13 disagreement.

Check 1 — published values: Guimaraes 2020 quotes BS Hs 0.30 / Tp 3.0,
          AA Hs 1.36 / Tp 4.5, YS Hs 1.94 / Tp 6.7, LJ Hs 10.0 / Tp 13.3.
Check 2 — same-sea invariance: LJ Grid1 (40 m patch) and LJ Grid2 (250 m patch)
          are the SAME sea at the SAME timestamp. A patch-independent estimator
          must return the same lam_p on both. The codebase's spatial values do
          not (40.6 m vs 300 m).
Check 3 — is the BS13a peak (Tp 5.3 s) a real spectral peak or a shoulder?
"""
import sys
import numpy as np
from scipy.signal import welch

sys.path.insert(0, 'scripts')
from d3_sa_multirecord import MRecord  # noqa: E402

# codebase's hardcoded spatial characterization: fp = 0.18 * lam_p
CODEBASE_FP = {'BS11a': 2.5, 'BS11b': 1.7, 'BS13a': 3.4, 'BS13b': 3.1,
               'AA15a': 9.3, 'AA14': 6.3, 'AA15b': 7.2, 'YS17': 10.5,
               'LJ18n': 7.3, 'LJ18g': 54.0}
PATHS = {
    'BS13a': ('data/real_bergamasco/BS_2013/Surfaces_20130925_121501_short.nc', 30.0, 1),
    'BS11a': ('data/real_bergamasco/BS_2011/Surfaces_20111004_113800_short.nc', 30.0, 1),
    'LJ18n': ('data/real_bergamasco/LJ_2018/Surfaces_20180103_093938_Grid1_short.nc', 45.0, 2),
    'LJ18g': ('data/real_bergamasco/LJ_2018/Surfaces_20180103_093938_Grid2_short.nc', 45.0, 2),
}
G = 9.81


def k_from_omega(w, d):
    k = w ** 2 / G
    for _ in range(80):
        t = np.tanh(k * d)
        k -= (G * k * t - w ** 2) / (G * (t + k * d * (1 - t ** 2)))
        k = max(k, 1e-6)
    return k


print(f"{'rec':<8}{'patch':>10}{'Tp':>7}{'lam_T':>8}{'lam_codebase':>14}{'ratio':>8}")
print('-' * 56)
spec = {}
for name, (path, depth, stride) in PATHS.items():
    rec = MRecord(path, stride=stride, robust=True)
    n = min(4096, rec.T_all)
    Z = rec.window(0, n)
    iy, ix = np.nonzero(rec.valid)
    rng = np.random.default_rng(0)
    sel = rng.choice(len(iy), size=min(300, len(iy)), replace=False)
    ts = Z[:, iy[sel], ix[sel]]
    ts = ts - np.nanmean(ts, axis=0, keepdims=True)
    ts = ts[:, np.all(np.isfinite(ts), axis=0)]
    f, P = welch(ts, fs=1 / rec.dt, nperseg=min(1024, n), axis=0)
    Pm = P.mean(axis=1)
    band = (f > 0.06) & (f < 0.4 / rec.dt)
    fp = float(f[band][np.argmax(Pm[band])])
    lam_t = 2 * np.pi / k_from_omega(2 * np.pi * fp, depth)
    lam_c = CODEBASE_FP[name] / 0.18
    L = min(np.ptp(rec.gx), np.ptp(rec.gy))
    print(f'{name:<8}{L:9.0f}m{1/fp:7.1f}{lam_t:8.1f}{lam_c:14.1f}'
          f'{lam_t/lam_c:8.2f}')
    spec[name] = (f[band], Pm[band], fp, L)
    rec.f.close()

print('\n--- Check 2: LJ Grid1 (40 m) vs Grid2 (250 m), same sea, same instant ---')
print('  temporal lam_p :  Grid1 %.1f m   Grid2 %.1f m   -> ratio %.2f'
      % (2 * np.pi / k_from_omega(2 * np.pi * spec["LJ18n"][2], 45.0),
         2 * np.pi / k_from_omega(2 * np.pi * spec["LJ18g"][2], 45.0),
         spec['LJ18n'][2] / spec['LJ18g'][2]))
print('  codebase lam_p :  Grid1 %.1f m   Grid2 %.1f m   -> ratio %.2f'
      % (7.3 / 0.18, 54.0 / 0.18, (7.3 / 0.18) / (54.0 / 0.18)))

print('\n--- Check 3: BS13a spectrum, top peaks (is Tp 5.3 s real?) ---')
f, Pm, fp, L = spec['BS13a']
order = np.argsort(Pm)[::-1][:6]
tot = Pm.sum()
for j in sorted(order):
    star = ' <-- peak' if abs(f[j] - fp) < 1e-9 else ''
    print(f'   f={f[j]:.3f} Hz  T={1/f[j]:5.1f} s  '
          f'PSD={Pm[j]:.4g}  ({100*Pm[j]/tot:4.1f}% of band){star}')
print(f'   patch = {L:.0f} m  ->  largest wavelength the spatial spectrum '
      f'can represent = {L:.0f} m')
