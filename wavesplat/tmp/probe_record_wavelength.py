"""Peak wavelength of every real record — two independent estimators.

Why: the E-10 power fix ("records 5 -> 13") wants to add the BS records, whose
patches are only 13-19 m. Whether that is legitimate depends on the aperture /
wavelength ratio, so we need lam_p per record — measured, not assumed.

Estimator A (temporal, trustworthy on small patches):
    Welch PSD of the surface time series at many valid pixels -> f_p -> T_p,
    then invert the linear dispersion omega^2 = g k tanh(k d) -> lam_p.
Estimator B (spatial, the one the codebase already used):
    2D spatial spectrum of single frames -> radial peak -> lam_p.
    B is floored by the patch: it cannot see lam > L. Disagreement A >> B on a
    small patch means B was a patch artifact (as already flagged for LJ18n).

Output: tmp/record_wavelength.json + a table.
"""
import sys
import glob
import json
import numpy as np
from scipy.signal import welch

sys.path.insert(0, 'scripts')
from d3_sa_multirecord import MRecord  # noqa: E402

G = 9.81

# depth [m] per site (Guimaraes 2020; same values the registries use)
DEPTH = {'BS_2011': 30.0, 'BS_2013': 30.0, 'AA_2014': 17.0, 'AA_2015': 17.0,
         'YS_2017': 50.0, 'LJ_2018': 45.0}

# registered fan outer radius (d3_falcon_real_operator.RECORDS), else None
FAN = {
    'YS_2017/Surfaces_20170513_050000.nc': ('YS17', 100.0),
    'AA_2015/Surfaces_20150305_103500.nc': ('AA15a', 40.0),
    'AA_2014/Surfaces_20140327_091000.nc': ('AA14', 40.0),
    'AA_2015/Surfaces_20150515_090000.nc': ('AA15b', 40.0),
    'LJ_2018/Surfaces_20180103_093938_Grid2_short.nc': ('LJ18g', 100.0),
}
SA_NAME = {  # names used by the 8-record set in d3_sa_multirecord
    'BS_2011/Surfaces_20111004_113800_short.nc': 'BS11a',
    'BS_2011/Surfaces_20111001_161800_short.nc': 'BS11b',
    'BS_2013/Surfaces_20130925_121501_short.nc': 'BS13a',
    'BS_2013/Surfaces_20130930_102001_short.nc': 'BS13b',
    'BS_2011/Surfaces_20111004_130700_short.nc': 'BS11c',
    'BS_2011/Surfaces_20111004_153000_short.nc': 'BS11d',
    'BS_2013/Surfaces_20130922_130001_short.nc': 'BS13c',
    'LJ_2018/Surfaces_20180103_093938_Grid1_short.nc': 'LJ18n',
}


def k_from_omega(w, d):
    """Solve omega^2 = g k tanh(k d) by Newton, deep-water start."""
    k = w ** 2 / G
    for _ in range(80):
        t = np.tanh(k * d)
        f = G * k * t - w ** 2
        df = G * (t + k * d * (1 - t ** 2))
        k = k - f / df
        k = max(k, 1e-6)
    return k


def temporal_lam(rec, depth, n_fr=4096, n_px=300, seed=0):
    n_fr = min(n_fr, rec.T_all)
    Z = rec.window(0, n_fr)                      # (n_fr, n0, n1)
    v = rec.valid
    iy, ix = np.nonzero(v)
    rng = np.random.default_rng(seed)
    sel = rng.choice(len(iy), size=min(n_px, len(iy)), replace=False)
    ts = Z[:, iy[sel], ix[sel]]                  # (n_fr, n_px)
    ts = ts - np.nanmean(ts, axis=0, keepdims=True)
    ok = np.all(np.isfinite(ts), axis=0)
    ts = ts[:, ok]
    fs = 1.0 / rec.dt
    nper = min(1024, n_fr)
    f, P = welch(ts, fs=fs, nperseg=nper, axis=0)
    Pm = P.mean(axis=1)
    band = (f > 0.06) & (f < 0.5 * fs * 0.8)     # drop trend + Nyquist junk
    fp = float(f[band][np.argmax(Pm[band])])
    Tp = 1.0 / fp
    kp = k_from_omega(2 * np.pi * fp, depth)
    Hs = 4.0 * float(np.nanstd(ts))
    return fp, Tp, 2 * np.pi / kp, Hs, kp * depth


def spatial_lam(rec, n_fr=12):
    idx = np.linspace(0, rec.T_all - 1, n_fr).astype(int)
    v = rec.valid
    ny, nx = v.shape
    win = np.outer(np.hanning(ny), np.hanning(nx))
    acc = None
    for i in idx:
        Z = rec.window(i, 1)[0]
        Zf = np.where(v & np.isfinite(Z), Z, 0.0)
        Zf = Zf - Zf[v & np.isfinite(Z)].mean()
        S = np.abs(np.fft.fftshift(np.fft.fft2(Zf * win))) ** 2
        acc = S if acc is None else acc + S
    kx = 2 * np.pi * np.fft.fftshift(np.fft.fftfreq(nx, rec.dx))
    ky = 2 * np.pi * np.fft.fftshift(np.fft.fftfreq(ny, rec.dy))
    KX, KY = np.meshgrid(kx, ky)
    K = np.hypot(KX, KY)
    m = K > 1e-9
    j = np.argmax(np.where(m, acc, -np.inf))
    kp = float(K.flat[j])
    return 2 * np.pi / kp


rows = []
for path in sorted(glob.glob('data/real_bergamasco/*/*.nc')):
    key = '/'.join(path.split('/')[-2:])
    site = key.split('/')[0]
    depth = DEPTH[site]
    reg = FAN.get(key)
    name = reg[0] if reg else SA_NAME.get(key, '?')
    stride = 2 if site in ('YS_2017', 'LJ_2018') else 1

    rec = MRecord(path, stride=stride, robust=True)
    Lx, Ly = float(np.ptp(rec.gx)), float(np.ptp(rec.gy))
    fp, Tp, lam_t, Hs, kd = temporal_lam(rec, depth)
    lam_s = spatial_lam(rec)
    r_fan = reg[1] if reg else min(Ly, Lx / 2.0)   # max hostable if unregistered
    rows.append(dict(key=key, name=name, registered=reg is not None,
                     Lx=Lx, Ly=Ly, depth=depth, Hs=Hs, fp=fp, Tp=Tp, kd=kd,
                     lam_temporal=lam_t, lam_spatial=lam_s,
                     L_over_lam=min(Lx, Ly) / lam_t,
                     r_fan=r_fan, aperture_ratio=r_fan / lam_t))
    rec.f.close()

rows.sort(key=lambda r: (not r['registered'], r['name']))
json.dump(rows, open('tmp/record_wavelength.json', 'w'), indent=1)

print(f"\n{'rec':<8}{'reg':<5}{'patch [m]':>13}{'Hs':>6}{'Tp':>6}{'kd':>6}"
      f"{'lam_T':>8}{'lam_S':>8}{'patch/lam':>11}{'r_fan':>7}{'r/lam':>8}")
print('-' * 92)
for r in rows:
    print(f"{r['name']:<8}{'Y' if r['registered'] else '-':<5}"
          f"{r['Lx']:6.0f}x{r['Ly']:<6.0f}"
          f"{r['Hs']:6.2f}{r['Tp']:6.1f}{r['kd']:6.2f}"
          f"{r['lam_temporal']:8.1f}{r['lam_spatial']:8.1f}"
          f"{r['L_over_lam']:11.2f}{r['r_fan']:7.0f}{r['aperture_ratio']:8.2f}")
print("\nlam_T = temporal PSD + dispersion (valid on small patches)")
print("lam_S = spatial 2D spectrum (floored by patch size -> artifact if ~L)")
