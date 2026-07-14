"""AA15a: is kp_pipe (0.243, lam 26 m) a genuine wind-sea peak, or an artifact?

AA15a is the one fan-protocol record where kp_pipe is 2.4x kp_true but the true
peak is NOT below the patch fundamental (k_min 0.097 < kp_true 0.100 — by ONE
spectral bin). Two competing stories:

  (a) genuine bimodal sea: 63 m swell (temporal peak) + 26 m wind sea (spatial
      peak). Then kp_pipe is a defensible choice, just a different peak.
  (b) artifact: the true peak sits at bin 1 of the patch, so the spatial
      spectrum cannot locate it, and the estimator falls onto a higher-k
      feature.

Discriminator: the frequency each spatial-peak wavenumber implies. If the sea
really carries 26 m energy, the temporal PSD must show a peak at the dispersion
frequency of k=0.243 (f=0.246 Hz, T=4.1 s). If that frequency is only a shoulder
on the flank of the 0.152 Hz peak, story (b).
"""
import sys
import numpy as np
from scipy.signal import welch

sys.path.insert(0, 'scripts')
from d3_sa_multirecord import MRecord  # noqa: E402

G = 9.81
PATH = 'data/real_bergamasco/AA_2015/Surfaces_20150305_103500.nc'
DEPTH = 17.0


def f_of_k(k, d):
    return float(np.sqrt(G * k * np.tanh(k * d)) / (2 * np.pi))


rec = MRecord(PATH, stride=1, robust=True)
n = min(8192, rec.T_all)
Z = rec.window(0, n)
iy, ix = np.nonzero(rec.valid)
rng = np.random.default_rng(0)
sel = rng.choice(len(iy), size=500, replace=False)
ts = Z[:, iy[sel], ix[sel]]
ts = ts - np.nanmean(ts, axis=0, keepdims=True)
ts = ts[:, np.all(np.isfinite(ts), axis=0)]
f, P = welch(ts, fs=1 / rec.dt, nperseg=2048, axis=0)
Pm = P.mean(axis=1)

f_true = f_of_k(0.100, DEPTH)     # temporal peak -> lam 63 m
f_pipe = f_of_k(0.243, DEPTH)     # pipeline spatial peak -> lam 26 m
band = (f > 0.05) & (f < 0.8)
m = np.trapezoid(Pm[band], f[band])

print(f"AA15a temporal PSD (500 px, {n} frames = {n*rec.dt:.0f} s)\n")
print(f"{'f [Hz]':>8}{'T [s]':>7}{'lam [m]':>9}{'PSD':>10}{'% of var':>10}")
print('-' * 46)
for ftgt, tag in ((f_true, '<- kp_true (temporal peak, lam 63 m)'),
                  (f_pipe, '<- kp_pipe (pipeline spatial, lam 26 m)')):
    j = int(np.argmin(np.abs(f - ftgt)))
    om = 2 * np.pi * f[j]
    k = om ** 2 / G
    for _ in range(60):
        t = np.tanh(k * DEPTH)
        k -= (G * k * t - om ** 2) / (G * (t + k * DEPTH * (1 - t ** 2)))
    print(f"{f[j]:8.3f}{1/f[j]:7.1f}{2*np.pi/k:9.1f}{Pm[j]:10.4g}"
          f"{100*Pm[j]*np.median(np.diff(f))/m:10.2f}  {tag}")

print("\nis f_pipe a local maximum of the temporal PSD?")
j = int(np.argmin(np.abs(f - f_pipe)))
w = 4
loc = Pm[j - w:j + w + 1]
is_peak = int(np.argmax(loc)) == w
print(f"   PSD around f={f[j]:.3f}: {np.array2string(loc, precision=4)}")
print(f"   local max at f_pipe? {'YES -> genuine second peak' if is_peak else 'NO -> monotone flank, i.e. a SHOULDER'}")

jt = int(np.argmax(Pm[band]))
print(f"\n   global peak of the temporal PSD: f={f[band][jt]:.3f} Hz "
      f"(T={1/f[band][jt]:.1f} s)")
print(f"   PSD(f_true)/PSD(f_pipe) = {Pm[int(np.argmin(np.abs(f-f_true)))]/Pm[j]:.2f}x")
rec.f.close()
