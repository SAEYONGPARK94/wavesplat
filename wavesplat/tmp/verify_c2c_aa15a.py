"""AA15a, settled. Two questions the prior agent conflated:

 Q1: WHICH k does the spatial estimator actually return on AA15a?
     He used window-0 only -> 0.2432 (lam 25.8 m) and built the whole
     "bimodal artifact" story on it. What is the representative value?

 Q2: is the returned k a genuine 2nd peak, pure variance, or BIN QUANTISATION?
     Frame-averaging kills variance but NOT resolution. The right diagnostic is
     the DIRECTIONAL floor: on a Cartesian FFT grid the smallest representable
     |k| ALONG A GIVEN DIRECTION theta is not 2*pi/L -- for a diagonal peak it
     is sqrt(dkx^2 + dky^2), much larger. Check whether the argmax bin is the
     diagonal fundamental (m,n) = (+-1,+-1).
"""
import sys, os, json
import numpy as np
from scipy import signal
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from d3_sa_multirecord import MRecord

G = 9.81
DEPTH = 17.0


def f_of_k(k, d):
    return float(np.sqrt(G * k * np.tanh(k * d)) / (2 * np.pi))


def k_of_f(f, d):
    om = 2 * np.pi * f
    lo, hi = 1e-5, 20.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if G * mid * np.tanh(mid * d) - om ** 2 > 0:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


rec = MRecord('data/real_bergamasco/AA_2015/Surfaces_20150305_103500.nc', stride=1)
nf = 580
Z = rec.window(200, nf)
ny, nx = rec.n0, rec.n1
dkx = 2 * np.pi / (nx * rec.dx)
dky = 2 * np.pi / (ny * rec.dy)
kx = np.fft.fftshift(np.fft.fftfreq(nx, d=rec.dx)) * 2 * np.pi
ky = np.fft.fftshift(np.fft.fftfreq(ny, d=rec.dy)) * 2 * np.pi
KX, KY = np.meshgrid(kx, ky)
K = np.hypot(KX, KY)
m = K > 1e-6
W = np.hanning(ny)[:, None] * np.hanning(nx)[None, :]

acc = np.zeros((ny, nx))
u = 0
for j in range(0, nf, 3):
    F = np.fft.fftshift(np.fft.fft2(np.nan_to_num(Z[j]) * W))
    acc += np.abs(F) ** 2
    u += 1
acc /= u

ipk = int(np.argmax(np.where(m, acc, -1)))
iy, ix = np.unravel_index(ipk, acc.shape)
kpx, kpy = KX[iy, ix], KY[iy, ix]
kp2d = float(np.hypot(kpx, kpy))
mbin, nbin = kpx / dkx, kpy / dky
theta = float(np.degrees(np.arctan2(kpy, kpx)))

print("=== AA15a grid ===")
print(f"  grid {ny} x {nx}, dx={rec.dx:.3f} dy={rec.dy:.3f}")
print(f"  dkx = {dkx:.5f}   dky = {dky:.5f}")
print(f"  scalar floor  min(dkx,dky)        = {min(dkx,dky):.5f}   "
      f"(prior agent used 2*pi/L = {2*np.pi/65.0:.5f})")
print(f"  DIAGONAL floor sqrt(dkx^2+dky^2)  = {np.hypot(dkx,dky):.5f}  <-- floor ALONG A DIAGONAL")

print("\n=== 2D frame-averaged tapered spectrum: where is the peak? ===")
print(f"  argmax at bin (m,n) = ({mbin:+.2f}, {nbin:+.2f})   |k| = {kp2d:.4f}   "
      f"theta = {theta:.1f} deg")
print(f"  -> lam = {2*np.pi/kp2d:.1f} m,  implied f = {f_of_k(kp2d, DEPTH):.4f} Hz")
if abs(abs(mbin) - 1) < 0.1 and abs(abs(nbin) - 1) < 0.1:
    print("  *** the peak IS the DIAGONAL FUNDAMENTAL bin (+-1,+-1): "
          "this is the resolution floor along theta, not an estimate ***")

# temporal spectrum
Zs = Z[:, ::12, ::12]
ts = Zs.reshape(nf, -1)
good = np.isfinite(ts).all(axis=0)
ts = ts[:, good] - ts[:, good].mean(axis=0, keepdims=True)
fr, P = signal.welch(ts, fs=1.0 / rec.dt, nperseg=512, axis=0)
Pm = P.mean(axis=1)
band = (fr > 0.03) & (fr < 1.0)
fp = float(fr[np.argmax(np.where(band, Pm, -1))])
kp_t = k_of_f(fp, DEPTH)

print("\n=== temporal (Welch, %d px) ===" % ts.shape[1])
print(f"  f_p = {fp:.4f} Hz  (Tp {1/fp:.2f} s) -> kp_temporal = {kp_t:.4f} "
      f"(lam {2*np.pi/kp_t:.1f} m)")

print("\n=== the two candidate spatial k, mapped to frequency ===")
for label, kk in [("prior agent's window-0 kp", 0.2432),
                  ("representative 2D-avg kp", kp2d),
                  ("temporal kp (reference)", kp_t)]:
    f_imp = f_of_k(kk, DEPTH)
    i = int(np.argmin(np.abs(fr - f_imp)))
    # local max?
    loc = (Pm[i] >= Pm[max(i - 2, 0)]) and (Pm[i] >= Pm[min(i + 2, len(Pm) - 1)])
    ratio = Pm[np.argmax(np.where(band, Pm, -1))] / Pm[i]
    print(f"  {label:26s} k={kk:.4f} -> f={f_imp:.4f} Hz | "
          f"PSD_peak/PSD_here = {ratio:5.2f}x | local max? {loc} | "
          f"k/kp_temporal = {kk/kp_t:.2f}x")

print("\n=== S(f) around the peak (is there a 2nd peak?) ===")
sel = (fr > 0.05) & (fr < 0.40)
for f_, p_ in zip(fr[sel], Pm[sel]):
    bar = '#' * int(60 * p_ / Pm[sel].max())
    mark = ''
    if abs(f_ - fp) < 1e-9:
        mark = ' <= f_p (temporal peak)'
    if abs(f_ - f_of_k(kp2d, DEPTH)) < (fr[1] - fr[0]) / 2:
        mark += ' <= f implied by 2D-avg spatial kp'
    if abs(f_ - f_of_k(0.2432, DEPTH)) < (fr[1] - fr[0]) / 2:
        mark += " <= f implied by prior agent's window-0 kp"
    print(f"  f={f_:.4f}  {bar}{mark}")

json.dump(dict(dkx=dkx, dky=dky, scalar_floor=min(dkx, dky),
               diagonal_floor=float(np.hypot(dkx, dky)),
               kp_2Davg=kp2d, argmax_bin=[float(mbin), float(nbin)],
               theta_deg=theta, fp=fp, kp_temporal=kp_t,
               ratio_2Davg_over_temporal=kp2d / kp_t),
          open(os.path.join(os.path.dirname(__file__), 'verify_c2c_aa15a.json'), 'w'), indent=1)
rec.f.close()
