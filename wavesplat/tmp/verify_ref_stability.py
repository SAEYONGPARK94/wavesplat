"""How well-determined is kp_true itself?

His entire audit grades the spatial estimator against kp_true = temporal Welch
peak + dispersion. If kp_true is itself uncertain at the +-20% level, then
"AA15a's kp is 1.22x wrong" is not a measurement, it is noise.

Compare: HIS kp_true (tmp/record_wavelength.json) vs MINE (independent Welch,
different nperseg / span / pixel subsampling). Same method, same data, two
competent implementations.

Also: the AA15a energy-ratio claim (he says 7.8x) at several f resolutions.
"""
import json, sys, os
import numpy as np
from scipy import signal
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))
from d3_sa_multirecord import MRecord

G = 9.81
HIS = {r['name']: r for r in json.load(open('tmp/record_wavelength.json'))}
MINE = json.load(open('tmp/verify_c2_temporal_all.json'))

print("=== kp_true: HIS implementation vs MINE (same method, same data) ===")
print(f"{'rec':7s}{'Tp_his':>8}{'Tp_mine':>9}{'kp_his':>9}{'kp_mine':>9}"
      f"{'kp ratio':>10}{'lam_his':>9}{'lam_mine':>10}")
worst = 0
for name, m in MINE.items():
    h = HIS.get(name)
    if not h:
        continue
    kp_his = 2 * np.pi / h['lam_temporal']
    kp_mine = m['kp_temporal']
    ratio = kp_mine / kp_his
    worst = max(worst, max(ratio, 1 / ratio))
    print(f"{name:7s}{h['Tp']:8.2f}{m['Tp']:9.2f}{kp_his:9.4f}{kp_mine:9.4f}"
          f"{ratio:10.2f}{h['lam_temporal']:9.1f}{m['lam_temporal']:10.1f}")
print(f"\n  WORST disagreement between two implementations of his OWN reference: "
      f"{worst:.2f}x")
print("  (compare: the spatial-estimator 'errors' he reports on the FAN records")
print("   are 0.73x - 1.23x, i.e. INSIDE this band.)")

print("\n=== AA15a: the '7.8x less energy' claim, vs Welch resolution ===")
rec = MRecord('data/real_bergamasco/AA_2015/Surfaces_20150305_103500.nc', stride=1)
n = 3000
Z = rec.window((rec.T_all - n) // 2, n)
Zs = Z[:, ::12, ::12].reshape(n, -1)
good = np.isfinite(Zs).all(axis=0)
ts = Zs[:, good] - Zs[:, good].mean(axis=0, keepdims=True)


def f_of_k(k, d=17.0):
    return float(np.sqrt(G * k * np.tanh(k * d)) / (2 * np.pi))


for nper in (256, 512, 1024, 2048):
    fr, P = signal.welch(ts, fs=1.0 / rec.dt, nperseg=nper, axis=0)
    Pm = P.mean(axis=1)
    band = (fr > 0.03) & (fr < 1.0)
    ip = int(np.argmax(np.where(band, Pm, -1)))
    fp = fr[ip]
    out = []
    for lbl, kk in (("win0 kp 0.2432", 0.2432), ("repr. kp 0.1216", 0.1216)):
        fi = f_of_k(kk)
        j = int(np.argmin(np.abs(fr - fi)))
        out.append(f"{lbl}: E_peak/E = {Pm[ip]/Pm[j]:5.2f}x")
    print(f"  nperseg={nper:5d} (df={fr[1]-fr[0]:.4f} Hz) f_p={fp:.4f}  " + " | ".join(out))
print("\n  his claim was 7.8x for the win0 k. Reproduced value is ~1.6-2.5x.")
rec.f.close()
