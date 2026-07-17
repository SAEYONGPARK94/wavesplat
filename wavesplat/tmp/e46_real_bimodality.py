"""e46 — PRECONDITION for the real-data leg: do the real records even CONTAIN
cross-Tp bimodal seas?

WHY THIS FIRST. The arc's one surviving deliverable is about **cross-Tp seas**
(two wave systems at DIFFERENT peak periods and different directions -- swell +
windsea). The whole justification for caring is "real seas are cross-Tp". That
sentence has been asserted throughout this project and **never measured on the
records we actually hold**.

Before building a real-data version of the encounter experiment, answer the
precondition: is the target regime present in the 13 stereo records at all?

  - If several records are genuinely cross-Tp -> the real leg is worth building,
    and this tells us which records to use.
  - If none are -> the real leg CANNOT test the deliverable, and that is itself a
    finding: the arc's target regime is absent from the available data, so the
    claim stays synthetic-scoped for a reason that has nothing to do with the
    estimator.

METHOD (deliberately crude and conservative -- this is a screen, not a verdict).
Per record, on a mean 2-D wavenumber spectrum over frames:
  1. E(kx,ky) = <|FFT2(Z_t)|^2> over frames, detrended/windowed, ring-limited.
  2. Find the dominant peak (k1, th1).
  3. Mask a neighbourhood of it, find the best secondary peak (k2, th2).
  4. Declare CROSS-Tp if angular separation >= 45 deg AND |k| ratio outside
     [0.85, 1.18] (i.e. Tp2/Tp1 outside ~[0.92,1.08], a real period difference)
     AND the secondary lobe holds >= 10% of the energy in the two lobes.
  5. Report the implied (Tp1, Tp2, dsep, r2) so we can compare against the
     synthetic arena's ranges (build_sea_crosstp: Tp1=7, Tp2=Tp1*U(0.75,1.4),
     dsep in [60,150] deg, r2 in [0.2,0.5]).

HONESTY. A 2-D snapshot spectrum has a 180-degree ambiguity (a wave going +x and
-x give the same |FFT|^2). The synthetic arena resolves this with the time axis.
Here the screen is direction-folded, so "angular separation" is measured modulo
180 deg. That makes the screen CONSERVATIVE for detecting two systems that are
nearly opposed, and it cannot distinguish a single system from its mirror. Stated
up front; a positive screen still needs the time axis to confirm.

No verdict bar is attached: this is a screen to decide whether the real leg is
buildable and on which records.

Usage: python3 tmp/e46_real_bimodality.py [--frames 200]
Out:   results/direction3/e46_real_bimodality.json
"""
import os
import sys
import json
import glob
import time
import argparse

import numpy as np

sys.path.insert(0, '/home/vexit/wavesplat/wavesplat')
sys.path.insert(0, 'scripts')

from scripts.realdata_analysis import load_surface_nc, peak_period

OUT = 'results/direction3/e46_real_bimodality.json'
G = 9.81
ROOT = 'data/real_bergamasco'
SEP_MIN = 45.0          # deg, folded mod 180
KRATIO_LO, KRATIO_HI = 0.85, 1.18
R2_MIN = 0.10


def tp_of_k(k, depth):
    om = np.sqrt(G * k * np.tanh(k * depth))
    return 2 * np.pi / om


def spectrum(Z, gx, gy, frames_max=200):
    T = Z.shape[0]
    idx = np.linspace(0, T - 1, min(T, frames_max)).astype(int)
    dy = float(np.median(np.diff(gy)))
    dx = float(np.median(np.diff(gx)))
    Ny, Nx = Z.shape[1], Z.shape[2]
    wy = np.hanning(Ny)[:, None]
    wx = np.hanning(Nx)[None, :]
    acc = None
    used = 0
    for i in idx:
        z = Z[i].astype(np.float64)
        if not np.isfinite(z).all():
            m = np.isfinite(z)
            if m.mean() < 0.7:
                continue
            z = np.where(m, z, np.nanmean(z))
        z = z - z.mean()
        F = np.fft.fftshift(np.abs(np.fft.fft2(z * wy * wx)) ** 2)
        acc = F if acc is None else acc + F
        used += 1
    if used == 0:
        return None
    acc /= used
    kx = np.fft.fftshift(np.fft.fftfreq(Nx, d=dx)) * 2 * np.pi
    ky = np.fft.fftshift(np.fft.fftfreq(Ny, d=dy)) * 2 * np.pi
    return acc, kx, ky


def analyse(E, kx, ky, depth):
    KX, KY = np.meshgrid(kx, ky)
    K = np.hypot(KX, KY)
    band = (K > 0.02) & (K < 0.5)          # 12 m .. 300 m wavelengths
    Em = np.where(band, E, 0.0)
    i1 = np.unravel_index(np.argmax(Em), Em.shape)
    k1, th1 = float(K[i1]), float(np.degrees(np.arctan2(KY[i1], KX[i1])))
    # fold direction mod 180 (snapshot spectra are direction-ambiguous)
    TH = np.degrees(np.arctan2(KY, KX))
    dth = np.abs(((TH - th1 + 90) % 180) - 90)
    near = (dth < SEP_MIN) | (np.abs(K / max(k1, 1e-9) - 1) < 0.15)
    E2 = np.where(near, 0.0, Em)
    if E2.max() <= 0:
        return dict(bimodal=False, reason='no secondary peak outside mask',
                    k1=k1, th1=th1, Tp1=tp_of_k(k1, depth))
    i2 = np.unravel_index(np.argmax(E2), E2.shape)
    k2, th2 = float(K[i2]), float(np.degrees(np.arctan2(KY[i2], KX[i2])))
    sep = float(np.abs(((th2 - th1 + 90) % 180) - 90))
    ratio = float(k2 / k1)
    # crude lobe energies: sum within an angular wedge +-30 deg of each peak
    def wedge(th0):
        d = np.abs(((TH - th0 + 90) % 180) - 90)
        return float(Em[(d < 30)].sum())
    e1, e2 = wedge(th1), wedge(th2)
    r2 = float(e2 / max(e1 + e2, 1e-30))
    Tp1, Tp2 = tp_of_k(k1, depth), tp_of_k(k2, depth)
    cross = bool(sep >= SEP_MIN
                 and not (KRATIO_LO <= ratio <= KRATIO_HI)
                 and r2 >= R2_MIN)
    return dict(bimodal=cross, k1=k1, k2=k2, th1=th1, th2=th2, sep=sep,
                kratio=ratio, r2=r2, Tp1=float(Tp1), Tp2=float(Tp2),
                Tp_ratio=float(Tp2 / Tp1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--frames', type=int, default=200)
    a = ap.parse_args()
    t0 = time.time()
    files = sorted(glob.glob(f'{ROOT}/*/*.nc'))
    print(f'found {len(files)} records', flush=True)
    rows = []
    for f in files:
        site = os.path.basename(os.path.dirname(f))
        try:
            Z, gx, gy, t, meta = load_surface_nc(f, tmax_frames=a.frames)
        except Exception as e:
            print(f'  {site:8s} {os.path.basename(f)[:38]:38s} LOAD FAIL {e}',
                  flush=True)
            rows.append(dict(file=f, site=site, error=str(e)))
            continue
        depth = float(meta.get('depth', 50.0)) if isinstance(meta, dict) else 50.0
        sp = spectrum(Z, gx, gy, a.frames)
        if sp is None:
            rows.append(dict(file=f, site=site, error='no usable frames'))
            continue
        E, kx, ky = sp
        r = analyse(E, kx, ky, depth)
        Tp_t, _ = peak_period(Z, t)
        r.update(file=f, site=site, depth=depth, Tp_time=float(Tp_t),
                 patch=f'{gx.max()-gx.min():.0f}x{gy.max()-gy.min():.0f}m',
                 nframes=int(Z.shape[0]))
        rows.append(r)
        tag = '★CROSS-Tp' if r.get('bimodal') else '  unimodal'
        if 'sep' in r:
            print(f"  {site:8s} {os.path.basename(f)[:30]:30s} {tag} | "
                  f"Tp1 {r['Tp1']:4.1f} Tp2 {r['Tp2']:4.1f} "
                  f"(ratio {r['Tp_ratio']:.2f}) sep {r['sep']:5.1f} "
                  f"r2 {r['r2']:.2f} | Tp_time {Tp_t:4.1f} "
                  f"[{time.time()-t0:.0f}s]", flush=True)
        else:
            print(f"  {site:8s} {os.path.basename(f)[:30]:30s} {tag} | "
                  f"{r.get('reason','')}", flush=True)

    ok = [r for r in rows if r.get('bimodal')]
    tot = [r for r in rows if 'bimodal' in r]
    print(f'\n=== SCREEN RESULT ===')
    print(f'  records analysed : {len(tot)} / {len(files)}')
    print(f'  ★cross-Tp bimodal: {len(ok)}')
    for r in ok:
        print(f'    {r["site"]:8s} Tp {r["Tp1"]:.1f}->{r["Tp2"]:.1f} '
              f'(ratio {r["Tp_ratio"]:.2f}) sep {r["sep"]:.0f} r2 {r["r2"]:.2f}')
    print(f'\n  synthetic arena for comparison: Tp1=7, Tp2/Tp1 ~ U(0.75,1.4), '
          f'sep ~ U(60,150) deg, r2 ~ U(0.2,0.5)')
    if not ok:
        print('\n  => the arc\'s TARGET REGIME may be ABSENT from the available '
              'records. If so the real leg cannot test the deliverable, and the '
              'claim stays synthetic-scoped for a reason unrelated to the '
              'estimator. That is itself a finding worth writing.')
    print('\n  CAVEAT: snapshot |FFT|^2 is direction-folded (180 deg ambiguity); '
          'separation measured mod 180. Conservative for near-opposed systems; '
          'a positive screen still needs the time axis to confirm.')
    json.dump(dict(rows=rows, n_cross=len(ok), n_analysed=len(tot)),
              open(OUT, 'w'), indent=1, default=float)
    print(f'saved -> {OUT}  ({time.time()-t0:.0f}s)')


if __name__ == '__main__':
    main()
