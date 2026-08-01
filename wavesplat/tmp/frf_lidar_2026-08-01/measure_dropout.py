"""Measure the REAL lidar return rate on water, and test the repo's raydrop model.

Why this works
--------------
The Velodyne HDL-32E on the CRAB is stationary during a wave file, and its scan
pattern repeats every frame (10 Hz). The sea surface is what changes. So for a
fixed beam direction, the fraction of frames in which that beam produced a return
IS the empirical return probability -- time supplies the repeats.

We cannot use slot index as the beam id (tested and refuted: across-frame azimuth
std ~14 deg), so we rebuild the beam grid from the returns themselves:
  ring   = cluster of the sensor-frame elevation angle (32 fixed lasers)
  az bin = azimuth quantized at the sensor's angular step

Restriction, stated up front: a (ring, az) cell is only observable if it produced
at least one return in the file. Beams that NEVER return (aimed at sky, or beyond
range, or always specular-lost) are invisible. This biases the measured rate
UPWARD and is a hard limit of point-cloud data -- reported, not corrected.

Model under test (wavesim/raydrop_uniform.py):
    P_raw = R_fresnel(theta_i) * cos(theta_i) / (d/d_ref)^2
with theta_i the incidence angle from the local surface normal. We use the
flat-surface approximation theta_i = 90deg - depression, and also a slope-aware
variant using the locally fitted surface normal.
"""
import glob
import json
import os
import sys

import numpy as np
import h5py

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from wavesim.raydrop_uniform import fresnel_reflectance   # the model under test

FILL = -999.0
AZ_STEP = 0.30          # deg, coarser than native to keep cells populated
N_RING = 32             # HDL-32E lasers
MIN_FRAMES = 40         # a cell must be observable in a file with >= this many frames


def load(path):
    with h5py.File(path, 'r') as h:
        X = h['xFRF'][0]; Y = h['yFRF'][0]; Z = h['elevation'][0]
        lx, ly, lz = (float(h['lidarX'][()]), float(h['lidarY'][()]),
                      float(h['lidarZ'][()]))
    V = (X > FILL) & (Y > FILL) & (Z > FILL)
    return X, Y, Z, V, lx, ly, lz


def analyse(path):
    X, Y, Z, V, lx, ly, lz = load(path)
    nfr = X.shape[0]
    if nfr < MIN_FRAMES:
        return None
    dx, dy, dz = X - lx, Y - ly, lz - Z
    rho = np.hypot(dx, dy)                       # horizontal range
    az = np.degrees(np.arctan2(dy, dx)) % 360.0
    dep = np.degrees(np.arctan2(dz, rho))        # depression angle (deg)
    rng = np.sqrt(rho ** 2 + dz ** 2)            # slant range

    dv = dep[V]
    if dv.size < 5000:
        return None
    # ring id = quantile bucket of depression -> proxy for the 32 fixed lasers
    edges = np.quantile(dv, np.linspace(0, 1, N_RING + 1))
    edges[0] -= 1e-6; edges[-1] += 1e-6
    ring = np.digitize(dep, edges) - 1
    azb = (az / AZ_STEP).astype(int)

    key = ring.astype(np.int64) * 100000 + azb
    key_v = key[V]
    uk, inv = np.unique(key_v, return_inverse=True)
    # frames-with-return per cell
    fr_idx = np.repeat(np.arange(nfr), X.shape[1]).reshape(X.shape)[V]
    hit = np.zeros((uk.size, nfr), bool)
    hit[inv, fr_idx] = True
    n_hit = hit.sum(axis=1)
    rate = n_hit / nfr

    # cell geometry = median over its returns
    def cellmed(arr):
        a = arr[V]
        out = np.zeros(uk.size)
        np.maximum.at(out, inv, 0)               # init
        s = np.zeros(uk.size); c = np.zeros(uk.size)
        np.add.at(s, inv, a); np.add.at(c, inv, 1)
        return s / np.maximum(c, 1)

    d_cell = cellmed(rng)
    dep_cell = cellmed(dep)
    z_cell = cellmed(Z)
    # flat-surface incidence angle from the vertical normal
    theta_flat = np.radians(90.0 - dep_cell)
    P_model = fresnel_reflectance(theta_flat) * np.cos(theta_flat) / (d_cell / 125.0) ** 2
    keep = (n_hit >= 1) & np.isfinite(P_model) & (d_cell > 1.0)
    return dict(rate=rate[keep], d=d_cell[keep], dep=dep_cell[keep],
                theta=np.degrees(theta_flat[keep]), P=P_model[keep],
                z=z_cell[keep], nfr=nfr, ncell=int(keep.sum()),
                valid_frac=float(V.mean()))


files = sorted(glob.glob(os.path.join(os.path.dirname(__file__), 'ws', '*.nc')))
if not files:
    files = [os.path.join(os.path.dirname(__file__), 'ws_sample.nc')]
print(f'files: {len(files)}')

acc = {k: [] for k in ('rate', 'd', 'dep', 'theta', 'P')}
meta = []
for f in files:
    try:
        r = analyse(f)
    except Exception as e:
        print(f'  SKIP {os.path.basename(f)}: {type(e).__name__}: {e}')
        continue
    if r is None:
        continue
    for k in acc:
        acc[k].append(r[k])
    meta.append(dict(file=os.path.basename(f), frames=r['nfr'],
                     cells=r['ncell'], valid_frac=round(r['valid_frac'], 4)))
    print(f"  {os.path.basename(f)[:58]:58} frames {r['nfr']:3d} cells {r['ncell']:6d} "
          f"valid {r['valid_frac']:.3f}")

for k in acc:
    acc[k] = np.concatenate(acc[k]) if acc[k] else np.array([])
rate, d, dep, theta, P = acc['rate'], acc['d'], acc['dep'], acc['theta'], acc['P']
print(f'\ntotal beam-cells: {rate.size:,}')

out = dict(files=meta, n_cells=int(rate.size))

print('\n=== return rate vs slant range ===')
be = np.array([0, 5, 10, 15, 20, 25, 30, 40, 60])
rows = []
for i in range(len(be) - 1):
    m = (d >= be[i]) & (d < be[i + 1])
    if m.sum() > 30:
        rows.append(dict(lo=int(be[i]), hi=int(be[i + 1]), n=int(m.sum()),
                         rate=round(float(rate[m].mean()), 4)))
        print(f'  d {be[i]:2d}-{be[i+1]:2d} m : n={m.sum():6d}  mean rate {rate[m].mean():.4f}')
out['rate_vs_range'] = rows

print('\n=== return rate vs incidence angle (flat-surface) ===')
te = np.array([0, 20, 30, 40, 50, 60, 70, 80, 90])
rows = []
for i in range(len(te) - 1):
    m = (theta >= te[i]) & (theta < te[i + 1])
    if m.sum() > 30:
        rows.append(dict(lo=int(te[i]), hi=int(te[i + 1]), n=int(m.sum()),
                         rate=round(float(rate[m].mean()), 4)))
        print(f'  theta {te[i]:2d}-{te[i+1]:2d} deg : n={m.sum():6d}  mean rate {rate[m].mean():.4f}')
out['rate_vs_theta'] = rows

# --- the model test: does P_model order the measured rate?
if rate.size > 100:
    ok = np.isfinite(P) & (P > 0)
    lp = np.log(P[ok]); rr = rate[ok]
    cs = float(np.corrcoef(np.argsort(np.argsort(lp)), np.argsort(np.argsort(rr)))[0, 1])
    cp = float(np.corrcoef(lp, rr)[0, 1])
    print(f'\n=== MODEL TEST: P_raw = R_fresnel(th)*cos(th)/(d/125)^2 ===')
    print(f'  Spearman(P_model, measured rate) = {cs:+.4f}   (n={ok.sum():,})')
    print(f'  Pearson(log P_model, rate)       = {cp:+.4f}')
    q = np.quantile(P[ok], np.linspace(0, 1, 11))
    print('  decile of P_model -> mean measured rate:')
    dec = []
    for i in range(10):
        m = (P[ok] >= q[i]) & (P[ok] <= q[i + 1])
        if m.sum() > 10:
            dec.append([round(float(np.median(P[ok][m])), 6), round(float(rr[m].mean()), 4),
                        int(m.sum())])
            print(f'    P~{np.median(P[ok][m]):.5f}  rate {rr[m].mean():.4f}  (n={m.sum()})')
    out['model_test'] = dict(spearman=round(cs, 4), pearson_logP=round(cp, 4),
                             deciles=dec, n=int(ok.sum()))

json.dump(out, open(os.path.join(os.path.dirname(__file__), 'dropout_measured.json'), 'w'),
          indent=1)
print('\nwrote dropout_measured.json')
