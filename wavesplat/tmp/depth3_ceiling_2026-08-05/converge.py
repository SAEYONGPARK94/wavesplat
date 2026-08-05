"""Confirm the N -> infinity plateau of our generator's grid applied to real records."""
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
FMIN, FMAX = 0.04, 0.50
STATIONS = ['028p1', '094p1', '142p1', '194p1', '243p1', '278p1']
NS = (15, 30, 60, 120, 200, 400)


def s1(r):
    return np.degrees(np.sqrt(np.clip(2.0 * (1.0 - np.clip(r, 0, 1)), 0, None)))


acc = {n: [] for n in NS}
acc_bandonly = []
for st in STATIONS:
    z = np.load(os.path.join(HERE, f'spec_{st}.npz'))
    f, E, a1, b1 = z['f'], z['E'].copy(), z['a1'].copy(), z['b1'].copy()
    Hs, Tp, fl = z['Hs'], z['Tp'], z['flag']
    for arr in (E, a1, b1):
        arr[arr <= -999] = np.nan
    good = (fl == 1) & (Hs > -999) & (Tp > 0) & np.isfinite(Tp)
    E, a1, b1, Tp = E[good], a1[good], b1[good], Tp[good]
    band = (f >= FMIN) & (f <= FMAX)
    r1f = np.sqrt(np.clip(a1 ** 2 + b1 ** 2, 0, 1))
    th = np.arctan2(b1, a1)
    W = np.where(band[None, :] & np.isfinite(E) & (E > 0) & np.isfinite(r1f), E, 0.0)
    for i in range(E.shape[0]):
        w = W[i]
        if w.sum() <= 0:
            continue
        w = w / w.sum()
        rr, tt = np.nan_to_num(r1f[i]), np.nan_to_num(th[i])
        wp = 2 * np.pi / Tp[i]
        for n in NS:
            edges = np.linspace(0.5 * wp, 3.0 * wp, n + 1) / (2 * np.pi)
            ee, aa = [], []
            for lo, hi in zip(edges[:-1], edges[1:]):
                m = (f >= lo) & (f < hi) & (w > 0)
                if not m.any():
                    continue
                bx = (w[m] * rr[m] * np.cos(tt[m])).sum()
                by = (w[m] * rr[m] * np.sin(tt[m])).sum()
                if bx == 0 and by == 0:
                    continue
                ee.append(w[m].sum()); aa.append(np.arctan2(by, bx))
            if not ee:
                continue
            ee = np.array(ee) / np.sum(ee); aa = np.array(aa)
            acc[n].append(float(s1(np.hypot((ee * np.cos(aa)).sum(), (ee * np.sin(aa)).sum()))))
        # across term restricted to OUR band, no binning artefact
        bm = (f >= 0.5 * wp / (2 * np.pi)) & (f < 3.0 * wp / (2 * np.pi)) & (w > 0)
        if bm.any():
            ww = w[bm] / w[bm].sum()
            acc_bandonly.append(float(s1(np.hypot((ww * np.cos(tt[bm])).sum(),
                                                  (ww * np.sin(tt[bm])).sum()))))

out = {'gen_grid_median_by_N': {str(n): float(np.median(acc[n])) for n in NS},
       'across_restricted_to_generator_band_median': float(np.median(acc_bandonly)),
       'n_records': len(acc_bandonly)}
print(json.dumps(out, indent=2))
json.dump(out, open(os.path.join(HERE, 'converge.json'), 'w'), indent=2)
