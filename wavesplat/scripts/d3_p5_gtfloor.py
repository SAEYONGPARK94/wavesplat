"""P5 (2026-07-08): GT/model-class floor vs sparse-sampling penalty.

falcon_real_operator_<REC>_dense.json stores, per window/W, both the sparse
forward-fan prediction (`pred`) and a dense-oracle prediction (`pred_dense`)
from ALL valid grid pixels (fan cone dropped, ~5x obs). Decompose the
real-data encounter-forecast residual:

  nRMSE(sparse)^2  ~  floor^2 (GT + free-wave-model mismatch, = dense)
                      + sampling_penalty (sparse fan vs dense grid)

If dense nRMSE ~ 0 the stereo GT is clean and free-wave-representable, so the
real-data shortfall is all aperture/sampling; if dense nRMSE is large the GT
carries irreducible non-free-wave content (nonlinearity + stereo noise) that
NO dispersion-constrained estimator can beat — a genuine floor.
"""
import glob
import json
import os
import re

import numpy as np

RES = 'results/direction3'
os.chdir(os.path.join(os.path.dirname(__file__), '..'))
BANDS = [(0, 2, '0-2s'), (0, 5, '0-5s'), (0, 10, '0-10s')]


def band(w, leads, W, lo, hi, key):
    gt = np.asarray(w['gt'], float)
    src = w['by_W'][W].get(key)
    if src is None:
        return None, None
    pr = np.asarray(src, float)
    n = min(len(gt), len(pr), len(leads))
    m = (leads[:n] >= lo) & (leads[:n] <= hi) & \
        np.isfinite(gt[:n]) & np.isfinite(pr[:n])
    return pr[:n][m], gt[:n][m]


def nrmse(p, g):
    if p is None or len(g) < 4 or np.std(g) == 0:
        return None
    return float(np.sqrt(np.mean((p - g) ** 2)) / np.std(g))


out = {}
print(f"{'rec':6} {'W':>3} {'band':5} | {'sparse':>6} {'dense':>6} "
      f"{'penalty':>7} | n_dense")
for path in sorted(glob.glob(f'{RES}/falcon_real_operator_*_dense.json')):
    rec = re.search(r'operator_(.+?)_dense', path).group(1)
    d = json.load(open(path))
    leads = np.asarray(d['leads'], float)
    wins = d['windows']
    ent = {}
    for W in [str(float(x)) for x in d['w_mems']]:
        nd = int(np.median([w['by_W'][W].get('n_dense', 0) for w in wins]))
        for lo, hi, bl in BANDS:
            ps, gs, pds = [], [], []
            for w in wins:
                p, g = band(w, leads, W, lo, hi, 'pred')
                pd, gd = band(w, leads, W, lo, hi, 'pred_dense')
                if p is not None:
                    ps.append(p); gs.append(g)
                if pd is not None:
                    pds.append(pd)
            g = np.concatenate(gs)
            r_sp = nrmse(np.concatenate(ps), g)
            r_dn = nrmse(np.concatenate(pds), g) if pds else None
            pen = (r_sp - r_dn) if (r_sp and r_dn) else None
            ent.setdefault(W, {})[bl] = dict(sparse=r_sp, dense=r_dn,
                                             penalty=pen, n_dense=nd)
            ps_s = f"{r_sp:.2f}" if r_sp else "  -- "
            dn_s = f"{r_dn:.2f}" if r_dn else "  -- "
            pn_s = f"{pen:+.2f}" if pen is not None else "  -- "
            print(f"{rec:6} {float(W):3.0f} {bl:5} | {ps_s:>6} {dn_s:>6} "
                  f"{pn_s:>7} | {nd}")
    out[rec] = ent
json.dump(out, open(f'{RES}/p5_gtfloor.json', 'w'), indent=1)
print(f"\nsaved {RES}/p5_gtfloor.json")
