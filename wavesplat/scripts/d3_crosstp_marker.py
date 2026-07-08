"""K6 (2026-07-08): does the K3 mechanism mark REAL bad windows?

K3 reproduced bad-window collapse with cross-Tp crossing seas. If that is
the real failure mode, real windows with stronger FREQUENCY-bimodality
(secondary spectral peak away from the primary) should have worse skill.
This is a different axis from the directional bimod marker (which held on
YS17 but failed to transfer to AA15a).

For every window of every falcon_real_operator record: take GT time series
at NP fixed points inside the forward-fan region over the memory span,
Welch-average S(f), find the primary peak, and compute
  crosstp = max_{f in [0.55,1.9]*fp, |f-fp|>0.25*fp} S(f) / S(fp).
Correlate with per-window nRMSE (0-10s, mid W) across records
(pooled + within-record ranks).
"""
import json
import os
import sys

import numpy as np
from scipy import stats as st
from scipy.signal import welch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
os.chdir(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, 'scripts')

from d3_sa_multirecord import MRecord
from d3_falcon_real_operator import RECORDS

RES = 'results/direction3'
NP_PTS = 16
SPAN_S = 40.0            # analysis span behind t0 (matches max W)
OUT = f'{RES}/crosstp_marker.json'

rows = []
for rec_name in ['YS17', 'AA15a', 'AA14', 'AA15b', 'LJ18g']:
    path = f'{RES}/falcon_real_operator_{rec_name}.json'
    if not os.path.exists(path):
        continue
    d = json.load(open(path))
    leads = np.asarray(d['leads'], float)
    Wmid = str(float(d['w_mems'][1]))
    cfg = RECORDS[rec_name]
    rec = MRecord(cfg['path'], stride=cfg['stride'])
    nspan = int(round(SPAN_S / rec.dt))
    rng = np.random.default_rng(3)
    for wi, w in enumerate(d['windows']):
        i0 = w['i0']
        Z = rec.window(max(0, i0 - nspan + 1), min(nspan, i0))
        # NP fixed points with mostly-finite time series
        ny, nx = Z.shape[1:]
        cand = rng.choice(ny * nx, 4000, replace=False)
        finite_frac = np.isfinite(Z.reshape(len(Z), -1)[:, cand]).mean(0)
        good = cand[finite_frac > 0.95][:NP_PTS]
        if len(good) < 6:
            continue
        S_acc, f_acc = None, None
        for g in good:
            iy, ix = np.unravel_index(g, (ny, nx))
            ts = Z[:, iy, ix]
            ts = np.interp(np.arange(len(ts)),
                           np.flatnonzero(np.isfinite(ts)),
                           ts[np.isfinite(ts)])
            f, S = welch(ts - ts.mean(), fs=1.0 / rec.dt,
                         nperseg=min(len(ts), 256))
            S_acc = S if S_acc is None else S_acc + S
            f_acc = f
        S, f = S_acc / len(good), f_acc
        band = (f > 0.04) & (f < 1.0)
        fp = f[band][np.argmax(S[band])]
        sp = S[band][np.argmax(S[band])]
        m2 = band & (f > 0.55 * fp) & (f < 1.9 * fp) & \
            (np.abs(f - fp) > 0.25 * fp)
        crosstp = float(S[m2].max() / sp) if m2.any() else 0.0
        # skill: per-window nRMSE 0-10s mid W
        gt = np.asarray(w['gt'], float)
        pr = np.asarray(w['by_W'][Wmid]['pred'], float)
        n = min(len(gt), len(pr), len(leads))
        m = (leads[:n] <= 10) & np.isfinite(gt[:n]) & np.isfinite(pr[:n])
        if m.sum() < 4:
            continue
        nr = float(np.sqrt(np.mean((pr[:n][m] - gt[:n][m]) ** 2))
                   / np.std(gt[:n][m]))
        rows.append(dict(rec=rec_name, wi=wi, fp=float(fp),
                         crosstp=crosstp, nrmse=nr))
        print(f"{rec_name} wi{wi}: fp={fp:.3f}Hz crosstp={crosstp:.2f} "
              f"nRMSE={nr:.2f}", flush=True)
    del rec

x = [r['crosstp'] for r in rows]
y = [r['nrmse'] for r in rows]
rho, p = st.spearmanr(x, y)
print(f"\npooled ({len(rows)} windows): crosstp vs nRMSE "
      f"Spearman {rho:+.2f} (p={p:.3f})")
xs, ys = [], []
for rec_name in set(r['rec'] for r in rows):
    sub = [r for r in rows if r['rec'] == rec_name]
    xs += list(st.rankdata([r['crosstp'] for r in sub]))
    ys += list(st.rankdata([r['nrmse'] for r in sub]))
rho_w, p_w = st.spearmanr(xs, ys)
print(f"within-record ranks: Spearman {rho_w:+.2f} (p={p_w:.3f})")
json.dump(dict(rows=rows,
               pooled=dict(rho=float(rho), p=float(p)),
               within=dict(rho=float(rho_w), p=float(p_w))),
          open(OUT, 'w'), indent=1)
print('saved', OUT)
