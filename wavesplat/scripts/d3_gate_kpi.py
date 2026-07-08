"""K5 (2026-07-08): gate-conditional KPI. The onboard causal gate signal is
the estimator's held-out val residual vr_rel (saved per window/W by
d3_falcon_real_operator.py). Site-normalize it within each record
(running-median/MAD analogue: (vr - med)/MAD) — the 2026-07-06 finding was
that absolute vr thresholds do not transfer across sites, relative ones do.

Deliverable: on the gate-PASS (confident) half of windows, what is the
requirement-band (0-2 s) nRMSE? This is the number an operator would quote:
"when the onboard gate says GO, encounter-forecast nRMSE is X."
"""
import glob
import json
import os
import re

import numpy as np
from scipy import stats as st

RES = 'results/direction3'
os.chdir(os.path.join(os.path.dirname(__file__), '..'))
GATE_W = '20.0'      # mid memory (the gate-development W)
OUT = f'{RES}/gate_kpi.json'


def band_nrmse(w, leads, lo, hi, key='pred'):
    gt = np.asarray(w['gt'], float)
    pr = np.asarray(w['by_W'][GATE_W][key], float)
    n = min(len(gt), len(pr), len(leads))
    m = (leads[:n] >= lo) & (leads[:n] <= hi) & \
        np.isfinite(gt[:n]) & np.isfinite(pr[:n])
    if m.sum() < 3 or np.std(gt[:n][m]) == 0:
        return None
    return float(np.sqrt(np.mean((pr[:n][m] - gt[:n][m]) ** 2))
                 / np.std(gt[:n][m]))


rows = []
for path in sorted(glob.glob(f'{RES}/falcon_real_operator_*_damp.json')):
    rec = re.search(r'operator_(.+?)_damp', path).group(1)
    d = json.load(open(path))
    leads = np.asarray(d['leads'], float)
    for wi, w in enumerate(d['windows']):
        vr = w['by_W'][GATE_W].get('vr_rel')
        if vr is None:
            continue
        rows.append(dict(
            rec=rec, wi=wi, vr=float(vr),
            n02=band_nrmse(w, leads, 0, 2),
            n010=band_nrmse(w, leads, 0, 10),
            n010_damp=band_nrmse(w, leads, 0, 10, 'pred_damp')))

# site-normalize vr within record
for rec in set(r['rec'] for r in rows):
    sub = [r for r in rows if r['rec'] == rec]
    vs = np.array([r['vr'] for r in sub])
    med = np.median(vs)
    mad = np.median(np.abs(vs - med)) + 1e-12
    for r in sub:
        r['vr_norm'] = (r['vr'] - med) / mad

valid = [r for r in rows if r['n010'] is not None]
vn = np.array([r['vr_norm'] for r in valid])
n010 = np.array([r['n010'] for r in valid])
rho, p = st.spearmanr(vn, n010)

# median split on vr_norm
thr = np.median(vn)
conf = [r for r in valid if r['vr_norm'] <= thr]
risk = [r for r in valid if r['vr_norm'] > thr]


def pooled(sub, key):
    vs = [r[key] for r in sub if r[key] is not None]
    return (float(np.mean(vs)), float(np.median(vs)), len(vs)) if vs \
        else (None, None, 0)


out = dict(gate_W=GATE_W, n=len(valid),
           spearman_vrnorm_nrmse=dict(rho=float(rho), p=float(p)),
           confident=dict(
               n02_mean=pooled(conf, 'n02'), n010_mean=pooled(conf, 'n010'),
               n010_damp=pooled(conf, 'n010_damp')),
           risky=dict(
               n02_mean=pooled(risk, 'n02'), n010_mean=pooled(risk, 'n010'),
               n010_damp=pooled(risk, 'n010_damp')),
           rows=rows)
json.dump(out, open(OUT, 'w'), indent=1)

print(f"K5 gate-conditional KPI (site-normalized vr_rel, W={GATE_W}, "
      f"n={len(valid)} windows)")
print(f"  Spearman(vr_norm, nRMSE 0-10s) = {rho:+.2f} (p={p:.3f})")
def f2(x):
    return f"{x:.2f}" if x is not None else " -- "


print(f"\n  {'half':10} {'n':>3} | 0-2s nRMSE | 0-10s nRMSE | 0-10s damp")
for name, sub in [('CONFIDENT', conf), ('RISKY', risk)]:
    a = pooled(sub, 'n02'); b = pooled(sub, 'n010'); c = pooled(sub, 'n010_damp')
    print(f"  {name:10} {len(sub):>3} | "
          f"mean {f2(a[0])} med {f2(a[1])} | "
          f"mean {f2(b[0])} med {f2(b[1])} | mean {f2(c[0])}")
# per-record within-gate strength
print("\n  within-record Spearman(vr_norm, nRMSE 0-10s):")
for rec in sorted(set(r['rec'] for r in valid)):
    sub = [r for r in valid if r['rec'] == rec]
    if len(sub) >= 4:
        rr, pp = st.spearmanr([r['vr_norm'] for r in sub],
                              [r['n010'] for r in sub])
        print(f"    {rec:6}: {rr:+.2f} (p={pp:.2f}, n={len(sub)})")
print(f"\nsaved {OUT}")
