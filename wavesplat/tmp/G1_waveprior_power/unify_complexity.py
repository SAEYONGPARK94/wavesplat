"""UNIFY experiment — does the learned prior's real-data advantage GROW with
spectral complexity? GT-FREE complexity metric (dispersion-constrained beamform-map
spectral entropy, obs only, no stereo GT anywhere), correlated with per-window
(analytic - learned) nRMSE delta from wp_realdata_fixed.log (net=prior_net_2ndlobeTF).
Pre-specified: PRIMARY metric = normalized spectral entropy over the annulus.
Secondary (reported, not fished): angular entropy. Guards: record-demeaned corr,
non-degenerate (WS<1.0) subset, entropy-vs-obs_std confound, per-record ranking.
Incremental JSON so a rerun skips computed windows.
"""
import json
import os
import re

import numpy as np
from scipy import stats as ss

import d3_compare_gifs as cg
from d3_prior_net_eval import beamform_top, beamform_power, KREL, NGRID

OUT = 'tmp/G1_waveprior_power/unify_complexity.json'
LOG = 'tmp/G1_waveprior_power/wp_realdata_fixed.log'

# 1) per-window deltas from the fixed-net real run
rows = []
for l in open(LOG):
    m = re.search(r'\[(\w+) w(\d)\] WS ([0-9.]+) / WP ([0-9.]+)', l)
    if m:
        rows.append((m.group(1), int(m.group(2)), float(m.group(3)), float(m.group(4))))
assert len(rows) == 30, f"expected 30 windows, got {len(rows)}"

done = json.load(open(OUT)) if os.path.exists(OUT) else {}

# 2) GT-free complexity per window (same beamform recipe as the net input path)
for rec, w, WS, WP in rows:
    key = f"{rec}_w{w}"
    if key in done:
        continue
    sc = cg.scene_real(rec, win=w)
    P, z = sc['P'].astype(np.float32), sc['z'].astype(np.float32)
    bx, by = beamform_top(P, z, sc['depth'], klim=2.0, n=120)
    kp = max(np.hypot(bx, by), 1e-3)
    bx, by = beamform_top(P, z, sc['depth'], klim=3.5 * kp, n=120)
    kp = max(np.hypot(bx, by), 1e-3)
    _, _, p = beamform_power(P, z, sc['depth'], KREL * kp, NGRID)
    p = np.asarray(p).reshape(NGRID, NGRID)
    ax = np.linspace(-KREL * kp, KREL * kp, NGRID)
    KX, KY = np.meshgrid(ax, ax)
    kk = np.hypot(KX, KY)
    sel = (kk > 0.35 * kp) & (kk < KREL * kp)
    pv = np.maximum(p[sel], 0.0)
    pv = pv / max(pv.sum(), 1e-12)
    ent = float(-(pv * np.log(pv + 1e-15)).sum() / np.log(len(pv)))          # PRIMARY
    ang = np.arctan2(KY, KX)[sel]
    ab = np.digitize(ang, np.linspace(-np.pi, np.pi, 25))
    aw = np.array([pv[ab == i].sum() for i in range(1, 25)])
    aw = aw / max(aw.sum(), 1e-12)
    aent = float(-(aw * np.log(aw + 1e-15)).sum() / np.log(24))              # secondary
    done[key] = dict(rec=rec, w=w, WS=WS, WP=WP, delta=WS - WP,
                     entropy=ent, ang_entropy=aent,
                     obs_std=float(np.std(z)), kp=float(kp))
    json.dump(done, open(OUT, 'w'), indent=1)
    print(f"{key:10s} ent {ent:.3f} aent {aent:.3f} d {WS-WP:+.3f}", flush=True)

# 3) analysis
D = [done[f"{rec}_w{w}"] for rec, w, _, _ in rows]
delta = np.array([d['delta'] for d in D])
ent = np.array([d['entropy'] for d in D])
aent = np.array([d['ang_entropy'] for d in D])
ws = np.array([d['WS'] for d in D])
ostd = np.array([d['obs_std'] for d in D])
recs = np.array([d['rec'] for d in D])


def corr(x, y, lbl):
    pr, pp = ss.pearsonr(x, y)
    sr, sp = ss.spearmanr(x, y)
    print(f"  {lbl:34s} pearson {pr:+.3f} (p={pp:.3f})  spearman {sr:+.3f} (p={sp:.3f})  n={len(x)}")
    return sr, sp


print("\n=== UNIFY: delta(analytic-learned, +=learned better) vs GT-free complexity ===")
corr(ent, delta, "POOLED entropy vs delta")
corr(aent, delta, "POOLED angular-entropy vs delta")
nd = ws < 1.0
corr(ent[nd], delta[nd], f"NON-DEGENERATE (WS<1.0)")
# within-record (record-demeaned)
de = delta.copy().astype(float); ee = ent.copy().astype(float)
for r in np.unique(recs):
    m = recs == r
    de[m] -= de[m].mean(); ee[m] -= ee[m].mean()
corr(ee, delta * 0 + de, "WITHIN-RECORD (demeaned)")
corr(ee[nd], de[nd], "WITHIN-RECORD, non-degenerate")
print("\n  confound: entropy vs obs_std:", end="")
corr(ostd, ent, "")
print("  per-record complexity (mean entropy) & mean delta:")
for r in ['YS17', 'AA15a', 'AA14', 'AA15b', 'LJ18g']:
    m = recs == r
    print(f"    {r:7s} ent {ent[m].mean():.3f}±{ent[m].std():.3f}  delta {delta[m].mean():+.3f}")
print("\n  per-record Spearman (n=6 each, low power):")
for r in ['YS17', 'AA15a', 'AA14', 'AA15b', 'LJ18g']:
    m = recs == r
    sr, sp = ss.spearmanr(ent[m], delta[m])
    print(f"    {r:7s} {sr:+.3f} (p={sp:.3f})")
print("\n  scatter (sorted by entropy): ent | delta | rec_w | WS")
for i in np.argsort(ent):
    print(f"    {ent[i]:.3f} | {delta[i]:+.3f} | {recs[i]}_w{D[i]['w']} | WS {ws[i]:.2f}")
