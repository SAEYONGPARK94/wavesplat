#!/usr/bin/env python3
"""Topic C NEUTRAL rederivation from raw jsonl. Read-only. No experiments launched."""
import json, itertools, statistics as st
from math import comb

RES2 = 'tmp/rx1_tracker_2026-08-05/results2.jsonl'

def load(fn):
    rows = []
    for line in open(fn):
        line = line.strip()
        if not line: continue
        rows.append(json.loads(line))
    return rows

rows = load(RES2)

def sign_test_two_sided(k, n):
    """P(X<=min(k,n-k) or X>=max) under Binom(n,0.5)."""
    lo = min(k, n-k)
    p = sum(comb(n, i) for i in range(0, lo+1)) / 2**n
    p2 = 2*p
    return min(1.0, p2)

def cell_table(metric, agg='median'):
    """per (cell, episode) aggregate over seeds"""
    d = {}
    for r in rows:
        d.setdefault((r['cell'], r['episode']), []).append(r[metric])
    out = {}
    for (c, e), v in d.items():
        out[(c, e)] = st.median(v) if agg == 'median' else st.mean(v)
    return out

EPS = sorted({r['episode'] for r in rows})

def paired(cell, base, metric, seedagg='median', epagg='median'):
    t = cell_table(metric, seedagg)
    deltas = [t[(cell, e)] - t[(base, e)] for e in EPS]
    n = len(deltas)
    k_improve = sum(1 for x in deltas if x < 0)  # lower better for nRMSE-like
    med = st.median(deltas) if epagg == 'median' else st.mean(deltas)
    return med, k_improve, n, sign_test_two_sided(k_improve, n), deltas

print('='*90)
print('CONVENTION (prereg §1): seed median-of-3 -> per-episode paired delta -> MEDIAN over 4 eps')
print('sign test TWO-SIDED, n=4 -> floor p = 0.125')
print('='*90)

ORACLE = ['K1','K2','K3','K4','K5','K6','K7','K12','K12-S1','X-NOTRACK']
DEPLOY = ['D-K12','D-K6','D-NOTRACK']

print('\n--- ORACLE arm (init_method=oracle, n_gaussians=[15]), base = X-BASE ---')
print(f"{'cell':12s} {'medD nRMSE':>11s} {'k/n':>5s} {'p':>6s} {'meanD':>9s} {'medD rho':>9s} {'medD |VR-1|':>11s}")
for c in ORACLE:
    med, k, n, p, dl = paired(c, 'X-BASE', 'nRMSE')
    mean_, _, _, _, _ = paired(c, 'X-BASE', 'nRMSE', epagg='mean')
    mr, kr, _, pr, _ = paired(c, 'X-BASE', 'Pearson_rho')
    mv, kv, _, pv, _ = paired(c, 'X-BASE', 'absVR1_framemean')
    print(f"{c:12s} {med:+11.4f} {k}/{n:<3d} {p:6.3f} {mean_:+9.4f} {mr:+9.4f} {mv:+11.4f}")

print('\n--- DEPLOY arm (init_method=v2_konly, ladder [3,8,15]), base = D-BASE ---')
print(f"{'cell':12s} {'medD nRMSE':>11s} {'k/n':>5s} {'p':>6s} {'meanD':>9s} {'medD rho':>9s} {'medD |VR-1|':>11s}")
for c in DEPLOY:
    med, k, n, p, dl = paired(c, 'D-BASE', 'nRMSE')
    mean_, _, _, _, _ = paired(c, 'D-BASE', 'nRMSE', epagg='mean')
    mr, kr, _, pr, _ = paired(c, 'D-BASE', 'Pearson_rho')
    mv, kv, _, pv, _ = paired(c, 'D-BASE', 'absVR1_framemean')
    print(f"{c:12s} {med:+11.4f} {k}/{n:<3d} {p:6.3f} {mean_:+9.4f} {mr:+9.4f} {mv:+11.4f}")
    parts = []
    for e, x in zip(EPS, dl):
        tag = '_'.join(e.split('_')[1:3])
        parts.append(f"{tag}={x:+.4f}")
    print(' ' * 12 + ' per-episode deltas: ' + ', '.join(parts))

print('\n--- CONVENTION SENSITIVITY on D-NOTRACK (the headline "+0.0000") ---')
for seedagg in ['median','mean']:
    for epagg in ['median','mean']:
        med, k, n, p, dl = paired('D-NOTRACK','D-BASE','nRMSE', seedagg, epagg)
        print(f"  seed={seedagg:6s} ep={epagg:6s} -> {med:+.5f}  k/n={k}/{n}  p={p:.3f}")
# unpaired run-level (12 vs 12)
a = [r['nRMSE'] for r in rows if r['cell']=='D-NOTRACK']
b = [r['nRMSE'] for r in rows if r['cell']=='D-BASE']
print(f"  run-level (n=12 each) median D-NOTRACK={st.median(a):.5f} D-BASE={st.median(b):.5f} diff={st.median(a)-st.median(b):+.5f}")
print(f"  run-level  mean  D-NOTRACK={st.mean(a):.5f} D-BASE={st.mean(b):.5f} diff={st.mean(a)-st.mean(b):+.5f}")
# paired at run level (same episode+seed)
pa = {}
for r in rows:
    pa.setdefault((r['episode'], r['seed']), {})[r['cell']] = r['nRMSE']
dl12 = [v['D-NOTRACK']-v['D-BASE'] for v in pa.values() if 'D-NOTRACK' in v and 'D-BASE' in v]
k12 = sum(1 for x in dl12 if x < 0)
print(f"  run-level PAIRED n={len(dl12)}: median={st.median(dl12):+.5f} mean={st.mean(dl12):+.5f} k/n={k12}/{len(dl12)} p={sign_test_two_sided(k12,len(dl12)):.4f}")
print(f"    deltas: {[round(x,4) for x in sorted(dl12)]}")

print('\n--- ABSOLUTE LEVELS: oracle vs deploy nRMSE (why the same intervention can be 0) ---')
for c in ['X-BASE','X-NOTRACK','K4','D-BASE','D-NOTRACK','D-K12','D-K6']:
    t = cell_table('nRMSE')
    vals = [t[(c,e)] for e in EPS]
    ti = cell_table('init_nrmse')
    ivals = [ti[(c,e)] for e in EPS]
    print(f"  {c:11s} nRMSE med-of-ep={st.median(vals):.4f}  per-ep={[round(v,4) for v in vals]}  init_nrmse med={st.median(ivals):.4f}")

print('\n--- seed dispersion (power) ---')
import math
sds = []
for c in sorted({r['cell'] for r in rows}):
    for e in EPS:
        v = [r['nRMSE'] for r in rows if r['cell']==c and r['episode']==e]
        if len(v)>1: sds.append(st.stdev(v))
print(f"  pooled sigma_seed over {len(sds)} cells x eps: median={st.median(sds):.4f} mean={st.mean(sds):.4f}")
print(f"  median-of-3 sigma approx = sigma*1.16/sqrt(3) = {st.mean(sds)*1.16/math.sqrt(3):.4f}")
