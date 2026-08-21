#!/usr/bin/env python3
"""Reconcile rx1 ('deployment harm = 0') with rx2 ('tracker eats 22-56% of placement gain')
via the 1-parameter contraction operator tracked = a + b*init.  All from raw jsonl."""
import json, statistics as st
import numpy as np

def load(fn):
    out = []
    for l in open(fn):
        l = l.strip()
        if not l: continue
        d = json.loads(l)
        if 'error' in d: continue
        out.append(d)
    return out

r1 = load('tmp/rx1_tracker_2026-08-05/results2.jsonl')
r2 = load('tmp/rx2_init_2026-08-05/results.jsonl')
EPS = sorted({d['episode'] for d in r2})

print('='*90)
print('STEP 0 -- VALIDATE: is init_nrmse a good stand-in for the tracker-OFF counterfactual?')
print('  (rx1 has NOTRACK cells; compare their frame-mean nRMSE against their init_nrmse)')
print('='*90)
for c in ['X-NOTRACK','D-NOTRACK']:
    per = {}
    for d in r1:
        if d['cell']==c: per.setdefault(d['episode'],[]).append((d['init_nrmse'], d['nRMSE']))
    for e in sorted(per):
        i = st.median([x[0] for x in per[e]]); f = st.median([x[1] for x in per[e]])
        print(f"  {c:10s} {'_'.join(e.split('_')[1:3]):11s} init={i:.4f} NOTRACK frame-mean={f:.4f} diff={f-i:+.4f}")
    allv = [(x[1]-x[0]) for e in per for x in per[e]]
    print(f"  {c:10s} => median(frame-mean - init) = {st.median(allv):+.4f}  (n={len(allv)} runs)\n")

print('='*90)
print('STEP 1 -- FIT the contraction operator on rx2 (18 cells, tracker ON, init spans 0.07-0.34)')
print('='*90)
cells = sorted({d['cell'] for d in r2})
pts = []
for c in cells:
    per = {}
    for d in r2:
        if d['cell']==c: per.setdefault(d['episode'],[]).append((d['init_nrmse'], d['nRMSE']))
    for e in EPS:
        i = st.median([x[0] for x in per[e]]); f = st.median([x[1] for x in per[e]])
        pts.append((c, e, i, f))
X = np.array([p[2] for p in pts]); Y = np.array([p[3] for p in pts])
b, a = np.polyfit(X, Y, 1)
r = np.corrcoef(X, Y)[0,1]
fp = a/(1-b)
print(f"  POOLED (n={len(pts)} cell x episode): tracked = {a:.4f} + {b:.4f} * init   r={r:.3f}  fixed point = {fp:.4f}")
print('  per-episode fits (removes episode as a confounder):')
fps = []
for e in EPS:
    xs = np.array([p[2] for p in pts if p[1]==e]); ys = np.array([p[3] for p in pts if p[1]==e])
    bb, aa = np.polyfit(xs, ys, 1); rr = np.corrcoef(xs, ys)[0,1]
    print(f"    {'_'.join(e.split('_')[1:3]):11s} n={len(xs):2d}  b={bb:.4f} a={aa:.4f} r={rr:.3f} fixed point={aa/(1-bb):.4f}")
    fps.append((bb, aa/(1-bb)))
bmed = st.median([x[0] for x in fps]); fpmed = st.median([x[1] for x in fps])
print(f"  => median within-episode b = {bmed:.4f}, median fixed point = {fpmed:.4f}")

print('\n' + '='*90)
print('STEP 2 -- PREDICT harm  = (1-b)*(fixedpoint - init)  and compare with MEASURED harm')
print('  MEASURED harm (rx1) = frame-mean nRMSE(tracker ON) - frame-mean nRMSE(tracker OFF), same cfg')
print('  MEASURED harm (rx2) = frame-mean nRMSE - init_nrmse  (init validated as OFF-proxy in STEP 0)')
print('='*90)
def epmed(rows, c, k):
    per = {}
    for d in rows:
        if d['cell']==c: per.setdefault(d['episode'],[]).append(d[k])
    return {e: st.median(per[e]) for e in per}

print(f"  {'arm':12s} {'src':4s} {'init':>7s} {'pred harm':>10s} {'measured':>9s} {'err':>8s}")
# rx1: true ON-vs-OFF pairs
for on, off, lbl in [('X-BASE','X-NOTRACK','oracle'), ('D-BASE','D-NOTRACK','deploy')]:
    ion = epmed(r1, on, 'init_nrmse'); fon = epmed(r1, on, 'nRMSE'); foff = epmed(r1, off, 'nRMSE')
    i = st.median(list(ion.values()))
    meas = st.median([fon[e]-foff[e] for e in fon])
    pred = (1-bmed)*(fpmed - i)
    print(f"  {lbl:12s} rx1  {i:7.4f} {pred:+10.4f} {meas:+9.4f} {pred-meas:+8.4f}   <-- TRUE ON/OFF pair")
# rx2: init-proxy
for c in ['X-BASE','P3','G1','G1P1','G1P3','AUTO','ORACLE-K','ORACLE']:
    ii = epmed(r2, c, 'init_nrmse'); ff = epmed(r2, c, 'nRMSE')
    i = st.median(list(ii.values()))
    meas = st.median([ff[e]-ii[e] for e in ff])
    pred = (1-bmed)*(fpmed - i)
    print(f"  {c:12s} rx2  {i:7.4f} {pred:+10.4f} {meas:+9.4f} {pred-meas:+8.4f}")

print('\n' + '='*90)
print('STEP 3 -- so what would the DECISIVE missing cell buy?  (good placement x tracker fix)')
print('='*90)
for c, i in [('AUTO', st.median(list(epmed(r2,'AUTO','init_nrmse').values()))),
             ('G1P1', st.median(list(epmed(r2,'G1P1','init_nrmse').values())))]:
    cur = st.median(list(epmed(r2, c, 'nRMSE').values()))
    print(f"  {c}: shipped-tracker frame-mean = {cur:.4f}; init = {i:.4f}")
    print(f"      if tracker were a no-op (b=1): {i:.4f}   -> headroom {i-cur:+.4f}")
    # K4 in rx1 beat NOTRACK by this ratio at oracle
i_x = st.median(list(epmed(r1,'X-BASE','init_nrmse').values()))
f_nt = st.median(list(epmed(r1,'X-NOTRACK','nRMSE').values()))
f_k4 = st.median(list(epmed(r1,'K4','nRMSE').values()))
print(f"  rx1 oracle: init={i_x:.4f} NOTRACK={f_nt:.4f} K4(penalty off)={f_k4:.4f} "
      f"-> K4 beats freeze by {f_nt-f_k4:+.4f} ({100*(f_nt-f_k4)/f_nt:.1f}%)")
print('  Extrapolating that ratio to AUTO is NOT supported by any measurement -- that is the gap.')

print('\n' + '='*90)
print('STEP 4 -- MDE / power at the deployment operating point')
print('='*90)
sds = []
for rows, lbl in [(r1,'rx1'), (r2,'rx2')]:
    for c in sorted({d['cell'] for d in rows}):
        for e in sorted({d['episode'] for d in rows}):
            v = [d['nRMSE'] for d in rows if d['cell']==c and d['episode']==e]
            if len(v) > 1: sds.append(st.stdev(v))
sd = st.median(sds)
print(f"  sigma_seed (median over {len(sds)} cell x episode groups) = {sd:.4f}")
sd3 = sd*1.16/np.sqrt(3)
print(f"  sigma of median-of-3 ~= {sd3:.4f}")
print(f"  paired n=4, 80% power, two-sided alpha=0.05 requires |d| >= {2.87*sd3/np.sqrt(4):.4f} (t-approx)")
print("  BUT: n=4 two-sided sign test attains only p in {0.125, 0.625, 1.000} -> alpha=0.05 UNREACHABLE.")
print("  To reach a two-sided sign-test p<0.05 you need n>=6 paired units (2*(1/2)^6 = 0.031).")
