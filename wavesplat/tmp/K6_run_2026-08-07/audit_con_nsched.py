#!/usr/bin/env python3
"""CON audit part 3:
  (a) Is K6-NSCHED simply a BROKEN / under-converged arm?  (its own null margin,
      init_nrmse, per-config breakdown, N=15 vs N=30, N vs max_n=30 saturation)
  (b) What does the 'placement factor -0.1023' become if the denominator is an arm
      that is NOT broken (AUTO, or ORACLE-K vs AUTO)?
  (c) Full multiplicity accounting over the declared test volume (BH + Bonferroni).
  (d) Null-exclusion sensitivity: n=14 vs n=16 vs leave-one-out.
  (e) Config-fold (n=4) aggregation, as PREREG requires be reported alongside.
  (f) Additivity check: does -0.0533 + (ORACLE-K - AUTO) equal -0.0640?
"""
import json, os, math
from collections import defaultdict
import itertools

HERE = os.path.dirname(os.path.abspath(__file__))
rows = [json.loads(l) for l in open(os.path.join(HERE,'results.jsonl')) if l.strip()]
nulls = {n['episode']: n for n in json.load(open(os.path.join(HERE,'nulls_k6.json')))}

def med(v):
    v = sorted(v); n = len(v)
    return v[n//2] if n % 2 else 0.5*(v[n//2-1]+v[n//2])

M = defaultdict(dict)
buck = defaultdict(list)
for r in rows:
    buck[(r['cell'], r['episode'])].append(r)
METRICS = ['nRMSE','init_nrmse','Pearson_rho','absVR_framewise','m_mean',
           'decorr_mean','shrink_mean']
for (c,e), rs in buck.items():
    M[c][e] = {m: med([x[m] for x in rs]) for m in METRICS}
    M[c][e]['N'] = med([x['N'] for x in rs])
    M[c][e]['det'] = med([x['spec']['detection_rate'] for x in rs if 'detection_rate' in x.get('spec',{})])
    M[c][e]['dir_err'] = med([x['spec']['direction_error_deg_mean'] for x in rs if 'direction_error_deg_mean' in x.get('spec',{})])
    M[c][e]['kmag_err'] = med([x['spec']['k_mag_rel_error_mean'] for x in rs if 'k_mag_rel_error_mean' in x.get('spec',{})])

eps = sorted(M['AUTO'])
KEEP = [e for e in eps if M['AUTO'][e]['nRMSE'] < nulls[e]['zero_null_tracking_nRMSE_mean']]
EXCL = [e for e in eps if e not in KEEP]

def C(n,k): return math.comb(n,k)
def sgn(ds):
    d = [x for x in ds if x != 0]; n=len(d)
    if not n: return 0,0,1.0,float('nan')
    w = sum(1 for x in d if x<0); k=max(w,n-w)
    return n, w, min(1.0, 2*sum(C(n,i) for i in range(k,n+1))/2**n), med(ds)

def con(a,b,blk,m='nRMSE'):
    return sgn([M[a][e][m]-M[b][e][m] for e in blk])

print("="*96)
print("(a) IS K6-NSCHED BROKEN?   m>1 == worse than predicting a flat sea")
print("="*96)
CFG = lambda e: e.split('episode_')[1].split('_seed')[0]
cfgs = sorted({CFG(e) for e in eps})
print(f"{'cell':11s} " + " ".join(f"{c:>13s}" for c in cfgs) + "   ALL16")
for c in ['AUTO','K6-NSCHED','ORACLE-K','ORACLE']:
    line = f"{c:11s} "
    for g in cfgs:
        sub = [e for e in eps if CFG(e)==g]
        line += f" {med([M[c][e]['nRMSE'] for e in sub]):6.4f}/{med([M[c][e]['m_mean'] for e in sub]):5.2f}"
    line += f"   {med([M[c][e]['nRMSE'] for e in eps]):.4f}/{med([M[c][e]['m_mean'] for e in eps]):.2f}"
    print(line)
print("   (cell = median nRMSE / median m_mean;  m>1.00 == worse than the flat-sea null)")

print(f"\n  K6-NSCHED beats flat-sea null (nRMSE): "
      f"{sum(1 for e in eps if M['K6-NSCHED'][e]['nRMSE'] < nulls[e]['zero_null_tracking_nRMSE_mean'])}/16")
print(f"  K6-NSCHED m_mean > 1 (diverged)      : {sum(1 for e in eps if M['K6-NSCHED'][e]['m_mean']>1)}/16")
print(f"  AUTO      m_mean > 1                 : {sum(1 for e in eps if M['AUTO'][e]['m_mean']>1)}/16")
print(f"  ORACLE-K  m_mean > 1                 : {sum(1 for e in eps if M['ORACLE-K'][e]['m_mean']>1)}/16")

print("\n  -- convergence: init_nrmse (end of Phase 1, SAME 900 gradient steps for all) --")
zi = {e: nulls[e]['zero_null_init_nrmse'] for e in eps}
for c in ['AUTO','K6-NSCHED','ORACLE-K','ORACLE','K6-KLR00']:
    v = [M[c][e]['init_nrmse'] for e in eps]
    b = sum(1 for e in eps if M[c][e]['init_nrmse'] < zi[e])
    print(f"   {c:11s} med={med(v):.4f}  min={min(v):.4f} max={max(v):.4f}"
          f"  beats-init-null {b:2d}/16")
print(f"   {'FLAT NULL':11s} med={med(list(zi.values())):.4f}")

print("\n  -- placement quality of the SAME blind beamformer, AUTO vs K6-NSCHED --")
print(f"   {'cell':11s} {'det_rate':>9s} {'dir_err_deg':>12s} {'kmag_rel':>10s} {'final N':>8s}")
for c in ['AUTO','K6-NSCHED','ORACLE-K','ORACLE']:
    print(f"   {c:11s} {med([M[c][e]['det'] for e in eps]):9.3f}"
          f" {med([M[c][e]['dir_err'] for e in eps]):12.2f}"
          f" {med([M[c][e]['kmag_err'] for e in eps]):10.3f}"
          f" {med([M[c][e]['N'] for e in eps]):8.1f}")

print("\n  -- split by N_gt (15 vs 30). max_n=30, so the 30-comp cells start AT the cap --")
for tag, sel in (('N_gt=15', [e for e in eps if '15comp' in e]),
                 ('N_gt=30', [e for e in eps if '30comp' in e])):
    n,w,p,m = con('ORACLE-K','K6-NSCHED',sel)
    n2,w2,p2,m2 = con('K6-NSCHED','AUTO',sel)
    print(f"   {tag}: ORACLE-K - NSCHED  med={m:+.4f} {w}/{n}   |   NSCHED - AUTO med={m2:+.4f} {w2}/{n2}"
          f"   NSCHED med m={med([M['K6-NSCHED'][e]['m_mean'] for e in sel]):.2f}")

print("\n" + "="*96)
print("(b) THE PLACEMENT FACTOR AGAINST A NON-BROKEN DENOMINATOR")
print("="*96)
for a,b in [('ORACLE-K','K6-NSCHED'), ('ORACLE-K','AUTO'), ('ORACLE','AUTO'),
            ('ORACLE','K6-NSCHED'), ('AUTO','K6-NSCHED')]:
    for blk,nm in ((KEEP,'n=14'),(eps,'n=16')):
        n,w,p,m = con(a,b,blk)
        print(f"  {a:10s} - {b:10s} [{nm}]  med={m:+.4f}  {w}/{n}  p={p:.5g}")

print("\n" + "="*96)
print("(f) ADDITIVITY: medians of differences do NOT decompose")
print("="*96)
_,_,_,d1 = con('ORACLE','ORACLE-K',KEEP)
_,_,_,d2 = con('ORACLE-K','AUTO',KEEP)
_,_,_,d3 = con('ORACLE','AUTO',KEEP)
print(f"  med(ORACLE-ORACLE-K)={d1:+.4f}  + med(ORACLE-K-AUTO)={d2:+.4f}  = {d1+d2:+.4f}")
print(f"  but med(ORACLE-AUTO)                                         = {d3:+.4f}"
      f"   (mismatch {d3-(d1+d2):+.4f})")

print("\n" + "="*96)
print("(d) NULL-EXCLUSION SENSITIVITY + leave-one-episode-out on the headlines")
print("="*96)
for a,b in [('ORACLE','ORACLE-K'), ('ORACLE-K','K6-NSCHED'), ('ORACLE','AUTO'),
            ('K6-KLR01','AUTO'), ('K6-KLR03','AUTO')]:
    ms=[]
    for drop in KEEP:
        blk=[e for e in KEEP if e!=drop]
        ms.append(con(a,b,blk)[3])
    n,w,p,m = con(a,b,KEEP)
    print(f"  {a:10s}-{b:10s} n=14 med={m:+.4f}  LOO range [{min(ms):+.4f},{max(ms):+.4f}]"
          f"  n=16 med={con(a,b,eps)[3]:+.4f}")
print(f"  excluded episodes = {[e[8:-13] for e in EXCL]}")

print("\n" + "="*96)
print("(e) CONFIG FOLD (n=4, floor p = 0.125 -- PREREG forbids the word 'significant')")
print("="*96)
for a,b in [('K6-KLR01','AUTO'),('K6-KLR03','AUTO'),('K6-KLR00','AUTO'),
            ('ORACLE','ORACLE-K'),('ORACLE-K','K6-NSCHED'),('ORACLE','AUTO')]:
    ds=[]
    for g in cfgs:
        sub=[e for e in eps if CFG(e)==g]
        ds.append(med([M[a][e]['nRMSE'] for e in sub])-med([M[b][e]['nRMSE'] for e in sub]))
    n,w,p,m = sgn(ds)
    print(f"  {a:10s} - {b:10s}  med={m:+.4f}  {w}/{n}  floor-p={2/2**4:.3f}")

print("\n" + "="*96)
print("(c) MULTIPLICITY over the PREREG-declared test volume")
print("="*96)
CONTRASTS = [('K6-KLR03','AUTO'),('K6-KLR01','AUTO'),('K6-KLR00','AUTO'),
             ('ORACLE-K','K6-NSCHED'),('ORACLE','ORACLE-K')]
PANEL7 = ['nRMSE','init_nrmse','Pearson_rho','absVR_framewise','m_mean',
          'decorr_mean','shrink_mean']
tests=[]
for (a,b) in CONTRASTS:
    for m in PANEL7:
        for blk,bn in ((KEEP,'n14'),(eps,'n16')):
            n,w,p,md = con(a,b,blk,m)
            tests.append((f"{a}-{b}|{m}|{bn}", p, md, w, n))
print(f"  declared family = {len(CONTRASTS)} contrasts x {len(PANEL7)} metrics x 2 blocks"
      f" = {len(tests)} tests")
tests.sort(key=lambda t:t[1])
alpha=0.05
bonf = alpha/len(tests)
# Benjamini-Hochberg
m_=len(tests); bh_cut=0
for i,(nm,p,md,w,n) in enumerate(tests,1):
    if p <= i/m_*alpha: bh_cut=i
print(f"  Bonferroni threshold = {bonf:.3g}   BH(0.05) rejects the {bh_cut} smallest p")
print(f"  {'test':46s} {'p':>10s} {'medD':>9s} {'wins':>7s}  Bonf  BH")
for i,(nm,p,md,w,n) in enumerate(tests,1):
    if i>18 and p>0.05: break
    print(f"  {nm:46s} {p:10.3g} {md:+9.4f} {w:4d}/{n:<2d}"
          f"  {'Y' if p<=bonf else '.':4s}  {'Y' if i<=bh_cut else '.'}")

print("\n  -- restricted family: nRMSE only, 5 declared contrasts, PREREG block (n=14) --")
tt=[(f"{a}-{b}", con(a,b,KEEP)[2], con(a,b,KEEP)[3], con(a,b,KEEP)[1]) for a,b in CONTRASTS]
tt.sort(key=lambda t:t[1])
b5=alpha/len(tt)
for i,(nm,p,md,w) in enumerate(tt,1):
    print(f"    {nm:26s} p={p:9.4g}  medD={md:+.4f}  wins={w}/14"
          f"   Bonf(a={b5:.3f}) {'PASS' if p<=b5 else 'fail'}")

print("\n" + "="*96)
print("(g) IS -0.0640 (ORACLE-AUTO) EVEN IN THE PREREG's JUDGMENT LIST?")
print("="*96)
prereg = open(os.path.join(HERE,'PREREG.md')).read()
for pat in ['ORACLE-K − K6-NSCHED','ORACLE − ORACLE-K','ORACLE − AUTO','ORACLE-K − AUTO',
            'K6-KLR01', 'ORACLE − X-BASE']:
    print(f"   '{pat}' appears in PREREG.md : {prereg.count(pat)} time(s)")
