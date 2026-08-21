#!/usr/bin/env python3
"""CON audit part 4:
  (h) effective-N (participation ratio) shift OLD->NEW and whether the headline
      contrasts depend on it (stratify; PREREG/feedback rule: control the
      stratifying variable before promoting a mechanism).
  (i) routing confound: which AUTO runs fell back to v2_konly, and does dropping
      them move anything.
  (j) the 2x2 factorial: which cell is MISSING.
"""
import json, os, math
from collections import defaultdict
import numpy as np, h5py

REPO='/home/vexit/wavesplat/wavesplat'; HERE=os.path.join(REPO,'tmp/K6_run_2026-08-07')
rows=[json.loads(l) for l in open(os.path.join(HERE,'results.jsonl')) if l.strip()]
nulls={n['episode']:n for n in json.load(open(os.path.join(HERE,'nulls_k6.json')))}
def med(v):
    v=sorted(v); n=len(v); return v[n//2] if n%2 else .5*(v[n//2-1]+v[n//2])
def C(n,k): return math.comb(n,k)
def sgn(ds):
    d=[x for x in ds if x!=0]; n=len(d)
    if not n: return 0,0,1.,float('nan')
    w=sum(1 for x in d if x<0); k=max(w,n-w)
    return n,w,min(1.,2*sum(C(n,i) for i in range(k,n+1))/2**n),med(ds)

B=defaultdict(list)
for r in rows: B[(r['cell'],r['episode'])].append(r)
M=defaultdict(dict)
for (c,e),rs in B.items(): M[c][e]={'nRMSE':med([x['nRMSE'] for x in rs]),
                                    'init':med([x['init_nrmse'] for x in rs])}
eps=sorted(M['AUTO'])
KEEP=[e for e in eps if M['AUTO'][e]['nRMSE']<nulls[e]['zero_null_tracking_nRMSE_mean']]

def PR(path):
    with h5py.File(path,'r') as f: A=np.asarray(f['config/wave_components'])[:,0]
    P=A**2; P=P/P.sum(); return 1./np.sum(P**2), len(A)
pr={}
for e in eps:
    p,n=PR(os.path.join(REPO,'data_indep2_2026-08-07',e)); pr[e]=(p,n)
CROWDED16=[f'episode_BF{bf}_{nc}comp_seed{es}_uniform.h5' for bf in (4,5) for nc in (15,30) for es in (0,1,2,3)]
proold={}
for e in CROWDED16:
    p,n=PR(os.path.join(REPO,'data_finemesh_2026-07-23',e)); proold[e]=(p,n)

print("="*92); print("(h) EFFECTIVE component count (participation ratio 1/sum(p^2)) -- did the");
print("    'crowded16' block stay crowded?"); print("="*92)
print(f"{'episode':30s} {'N':>4s} {'PR_old':>8s} {'PR_new':>8s}   verdict")
for e in CROWDED16:
    ne=e.replace('.h5','_ra_rf_ds.h5')
    po,_=proold[e]; pn,n=pr[ne]
    v = 'NEW is effectively SPARSE (<5)' if pn<5 else ''
    print(f"{e[8:38]:30s} {n:4d} {po:8.2f} {pn:8.2f}   {v}")
print(f"\n  median PR   old={med([proold[e][0] for e in CROWDED16]):.2f}"
      f"   new={med([pr[e.replace('.h5','_ra_rf_ds.h5')][0] for e in CROWDED16]):.2f}")
print(f"  episodes with PR < 5 :  old={sum(1 for e in CROWDED16 if proold[e][0]<5)}/16"
      f"   new={sum(1 for e in CROWDED16 if pr[e.replace('.h5','_ra_rf_ds.h5')][0]<5)}/16")
print(f"  episodes with PR < 8 :  old={sum(1 for e in CROWDED16 if proold[e][0]<8)}/16"
      f"   new={sum(1 for e in CROWDED16 if pr[e.replace('.h5','_ra_rf_ds.h5')][0]<8)}/16")
print(f"  PR range             :  old=[{min(proold[e][0] for e in CROWDED16):.2f},"
      f"{max(proold[e][0] for e in CROWDED16):.2f}]"
      f"   new=[{min(pr[e.replace('.h5','_ra_rf_ds.h5')][0] for e in CROWDED16):.2f},"
      f"{max(pr[e.replace('.h5','_ra_rf_ds.h5')][0] for e in CROWDED16):.2f}]")

prv=sorted(eps,key=lambda e:pr[e][0])
lo,hi=prv[:8],prv[8:]
print(f"\n  -- stratify the 16 new episodes by PR (low 8 = 'effectively sparser') --")
print(f"     low-PR  8: PR {pr[lo[0]][0]:.1f}..{pr[lo[-1]][0]:.1f}")
print(f"     high-PR 8: PR {pr[hi[0]][0]:.1f}..{pr[hi[-1]][0]:.1f}")
def con(a,b,blk): return sgn([M[a][e]['nRMSE']-M[b][e]['nRMSE'] for e in blk])
for a,b in [('ORACLE','AUTO'),('ORACLE','ORACLE-K'),('ORACLE-K','K6-NSCHED'),
            ('ORACLE-K','AUTO'),('K6-KLR01','AUTO')]:
    n1,w1,p1,m1=con(a,b,lo); n2,w2,p2,m2=con(a,b,hi)
    print(f"     {a:9s}-{b:10s} lowPR med={m1:+.4f} {w1}/{n1} | highPR med={m2:+.4f} {w2}/{n2}")
# spearman of AUTO nRMSE vs PR
def spear(x,y):
    rx=np.argsort(np.argsort(x)); ry=np.argsort(np.argsort(y))
    return float(np.corrcoef(rx,ry)[0,1])
print(f"\n     Spearman(PR, AUTO nRMSE)            = {spear([pr[e][0] for e in eps],[M['AUTO'][e]['nRMSE'] for e in eps]):+.3f}")
print(f"     Spearman(PR, ORACLE-AUTO delta)     = {spear([pr[e][0] for e in eps],[M['ORACLE'][e]['nRMSE']-M['AUTO'][e]['nRMSE'] for e in eps]):+.3f}")
print(f"     Spearman(PR, ORACLE-K - NSCHED)     = {spear([pr[e][0] for e in eps],[M['ORACLE-K'][e]['nRMSE']-M['K6-NSCHED'][e]['nRMSE'] for e in eps]):+.3f}")

print("\n"+"="*92); print("(i) ROUTING CONFOUND"); print("="*92)
fb=[(r['episode'],r['seed']) for r in rows if r['cell']=='AUTO' and r.get('auto_selected')=='v2_konly']
print(f"  AUTO fell back to v2_konly on {len(fb)}/48 runs: {[(e[8:-13],s) for e,s in fb]}")
fbeps=sorted({e for e,_ in fb})
print(f"  affected episodes: {[e[8:-13] for e in fbeps]}")
blk=[e for e in KEEP if e not in fbeps]
for a,b in [('ORACLE','AUTO'),('ORACLE-K','AUTO'),('K6-KLR01','AUTO')]:
    n,w,p,m=con(a,b,blk); n0,w0,p0,m0=con(a,b,KEEP)
    print(f"  {a:9s}-{b:9s}  n=14 med={m0:+.4f} {w0}/{n0}   |  drop-fallback n={len(blk)} med={m:+.4f} {w}/{n}")
print("  NOTE: ORACLE / ORACLE-K / K6-NSCHED never touch the router "
      "(auto_selected=None, ensemble_k=1);")
print("        AUTO / K6-KLR* do (learned 45 / v2_konly 3, ensemble_k in {1,3}).")
ek=defaultdict(int)
for r in rows:
    if r['cell']=='AUTO': ek[r.get('ensemble_k')]+=1
print(f"        AUTO ensemble_k histogram over 48 runs: {dict(ek)}"
      "   -> the oracle cells also lose median-of-k ensembling.")

print("\n"+"="*92); print("(j) THE 2x2 FACTORIAL HAS A MISSING CELL"); print("="*92)
print(f"                       | Phase-1 schedule = shipped ladder [3,8,15]/[200,200,500]"
      f"  |  1 stage [N_gt]/[900]")
print(f"   placement = blind   |  AUTO      {med([M['AUTO'][e]['nRMSE'] for e in eps]):.4f}"
      f"                              |  K6-NSCHED {med([M['K6-NSCHED'][e]['nRMSE'] for e in eps]):.4f}")
print(f"   placement = ORACLE  |  ***NEVER RUN***"
      f"                                |  ORACLE-K  {med([M['ORACLE-K'][e]['nRMSE'] for e in eps]):.4f}")
print("   => 'the placement factor' is a SIMPLE effect inside the 1-stage column only.")
print(f"   simple effect of placement | 1-stage   = {con('ORACLE-K','K6-NSCHED',KEEP)[3]:+.4f} "
      f"({con('ORACLE-K','K6-NSCHED',KEEP)[1]}/14)")
print(f"   simple effect of schedule  | blind     = {con('AUTO','K6-NSCHED',KEEP)[3]:+.4f} "
      f"({con('AUTO','K6-NSCHED',KEEP)[1]}/14)")
print(f"   diagonal ORACLE-K - AUTO              = {con('ORACLE-K','AUTO',KEEP)[3]:+.4f} "
      f"({con('ORACLE-K','AUTO',KEEP)[1]}/14, p={con('ORACLE-K','AUTO',KEEP)[2]:.3g})")
print("   => the placement gain and the schedule loss very nearly CANCEL.")
