import json, statistics as st
from math import comb
R=[json.loads(l) for l in open('tmp/rx1_tracker_2026-08-05/results2.jsonl')]
EPS=sorted(set(r['episode'] for r in R)); SEEDS=sorted(set(r['seed'] for r in R))
def sign_p(d):
    nz=[x for x in d if x!=0]; n=len(nz); k=sum(1 for x in nz if x<0)
    p=sum(comb(n,i) for i in range(n+1) if abs(i-n/2)>=abs(k-n/2))/2**n if n else 1.0
    return k,n,min(1.0,p)
def get(c,e,s,f='nRMSE'):
    v=[r[f] for r in R if r['cell']==c and r['episode']==e and r['seed']==s]; return v[0]
def conv_sweep(A,B,f='nRMSE'):
    out={}
    # C1 seed-median -> per-ep delta -> median
    d=[st.median([get(A,e,s,f) for s in SEEDS])-st.median([get(B,e,s,f) for s in SEEDS]) for e in EPS]
    out['C1 seedMED->epMED']=(st.median(d),)+sign_p(d)
    out['C2 seedMED->epMEAN']=(st.mean(d),)+sign_p(d)
    dm=[st.mean([get(A,e,s,f) for s in SEEDS])-st.mean([get(B,e,s,f) for s in SEEDS]) for e in EPS]
    out['C3 seedMEAN->epMED']=(st.median(dm),)+sign_p(dm)
    out['C4 seedMEAN->epMEAN']=(st.mean(dm),)+sign_p(dm)
    # C5 per-(ep,seed) paired delta -> median over seeds -> median over eps
    d5=[st.median([get(A,e,s,f)-get(B,e,s,f) for s in SEEDS]) for e in EPS]
    out['C5 pairMEDseed->epMED']=(st.median(d5),)+sign_p(d5)
    d6=[get(A,e,s,f)-get(B,e,s,f) for e in EPS for s in SEEDS]
    out['C6 12pairs MED']=(st.median(d6),)+sign_p(d6)
    out['C7 12pairs MEAN']=(st.mean(d6),)+sign_p(d6)
    return out,d,d6
for A,B in [('D-NOTRACK','D-BASE'),('D-K12','D-BASE'),('D-K6','D-BASE')]:
    o,d,d6=conv_sweep(A,B)
    print(f"--- {A} - {B} ---")
    for k,(v,kk,n,p) in o.items(): print(f"   {k:24s} {v:+.4f}  {kk}/{n} p={p:.4f}")
    print(f"   per-ep deltas (C1): {[round(x,4) for x in d]}")
    print(f"   sd per-ep delta (sample)={st.stdev(d):.4f}  sd 12 run-pairs={st.stdev(d6):.4f}")
    print()
# t CI for D-NOTRACK
d=[st.median([get('D-NOTRACK',e,s) for s in SEEDS])-st.median([get('D-BASE',e,s) for s in SEEDS]) for e in EPS]
m=st.mean(d); sd=st.stdev(d); se=sd/2
print(f"D-NOTRACK per-ep delta: mean={m:+.5f} sd={sd:.5f} se={se:.5f} 95%CI(t3=3.1824)=[{m-3.1824*se:+.4f},{m+3.1824*se:+.4f}]")
# MDE
import math
def mde(sd,n,power=0.80,alpha=0.05):
    from statistics import NormalDist
    z=NormalDist().inv_cdf(1-alpha/2)+NormalDist().inv_cdf(power)
    return z*sd/math.sqrt(n)
for lbl,sd in [('rx1 deploy per-ep sd',st.stdev(d))]:
    for n in [4,6,8,12,20]:
        print(f"  {lbl}={sd:.4f}  n={n:2d} MDE(80%,2sided)={mde(sd,n):.4f}")
# oracle sd
do=[st.median([get('X-NOTRACK',e,s) for s in SEEDS])-st.median([get('X-BASE',e,s) for s in SEEDS]) for e in EPS]
print(f"  oracle X-NOTRACK per-ep sd={st.stdev(do):.4f}")
