import json, collections, statistics as st, math
from math import comb
R=[json.loads(l) for l in open('tmp/rx1_tracker_2026-08-05/results2.jsonl')]
EPS=sorted(set(r['episode'] for r in R))

def sign_p(deltas, direction=-1):
    """two-sided exact sign test; k = # in improving direction"""
    nz=[d for d in deltas if d!=0.0]
    n=len(nz); k=sum(1 for d in nz if (d<0 if direction<0 else d>0))
    if n==0: return 0,0,1.0
    # two-sided
    p=sum(comb(n,i) for i in range(0,n+1) if abs(i-n/2)>=abs(k-n/2))/2**n
    return k,n,min(1.0,p)

def cellmed(cell, field, ep):
    v=[r[field] for r in R if r['cell']==cell and r['episode']==ep]
    assert len(v)==3, (cell,ep,len(v))
    return st.median(v)

def paired(cellA, cellB, field='nRMSE'):
    """delta = A - B per episode (seed median-of-3 first)"""
    d=[cellmed(cellA,field,e)-cellmed(cellB,field,e) for e in EPS]
    return d

def rep(cellA,cellB,field='nRMSE',direction=-1,label=''):
    d=paired(cellA,cellB,field)
    med=st.median(d); k,n,p=sign_p(d,direction)
    print(f"{label or (cellA+'-'+cellB):22s} {field:18s} median={med:+.4f}  {k}/{n}  p={p:.4f}  per-ep={[round(x,4) for x in d]}")
    return d,med

print("=== ORACLE arm (vs X-BASE), nRMSE, lower=better ===")
for c in ['K1','K2','K3','K4','K5','K6','K7','K12','K12-S1','X-NOTRACK']:
    rep(c,'X-BASE')
print()
print("=== DEPLOY arm (vs D-BASE), nRMSE ===")
for c in ['D-NOTRACK','D-K12','D-K6']:
    rep(c,'D-BASE')
print()
print("=== recovery rate vs X-NOTRACK floor (oracle) ===")
_,base_floor=paired('X-NOTRACK','X-BASE'),None
fl=st.median(paired('X-NOTRACK','X-BASE'))
for c in ['K4','K6','K7','K12','K12-S1']:
    m=st.median(paired(c,'X-BASE'))
    print(f"  {c:8s} {m:+.4f}  recovery={100*m/fl:.1f}% of NOTRACK floor {fl:+.4f}")
print()
print("=== deploy rho / |VR-1| ===")
for c in ['D-NOTRACK','D-K12','D-K6']:
    rep(c,'D-BASE','Pearson_rho',direction=+1)
    rep(c,'D-BASE','absVR1_framemean',direction=-1)
print()
print("=== absolute cell levels (median of 12 runs) ===")
for c in sorted(set(r['cell'] for r in R)):
    n=[r['nRMSE'] for r in R if r['cell']==c]; i=[r['init_nrmse'] for r in R if r['cell']==c]
    nn=[r['N'] for r in R if r['cell']==c]
    print(f"  {c:10s} nRMSE={st.median(n):.4f}  init={st.median(i):.4f}  N={st.median(nn):.1f}")
