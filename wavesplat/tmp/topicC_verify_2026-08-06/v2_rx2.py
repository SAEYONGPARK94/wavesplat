import json, statistics as st
from math import comb
R2=[json.loads(l) for l in open('tmp/rx2_init_2026-08-05/results.jsonl')]
EPS=sorted(set(r['episode'] for r in R2))
def sign_p(d,direction=-1):
    nz=[x for x in d if x!=0]; n=len(nz); k=sum(1 for x in nz if (x<0 if direction<0 else x>0))
    p=sum(comb(n,i) for i in range(n+1) if abs(i-n/2)>=abs(k-n/2))/2**n if n else 1.0
    return k,n,min(1.0,p)
def med(cell,f,ep,R=R2):
    v=[r[f] for r in R if r['cell']==cell and r['episode']==ep]; assert len(v)==3,(cell,ep,len(v)); return st.median(v)

print("=== rx2 cells: absolute + gap(nRMSE - init_nrmse) per episode ===")
print(f"{'cell':10s} {'nRMSE_med':>9s} {'init_med':>9s} {'gap_medOfEp':>11s} {'gap_cellmed':>11s} {'k/4':>5s} {'sd':>7s}")
gaps={}
for c in sorted(set(r['cell'] for r in R2)):
    g=[med(c,'nRMSE',e)-med(c,'init_nrmse',e) for e in EPS]
    nn=st.median([med(c,'nRMSE',e) for e in EPS]); ii=st.median([med(c,'init_nrmse',e) for e in EPS])
    k,n,p=sign_p(g,direction=+1)   # positive gap = tracker hurts
    gaps[c]=g
    print(f"{c:10s} {nn:9.4f} {ii:9.4f} {st.median(g):+11.4f} {nn-ii:+11.4f} {k}/{n} {st.pstdev(g):7.4f}")

print()
print("=== rx2 arm gains vs X-BASE (nRMSE) and vs X-BASE init ===")
for c in ['AUTO','G1','G1P1','G1P3','P1','P2','P3','ORACLE','ORACLE-K']:
    dn=[med(c,'nRMSE',e)-med('X-BASE','nRMSE',e) for e in EPS]
    di=[med(c,'init_nrmse',e)-med('X-BASE','init_nrmse',e) for e in EPS]
    k,n,p=sign_p(dn)
    surv = st.median(dn)/st.median(di) if st.median(di)!=0 else float('nan')
    print(f"{c:9s} dnRMSE={st.median(dn):+.4f} {k}/{n} p={p:.3f} | dinit={st.median(di):+.4f} | survival={100*surv:5.1f}% eaten={100*(1-surv):5.1f}%")

print()
print("=== rx1 D-BASE vs rx2 X-BASE cfg comparison ===")
R1=[json.loads(l) for l in open('tmp/rx1_tracker_2026-08-05/results2.jsonl')]
d=[r for r in R1 if r['cell']=='D-BASE'][0]['cfg']
x=[r for r in R2 if r['cell']=='X-BASE'][0]['cfg']
allk=sorted(set(d)|set(x))
diff=[(k,d.get(k,'<MISSING>'),x.get(k,'<MISSING>')) for k in allk if d.get(k,'<MISSING>')!=x.get(k,'<MISSING>')]
print(' rx1 D-BASE cfg:',d)
print(' rx2 X-BASE cfg:',x)
print(' DIFFS:',diff if diff else 'NONE')
print()
print("=== levels: median-of-12 vs mean-of-12 vs medOfEpSeedMed ===")
for lbl,R,cells in [('rx1',R1,['D-BASE','D-NOTRACK','K4','X-BASE','X-NOTRACK']),('rx2',R2,['X-BASE','AUTO','G1P1','ORACLE'])]:
    for c in cells:
        v=[r['nRMSE'] for r in R if r['cell']==c]; i=[r['init_nrmse'] for r in R if r['cell']==c]
        N=[r['N'] for r in R if r['cell']==c]
        eps=sorted(set(r['episode'] for r in R))
        vm=st.median([st.median([r['nRMSE'] for r in R if r['cell']==c and r['episode']==e]) for e in eps])
        im=st.median([st.median([r['init_nrmse'] for r in R if r['cell']==c and r['episode']==e]) for e in eps])
        print(f"{lbl} {c:10s} med12={st.median(v):.4f} mean12={st.mean(v):.4f} medEpSeedMed={vm:.4f} | init med12={st.median(i):.4f} mean12={st.mean(i):.4f} medEp={im:.4f} | N med={st.median(N):.1f} mean={st.mean(N):.2f}")
