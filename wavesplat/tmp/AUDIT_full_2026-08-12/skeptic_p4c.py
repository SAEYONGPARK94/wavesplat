"""S11 살아있는 헤드라인 C_matched 를 10개 3/3 분할 전수로 재검정."""
import json, os, itertools, statistics as st
from collections import defaultdict
from math import comb
def p2(k,n):
    if n==0: return float('nan')
    return sum(comb(n,i) for i in range(n+1) if abs(i-n/2)>=abs(k-n/2))/2.0**n

M=defaultdict(lambda: defaultdict(list)); reg={}
for path in ('tmp/S9_lr_x_l1_2026-08-11/results.jsonl','tmp/X2_no_learned_2026-08-11/results.jsonl'):
    for l in open(path):
        r=json.loads(l)
        if 'error' in r: continue
        ep=os.path.basename(r['episode']); reg[ep]=r['regime']
        M[(r['cell'],ep)][r['seed']]=r
seeds=sorted({s for d in M.values() for s in d if True})
S6=sorted({s for k,d in M.items() if k[0] in ('AUTO','AUTO-SEED-NULL') for s in d})
print('AUTO/SEED-NULL 시드 6개:',S6)
print('V2K 시드:',sorted({s for k,d in M.items() if k[0]=='V2K' for s in d}))

def med(cell,ep,seedset,key):
    v=[M[(cell,ep)][s][key] for s in seedset if s in M.get((cell,ep),{})]
    v=[x for x in v if x is not None and x==x]
    return st.median(v) if v else None

def pooled(ep,seedset,key):
    """AUTO 와 AUTO-SEED-NULL 은 cfg 비트 동일 -> 6시드를 한 풀로 보고 임의 3개를 뽑는다."""
    v=[]
    for c in ('AUTO','AUTO-SEED-NULL'):
        for s in seedset:
            if s in M.get((c,ep),{}): v.append(M[(c,ep)][s][key])
    v=[x for x in v if x is not None and x==x]
    return st.median(v) if v else None

print()
print('C_matched = SELECT{A,V2K} vs SELECT{A,B}   (A,B = 6시드의 3/3 분할)   stat=window_resid, 결과=nRMSE')
for rg in ('sparse','crowded'):
    eps=[e for e in reg if reg[e]==rg and ('V2K',e) in M]
    res=[]
    for A in itertools.combinations(S6,3):
        Bs=tuple(s for s in S6 if s not in A)
        rel=[]
        for e in eps:
            aw=pooled(e,A,'window_resid'); bw=pooled(e,Bs,'window_resid')
            an=pooled(e,A,'nRMSE');       bn=pooled(e,Bs,'nRMSE')
            vw=med('V2K',e,(1575,1576,1577),'window_resid'); vn=med('V2K',e,(1575,1576,1577),'nRMSE')
            if None in (aw,bw,an,bn,vw,vn): continue
            armA = an if aw<=vw else vn        # SELECT{A, V2K}
            armB = an if aw<=bw else bn        # SELECT{A, B}  (널)
            rel.append((armA-armB)/armB if armB else 0.0)
        i=sum(1 for x in rel if x<0); w=sum(1 for x in rel if x>0); t=len(rel)-i-w
        nz=[x for x in rel if x]
        res.append((p2(max(i,w),i+w) if i+w else 1.0, i, w, t, A, st.median(nz)*100 if nz else 0.0))
    res.sort()
    print(f'  --- {rg}  n={len(eps)} ---')
    for p,i,w,t,A,m in res:
        tag=' <-S9/S11 이 쓴 분할' if A==(1575,1576,1577) else ''
        print(f'    A={A}  {i:2d}/{w:2d} tie{t:2d}  p={p:.5g}  med_nonzero={m:+.2f}%{"  *" if p<=0.05 else "   "}{tag}')
    print(f'    유의 분할 {sum(1 for r in res if r[0]<=0.05)}/{len(res)}')
