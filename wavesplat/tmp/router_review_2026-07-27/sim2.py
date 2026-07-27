import json, re
import numpy as np
from scipy.stats import wilcoxon, spearmanr
exec(open('/home/vexit/wavesplat/wavesplat/tmp/router_review_2026-07-27/sim_router.py').read().split('rows=[]')[0])

def opt(net, thresh, guard):
    L,N=NETS[net]; sc={}; routes={}
    for ep in eps:
        lane=[]; r=[]
        for (E,D) in LANES:
            vv=v2.get((ep,E)); lv=L.get((ep,E,D)); ns=N.get((ep,D))
            u=(ns is not None and ns>=thresh)
            if u and guard is not None and bfr.get(ep,1.0)<guard: u=False
            lane.append(lv if u else vv); r.append(u)
        sc[ep]=med(lane); routes[ep]=sum(r)
    return sc, routes
def pure(net):
    L,_=NETS[net]; return {ep: med([L.get((ep,E,D)) for (E,D) in LANES]) for ep in eps}

PV2={ep: med([v2.get((ep,E)) for (E,_) in LANES]) for ep in eps}
ORCn={n: {ep: med([min(v2.get((ep,E)), NETS[n][0].get((ep,E,D))) for (E,D) in LANES]) for ep in eps} for n in NETS}

print('=== router-signal quality: spearman(median n_selected, true n_components) ===')
for n in ('old','v1','v3'):
    ns=[med([NETS[n][1].get((ep,D)) for (_,D) in LANES]) for ep in eps]
    cs=[comp(ep) for ep in eps]
    rho,p=spearmanr(ns,cs)
    # separation crowded vs not, at threshold 8, per episode-median
    tp=sum(1 for x,c in zip(ns,cs) if c>=15 and x>=8); fp=sum(1 for x,c in zip(ns,cs) if c<15 and x>=8)
    print(f'  {n}: rho={rho:+.3f} p={p:.1e} | at ns>=8: crowded caught {tp}/16, non-crowded false {fp}/24')

print('\n=== lane-level routing instability (episodes where the 5 lanes disagree), th=8 ===')
for n in ('old','v1','v3'):
    _,r=opt(n,8,None)
    split=[e for e in eps if 0<r[e]<5]
    print(f'  {n}: {len(split)}/40 episodes split; unanimous-learned {sum(1 for e in eps if r[e]==5)}, unanimous-v2 {sum(1 for e in eps if r[e]==0)}')
    print('     split eps:', [f"{e.split('episode_')[1][:22]}({r[e]}/5)" for e in split][:10])

print('\n=== per-episode: analytic vs v1 vs v3 (sparse + moderate slices) ===')
print(f"  {'episode':46s} {'v2':>8s} {'v1':>8s} {'v3':>8s}")
for ep in eps:
    if comp(ep) in (1,2,5):
        print(f"  {ep:46s} {PV2[ep]:8.4f} {pure('v1')[ep]:8.4f} {pure('v3')[ep]:8.4f}")

print('\n=== candidate options: regret vs per-net oracle, tail, paired tests ===')
CAND={
 '1 v3 + router th8+guard (status quo router)': opt('v3',8,0.05)[0],
 '2 v3 learned-everywhere':                     pure('v3'),
 '3 v1 learned-everywhere':                     pure('v1'),
 '4 v1 + router th8+guard':                     opt('v1',8,0.05)[0],
 '4b v1 + router th10 noguard (retuned)':       opt('v1',10,None)[0],
 '5 old + router th8+guard (status quo)':       opt('old',8,0.05)[0],
 '6a v3 + router th8, guard retired':           opt('v3',8,None)[0],
 '6b analytic everywhere (v2_konly)':           PV2,
}
base=CAND['5 old + router th8+guard (status quo)']
for k,sc in CAND.items():
    a=np.array([sc[e] for e in eps])
    d=np.array([sc[e]-base[e] for e in eps]); nz=d[np.abs(d)>1e-12]
    p=wilcoxon(nz).pvalue if len(nz)>2 else float('nan')
    dv=np.array([sc[e]-PV2[e] for e in eps]); nzv=dv[np.abs(dv)>1e-12]
    pv=wilcoxon(nzv).pvalue if len(nzv)>2 else float('nan')
    # bootstrap CI on mean diff vs status quo
    rng=np.random.default_rng(7); bs=[np.mean(rng.choice(d,40,replace=True)) for _ in range(20000)]
    lo,hi=np.percentile(bs,[2.5,97.5])
    print(f"  {k:44s} mean={a.mean():.5f} p95={np.percentile(a,95):.4f} max={a.max():.4f} "
          f"| vs SQ: dmean={d.mean():+.5f} CI[{lo:+.4f},{hi:+.4f}] medd={np.median(d):+.5f} p={p:.3f} "
          f"| vs v2: medd={np.median(dv):+.5f} p={pv:.3f}")

print('\n=== worst-case profile: sorted top-5 episode errors per option ===')
for k,sc in CAND.items():
    top=sorted(eps,key=lambda e:-sc[e])[:5]
    print(f"  {k:44s} " + ' '.join(f"{sc[e]:.3f}({comp(e)}c)" for e in top))

print('\n=== how much does routing sparse->analytic cost each net (the "retrain thrown away" term)? ===')
for n in ('v1','v3'):
    sp1=[e for e in eps if comp(e)==1]; sp2=[e for e in eps if comp(e)==2]; mo=[e for e in eps if comp(e)==5]
    for name,S in (('1comp(12)',sp1),('2comp(4)',sp2),('5comp(8)',mo)):
        dl=np.mean([pure(n)[e]-PV2[e] for e in S])
        print(f'  {n} {name}: learned-minus-analytic mean delta = {dl:+.5f} (neg = learned better); '
              f'weight on all-40 mean = {dl*len(S)/40:+.5f}')
