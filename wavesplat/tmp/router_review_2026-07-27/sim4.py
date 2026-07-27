import json
import numpy as np
from scipy.stats import wilcoxon
exec(open('/home/vexit/wavesplat/wavesplat/tmp/router_review_2026-07-27/sim_router.py').read().split('rows=[]')[0])
def lanes_of(net,th,g):
    L,N=NETS[net]; out={}
    for ep in eps:
        lane=[]
        for (E,D) in LANES:
            vv=v2.get((ep,E)); lv=L.get((ep,E,D)); ns=N.get((ep,D))
            u=(ns is not None and ns>=th)
            if u and g is not None and bfr.get(ep,1.0)<g: u=False
            lane.append(lv if u else vv)
        out[ep]=lane
    return out
def epmed(LL): return {ep: float(np.median(LL[ep])) for ep in eps}
V2L={ep:[v2.get((ep,E)) for (E,_) in LANES] for ep in eps}

print('analytic sparse worst lanes:')
for ep in eps:
    if comp(ep) in (1,2):
        m=max(V2L[ep])
        if m>0.05: print(f'  {ep}  lanes={["%.4f"%x for x in V2L[ep]]}')
print('\nv1/v3 learned sparse worst lanes >0.04:')
for net in ('v1','v3'):
    LL=lanes_of(net,0,None)
    for ep in eps:
        if comp(ep) in (1,2) and max(LL[ep])>0.04:
            print(f'  {net} {ep} lanes={["%.4f"%x for x in LL[ep]]}')

print('\n=== head-to-head paired (episode medians, n=40) ===')
def cmp(a,b,na,nb):
    A=np.array([a[e] for e in eps]); B=np.array([b[e] for e in eps]); d=A-B
    nz=d[np.abs(d)>1e-12]
    p=wilcoxon(nz).pvalue if len(nz)>2 else float('nan')
    rng=np.random.default_rng(11); bs=[np.mean(rng.choice(d,40,replace=True)) for _ in range(20000)]
    print(f'  {na:32s} vs {nb:32s} dmean={d.mean():+.5f} CI[{np.percentile(bs,2.5):+.4f},{np.percentile(bs,97.5):+.4f}] '
          f'wins {int((d<-1e-12).sum())}/{int((np.abs(d)>1e-12).sum())} discordant  p={p:.3f}')
A=epmed(lanes_of('v1',8,0.05)); B=epmed(lanes_of('v3',8,0.05))
cmp(A,B,'v1+router8','v3+router8')
cmp(epmed(lanes_of('v1',0,None)), epmed(lanes_of('v3',0,None)),'v1 everywhere','v3 everywhere')
cmp(epmed(lanes_of('v1',8,0.05)), epmed(lanes_of('v1',0,None)),'v1+router8','v1 everywhere')
cmp(epmed(lanes_of('v3',8,0.05)), epmed(lanes_of('v3',0,None)),'v3+router8','v3 everywhere')
cmp(epmed(lanes_of('v3',8,0.05)), epmed(V2L),'v3+router8','analytic everywhere')
cmp(epmed(lanes_of('v1',8,0.05)), epmed(V2L),'v1+router8','analytic everywhere')
cmp(epmed(lanes_of('v3',8,0.05)), epmed(lanes_of('old',8,0.05)),'v3+router8','old+router8 (SQ)')
cmp(epmed(lanes_of('v1',8,0.05)), epmed(lanes_of('old',8,0.05)),'v1+router8','old+router8 (SQ)')

print('\n=== crowded slice only (n=16), v1 vs v3 learned ===')
cr=[e for e in eps if comp(e) in (15,30)]
A=epmed(lanes_of('v1',0,None)); B=epmed(lanes_of('v3',0,None))
d=np.array([A[e]-B[e] for e in cr]); print(f'  dmean={d.mean():+.5f} v1 better on {int((d<0).sum())}/16 p={wilcoxon(d).pvalue:.3f}')
