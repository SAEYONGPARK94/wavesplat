import json, re
import numpy as np
exec(open('/home/vexit/wavesplat/wavesplat/tmp/router_review_2026-07-27/sim_router.py').read().split('rows=[]')[0])

def lanes_of(net, thresh, guard):
    L,N=NETS[net]; out={}
    for ep in eps:
        lane=[]
        for (E,D) in LANES:
            vv=v2.get((ep,E)); lv=L.get((ep,E,D)); ns=N.get((ep,D))
            u=(ns is not None and ns>=thresh)
            if u and guard is not None and bfr.get(ep,1.0)<guard: u=False
            lane.append(lv if u else vv)
        out[ep]=lane
    return out
V2L={ep:[v2.get((ep,E)) for (E,_) in LANES] for ep in eps}

OPTS={
 '1 v3 + shipped router(8,guard)': lanes_of('v3',8,0.05),
 '2 v3 learned-everywhere':        lanes_of('v3',0,None),
 '3 v1 learned-everywhere':        lanes_of('v1',0,None),
 '4 v1 + shipped router(8,guard)': lanes_of('v1',8,0.05),
 '5 old + shipped router (SQ)':    lanes_of('old',8,0.05),
 '6b analytic everywhere':         V2L,
}
print('=== SINGLE-DRAW risk (one diffusion seed, as a real-time deployment would run) ===')
print(f"{'option':34s} {'mean':>7s} {'p90':>7s} {'p99':>7s} {'max':>7s} {'P(>0.05|sparse)':>16s} {'P(>0.30)':>9s} {'spread>0.02':>12s}")
for k,LL in OPTS.items():
    allv=np.array([x for ep in eps for x in LL[ep]])
    sp=np.array([x for ep in eps if comp(ep) in (1,2) for x in LL[ep]])
    spread=[max(LL[ep])-min(LL[ep]) for ep in eps]
    print(f"{k:34s} {allv.mean():7.4f} {np.percentile(allv,90):7.4f} {np.percentile(allv,99):7.4f} {allv.max():7.4f} "
          f"{(sp>0.05).mean():16.3f} {(allv>0.30).mean():9.3f} {sum(1 for s in spread if s>0.02):12d}/40")

print('\n=== per-regime single-draw mean / worst ===')
for k,LL in OPTS.items():
    line=f"{k:34s}"
    for rg,cs in (('sparse',(1,2)),('moderate',(5,)),('crowded',(15,30))):
        v=np.array([x for ep in eps if comp(ep) in cs for x in LL[ep]])
        line+=f" | {rg}: {v.mean():.4f}/{v.max():.4f}"
    print(line)

print('\n=== proposal wall-clock (GPU) per diffusion draw, from atoms caches ===')
import statistics as st
for n,path in (('old','/home/vexit/wavesplat/wavesplat/tmp/heldout40_2026-07-26/atoms_cache.jsonl'),
               ('v1','/home/vexit/wavesplat/wavesplat/tmp/mixed_retrain_2026-07-26/atoms_cache_mixed_v1.jsonl'),
               ('v3','/home/vexit/wavesplat/wavesplat/tmp/mixed_retrain_2026-07-26/atoms_cache_mixed_v3.jsonl')):
    s=[json.loads(l)['sec'] for l in open(path) if json.loads(l).get('sec')]
    print(f'  {n}: median {st.median(s):.2f}s  p90 {np.percentile(s,90):.2f}s  n={len(s)}')
