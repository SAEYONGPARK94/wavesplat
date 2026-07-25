"""DESIGN-PHASE: inspect the crowded episodes a beamform-only guard would false-veto,
and search for a COMBINED signature (proposal ceiling + beamform dominance + atom merge)
that fires on the 10 sparse misroutes but ZERO crowded episodes.

Pulls arm results too, to see whether learned even beats v2 on the false-veto crowded
(context; the frozen bar is 0 flips regardless)."""
import json, re
import numpy as np
from wavesplat.optimizer.learned_init import pool_and_select

def comp(ep): return int(re.search(r'_(\d+)comp', ep).group(1))
def med(xs):
    xs=[x for x in xs if x is not None and np.isfinite(x)]
    return float(np.median(xs)) if xs else float('nan')

LANES=[(42,0),(1,1),(2,2),(3,3),(4,4)]
# results
v2={}
for l in open('tmp/rerun_2026-07-25/batchC_ws_phasefix.jsonl'):
    d=json.loads(l); v2[(d['episode'],d['seed'])]=d['nRMSE']
learned={}
for src in ('tmp/wp_integration_2026-07-25/arm_runs.jsonl','tmp/unified_2026-07-26/arm_runs_extra.jsonl'):
    for l in open(src):
        d=json.loads(l)
        if d.get('arm')=='learned' and d.get('err') is None and d.get('avg_nRMSE') is not None:
            learned[(d['episode'],d['est_seed'],d['diff_seed'])]=d['avg_nRMSE']
# atom cache raw
cache={}
for src in ('tmp/wp_integration_2026-07-25/atoms_cache.jsonl','tmp/unified_2026-07-26/atoms_cache_extra.jsonl'):
    for l in open(src):
        d=json.loads(l)
        if d.get('kx') is not None and d.get('err') is None:
            cache[(d['episode'],d['diff_seed'])]=d
bf={r['ep']:r for r in json.load(open('tmp/unified_guard_2026-07-26/explore_beamform.json'))}
sig=json.load(open('tmp/unified_guard_2026-07-26/explore_signals.json'))
by_ep={}
for r in sig: by_ep.setdefault(r['ep'],[]).append(r)

def kp_merge_count(recs_ep, radius_frac):
    """median over diff_seeds of merged count using robust kp = beamform dominant |k|...
    here we use per-record dominant-atom kp (as in explore) for the atom-only variant."""
    cnts=[]
    for r in recs_ep:
        d=cache.get((r['ep'],r['ds']))
        if d is None: continue
        sk,sy,se=pool_and_select(np.asarray(d['kx'],float),np.asarray(d['ky'],float),np.asarray(d['energy'],float))
        if len(sk)==0: cnts.append(0); continue
        kmag=np.sqrt(sk**2+sy**2); kp=kmag[np.argmax(se)]
        rad=radius_frac*kp
        # union-find single linkage
        n=len(sk); par=list(range(n))
        def f(a):
            while par[a]!=a: par[a]=par[par[a]]; a=par[a]
            return a
        for i in range(n):
            for j in range(i+1,n):
                if (sk[i]-sk[j])**2+(sy[i]-sy[j])**2<rad*rad:
                    ri,rj=f(i),f(j)
                    if ri!=rj: par[ri]=rj
        cnts.append(len({f(i) for i in range(n)}))
    return med(cnts)

# per-episode assemble
rows=[]
for ep,recs in by_ep.items():
    ns_med=med([r['n_sel'] for r in recs])
    b=bf[ep]
    v2m=med([v2.get((ep,E)) for (E,D) in LANES])
    lnm=med([learned.get((ep,E,D)) for (E,D) in LANES])
    rows.append(dict(ep=ep,comp=comp(ep),ns_med=ns_med,ns_max=max(r['n_sel'] for r in recs),
                     npk10=b['npk_10'],p2p1=b['p2_over_p1'],p3p1=b['p3_over_p1'],
                     nmerge25=kp_merge_count(recs,0.25),
                     v2=v2m,learned=lnm,learn_better=(lnm<v2m)))

routed=[r for r in rows if r['ns_med']>=8]
sp=[r for r in routed if r['comp'] in (1,2)]
cr=[r for r in routed if r['comp'] in (15,30)]

print("=== The crowded episodes a beamform-only guard (p2p1<0.1) would FALSE-VETO ===")
fv=[r for r in cr if r['p2p1']<0.1]
for r in sorted(fv,key=lambda x:x['p2p1']):
    print(f"  {r['ep']:34s} c{r['comp']} ns_med={r['ns_med']:.0f} ns_max={r['ns_max']} npk10={r['npk10']} "
          f"p2/p1={r['p2p1']:.3f} nmerge25={r['nmerge25']:.0f} | v2={r['v2']:.3f} learned={r['learned']:.3f} learn_better={r['learn_better']}")

print("\n=== The 10 sparse MISROUTES: same columns ===")
for r in sorted(sp,key=lambda x:x['ep']):
    print(f"  {r['ep']:34s} c{r['comp']} ns_med={r['ns_med']:.0f} ns_max={r['ns_max']} npk10={r['npk10']} "
          f"p2/p1={r['p2p1']:.3f} nmerge25={r['nmerge25']:.0f} | v2={r['v2']:.3f} learned={r['learned']:.3f} learn_better={r['learn_better']}")

print("\n=== COMBINED-SIGNATURE SEARCH: fire veto iff (p2p1 < A) AND (nmerge25 <= B) ===")
for A in (0.1,0.15,0.2):
    for B in (2,3,4,5):
        vs=[r for r in sp if r['p2p1']<A and r['nmerge25']<=B]
        vc=[r for r in cr if r['p2p1']<A and r['nmerge25']<=B]
        print(f"  p2p1<{A} & nmerge25<={B}: misroutes {len(vs)}/{len(sp)}, crowded-flips {len(vc)}/{len(cr)}"
              f"{'  <-- 0 crowded flips' if len(vc)==0 else ''}")

print("\n=== ALT: fire iff (npk10 <= C) AND (nmerge25 <= B) ===")
for C in (1,2,3):
    for B in (2,3,4,5):
        vs=[r for r in sp if r['npk10']<=C and r['nmerge25']<=B]
        vc=[r for r in cr if r['npk10']<=C and r['nmerge25']<=B]
        print(f"  npk10<={C} & nmerge25<={B}: misroutes {len(vs)}/{len(sp)}, crowded-flips {len(vc)}/{len(cr)}"
              f"{'  <-- 0 crowded flips' if len(vc)==0 else ''}")

# Which misroutes are missed by best clean combo? examine nmerge25 of slope cases
print("\n=== misroute nmerge25 detail (slope cases have corrupted kp -> high nmerge) ===")
for r in sorted(sp,key=lambda x:-x['nmerge25']):
    print(f"  {r['ep']:34s} nmerge25={r['nmerge25']:.0f} p2p1={r['p2p1']:.3f} npk10={r['npk10']}")
