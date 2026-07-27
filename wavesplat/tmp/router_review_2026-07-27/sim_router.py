"""EXPLORATORY (not a gate): lane-level replication of the heldout-40 router protocol
for the old net vs mixed_v1 vs mixed_v3, plus a threshold/guard sweep.
Protocol copied from tmp/heldout40_2026-07-26/score_heldout40.py (LANES, THRESH, GUARD_RATIO).
Read-only: reads existing caches, writes only this directory."""
import json, re, itertools
import numpy as np
from scipy.stats import wilcoxon

H = '/home/vexit/wavesplat/wavesplat/tmp/heldout40_2026-07-26/'
M = '/home/vexit/wavesplat/wavesplat/tmp/mixed_retrain_2026-07-26/'
LANES = [(42,0),(1,1),(2,2),(3,3),(4,4)]

def comp(ep): return int(re.search(r'_(\d+)comp', ep).group(1))
def regime(c): return 'sparse' if c in (1,2) else ('moderate' if c==5 else 'crowded')
def med(xs):
    xs=[x for x in xs if x is not None and np.isfinite(x)]
    return float(np.median(xs)) if xs else float('nan')

v2={}; eps=set()
for l in open(H+'v2_arm.jsonl'):
    d=json.loads(l); v2[(d['episode'],d['seed'])]=d['nRMSE']; eps.add(d['episode'])
eps=sorted(eps)
bfr=json.load(open(H+'bf_ratio.json'))

def load_learned(path):
    out={}
    for l in open(path):
        d=json.loads(l)
        if d.get('arm')=='learned' and d.get('err') is None and d.get('avg_nRMSE') is not None:
            out[(d['episode'],d['est_seed'],d['diff_seed'])]=d['avg_nRMSE']
    return out
def load_ns(path):
    out={}
    for l in open(path):
        d=json.loads(l)
        if d.get('n_selected') is not None: out[(d['episode'],d['diff_seed'])]=d['n_selected']
    return out

NETS={
 'old': (load_learned(H+'arm_runs.jsonl'), load_ns(H+'atoms_cache.jsonl')),
 'v1' : (load_learned(M+'arm_runs_mixed_v1.jsonl'), load_ns(M+'atoms_cache_mixed_v1.jsonl')),
 'v3' : (load_learned(M+'arm_runs_mixed_v3.jsonl'), load_ns(M+'atoms_cache_mixed_v3.jsonl')),
}
for n,(L,N) in NETS.items():
    miss=[(e,E,D) for e in eps for (E,D) in LANES if (e,E,D) not in L]
    print(f'coverage {n}: learned lanes {len(L)} missing {len(miss)}, ns keys {len(N)}')

def score(net, thresh, guard):
    """returns dict ep -> per-episode median-of-lanes score, plus routing stats"""
    L,N = NETS[net]; out={}; routed_learned={}
    for ep in eps:
        lane=[]; rl=0
        for (E,D) in LANES:
            vv=v2.get((ep,E)); lv=L.get((ep,E,D)); ns=N.get((ep,D))
            use_learned = (ns is not None and ns>=thresh)
            if use_learned and guard is not None and bfr.get(ep,1.0)<guard:
                use_learned=False
            lane.append(lv if use_learned else vv); rl+=int(use_learned)
        out[ep]=med(lane); routed_learned[ep]=rl
    return out, routed_learned

def pure(net):
    L,_=NETS[net]
    return {ep: med([L.get((ep,E,D)) for (E,D) in LANES]) for ep in eps}

PV2={ep: med([v2.get((ep,E)) for (E,_) in LANES]) for ep in eps}
ORC={}
for ep in eps:
    ORC[ep]=med([min(v2.get((ep,E)), NETS['old'][0].get((ep,E,D))) for (E,D) in LANES])

def summ(name, sc):
    a=np.array([sc[ep] for ep in eps])
    reg={r: np.mean([sc[ep] for ep in eps if regime(comp(ep))==r]) for r in ('sparse','moderate','crowded')}
    d=np.array([sc[ep]-PV2[ep] for ep in eps])
    nz=d[np.abs(d)>1e-12]
    p=wilcoxon(nz).pvalue if len(nz)>0 else float('nan')
    worst=max(eps, key=lambda e: sc[e])
    # regret vs analytic: episodes where option is worse than analytic by >0.01
    regret=[e for e in eps if sc[e]-PV2[e]>0.01]
    bigregret=[e for e in eps if sc[e]-PV2[e]>0.05]
    return dict(name=name, mean=float(a.mean()), sparse=reg['sparse'], moderate=reg['moderate'],
                crowded=reg['crowded'], max=float(a.max()), worst_ep=worst,
                p90=float(np.percentile(a,90)),
                sparse_max=max(sc[e] for e in eps if regime(comp(e))=='sparse'),
                med_delta_vs_v2=float(np.median(d)), p_vs_v2=float(p),
                n_worse_than_v2=int((d>1e-12).sum()), n_regret001=len(regret), n_regret005=len(bigregret),
                regret_eps=regret[:6])

rows=[]
rows.append(summ('pure_v2 (analytic everywhere)', PV2))
rows.append(summ('oracle (old, lane)', ORC))
for net in ('old','v1','v3'):
    rows.append(summ(f'{net}: learned everywhere', pure(net)))
    for th,g,tag in [(8,0.05,'router th8+guard'),(8,None,'router th8 noguard'),(1,0.05,'guard-only')]:
        sc,rl=score(net,th,g)
        r=summ(f'{net}: {tag}', sc)
        r['lanes_learned']=int(sum(rl.values()))
        r['eps_majority_learned']=int(sum(1 for e in eps if rl[e]>=3))
        r['sparse_eps_maj_learned']=int(sum(1 for e in eps if rl[e]>=3 and regime(comp(e))=='sparse'))
        rows.append(r)

hdr=f"{'option':42s} {'mean':>7s} {'sparse':>7s} {'mod':>7s} {'crowd':>7s} {'max':>7s} {'sp_max':>7s} {'>v2':>4s} {'reg.01':>6s} {'reg.05':>6s} {'p':>7s}"
print('\n'+hdr); print('-'*len(hdr))
for r in rows:
    print(f"{r['name']:42s} {r['mean']:7.4f} {r['sparse']:7.4f} {r['moderate']:7.4f} {r['crowded']:7.4f} "
          f"{r['max']:7.4f} {r['sparse_max']:7.4f} {r['n_worse_than_v2']:4d} {r['n_regret001']:6d} {r['n_regret005']:6d} {r['p_vs_v2']:7.3f}")

print('\nworst episode per option:')
for r in rows: print(f"  {r['name']:42s} {r['worst_ep']}  ({r['max']:.4f})")

# threshold sweep
print('\nthreshold sweep (mean all-40 / sparse mean / crowded mean / lanes routed learned):')
for net in ('v1','v3'):
    print(f'-- {net}')
    for th in [1,2,3,4,5,6,8,10,12,15,20,25,31]:
        for g in (None,0.05):
            sc,rl=score(net,th,g)
            a=np.array([sc[ep] for ep in eps])
            sp=np.mean([sc[e] for e in eps if regime(comp(e))=='sparse'])
            cr=np.mean([sc[e] for e in eps if regime(comp(e))=='crowded'])
            mo=np.mean([sc[e] for e in eps if regime(comp(e))=='moderate'])
            print(f'  th={th:2d} guard={"on " if g else "off"}  mean={a.mean():.5f} sparse={sp:.5f} mod={mo:.5f} crowd={cr:.5f} max={a.max():.4f} lanes={sum(rl.values()):3d}')

# per-episode n_selected medians
print('\nper-episode median n_selected (old / v1 / v3) and bfr:')
for ep in eps:
    ns={n: med([NETS[n][1].get((ep,D)) for (_,D) in LANES]) for n in ('old','v1','v3')}
    print(f"  {ep:44s} old={ns['old']:5.1f} v1={ns['v1']:5.1f} v3={ns['v3']:5.1f} bfr={bfr.get(ep,float('nan')):.4f} comp={comp(ep):2d}")
