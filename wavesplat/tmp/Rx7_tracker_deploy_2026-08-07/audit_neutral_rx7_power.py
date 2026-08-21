"""NEUTRAL — power/MDE + dependence of the two PRIMARY-B halves + robustness."""
import json,math,os,statistics as st,collections,itertools
import numpy as np
H=os.path.dirname(os.path.abspath(__file__))
NEED={16:13,15:12,14:12,13:11,12:10}
CELLS=['AUTO','AUTO-DUP','AUTO-LR10','AUTO-LR100','AUTO-LR1000','AUTO-NOTRACK']
CAND=['AUTO-LR10','AUTO-LR100','AUTO-LR1000']
def sp(k,n):
    lo=min(k,n-k); return min(1.0,2*sum(math.comb(n,i) for i in range(lo+1))/2**n)
rows=[json.loads(l) for l in open(H+'/results.jsonl') if l.strip()]
ok=[r for r in rows if 'error' not in r]
NULL={n['episode']:n['zero_null_tracking_nRMSE_mean'] for n in json.load(open(H+'/nulls_rx7.json'))}
IDX=collections.defaultdict(list)
for r in ok: IDX[(r['cell'],r['episode'])].append(r)
eps=sorted({r['episode'] for r in ok}); seeds=sorted({r['seed'] for r in ok})
def med(c,e,k='nRMSE'):
    v=[x[k] for x in IDX[(c,e)]]; return st.median(v)
kept=[e for e in eps if med('AUTO',e)<NULL[e]]; n=len(kept)
D=lambda a,b,k='nRMSE',blk=None:[med(a,e,k)-med(b,e,k) for e in (blk or kept)]

print('### POWER: exact sign test, rejection region = wins >= threshold')
for nn,thr in [(13,11),(16,13)]:
    print(f'  n={nn} thr={thr}/{nn} (two-sided p={sp(thr,nn):.4f}):')
    for pw in [0.6,0.7,0.8,0.9,0.95,0.99]:
        pow_=sum(math.comb(nn,i)*pw**i*(1-pw)**(nn-i) for i in range(thr,nn+1))
        print(f'     P(win)={pw:.2f} -> power={pow_:.3f}')
    # p_win needed for 80% power
    lo,hi=0.5,1.0
    for _ in range(60):
        mid=(lo+hi)/2
        pw_=sum(math.comb(nn,i)*mid**i*(1-mid)**(nn-i) for i in range(thr,nn+1))
        lo,hi=(mid,hi) if pw_<0.8 else (lo,mid)
    print(f'     -> P(win) needed for 80% power = {hi:.4f}')
    globals()[f'pwin{nn}']=hi

print('\n### MDE in nRMSE units, per comparison (nonparametric: shift needed so that')
print('###  P(paired Delta < 0) = P_win, using that comparison\'s own paired dispersion)')
from statistics import NormalDist
margin=st.median(D('AUTO','AUTO-NOTRACK'))
print(f'  incumbent tracker margin (AUTO - floor, nRMSE) = {margin:+.5f}')
for c in CAND+['AUTO']:
    d=D(c,'AUTO-NOTRACK'); s=st.pstdev(d); obs=st.median(d)
    z=NormalDist().inv_cdf(pwin13)
    m=z*s
    print(f'  {c:13s} paired sd={s:.5f}  observed |medΔ|={abs(obs):.5f}  '
          f'MDE(80%)={m:.5f}  MDE/margin={m/margin*100:6.2f}%  obs/MDE={abs(obs)/m:5.2f}x')
for k in ['avg_RMSE_observed','avg_RMSE_unobserved']:
    d=D('AUTO','AUTO-NOTRACK',k); s=st.pstdev(d)
    m=NormalDist().inv_cdf(pwin13)*s
    print(f'  AUTO vs floor [{k:20s}] (METRES) sd={s:.5f} obs|medΔ|={abs(st.median(d)):.5f} '
          f'MDE={m:.5f}  obs/MDE={abs(st.median(d))/m:5.2f}x')

print('\n### §3.8 dependence of the two PRIMARY-B halves (are they 1 finding or 2?)')
a=D('AUTO','AUTO-NOTRACK','avg_RMSE_observed'); b=D('AUTO','AUTO-NOTRACK','avg_RMSE_unobserved')
r=np.corrcoef(a,b)[0,1]
print(f'  Pearson r(obsΔ, unobsΔ) over the {n} kept episodes = {r:+.3f}')
print(f'  Spearman-ish (rank) r = {np.corrcoef(np.argsort(np.argsort(a)),np.argsort(np.argsort(b)))[0,1]:+.3f}')
print(f'  sign pairs observed: {collections.Counter((x<0,y<0) for x,y in zip(a,b))}')
print('  => same 13 episodes, same 78 runs, complementary cells of one grid.')
print(f'  conservative joint bound: p_joint <= min(marginals) = {sp(13,13):.4f}; '
      f'the product 0.0002^2 is NOT valid.')

print('\n### ROBUSTNESS of PRIMARY-A (LR1000 vs floor), leave-one-episode-out')
d=D('AUTO-LR1000','AUTO-NOTRACK')
print(f'  full n=13: med={st.median(d):+.6f} wins={sum(1 for x in d if x<0)}/13 p={sp(sum(1 for x in d if x<0),13):.4f}')
worst=0
for i in range(13):
    dd=d[:i]+d[i+1:]; w=sum(1 for x in dd if x<0); p=sp(w,12)
    ok_=(w>=NEED[12] and st.median(dd)<0 and p<=0.025)
    if not ok_: worst+=1
    print(f'    drop {kept[i][8:-17]:24s} -> {w}/12 med={st.median(dd):+.6f} p={p:.4f} '
          f'holm-step2(α=.025) {"PASS" if ok_ else "FAIL"}')
print(f'  LOO: verdict flips to fail in {worst}/13 leave-one-outs')

print('\n### per-episode LR1000 - floor deltas (nRMSE) and per-seed sign agreement')
for e in kept:
    per=[]
    for s in seeds:
        x=[r for r in IDX[('AUTO-LR1000',e)] if r['seed']==s][0]['nRMSE']
        y=[r for r in IDX[('AUTO-NOTRACK',e)] if r['seed']==s][0]['nRMSE']
        per.append(x-y)
    print(f'  {e[8:-17]:24s} medΔ={med("AUTO-LR1000",e)-med("AUTO-NOTRACK",e):+.6f}  '
          f'per-seed Δ={["%+.6f"%v for v in per]}  seeds better={sum(1 for v in per if v<0)}/3')

print('\n### PRIMARY-A on nRMSE with the OTHER two seed-aggregations (protocol sensitivity)')
for agg,fn in [('median (PREREG)',st.median),('mean',st.mean),('min',min),('max',max)]:
    def m2(c,e): return fn([r['nRMSE'] for r in IDX[(c,e)]])
    dd=[m2('AUTO-LR1000',e)-m2('AUTO-NOTRACK',e) for e in kept]
    w=sum(1 for x in dd if x<0)
    print(f'  {agg:16s} med={st.median(dd):+.6f} wins={w}/13 p={sp(w,13):.4f}')

print('\n### per-run (not per-episode) sign count, kept block, 39 paired runs (descriptive only)')
c=0;t=0
for e in kept:
    for s in seeds:
        x=[r for r in IDX[('AUTO-LR1000',e)] if r['seed']==s][0]['nRMSE']
        y=[r for r in IDX[('AUTO-NOTRACK',e)] if r['seed']==s][0]['nRMSE']
        t+=1; c+= (x<y)
print(f'  LR1000 better in {c}/{t} individual runs (runs are NOT independent -> no p)')
