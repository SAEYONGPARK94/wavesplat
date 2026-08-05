import json, itertools, math
import numpy as np
from scipy import stats

d = json.load(open('tmp/relax_extctrl_2026-08-05/relax_extctrl.json'))
eps = d['episodes']
ARMS = ['OURS','EXT-M','EXT-F']
CROWDED = [e for e in eps if any(s in e for s in ('15comp','30comp'))]
rows=[]
for ep,entry in eps.items():
    ncomp=int(ep.split('_')[1].replace('comp',''))
    for arm in ARMS:
        for s in entry[arm]['seed_results']:
            if 'error' in s: continue
            rows.append(dict(ep=ep,ncomp=ncomp,arm=arm,seed=s['seed'],
                             n=s['final_n'],y=s['avg_nRMSE'],
                             yi=s['init_nrmse'],crowd=ep in CROWDED))
print(f'rows={len(rows)}  crowded eps={CROWDED}')

def sp(x,y):
    if len(set(x))<2 or len(x)<3: return (float("nan"),float("nan"))
    r,p=stats.spearmanr(x,y); return r,p

print('\n=== A. per-arm, POOLED over crowded episodes (n=12 each) ===')
for arm in ARMS:
    R=[r for r in rows if r['crowd'] and r['arm']==arm]
    x=[r['n'] for r in R]; y=[r['y'] for r in R]
    r,p=sp(x,y); pr,pp=stats.pearsonr(x,y)
    print(f'{arm:6s} n={len(R)} spearman={r:+.3f} p={p:.3f} pearson={pr:+.3f} p={pp:.3f}'
          f'  final_n range {min(x)}-{max(x)}')

print('\n=== B. ALL arms pooled, crowded (n=36) ===')
R=[r for r in rows if r['crowd']]
x=[r['n'] for r in R]; y=[r['y'] for r in R]
r,p=sp(x,y); print(f'pooled spearman={r:+.3f} p={p:.4f}')

print('\n=== C. WITHIN-EPISODE (episode = stratifying variable, controlled) ===')
print('   per crowded episode, 9 runs (3 arms x 3 seeds):')
wz=[]
for ep in CROWDED:
    R=[r for r in rows if r['ep']==ep]
    x=[r['n'] for r in R]; y=[r['y'] for r in R]
    r,p=sp(x,y)
    print(f'  {ep:28s} n={len(R)} rho={r:+.3f} p={p:.3f}  final_n={sorted(x)}')
    if not math.isnan(r): wz.append(r)
print(f'  -> mean within-episode rho (crowded) = {np.mean(wz):+.3f}  (n_eps={len(wz)})')

print('\n   all 12 episodes:')
wz2=[]
for ep in eps:
    R=[r for r in rows if r['ep']==ep]
    x=[r['n'] for r in R]; y=[r['y'] for r in R]
    r,p=sp(x,y); wz2.append((ep,r,p))
    print(f'  {ep:28s} rho={r:+.3f} p={p:.3f}')
vals=[r for _,r,_ in wz2 if not math.isnan(r)]
print(f'  -> mean within-episode rho (all) = {np.mean(vals):+.3f}, sign {sum(v<0 for v in vals)}/{len(vals)} negative')

print('\n=== D. within-EPISODE-and-ARM (both strata controlled), 3 seeds/cell ===')
# episode+arm centred residuals, then pool
dx=[];dy=[];cells=0
for ep in eps:
    for arm in ARMS:
        R=[r for r in rows if r['ep']==ep and r['arm']==arm]
        if len(R)<3: continue
        n=np.array([r['n'] for r in R],float); y=np.array([r['y'] for r in R],float)
        if n.std()==0: continue
        cells+=1
        dx+= list(n-n.mean()); dy+= list(y-y.mean())
r,p=stats.spearmanr(dx,dy); pr,pp=stats.pearsonr(dx,dy)
print(f'cells with final_n variation = {cells}; pooled centred spearman={r:+.3f} p={p:.4f} pearson={pr:+.3f} p={pp:.4f}')
# crowded only
dx=[];dy=[];cells=0
for ep in CROWDED:
    for arm in ARMS:
        R=[r for r in rows if r['ep']==ep and r['arm']==arm]
        n=np.array([r['n'] for r in R],float); y=np.array([r['y'] for r in R],float)
        if n.std()==0: continue
        cells+=1; dx+=list(n-n.mean()); dy+=list(y-y.mean())
if len(dx)>2:
    r,p=stats.spearmanr(dx,dy)
    print(f'crowded-only: cells={cells} pooled centred spearman={r:+.3f} p={p:.4f}')

print('\n=== E. does the ARM effect survive conditioning on final_n? (crowded) ===')
for ep in CROWDED:
    entry=eps[ep]
    line=f'  {ep:28s}'
    for arm in ARMS:
        a=entry[arm]
        line+=f' | {arm} n={a["final_n"]["median"]:.0f} nRMSE={a["avg_nRMSE"]["median"]:.3f}'
    print(line)
print('\n  matched-final_n pairs across arms (|dn|<=1) within crowded episodes:')
cnt=0; wins=0; deltas=[]
for ep in CROWDED:
    O=[r for r in rows if r['ep']==ep and r['arm']=='OURS']
    E=[r for r in rows if r['ep']==ep and r['arm'] in ('EXT-M','EXT-F')]
    for a,b in itertools.product(O,E):
        if abs(a['n']-b['n'])<=1:
            cnt+=1; deltas.append(a['y']-b['y']); wins+= (a['y']<b['y'])
            print(f'    {ep:26s} OURS n={a["n"]} {a["y"]:.3f} (s{a["seed"]})  vs {b["arm"]} n={b["n"]} {b["y"]:.3f} (s{b["seed"]})  d={a["y"]-b["y"]:+.3f}')
if cnt:
    print(f'    -> {cnt} matched pairs, OURS better in {wins}, median delta={np.median(deltas):+.4f}')

print('\n=== F. per-seed detail, crowded ===')
for ep in CROWDED:
    for arm in ARMS:
        R=[r for r in rows if r['ep']==ep and r['arm']==arm]
        print(f'  {ep:28s} {arm:6s} ' + '  '.join(f's{r["seed"]}: n={r["n"]:2d} nRMSE={r["y"]:.3f} init={r["yi"]:.3f}' for r in R))

print('\n=== G. init_nrmse (post-init, pre-tracking) vs final_n, crowded ===')
for arm in ARMS:
    R=[r for r in rows if r['crowd'] and r['arm']==arm]
    r_,p_=sp([r['n'] for r in R],[r['yi'] for r in R])
    print(f'  {arm:6s} spearman(final_n, init_nrmse)={r_:+.3f} p={p_:.3f}')
