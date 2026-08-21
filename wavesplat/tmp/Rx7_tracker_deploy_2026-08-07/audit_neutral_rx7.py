"""NEUTRAL audit of Rx-7. Written from PREREG.md text only; reads results.jsonl +
nulls_rx7.json.  Does NOT import analyze_rx7.py."""
import json, math, os, statistics as st, itertools, collections
H = os.path.dirname(os.path.abspath(__file__))
NEED = {16:13, 15:12, 14:12, 13:11, 12:10}
CELLS = ['AUTO','AUTO-DUP','AUTO-LR10','AUTO-LR100','AUTO-LR1000','AUTO-NOTRACK']
CAND  = ['AUTO-LR10','AUTO-LR100','AUTO-LR1000']

def sp(k,n):
    if n==0: return 1.0
    lo=min(k,n-k)
    return min(1.0, 2*sum(math.comb(n,i) for i in range(lo+1))/2**n)
def sp1(k,n):  # one-sided (k = #wins), P[X>=k]
    if n==0: return 1.0
    return sum(math.comb(n,i) for i in range(k,n+1))/2**n

rows=[json.loads(l) for l in open(H+'/results.jsonl') if l.strip()]
ok=[r for r in rows if 'error' not in r]
NULL={n['episode']:n['zero_null_tracking_nRMSE_mean'] for n in json.load(open(H+'/nulls_rx7.json'))}
IDX=collections.defaultdict(list)
for r in ok: IDX[(r['cell'],r['episode'])].append(r)
eps=sorted({r['episode'] for r in ok}); seeds=sorted({r['seed'] for r in ok})
def med(c,e,k='nRMSE'):
    v=[x[k] for x in IDX[(c,e)] if x.get(k) is not None and x[k]==x[k]]
    return st.median(v) if v else None

print(f'rows={len(rows)} ok={len(ok)} eps={len(eps)} seeds={seeds} cells={sorted({r["cell"] for r in ok})}')
print(f'design check: 6x{len(eps)}x{len(seeds)}={6*len(eps)*len(seeds)}  actual={len(ok)}')
cnt=collections.Counter((r['cell'],r['episode']) for r in ok)
print(f'  cells-per-(cell,ep) all==3: {set(cnt.values())=={3}}   n(cell,ep) pairs={len(cnt)}')

print('\n### §5.1 gate16 (re-derived from nulls file cannot cover draws; see separate rerun)')

print('\n### §5.2 worker leak: AUTO-DUP vs AUTO, FULL record (not a 9-field subset)')
SKIP={'job_id','cell','wall_s','elapsed_ms'}
nc=nb=0; diffk=collections.Counter()
for e in eps:
    for s in seeds:
        a=[x for x in IDX[('AUTO',e)] if x['seed']==s][0]
        d=[x for x in IDX[('AUTO-DUP',e)] if x['seed']==s][0]
        for k in sorted(set(a)|set(d)):
            if k in SKIP: continue
            nc+=1
            if json.dumps(a.get(k),sort_keys=True)!=json.dumps(d.get(k),sort_keys=True):
                nb+=1; diffk[k]+=1
print(f'  {nc} field comparisons over {len(eps)*len(seeds)} (ep,seed), mismatches={nb} {dict(diffk)}')
wd=[abs(a['wall_s']-d['wall_s']) for e in eps for s in seeds
    for a in [ [x for x in IDX[('AUTO',e)] if x['seed']==s][0] ]
    for d in [ [x for x in IDX[('AUTO-DUP',e)] if x['seed']==s][0] ]]
print(f'  wall_s differs (proves separate execution): max|d|={max(wd):.2f}s  min={min(wd):.2f}s')

print('\n### §5.3 Phase-1 invariance across all 6 cells')
bad=0; ex=[]
for e in eps:
    for s in seeds:
        v={}
        for c in CELLS:
            r=[x for x in IDX[(c,e)] if x['seed']==s][0]
            v[c]=(r['init_nrmse'],r['init_rmse'],r['auto_selected'],r['N'],
                  r['auto_n_selected'],r['ensemble_k'])
        if len(set(v.values()))>1: bad+=1; ex.append((e,s,v))
print(f'  (ep,seed) cells with non-identical Phase-1 tuple: {bad}/{len(eps)*len(seeds)}')
print(f'  auto_selected values across all runs: {collections.Counter(r["auto_selected"] for r in ok)}')

print('\n### §5.4 one knob per cell, re-read from the RECORDED cfg (not the pre-run file)')
base=None
for c in CELLS:
    cf=[r['cfg'] for r in ok if r['cell']==c]
    same=all(json.dumps(x,sort_keys=True)==json.dumps(cf[0],sort_keys=True) for x in cf)
    if c=='AUTO': base=cf[0]
    d={k:(base.get(k,'-'),cf[0].get(k,'-')) for k in set(base)|set(cf[0])
       if json.dumps(base.get(k),sort_keys=True)!=json.dumps(cf[0].get(k),sort_keys=True)}
    print(f'  {c:14s} cfg identical across its {len(cf)} runs={same}  diff vs AUTO={d}')

print('\n### §5.6 complement: observed/unobserved are per-frame means; check frame counts')
# indirect: if some frames were dropped from one list only, the two means average
# different frame sets.  Check that both are finite everywhere.
nan_o=sum(1 for r in ok if r['avg_RMSE_observed']!=r['avg_RMSE_observed'])
nan_u=sum(1 for r in ok if r['avg_RMSE_unobserved']!=r['avg_RMSE_unobserved'])
print(f'  NaN avg_RMSE_observed={nan_o}  avg_RMSE_unobserved={nan_u}')
print(f'  UNITS: nRMSE = RMSE/Hs (dimensionless);  avg_RMSE_* = mean of per-frame '
      f'rmse() in METRES (metrics.py has no Hs division in rmse_observed/unobserved)')
for e in eps[:2]:
    r=[x for x in IDX[('AUTO-NOTRACK',e)]][0]
    print(f'   e.g. {e[8:-17]:24s} nRMSE={r["nRMSE"]:.4f}  obs={r["avg_RMSE_observed"]:.4f}  '
          f'unobs={r["avg_RMSE_unobserved"]:.4f}')

print('\n### §3.4 null rule')
kept=[e for e in eps if med('AUTO',e)<NULL[e]]; excl=[e for e in eps if e not in kept]
n=len(kept)
print(f'  n_kept={n}  need={NEED[n]}   excluded={len(excl)}')
for e in excl:
    print(f'    EXCL {e[8:-17]:24s} AUTO_med={med("AUTO",e):.4f} null={NULL[e]:.4f} '
          f'delta=+{med("AUTO",e)-NULL[e]:.4f} | NOTRACK={med("AUTO-NOTRACK",e):.4f} '
          f'(vs null {med("AUTO-NOTRACK",e)-NULL[e]:+.4f})')
print('  --- kept episodes AUTO margin below null ---')
for e in kept:
    print(f'    KEEP {e[8:-17]:24s} AUTO={med("AUTO",e):.4f} null={NULL[e]:.4f} '
          f'{med("AUTO",e)-NULL[e]:+.4f}')
print('  null-pass count per cell, on ALL 16:')
for c in CELLS:
    print(f'    {c:14s} pass {sum(1 for e in eps if med(c,e)<NULL[e])}/16  '
          f'medOfMed(kept)={st.median([med(c,e) for e in kept]):.4f}  '
          f'medOfMed(all16)={st.median([med(c,e) for e in eps]):.4f}')
print('  m>1 (worse than flat sea) run counts per cell:')
for c in CELLS:
    print(f'    {c:14s} m_mean>1 in {sum(1 for r in ok if r["cell"]==c and r["m_mean"]>1)}/48 runs')

def paired(a,b,key='nRMSE',blk=None,good_neg=True):
    blk=blk or kept
    raw=[med(a,e,key)-med(b,e,key) for e in blk]
    ties=sum(1 for x in raw if x==0); d=[x for x in raw if x!=0]
    if not d: return None
    w=sum(1 for x in d if (x<0)==good_neg)
    return st.median(d),w,len(d),sp(w,len(d)),ties,d

print('\n### PRIMARY-A  (candidate - AUTO-NOTRACK), nRMSE, kept block n=%d'%n)
res={}
for c in CAND:
    m,w,nn,p,t,d=paired(c,'AUTO-NOTRACK')
    res[c]=(m,w,nn,p)
    print(f'  {c:13s} medΔ={m:+.6f}  wins={w}/{nn} (need {NEED[nn]})  p2={p:.4f}  '
          f'p1={sp1(w,nn):.4f}  ties={t}')
print('\n  Holm as executed (two-sided p, ascending):')
order=sorted(res,key=lambda c:res[c][3]); kr=False
for i,c in enumerate(order):
    a=0.05/(3-i); m,w,nn,p=res[c]
    ok_=p<=a; passed=ok_ and m<0 and w>=NEED[nn]
    kr|=passed
    print(f'    step{i+1} {c:13s} p={p:.4f} <= α={a:.4f}? {ok_}   direction m<0? {m<0}   '
          f'wins>=thr? {w>=NEED[nn]}  -> {"PASS" if passed else "fail"}')
    if not ok_: print('    (step-down stops here)'); break
print(f'  => as executed: {"KEEP-REDUCED" if kr else "SHIP-OFF"}')
print('\n  ALTERNATIVE Holm orderings (sensitivity of the verdict):')
m,w,nn,p=res['AUTO-LR1000']
print(f'    (a) if LR10 (wrong-direction) is NOT allowed to consume a Holm step,')
print(f'        LR1000 is the first *directionally admissible* hypothesis -> α=0.05/3=0.0167')
print(f'        p2={p:.4f} > 0.0167  -> FAIL   |  p1={sp1(w,nn):.4f} <= 0.0167 -> PASS')
print(f'    (b) Bonferroni α/3 two-sided: {p:.4f} vs 0.0167 -> {"PASS" if p<=0.05/3 else "FAIL"}')
print(f'    (c) uncorrected two-sided: {p:.4f} vs 0.05 -> {"PASS" if p<=0.05 else "FAIL"}')

print('\n  PRIMARY-A on the UN-excluded n=16 block (§3.4 병기 의무):')
for c in CAND:
    m,w,nn,p,t,d=paired(c,'AUTO-NOTRACK',blk=eps)
    print(f'    {c:13s} medΔ={m:+.6f}  wins={w}/{nn} (need {NEED.get(nn,"?")})  p2={p:.4f}')

print('\n### dose-response ladder vs floor (nRMSE), kept block')
for c in ['AUTO','AUTO-DUP']+CAND:
    r=paired(c,'AUTO-NOTRACK')
    if r is None: print(f'  {c:13s} all ties (n=0)'); continue
    m,w,nn,p,t,d=r
    print(f'  {c:13s} medΔ={m:+.6f} wins={w}/{nn} p={p:.4f} ties={t}  '
          f'| % of AUTO-floor margin recovered = {100*(1-m/0.0):.1f}' if False else
          f'  {c:13s} medΔ={m:+.6f} wins={w}/{nn} p={p:.4f} ties={t}')
mA=paired('AUTO','AUTO-NOTRACK')[0]
print(f'  tracker margin (AUTO - floor) = {mA:+.6f}')
for c in CAND:
    m=paired(c,'AUTO-NOTRACK')[0]
    print(f'    {c:13s} |medΔ| / margin = {abs(m)/mA*100:6.2f}%   '
          f'gap recovered vs AUTO = {100*(1-m/mA):6.1f}%')

print('\n### PRIMARY-B  (AUTO - AUTO-NOTRACK) on the two halves')
for key in ['avg_RMSE_observed','avg_RMSE_unobserved']:
    m,w,nn,p,t,d=paired('AUTO','AUTO-NOTRACK',key)
    print(f'  {key:22s} medΔ={m:+.6f} improve {w}/{nn} p={p:.4f}')
print('  absolute levels (METRES, kept block, median of per-ep medians):')
for c in CELLS:
    print(f'    {c:14s} obs={st.median([med(c,e,"avg_RMSE_observed") for e in kept]):.4f}  '
          f'unobs={st.median([med(c,e,"avg_RMSE_unobserved") for e in kept]):.4f}')
print('  ladder on each half (cell - floor):')
for c in ['AUTO']+CAND:
    a=paired(c,'AUTO-NOTRACK','avg_RMSE_observed'); b=paired(c,'AUTO-NOTRACK','avg_RMSE_unobserved')
    print(f'    {c:13s} obsΔ={a[0]:+.5f} {a[1]:2d}/{a[2]} p={a[3]:.4f} | '
          f'unobsΔ={b[0]:+.5f} {b[1]:2d}/{b[2]} p={b[3]:.4f}')

print('\n### POWER / MDE (paired sign test, exact)')
def mde(nn):
    for k in range(nn,0,-1):
        if sp(k,nn)<=0.05: continue
        return k+1
for nn in [13,16]:
    need=None
    for k in range(nn//2,nn+1):
        if sp(k,nn)<=0.05: need=k; break
    print(f'  n={nn}: smallest win-count with two-sided p<=0.05 is {need} '
          f'({need}/{nn}); floor p (all wins) = {sp(nn,nn):.2e}')
print('  observed paired |Δ| dispersion (kept block):')
for c,key in [('AUTO-LR1000','nRMSE'),('AUTO','nRMSE'),('AUTO','avg_RMSE_observed')]:
    d=[med(c,e,key)-med('AUTO-NOTRACK',e,key) for e in kept]
    print(f'    {c:13s} {key:20s} med={st.median(d):+.5f} IQR=[{sorted(d)[3]:+.5f},'
          f'{sorted(d)[-4]:+.5f}] std={st.pstdev(d):.5f} min={min(d):+.5f} max={max(d):+.5f}')
print('  seed dispersion within (cell,episode) — the noise the median-of-3 fights:')
sds=[st.pstdev([x['nRMSE'] for x in IDX[(c,e)]]) for c in CELLS for e in eps]
print(f'    median per-(cell,ep) seed std of nRMSE = {st.median(sds):.5f}  max={max(sds):.5f}')
