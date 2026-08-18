import json, collections, statistics as st, math, itertools
B='/home/vexit/wavesplat/wavesplat/tmp/'
S16=[json.loads(l) for l in open(B+'S16_BASELINE_2026-08-14/results.jsonl')]
S13=[json.loads(l) for l in open(B+'S13_oos_2026-08-12/results.jsonl')]
S13b=[json.loads(l) for l in open(B+'S13b_seed_robust_2026-08-13/results_s13b.jsonl')]

# ---- SHIP med3 (crowded) ----
ship=collections.defaultdict(dict)   # ep -> seed -> nrmse
for r in S13:
    if r['arm']=='SHIP': ship[r['episode']][42]=r['avg_nRMSE']
for r in S13b:
    if r['arm'].startswith('SHIP@'): ship[r['episode']][r['est_seed']]=r['avg_nRMSE']
ship_med={ep:st.median(list(d.values())) for ep,d in ship.items() if len(d)==3}
print('SHIP eps with 3 seeds:',len(ship_med))

# ---- S16 arms med3 ----
arm=collections.defaultdict(lambda: collections.defaultdict(dict))
zero=collections.defaultdict(dict)
strat={}; ncomp={}
for r in S16:
    arm[r['arm']][r['episode']][r['filter_seed']]=r['avg_nRMSE']
    zero[r['episode']][(r['arm'],r['filter_seed'])]=r['zero_nRMSE']
    strat[r['episode']]=r['stratum']; ncomp[r['episode']]=r['ncomp']
CROWD=sorted([e for e in strat if strat[e]=='crowded'])
print('crowded eps in S16:',len(CROWD),'ncomps',sorted(set(ncomp[e] for e in CROWD)))
print('crowded eps also in SHIP med3:',len([e for e in CROWD if e in ship_med]))

def med3(a,ep):
    d=arm[a][ep]; return st.median(list(d.values())) if len(d)==3 else None

ARMS=['SPEC25','SPEC50','SPEC100','WHITE50','WHITER50']
print('\n--- 0.1 crowded mean nRMSE (med3) and W_a ---')
tab={}
for a in ARMS:
    vals=[med3(a,e) for e in CROWD]
    m=sum(vals)/len(vals)
    W=sum(1 for e in CROWD if ship_med[e]<med3(a,e))
    rel=[(ship_med[e]-med3(a,e))/med3(a,e) for e in CROWD]
    tab[a]=(m,W,st.median(rel))
    print(f'{a:10s} mean={m:.4f}  W={W}/64  med(rel)={st.median(rel)*100:+.1f}%')

# envelope: per-episode best (min) of 5 arms
print('\n--- envelope (per-ep min over 5 arms) ---')
env={e:min(med3(a,e) for a in ARMS) for e in CROWD}
Wenv=sum(1 for e in CROWD if ship_med[e]<env[e])
relenv=[(ship_med[e]-env[e])/env[e] for e in CROWD]
print(f'W_env={Wenv}/64 med(rel)={st.median(relenv)*100:+.1f}% mean_env={sum(env.values())/64:.4f}')
# envelope excluding WHITE50 (stock class)
ARMS4=['SPEC25','SPEC50','SPEC100','WHITER50']
env4={e:min(med3(a,e) for a in ARMS4) for e in CROWD}
W4=sum(1 for e in CROWD if ship_med[e]<env4[e])
print(f'W_env(4 arms, no WHITE50)={W4}/64 med(rel)={st.median([(ship_med[e]-env4[e])/env4[e] for e in CROWD])*100:+.1f}%')

def binom_p(k,n):
    # one-sided P(X>=k) under p=0.5
    return sum(math.comb(n,i) for i in range(k,n+1))/2**n
print('p(env)=%.3e'%binom_p(Wenv,64), ' p(58/64)=%.3e'%binom_p(58,64))

print('\n--- 0.2 factor ratios (crowded mean med3) ---')
m=lambda a: tab[a][0]
print('loc100->50  SPEC50/SPEC100 = %.4f'%(m('SPEC50')/m('SPEC100')))
print('loc50->25   SPEC25/SPEC50  = %.4f'%(m('SPEC25')/m('SPEC50')))
print('white/spec@50 WHITER50/SPEC50 = %.4f'%(m('WHITER50')/m('SPEC50')))
print('white/spec@50 WHITE50 /SPEC50 = %.4f'%(m('WHITE50')/m('SPEC50')))

print('\n--- 0.3 MDE: shrink SPEC25 by factor f ---')
for pct in [0,10,12,14,16,18,20,22,24,26]:
    f=1-pct/100
    W=sum(1 for e in CROWD if ship_med[e]<med3('SPEC25',e)*f)
    We=sum(1 for e in CROWD if ship_med[e]<env[e]*f)
    print(f'  {pct:3d}%  W_SPEC25={W:2d}  W_env={We:2d}  mean={m("SPEC25")*f:.4f}')
