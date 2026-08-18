import json,collections,statistics as st,numpy as np,math
B='/home/vexit/wavesplat/wavesplat/tmp/'
S16=[json.loads(l) for l in open(B+'S16_BASELINE_2026-08-14/results.jsonl')]
S13=[json.loads(l) for l in open(B+'S13_oos_2026-08-12/results.jsonl')]
S13b=[json.loads(l) for l in open(B+'S13b_seed_robust_2026-08-13/results_s13b.jsonl')]
ship=collections.defaultdict(dict)
for r in S13:
    if r['arm']=='SHIP': ship[r['episode']][42]=r['avg_nRMSE']
for r in S13b:
    if r['arm'].startswith('SHIP@'): ship[r['episode']][r['est_seed']]=r['avg_nRMSE']
sm={e:st.median(list(v.values())) for e,v in ship.items()}
arm=collections.defaultdict(lambda: collections.defaultdict(dict)); zero=collections.defaultdict(dict); strat={}
for r in S16:
    arm[r['arm']][r['episode']][r['filter_seed']]=r['avg_nRMSE']
    zero[r['episode']][(r['arm'],r['filter_seed'])]=r['zero_nRMSE']; strat[r['episode']]=r['stratum']
C=sorted(e for e in strat if strat[e]=='crowded')
ARMS=['SPEC25','SPEC50','SPEC100','WHITE50','WHITER50']
M=lambda a,e: st.median(list(arm[a][e].values()))

print('=== R-1a: a<ZERO global, crowded n=64 (med3 both sides) ===')
# ZERO per episode: is zero_nRMSE identical across arms/seeds?
sp=[]
for e in C:
    vs=[zero[e][k] for k in zero[e]]
    sp.append(max(vs)-min(vs))
print('  max spread of zero_nRMSE across 15 (arm,seed) cells per ep: %.3e'%max(sp))
Z={e: st.median([zero[e][(a,s)] for a in ARMS for s in (42,43,44)]) for e in C}
for a in ARMS:
    w=sum(1 for e in C if M(a,e)<Z[e]); print('  %-9s a<ZERO %d/64  (losses %d)'%(a,w,64-w))
print('  SHIP<ZERO %d/64'%sum(1 for e in C if sm[e]<Z[e]))

print('\n=== ties: SHIP_med3 == a_med3 exact float? ===')
for a in ARMS:
    t=sum(1 for e in C if sm[e]==M(a,e)); print('  %-9s exact ties: %d'%(a,t))
mind=min(abs(sm[e]-M(a,e)) for a in ARMS for e in C)
print('  smallest |SHIP-a| over 5x64 = %.3e'%mind)
# also a vs a' ties (R-2 steps)
print('  a==a\' exact ties (WHITER50 vs WHITE50):', sum(1 for e in C if M('WHITER50',e)==M('WHITE50',e)))

print('\n=== listwise completeness ===')
have=collections.Counter()
for a in ARMS:
    for e in C:
        if len(arm[a][e])==3: have[a]+=1
print('  crowded eps with all 3 seeds per arm:',dict(have))
print('  SHIP crowded eps with 3 seeds:',sum(1 for e in C if len(ship[e])==3))
print('  => n after listwise =',len([e for e in C if all(len(arm[a][e])==3 for a in ARMS) and len(ship[e])==3]))

print('\n=== BEST tiebreak on the 4 pool arms already observed ===')
pool=['SPEC25','SPEC50','SPEC100','WHITER50']
rows=[]
for a in pool:
    W=sum(1 for e in C if sm[e]<M(a,e)); rel=st.median([(sm[e]-M(a,e))/M(a,e) for e in C])
    loc={'SPEC25':25.,'SPEC50':50.,'SPEC100':100.,'WHITER50':50.}[a]
    rows.append((W,rel,loc,a))
for r in sorted(rows): print('  W=%d med(rel)=%+.4f loc=%5.1f  %s'%r)
print('  => argmin W has a TIE:', [r[3] for r in rows if r[0]==min(x[0] for x in rows)])
