import json,collections,statistics as st,numpy as np
B='/home/vexit/wavesplat/wavesplat/tmp/'
S16=[json.loads(l) for l in open(B+'S16_BASELINE_2026-08-14/results.jsonl')]
SH=['obs:0_5','obs:5_10','obs:10_20','obs:20_30','obs:30_50','obs:50_75','obs:75_100','obs:100_150','obs:150_inf']
d=collections.defaultdict(dict); strat={}
for r in S16:
    d[(r['arm'],r['episode'])][r['filter_seed']]=r
    strat[r['episode']]=r['stratum']
C=sorted([e for e in strat if strat[e]=='crowded'])
ARMS=['SPEC25','SPEC50','SPEC100','WHITE50','WHITER50']

# (1) is shell_frac identical across arms & seeds for the same episode?
print('=== shell_frac invariance across arm/seed (crowded) ===')
mx=0; mxs=0
for e in C:
    vals=collections.defaultdict(list)
    for a in ARMS:
        for s,r in d[(a,e)].items():
            for k in SH: vals[k].append(r['shell_frac'][k])
    for k in SH:
        v=vals[k]; mx=max(mx,max(v)-min(v))
print('  max spread of shell_frac across 15 (arm,seed) cells, worst shell/ep: %.3e'%mx)
# sum to 1?
sums=[sum(d[('SPEC25',e)][42]['shell_frac'][k] for k in SH) for e in C]
print('  sum over 9 shells: min=%.6f max=%.6f'%(min(sums),max(sums)))

def med3v(a,e,key,sh):
    v=[d[(a,e)][s][key][sh] for s in (42,43,44)]
    return float(np.median(v))

print('\n=== per-shell win counts a<ZERO (med3 on both), crowded n=64 ===')
area_ep = {k: float(np.mean([d[('SPEC25',e)][42]['shell_frac'][k] for e in C])) for k in SH}
print('  area (mean over 64 crowded, SPEC25@42): '+' '.join('%s=%.4f'%(k.split(':')[1],area_ep[k]) for k in SH))
res={}
for a in ARMS:
    wins=[]
    for k in SH:
        w=sum(1 for e in C if med3v(a,e,'shell_rmse',k) < med3v(a,e,'zero_shell',k))
        wins.append(w)
    passing=[k for k,w in zip(SH,wins) if w>=41]
    Aw=sum(area_ep[k] for k in passing)
    res[a]=(wins,Aw)
    print(f'  {a:9s} {"·".join(str(w) for w in wins):45s} A_win={Aw:.4f}')

print('\n=== VARIANT SENSITIVITY of A_win ===')
def awin(a, med3=True, seed=42, areasrc='self', norm=True, eps=C):
    wins={}
    for k in SH:
        if med3: w=sum(1 for e in eps if med3v(a,e,'shell_rmse',k)<med3v(a,e,'zero_shell',k))
        else:    w=sum(1 for e in eps if d[(a,e)][seed]['shell_rmse'][k]<d[(a,e)][seed]['zero_shell'][k])
        wins[k]=w
    src = a if areasrc=='self' else areasrc
    ar={k: float(np.mean([d[(src,e)][seed]['shell_frac'][k] for e in eps])) for k in SH}
    tot=sum(ar.values()) if norm else 1.0
    return sum(ar[k] for k in SH if wins[k]>=41)/tot, wins
for a in ARMS:
    v1,_=awin(a); v2,_=awin(a,med3=False,seed=42); v3,_=awin(a,med3=False,seed=43); v4,_=awin(a,med3=False,seed=44)
    v5,_=awin(a,norm=False); v6,_=awin(a,areasrc='WHITER50')
    ALL=sorted(strat); v7,_=awin(a,eps=ALL)
    print(f'  {a:9s} med3={v1:.4f} s42={v2:.4f} s43={v3:.4f} s44={v4:.4f} unnorm={v5:.4f} area_from_WHITER50={v6:.4f} all128ep={v7:.4f}')
