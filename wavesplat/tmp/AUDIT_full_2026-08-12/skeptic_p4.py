"""P4 회의론자 — 원자료 독립 재도출."""
import json, os, statistics as st
from collections import defaultdict
from math import comb

B={'s7':'tmp/S7_confirm_n32_2026-08-10/results.jsonl',
   's8':'tmp/S8_deploy_confirm_2026-08-10/results.jsonl',
   's9':'tmp/S9_lr_x_l1_2026-08-11/results.jsonl',
   'x2':'tmp/X2_no_learned_2026-08-11/results.jsonl'}
SH=('0_5','5_10','10_20','20_30','30_50','50_75','75_100','100_150','150_inf')
NEAR=('0_5','5_10','10_20')

def p2(k,n):
    if n==0: return float('nan')
    return sum(comb(n,i) for i in range(n+1) if abs(i-n/2)>=abs(k-n/2))/2.0**n

def load(p):
    rec=defaultdict(lambda: defaultdict(list)); reg={}
    for l in open(p):
        r=json.loads(l)
        if 'error' in r: continue
        rec[r['cell']][os.path.basename(r['episode'])].append(r)
        reg[os.path.basename(r['episode'])]=r['regime']
    return rec,reg

def ep_med(recs,g):
    v=[g(r) for r in recs]; v=[x for x in v if x is not None and x==x]
    return st.median(v) if v else None

def contrast(rec,reg,t,b,rg,g):
    out=[]
    if t not in rec or b not in rec: return out
    for ep in rec[t]:
        if reg.get(ep)!=rg or ep not in rec[b]: continue
        a=ep_med(rec[t][ep],g); c=ep_med(rec[b][ep],g)
        if a is None or c is None: continue
        out.append((a-c,(a-c)/c if c else float('nan')))
    return out

def stat(d):
    imp=sum(1 for x,_ in d if x<0); wor=sum(1 for x,_ in d if x>0); n=imp+wor
    return imp,wor,len(d),(p2(max(imp,wor),n) if n else float('nan')), (st.median([r for _,r in d])*100 if d else float('nan'))

G=lambda k:(lambda r:r.get(k))
GS=lambda s:(lambda r:(r.get('shell_rmse') or {}).get(s))

LOAD={k:load(v) for k,v in B.items()}

print('='*100)
print('[1] SUPPORT-TRADEOFF 게이트: 전 팔 × 전 레짐  (근거리 3셸 중 악화 있으면 발화)')
print('='*100)
ARMS={'s7':[('AUTO-NT-L1-1e4','AUTO-NT'),('AUTO-NT-L1-1e4','AUTO'),('AUTO-L1-1e4','AUTO'),('AUTO-NT','AUTO')],
      's8':[('AUTO-L1-1e4','AUTO'),('AUTO-L1-3e4','AUTO'),('AUTO-L1-0','AUTO'),('AUTO-NT','AUTO'),('AUTO-TRK-QTR','AUTO')],
      's9':[('AUTO-L1','AUTO'),('AUTO-QTR-L1','AUTO-L1'),('AUTO-QTR','AUTO'),('AUTO-NT','AUTO'),('AUTO-SEED-NULL','AUTO')]}
for b in ('s7','s8','s9'):
    rec,reg=LOAD[b]
    for t,base in ARMS[b]:
        if t not in rec or base not in rec: 
            print(f'  {b} {t}-{base}: 셀 없음'); continue
        for rg in ('crowded','sparse'):
            g=stat(contrast(rec,reg,t,base,rg,G('nRMSE')))
            if g[2]==0: continue
            row=[]
            fire=False
            for s in NEAR:
                i,w,n,p,rel=stat(contrast(rec,reg,t,base,rg,GS(s)))
                row.append(f'{s}:{i}/{n}({rel:+.1f}%)')
                if w>i: fire=True
            ar=stat(contrast(rec,reg,t,base,rg,G('all_resid')))
            trk = 'TRK-OFF/DOWN' if ('NT' in t or 'QTR' in t) and not ('NT' in base or 'QTR' in base) else '.'
            print(f'  {b} {t:16s}-{base:10s} {rg:8s} n={g[2]:2d} nRMSE {g[0]:2d}/{g[2]} ({g[4]:+6.1f}%) | '
                  f'{" ".join(row)} | all_resid {ar[0]}/{ar[2]}({ar[4]:+.1f}%) | {"FIRE" if fire else "pass"} | {trk}')
    print()
