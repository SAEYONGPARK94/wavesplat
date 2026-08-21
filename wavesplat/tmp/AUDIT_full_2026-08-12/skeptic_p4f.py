import json, os, statistics as st
from collections import defaultdict
from math import comb
exec(open('/home/vexit/wavesplat/wavesplat/tmp/AUDIT_full_2026-08-12/skeptic_p4.py').read().split("print('='*100)")[0])
rec,reg=LOAD['s9']
# learned 라우트 판정: auto_selected / auto_n_selected
r0=rec['AUTO-L1'][list(rec['AUTO-L1'])[0]][0]
print('auto_selected 예:', r0.get('auto_selected'), 'n_selected', r0.get('auto_n_selected'))
route=defaultdict(list)
for ep,rs in rec['AUTO-L1'].items():
    for r in rs: route[ep].append(r.get('auto_selected'))
maj={ep:sum(1 for x in v if x=='learned') for ep,v in route.items()}
sp_all=[e for e in maj if reg[e]=='sparse' and maj[e]==3]
sp_maj=[e for e in maj if reg[e]=='sparse' and maj[e]>=2]
print(f'희소: 3시드 전부 learned {len(sp_all)}ep · 과반 learned {len(sp_maj)}ep  (S11 은 53ep 인용)')
def sub(eps,t,b,g):
    out=[]
    for ep in eps:
        if ep not in rec[t] or ep not in rec[b]: continue
        a=ep_med(rec[t][ep],g); c=ep_med(rec[b][ep],g)
        if a is None or c is None: continue
        out.append((a-c,(a-c)/c if c else float('nan')))
    return out
print('\nS11 §5 추적축 = AUTO-NT − AUTO-L1  (희소 learned 라우트 부분집합)')
for nm,eps in (('과반규약 %dep'%len(sp_maj),sp_maj),('전부규약 %dep'%len(sp_all),sp_all),('희소 전체 64ep',[e for e in maj if reg[e]=='sparse'])):
    for k,g in (('nRMSE',G('nRMSE')),('0_5',GS('0_5')),('all_resid',G('all_resid'))):
        i,w,n,p,rel=stat(sub(eps,'AUTO-NT','AUTO-L1',g))
        print(f'  {nm:16s} {k:10s} {i:2d}/{w:2d}  p={p:.4g}  med_rel {rel:+7.2f}%')
    print()
