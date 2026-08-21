import json, os, statistics as st
from collections import defaultdict
from math import comb
exec(open('/home/vexit/wavesplat/wavesplat/tmp/AUDIT_full_2026-08-12/skeptic_p4.py').read().split("print('='*100)")[0])
HARM={'crowded':21,'sparse':39}   # MDE_prespec_s9.json (n=32/64); S8 PREREG 도 21/32
print('사전등록 SUPPORT-TRADEOFF 규칙: 근거리 3셸 중 하나라도 [악화 >= harm_k] 이면 발화')
print('harm_k: crowded 21/32, sparse 39/64\n')
ARMS={'s7':[('AUTO-L1-1e4','AUTO','L1'),('AUTO-NT-L1-1e4','AUTO-NT','L1'),
            ('AUTO-NT-L1-1e4','AUTO','TRK'),('AUTO-NT','AUTO','TRK')],
      's8':[('AUTO-L1-1e4','AUTO','L1'),('AUTO-L1-3e4','AUTO','L1'),('AUTO-L1-0','AUTO','L1'),
            ('AUTO-NT','AUTO','TRK'),('AUTO-TRK-QTR','AUTO','TRK')],
      's9':[('AUTO-L1','AUTO','L1'),('AUTO-QTR-L1','AUTO-L1','TRK'),('AUTO-QTR','AUTO','TRK'),
            ('AUTO-NT','AUTO','TRK'),('AUTO-SEED-NULL','AUTO','NULL')]}
tally=defaultdict(lambda:[0,0])
for b in ('s7','s8','s9'):
    rec,reg=LOAD[b]
    for t,base,kind in ARMS[b]:
        if t not in rec or base not in rec: continue
        for rg in ('crowded','sparse'):
            g=stat(contrast(rec,reg,t,base,rg,G('nRMSE')))
            if g[2]==0: continue
            fire=False; det=[]
            for s in NEAR:
                i,w,n,p,rel=stat(contrast(rec,reg,t,base,rg,GS(s)))
                det.append(f'{s} 악화{w}/{n}')
                if w>=HARM[rg]: fire=True
            tally[kind][0 if fire else 1]+=1
            print(f'  {b} {kind:4s} {t:16s}-{base:10s} {rg:8s} nRMSE {g[0]}/{g[2]} ({g[4]:+6.1f}%)  '
                  f'{" ".join(det):46s} -> {"★FIRE" if fire else "pass"}')
print()
for k,(f,p) in tally.items(): print(f'  {k:5s} 팔: FIRE {f} / pass {p}')
