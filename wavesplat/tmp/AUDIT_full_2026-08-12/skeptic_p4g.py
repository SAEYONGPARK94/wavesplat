import json, os, statistics as st
from collections import defaultdict
from math import comb
exec(open('/home/vexit/wavesplat/wavesplat/tmp/AUDIT_full_2026-08-12/skeptic_p4.py').read().split("print('='*100)")[0])
rec,reg=LOAD['s8']
print('S8 λ 용량-반응: 배포값 1e-4 가 다른 λ 를 정말 이기는가 (짝 비교, median-of-3)')
for rg in ('crowded','sparse'):
    print(f'  --- {rg} ---')
    for t,b in (('AUTO-L1-1e4','AUTO'),('AUTO-L1-3e4','AUTO'),('AUTO-L1-0','AUTO'),
                ('AUTO-L1-1e4','AUTO-L1-3e4'),('AUTO-L1-1e4','AUTO-L1-0'),('AUTO-L1-0','AUTO-L1-3e4')):
        for k,g in (('nRMSE',G('nRMSE')),('0_5',GS('0_5')),('all_resid',G('all_resid'))):
            i,w,n,p,rel=stat(contrast(rec,reg,t,b,rg,g))
            if n==0: continue
            print(f'    {t:12s} - {b:12s} {k:10s} {i:2d}/{w:2d}  p={p:.4g}  med_rel {rel:+7.2f}%')
        print()
