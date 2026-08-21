import json, os, statistics as st
from collections import defaultdict
from math import comb
exec(open('/home/vexit/wavesplat/wavesplat/tmp/AUDIT_full_2026-08-12/skeptic_p4.py').read().split("print('='*100)")[0])

rec,reg=LOAD['s9']
print('='*100); print('[2] S9 AUTO-L1 vs AUTO — 9셸 전부 개선인데 nRMSE 는 악화한 에피소드가 있는가?')
for rg,n in (('sparse',64),('crowded',32)):
    allimp_but_worse=0; allimp=0; tot=0; worse=0
    for ep in rec['AUTO-L1']:
        if reg[ep]!=rg: continue
        tot+=1
        d_n = ep_med(rec['AUTO-L1'][ep],G('nRMSE'))-ep_med(rec['AUTO'][ep],G('nRMSE'))
        ds=[ep_med(rec['AUTO-L1'][ep],GS(s))-ep_med(rec['AUTO'][ep],GS(s)) for s in SH]
        ai=all(x<0 for x in ds)
        allimp+=ai
        if d_n>0: worse+=1
        if ai and d_n>0: allimp_but_worse+=1
    print(f'  {rg}: n={tot} | nRMSE 악화 {worse} | 9셸 전부개선 {allimp} | ★9셸전부개선 & nRMSE악화 = {allimp_but_worse}')

print()
print('='*100); print('[3] 셸 면적분수 shell_frac (에피소드 중앙) — "9셸 전부 개선" 의 가중치')
fr=defaultdict(list)
for ep,rs in rec['AUTO'].items():
    for r in rs:
        if r.get('shell_frac'):
            for k,v in r['shell_frac'].items(): fr[(reg[ep],k)].append(v)
for rg in ('sparse','crowded'):
    print(f'  {rg}: '+'  '.join(f'{s}={st.median(fr[(rg,s)])*100:.1f}%' for s in SH if (rg,s) in fr))

print()
print('='*100); print('[4] ★기록됐으나 한 번도 집계 안 된 꼬리 지표 — AUTO-L1 − AUTO (S9)')
for rg in ('sparse','crowded'):
    for k in ('nRMSE','nRMSE_p95','nRMSE_max','frac_frames_nRMSE_gt_0p5','n_frames_nonfinite','init_iters','final_n','decorr_mean','window_resid','all_resid'):
        d=contrast(rec,reg,'AUTO-L1','AUTO',rg,G(k))
        if not d: continue
        i,w,n,p,rel=stat(d)
        print(f'  {rg:8s} {k:26s} imp{i:3d}/wor{w:3d}/n{n:3d}  p={p:.4g}  med_rel {rel:+8.2f}%')
    print()

print('='*100); print('[5] S8 도 같은 꼬리 지표 (AUTO-L1-1e4 − AUTO)')
rec8,reg8=LOAD['s8']
for rg in ('sparse','crowded'):
    for k in ('nRMSE','nRMSE_p95','nRMSE_max','frac_frames_nRMSE_gt_0p5','n_frames_nonfinite'):
        d=contrast(rec8,reg8,'AUTO-L1-1e4','AUTO',rg,G(k))
        if not d: continue
        i,w,n,p,rel=stat(d)
        print(f'  {rg:8s} {k:26s} imp{i:3d}/wor{w:3d}/n{n:3d}  p={p:.4g}  med_rel {rel:+8.2f}%')
    print()
