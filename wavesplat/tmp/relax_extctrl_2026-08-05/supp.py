"""보충 분석 — 사전등록 밖. **탐색적**이라고 명시하고, p 를 확증으로 쓰지 않는다."""
import json, os
from math import comb
import numpy as np
HERE='tmp/relax_extctrl_2026-08-05'
D=json.load(open(os.path.join(HERE,'relax_extctrl.json'))); EPS=D['episodes']
def st(w,n):
    k=max(w,n-w); return min(1.0, 2*sum(comb(n,i) for i in range(k,n+1))/2**n)
def m(ep,arm,k):
    v=EPS[ep][arm][k]; return None if v is None else v['median']

CROWD=[e for e in EPS if '15comp' in e or '30comp' in e]
SPARSE=[e for e in EPS if e not in CROWD]
print('탐색적 층화 — 이 프로그램의 기존 동결 분할(>=15comp) 이지만 **본 사전등록의 층은 아니다**')
print(f'  crowded(>=15comp) n={len(CROWD)}  (부호검정 바닥 {st(len(CROWD),len(CROWD)):.4f})')
print(f'  sparse (<=5comp)  n={len(SPARSE)} (바닥 {st(len(SPARSE),len(SPARSE)):.4f})')
for name,grp in [('crowded',CROWD),('sparse',SPARSE)]:
    for key in ('avg_nRMSE','init_nrmse'):
        d=[m(e,'EXT-M',key)-m(e,'OURS',key) for e in grp]
        rel=[(m(e,'EXT-M',key)-m(e,'OURS',key))/m(e,'OURS',key)*100 for e in grp]
        w=sum(1 for x in d if x<0)
        print(f'  {name:8s} {key:11s} 중앙Δ {np.median(d):+.4f} ({np.median(rel):+.1f}%)  '
              f'EXT-M승 {w}/{len(d)}  p={st(w,len(d)):.4f}')

print('\n에피소드별 상대 변화 [%] (음수 = 외부 RELAX 가 낫다)')
print(f"  {'episode':30s} {'e2e':>8s} {'init':>8s}")
for e in EPS:
    r1=(m(e,'EXT-M','avg_nRMSE')-m(e,'OURS','avg_nRMSE'))/m(e,'OURS','avg_nRMSE')*100
    r2=(m(e,'EXT-M','init_nrmse')-m(e,'OURS','init_nrmse'))/m(e,'OURS','init_nrmse')*100
    print(f'  {e:30s} {r1:+8.1f} {r2:+8.1f}')

print('\n★ 시드 안정성 (짝지음) — 에피소드별 3시드 std')
so=[EPS[e]['OURS']['avg_nRMSE']['std'] for e in EPS]
sm=[EPS[e]['EXT-M']['avg_nRMSE']['std'] for e in EPS]
sf=[EPS[e]['EXT-F']['avg_nRMSE']['std'] for e in EPS]
w=sum(1 for o, mm in zip(so, sm) if mm < o)      # EXT-M 이 더 안정한 에피소드 수
print(f'  중앙 std  OURS {np.median(so):.4f}  EXT-M {np.median(sm):.4f}  EXT-F {np.median(sf):.4f}')
print(f'  EXT-M std < OURS std : {w}/{len(so)}   부호검정(양측) p={st(w,len(so)):.4f}')
print('  per-ep std:  ' + '  '.join(f'{o:.4f}/{mm:.4f}' for o, mm in zip(so, sm)))
print(f'  비율 중앙 = {np.median([b/a if a>0 else np.nan for a,b in zip(so,sm)]):.3f}배')

print('\n⚠ 이상치 / 발산')
for e in EPS:
    for arm in ('OURS','EXT-M','EXT-F'):
        for s in EPS[e][arm]['seed_results']:
            if s.get('avg_nRMSE',0)>0.5: print(f'  발산 {e} {arm} seed={s["seed"]} nRMSE={s["avg_nRMSE"]:.3f}')
            if s.get('avg_inference_ms',0)<0: print(f'  ★음수 지연 {e} {arm} seed={s["seed"]} ms={s["avg_inference_ms"]:.2f}')
