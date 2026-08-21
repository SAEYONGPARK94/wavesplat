#!/usr/bin/env python3
"""FAIR 개입 실험 집계기 — `V1_refute_nodisp.md §7` 의 표를 `fair.jsonl` 에서 재도출한다.

런 0회. 이 파일은 2026-08-13 3라운드 감사(R3 §M-6, "32런의 원자료가 저장되지 않았다")의
처방으로 추가됐다. 규약은 결과 이후에 정하지 않았다 — V1 이 이미 쓴 규약을 그대로 코드로 옮긴 것이다:

  - 대응쌍 상대차 `rel = (a - b) / b`  (양수 = a 가 나쁘다)
  - 집계 = `med(rel)` (V1 §7 표 제목과 동일)
  - 검정 = 양측 이항 부호검정 (`binomtest(loss, n, 0.5)`), n = 16/층
  - ⚠ 사전등록 없음 = **탐색적**

사용법:
    python3 tmp/AUDIT_recheck_2026-08-13/fair_ablation/analyze_fair.py
"""
import json
import os
import statistics as st
import sys

from scipy.stats import binomtest

HERE = os.path.dirname(os.path.abspath(__file__))
PATH = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, 'fair.jsonl')

CONTRASTS = (
    ('NODISP vs SHIP', 'nod_fin', 'ship_fin'),
    ('FAIR   vs SHIP', 'fair_fin', 'ship_fin'),
    ('NODISP vs FAIR', 'nod_fin', 'fair_fin'),
)

rows = [json.loads(line) for line in open(PATH)]
print(f'rows = {len(rows)}  ({PATH})')

for stratum in ('sparse', 'crowded'):
    R = [r for r in rows if r['strat'] == stratum]
    print(f'\n--- {stratum}  n={len(R)}')
    for name, a, b in CONTRASTS:
        rel = [(r[a] - r[b]) / r[b] for r in R]
        loss = sum(1 for x in rel if x > 0)
        p = binomtest(loss, len(rel), 0.5).pvalue
        print(f'  {name}  med(rel)={st.median(rel) * 100:+8.1f}%   '
              f'패={loss}/{len(rel)}   p={p:.3g}')
    inits = {lab: st.median([r[k] for r in R])
             for k, lab in (('ship_init', 'SHIP'),
                            ('fair_init', 'FAIR'),
                            ('nod_init', 'NODISP'))}
    print('  init_nrmse 중앙: ' + ' · '.join(f'{k} {v:.4f}' for k, v in inits.items()))
