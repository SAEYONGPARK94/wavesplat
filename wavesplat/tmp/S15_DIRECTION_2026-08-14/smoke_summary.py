#!/usr/bin/env python3
"""스모크 실측 -> 전량 예상시간. `SMOKE.md` 의 표를 이 스크립트가 만든다. 판정과 무관."""
import json
import os
import statistics as st
import sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from run_s15 import ARMS, EST_SEEDS, SINGLE_SEED_ARMS   # noqa: E402

N_EP = {'crowded': 32, 'sparse': 32}          # 16 쌍 x 2 방향, 층별
GEN_S_PER_EP = 187.0                          # 기존 코퍼스 실측 (S1 manifest 중앙)
N_GEN = 32                                    # dir180 만 새로 만든다
N_G0 = 16                                     # G-0 은 혼잡 16 x 2 팔 x 1 시드


def load(p):
    out = []
    for line in open(p):
        line = line.strip()
        if line:
            r = json.loads(line)
            if 'error' not in r:
                out.append(r)
    return out


def main():
    rows = []
    for p in (os.path.join(HERE, 'smoke_results.jsonl'),
              os.path.join(HERE, 'g0_results.jsonl')):
        if os.path.exists(p):
            rows += load(p)
    w = defaultdict(list)
    thr = defaultdict(set)
    for r in rows:
        w[(r['arm'], r['stratum'])].append(r['wall_s'])
        thr[(r['arm'], r['stratum'])].add(r.get('n_threads'))
    print('## 팔별 실측 (스모크 2 짝 = 4 에피소드 파일)\n')
    print('| 팔 | 층 | n | 중앙 wall_s | 스레드 |')
    print('|---|---|---|---|---|')
    med = {}
    for k in sorted(w):
        med[k] = st.median(w[k])
        print(f'| `{k[0]}` | {k[1]} | {len(w[k])} | {med[k]:.1f} | '
              f'{sorted(x for x in thr[k] if x is not None)} |')

    print('\n## 전량 예상 (CPU-초 합 / 워커)\n')
    total = 0.0
    per = {}
    for arm, spec in ARMS.items():
        for stratum in spec['strata']:
            k = (arm, stratum)
            if k not in med:
                print(f'  ⚠{arm}/{stratum}: 스모크 실측 없음 — 추정 불가')
                continue
            n_seed = 1 if arm in SINGLE_SEED_ARMS else len(EST_SEEDS)
            s = med[k] * n_seed * N_EP[stratum]
            per[k] = s
            total += s
    g0 = med.get(('ORA-WPHI-NT@L3', 'crowded'), 0) * N_G0 \
        + med.get(('ORA-WPHI-0STEP@L3', 'crowded'), 0) * N_G0
    gen = GEN_S_PER_EP * N_GEN
    print(f'| 단계 | CPU-초 | 워커 | 벽시계 |')
    print(f'|---|---|---|---|')
    print(f'| 생성 dir180 x{N_GEN} | {gen:,.0f} | 1 (GPU) | **{gen/3600:.1f} h** |')
    print(f'| G-0 재현 (혼잡 16 x 2팔) | {g0:,.0f} | 6 | **{g0/6/3600:.2f} h** |')
    ws = sum(v for k, v in per.items() if ARMS[k[0]]['kind'] in ('ws', 'ship', 'zero'))
    lk = sum(v for k, v in per.items() if ARMS[k[0]]['kind'] == 'letkf')
    print(f'| 전량 wavesplat 팔 | {ws:,.0f} | 6 | **{ws/6/3600:.1f} h** |')
    print(f'| 전량 LETKF_PLUS | {lk:,.0f} | 2 | **{lk/2/3600:.1f} h** |')
    tot_wall = gen / 3600 + g0 / 6 / 3600 + ws / 6 / 3600 + lk / 2 / 3600
    print(f'| **합계** | {gen+g0+total:,.0f} | — | **{tot_wall:.1f} h** |')

    njobs = 0
    for arm, spec in ARMS.items():
        for stratum in spec['strata']:
            njobs += (1 if arm in SINGLE_SEED_ARMS else len(EST_SEEDS)) * N_EP[stratum]
    print(f'\n총 추정 런 = **{njobs}** (+ G-0 {2*N_G0}) = **{njobs + 2*N_G0}**')


if __name__ == '__main__':
    main()
