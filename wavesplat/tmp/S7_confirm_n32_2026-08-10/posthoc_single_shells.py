"""⚠**사후(POST-HOC) 안전점검** — 사전등록되지 않았다. 결과를 본 뒤 작성했다.

왜 필요한가: 동결 채점기(`analyze_s7.py`)는 A2 의 셸/지지집합 분해를 **두 PRIMARY 에만** 걸었다.
그런데 실제 배포에 가장 가까운 팔은 A17 이 뒤늦게 추가한 `AUTO-L1-1e4`(추적기 ON, 출하 yaml 한 줄)
이고, 그 팔은 두 레짐에서 `SINGLE-KNOB-SUPPORTED` 가 나왔다. PRIMARY-2 가 근거리 셸에서
**0/32 · +194%** 로 무너진 것을 보고도 같은 점검을 이 팔에 하지 않으면, 정확히 우리가 08-07 에
"E-B 오류"로 기록한 짓(유리한 쪽만 쪼개기)이 된다.

⛔이 파일의 어떤 결과도 **판정을 바꾸지 않는다**. `SINGLE-KNOB-SUPPORTED` 는 동결 채점기가 찍은
그대로이고, 여기서 나오는 것은 **그 라벨에 반드시 병기해야 할 제한**이다.
⛔사후 점검이므로 p 값을 확증 근거로 인용하지 않는다(다중비교 미보정·사후 선택).
"""
import json
import os
import re
from collections import defaultdict

import numpy as np
from scipy.stats import binomtest

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, 'results.jsonl')
SHELLS = ('0_5', '5_10', '10_20', '20_30', '30_50', '50_75', '75_100', '100_150', '150_inf')
OUT = []


def p(s=''):
    print(s, flush=True)
    OUT.append(s)


def regime(ep):
    return 'crowded' if ep.startswith('data_s7_crowded') else 'sparse'


def main():
    good = [json.loads(l) for l in open(RES) if l.strip()]
    good = [r for r in good if 'error' not in r]
    eps = sorted({r['episode'] for r in good})

    def per_ep(key):
        raw = defaultdict(list)
        for r in good:
            v = ((r.get('shell_rmse') or {}).get(key.split(':')[1])
                 if key.startswith('shell:')
                 else ((r.get('alpha_episode') or {}).get('0.2') if key == 'alpha'
                       else r.get(key)))
            if v is not None and np.isfinite(v):
                raw[(r['cell'], r['episode'])].append(float(v))
        return {k: float(np.median(v)) for k, v in raw.items()}

    def contrast(med, t, b, rg):
        d = np.array([med[(t, e)] - med[(b, e)] for e in eps
                      if regime(e) == rg and (t, e) in med and (b, e) in med])
        base = np.array([med[(b, e)] for e in eps
                         if regime(e) == rg and (t, e) in med and (b, e) in med])
        imp = int((d < 0).sum())
        return imp, len(d), float(np.median(d)), float(np.median(d / base)), \
            binomtest(imp, len(d), 0.5).pvalue

    p('⚠POST-HOC (사전등록 아님) — `AUTO-L1-1e4`(출하 yaml 한 줄, 추적기 ON) 의 지지집합 안/밖')
    p('  비교 대조군으로 동결 채점기가 이미 낸 PRIMARY-2 를 나란히 놓는다.')
    for rg in ('crowded', 'sparse'):
        p(f'\n--- {rg} ---')
        for t, b, lab in (('AUTO-L1-1e4', 'AUTO', 'SINGLE (추적기 ON, λ=1e-4)'),
                          ('AUTO-NT-L1-1e4', 'AUTO', 'PRIMARY-2 (추적기 off + λ)')):
            p(f'  [{lab}]')
            for key, name in (('RMSE_obs_r5', '관측 ≤5m'), ('RMSE_obs_r30', '관측 ≤30m'),
                              ('RMSE_unobs_r30', '미관측 >30m'), ('alpha', 'α_wave(0.2)')):
                m = per_ep(key)
                imp, n, md, mr, pv = contrast(m, t, b, rg)
                p(f'    {name:14s} 개선 {imp}/{n}  Δ중앙 {md:+.5f} ({mr * 100:+.1f}%)  p={pv:.5f}')
            row = []
            for sk in SHELLS:
                m = per_ep(f'shell:{sk}')
                imp, n, md, _mr, _pv = contrast(m, t, b, rg)
                row.append(f'{sk} {imp}/{n} {md:+.4f}')
            p('    셸: ' + ' | '.join(row))

    p('\n⛔이 점검은 판정을 바꾸지 않는다. `SINGLE-KNOB-SUPPORTED` 라벨은 동결 채점기의 것이다.')
    p('⛔사후 점검의 p 값을 확증 근거로 인용하지 않는다.')
    open(os.path.join(HERE, 'posthoc_single_shells.txt'), 'w').write('\n'.join(OUT) + '\n')


if __name__ == '__main__':
    main()
