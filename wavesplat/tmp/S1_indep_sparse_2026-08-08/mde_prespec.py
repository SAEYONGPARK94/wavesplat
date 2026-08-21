"""S1 사전등록용 MDE 산출 (결과 관측 **전**에 실행·동결).

## AMENDMENT 1 (2026-08-08, S1 결과 0건 상태에서 개정) — CON 지적 F1

v1 은 σ 를 **혼잡 4개에서만** 뽑아 단일 MDE 0.04616 을 냈다. 그런데 CON 이 B2 원자료에서
잰 대로 **희소의 기준선 전체(nRMSE 중앙 0.0153)가 그 MDE 보다 3.0배 작다.**
⟹ 희소에서 오차를 0 으로 만들어도 관측/MDE ≤ 0.33 ⟹ 사전등록 §5.5 의 "관측/MDE" 가
**희소 5개 대조 전부에 자동으로 '검정력 부족'을 찍는다.** 이건 E-D 오류 #18(셸 MDE 누락)의
*반대 방향* 재발이다: 있어야 할 곳에 없던 게 아니라, **엉뚱한 척도의 것을 붙인 것**이다.

그래서 v2 는 네 가지를 낸다:
  * **레짐별** σ_d (희소/혼잡 따로) — 기준선이 18.5배 다르므로 하나로 못 쓴다
  * `mde_ep`   : 단일 에피소드 탐지 하한 = MDE_MULT * sigma_d
  * `mde_agg`  : n 개 짝 Δ 의 **중앙**에 대한 하한 = MDE_MULT * 1.253 * sigma_d / sqrt(n)
                 (1.253 = 중앙값 SE 의 점근 상수 sqrt(pi/2))
  * **상대판**  : 각 레짐 X-BASE 기준선 중앙으로 나눈 값 (CON 지적 F3 — 절대 규약 단독은
                 희소 피해를 구조적으로 은폐한다)

분산 출처 = `tmp/B2_g1p1_ext_2026-08-06/results.jsonl`. B2 를 쓰는 이유는 **두 레짐을 모두
담은 유일한 배치**이기 때문이다(rx2 는 혼잡 4개뿐). 규약 상수는 rx2 `MDE_prespec.json` 승계.

⚠이 값들은 **설계용 사전 추정치**다(구 데이터·구 표본). S1 종료 후 신규 대조 팔(X-BASE)의
실제 시드분산으로 **재도출해 병기**한다 — 사후 재선택이 아니라 점검이다.
이 스크립트는 S1 결과 파일을 **읽지 않는다**.
"""
import json
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
B2 = os.path.abspath(os.path.join(HERE, '..', 'B2_g1p1_ext_2026-08-06', 'results.jsonl'))
RX2 = os.path.abspath(os.path.join(HERE, '..', 'rx2_init_2026-08-05', 'results.jsonl'))
OUT = os.path.join(HERE, 'MDE_prespec.json')

D2_N3 = 1.693
MDE_MULT = 2.1279493432578005          # rx2 승계, 재선택 금지
MEDIAN_SE = 1.2533141373155001         # sqrt(pi/2)
N_EPISODES = 16                        # 레짐당
CELLS = ('X-BASE', 'G1', 'P1', 'G1P1', 'AUTO')
METRICS = ('nRMSE',)
CONTROL = 'X-BASE'


def ncomp(ep):
    m = re.search(r'_(\d+)comp', ep)
    return int(m.group(1)) if m else -1


def regime(ep):
    n = ncomp(ep)
    return 'crowded' if n >= 15 else ('sparse' if n > 0 else '?')


def load(path):
    out = []
    for line in open(path):
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if 'error' not in r and r.get('cell') in CELLS:
            out.append(r)
    return out


def stats(recs, metric):
    """레짐별 (range_med -> sigma_hat -> sigma_d) 와 X-BASE 기준선 중앙."""
    out = {}
    for rg in ('sparse', 'crowded'):
        sel = [r for r in recs if regime(r['episode']) == rg]
        if not sel:
            continue
        ranges = []
        for cell in CELLS:
            for ep in sorted({r['episode'] for r in sel if r['cell'] == cell}):
                v = [r[metric] for r in sel if r['cell'] == cell and r['episode'] == ep
                     and isinstance(r.get(metric), (int, float)) and np.isfinite(r[metric])]
                if len(v) >= 2:
                    ranges.append(float(np.max(v) - np.min(v)))
        base = []
        for ep in sorted({r['episode'] for r in sel if r['cell'] == CONTROL}):
            v = [r[metric] for r in sel if r['cell'] == CONTROL and r['episode'] == ep
                 and isinstance(r.get(metric), (int, float)) and np.isfinite(r[metric])]
            if v:
                base.append(float(np.median(v)))
        if not ranges or not base:
            continue
        range_med = float(np.median(ranges))
        sigma_hat = range_med / D2_N3
        sigma_d = float(sigma_hat * np.sqrt(2.0) * 1.16 / np.sqrt(3.0))
        mde_ep = MDE_MULT * sigma_d
        mde_agg = MDE_MULT * MEDIAN_SE * sigma_d / np.sqrt(N_EPISODES)
        b = float(np.median(base))
        out[rg] = {
            'n_cells_episodes': len(ranges),
            'n_episodes': len(base),
            'baseline_median_XBASE': b,
            'range_med': range_med,
            'sigma_hat': sigma_hat,
            'sigma_d': sigma_d,
            'mde_ep': float(mde_ep),
            'mde_agg': float(mde_agg),
            'mde_ep_rel': float(mde_ep / b) if b else float('nan'),
            'mde_agg_rel': float(mde_agg / b) if b else float('nan'),
        }
    return out


def main():
    for p in (B2, RX2):
        if not os.path.exists(p):
            sys.exit(f'없음: {p}')
    b2 = load(B2)
    rx2 = load(RX2)

    out = {
        'amendment': 1,
        'source_primary': os.path.relpath(B2, os.path.dirname(HERE)),
        'source_check': os.path.relpath(RX2, os.path.dirname(HERE)),
        'n_records_b2': len(b2), 'n_records_rx2': len(rx2),
        'd2_n3': D2_N3, 'mde_mult': MDE_MULT, 'median_se_const': MEDIAN_SE,
        'n_episodes_per_regime': N_EPISODES,
        'note': ('regime-split sigma from OLD B2 data; design-time prior only. '
                 'obs/MDE must be compared WITHIN regime, never across.'),
        'by_regime': {},
        'crowded_crosscheck_rx2': {},
    }
    for m in METRICS:
        out['by_regime'][m] = stats(b2, m)
        out['crowded_crosscheck_rx2'][m] = stats(rx2, m).get('crowded')

    json.dump(out, open(OUT, 'w'), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
