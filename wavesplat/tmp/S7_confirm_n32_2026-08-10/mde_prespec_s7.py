"""S7 사전등록용 MDE 산출 (S7 결과 관측 **전**에 실행·동결).

## 무엇이 S1 과 다른가
- **n_episodes = 32** (S1 은 16). `mde_agg` 는 √n 로 줄어든다.
- **분산 출처 = S6 `results.jsonl`** — S7 이 쓸 **바로 그 셀들**(`AUTO`, `AUTO-NT`,
  `AUTO-NT-L1-*`)의 시드분산을 담은 유일한 배치다. S1 은 구 B2 를 썼다.
- **기준선을 둘 낸다**: PRIMARY-1 은 `AUTO-NT`, PRIMARY-2 는 `AUTO`(출하 기본값)를 기준선으로
  쓰므로 상대 MDE 의 분모가 다르다.

## 순환논리가 아닌 이유
여기서 뽑는 것은 **σ(시드 재현 산포)** 와 **기준선 수준**뿐이고, **처치 효과 크기는 쓰지 않는다.**
σ 는 S6 의 구 에피소드(시드 20-23)에서 나오고 S7 은 신선 에피소드(시드 24-31)에서 잰다
⟹ 설계용 **사전 추정치**다. ⚠S7 종료 후 신규 팔의 실제 시드분산으로 재도출해 **병기**한다
(사후 재선택이 아니라 점검). 이 스크립트는 S7 결과 파일을 **읽지 않는다**.

규약 상수(`D2_N3`, `MDE_MULT`, `MEDIAN_SE`)는 rx2 → S1 → S6 승계. **재선택 금지.**
"""
import json
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
S6 = os.path.abspath(os.path.join(HERE, '..', 'S6_l1_sweep_2026-08-08', 'results.jsonl'))
OUT = os.path.join(HERE, 'MDE_prespec_s7.json')

D2_N3 = 1.693                          # 3표본 범위 -> sigma 변환 (승계)
MDE_MULT = 2.1279493432578005          # rx2 승계, 재선택 금지
MEDIAN_SE = 1.2533141373155001         # sqrt(pi/2)
N_EPISODES = 32                        # ★S7: 레짐당 32
CELLS = ('AUTO', 'AUTO-NT', 'AUTO-NT-L1-3e4', 'AUTO-NT-L1-1e4', 'AUTO-NT-L1-0')
BASELINES = ('AUTO', 'AUTO-NT')        # PRIMARY-2 / PRIMARY-1 기준선
METRIC = 'nRMSE'


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


def stats(recs, rg):
    sel = [r for r in recs if regime(r['episode']) == rg]
    if not sel:
        return None
    ranges = []
    for cell in CELLS:
        for ep in sorted({r['episode'] for r in sel if r['cell'] == cell}):
            v = [r[METRIC] for r in sel if r['cell'] == cell and r['episode'] == ep
                 and isinstance(r.get(METRIC), (int, float)) and np.isfinite(r[METRIC])]
            if len(v) >= 2:
                ranges.append(float(np.max(v) - np.min(v)))
    if not ranges:
        return None
    range_med = float(np.median(ranges))
    sigma_hat = range_med / D2_N3
    sigma_d = float(sigma_hat * np.sqrt(2.0) * 1.16 / np.sqrt(3.0))
    mde_ep = MDE_MULT * sigma_d
    mde_agg = MDE_MULT * MEDIAN_SE * sigma_d / np.sqrt(N_EPISODES)

    base = {}
    for cell in BASELINES:
        per_ep = []
        for ep in sorted({r['episode'] for r in sel if r['cell'] == cell}):
            v = [r[METRIC] for r in sel if r['cell'] == cell and r['episode'] == ep
                 and isinstance(r.get(METRIC), (int, float)) and np.isfinite(r[METRIC])]
            if v:
                per_ep.append(float(np.median(v)))
        base[cell] = dict(
            n_episodes=len(per_ep),
            baseline_median=float(np.median(per_ep)) if per_ep else float('nan'),
        )
        b = base[cell]['baseline_median']
        base[cell]['mde_ep_rel'] = float(mde_ep / b) if b else float('nan')
        base[cell]['mde_agg_rel'] = float(mde_agg / b) if b else float('nan')

    return dict(n_cell_episode_ranges=len(ranges), range_med=range_med,
                sigma_hat=sigma_hat, sigma_d=sigma_d,
                mde_ep=float(mde_ep), mde_agg=float(mde_agg), baselines=base)


def main():
    if not os.path.exists(S6):
        sys.exit(f'없음: {S6}')
    recs = load(S6)
    out = {
        'batch': 'S7',
        'source': os.path.relpath(S6, os.path.dirname(HERE)),
        'source_note': ('sigma + baseline levels from S6 OLD episodes (seed 20-23); '
                        'S7 scores FRESH episodes (seed 24-31). Design-time prior only. '
                        'Effect sizes are NOT taken from S6. '
                        'obs/MDE must be compared WITHIN regime, never across.'),
        'n_records': len(recs),
        'd2_n3': D2_N3, 'mde_mult': MDE_MULT, 'median_se_const': MEDIAN_SE,
        'n_episodes_per_regime': N_EPISODES,
        'metric': METRIC,
        'by_regime': {rg: stats(recs, rg) for rg in ('crowded', 'sparse')},
    }
    json.dump(out, open(OUT, 'w'), indent=1)
    print(json.dumps(out, indent=1, ensure_ascii=False))


if __name__ == '__main__':
    main()
