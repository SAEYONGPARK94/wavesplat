"""B-8 단일출처 — ★**`B-7` 이 남긴 `UNVERIFIED` 를 측정으로 가른다**.

## 갈라야 할 두 가설
`B-7` 이 *"제안 표현은 추적해도 실측 기술이 없다"*(12/36)를 냈다. 그러나 그 팔은
`init_method='learned'` 강제였고 **그 체크포인트는 합성으로 학습됐다**. ⟹ 두 설명이 남는다:
- **H-A `REPRESENTATION-LIMITED`** — *격자 비의존 스펙트럼 표현/옵티마이저 자체*가 실측에서 안 된다.
- **H-B `CHECKPOINT-LIMITED`** — *표현은 되는데 합성 학습 체크포인트가 실측으로 전이되지 않는다.*

★**가르는 방법**: **체크포인트를 쓰지 않는 같은 표현**을 같은 프로토콜에서 돌린다.
`init_method='v2_konly'` = **해석적 빔포밍 + 코스투파인** 초기화. **같은 Gaussian 성분 표현**,
**같은 추적 경사하강**, **학습 0**. ⟹ 두 팔의 차이가 **체크포인트 하나**다.

## 팔 (12 레코드 × 창 {1,3,5} = 36; 나머지는 `B-7`/`B-3` 에서 재사용)
- `V2K-TRACK` (**신규**): `real_domain_gate=False`, `init_method='v2_konly'`
- 재사용: `PROP-TRACK`(학습) · `SHIP-TRACK`(배포) — `../B7_PROP_TRACK_2026-08-23/results_b7.jsonl`

## 판정 (결과 전 확정) — 참 제로필드 대비, **lead 1–3 s**, n=36
`PROP-TRACK` 은 그 자리에서 **12/36** 이었다(기준선).
- `V2K-TRACK` **≥24/36** ⟹ ★**`CHECKPOINT-LIMITED`** — 표현은 되고 **체크포인트가 문제**다.
  ⟹ 처방이 명확해진다: **혼합 커리큘럼 재학습**(상시지시와 정합).
- **19~23** ⟹ ⚠`SPLIT-UNDECIDED`
- **≤18/36** ⟹ ⛔**`REPRESENTATION-LIMITED`** — **학습을 빼도 안 된다** ⟹ 체크포인트 가설 기각.
  ★이 경우 `B-7` 의 음성은 **표현/옵티마이저 수준**이고 재학습으로 풀리지 않는다.

⚠**둘 다 12/36 근처면** 그것 자체가 답이다(**차이 0 = 체크포인트가 요인이 아니다**).
★**짝 검정도 결과 전에 적는다**: `V2K` vs `PROP` 짝(`fc_mean`) — 어느 쪽이 나은가.
가족단위 α = 0.05/12 = 0.00417.

★**개입 계수기(가지 단위)**: `V2K` 팔에서 **추적 경사스텝 > 0** 이어야 한다.
0 이면 판정은 ⛔`V2K-NO-INTERVENTION`(설계가 답을 막음).
★**음성 대조**: `V2K` 의 성분 수 > 0 이어야 한다(초기화가 실제로 성분을 만들었나).
"""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
B2 = os.path.abspath(os.path.join(HERE, '..', 'B2_REAL_2026-08-23'))
B3 = os.path.abspath(os.path.join(HERE, '..', 'B3_REFIT_2026-08-23'))
B7 = os.path.abspath(os.path.join(HERE, '..', 'B7_PROP_TRACK_2026-08-23'))
for p in (HERE, B2, B3, B7):
    if p not in sys.path: sys.path.insert(0, p)
from gates_b3 import RECORDS, WINDOWS, SEED, NOISE, N_PAIRS, sign_test_one_sided, key, paired  # noqa
from gates_b7 import beats_zero, LEAD_NEAR, LEAD_FAR                                            # noqa

ARMS = {'V2K-TRACK': {'real_domain_gate': False, 'init_method': 'v2_konly'}}
ARM_ORDER = ['V2K-TRACK']
REUSED = ['PROP-TRACK', 'SHIP-TRACK']
NOMINAL, CONFIRM = 24, 26
FAMILY_ALPHA = 0.05 / 12
PROP_BASELINE = 12          # B-7 의 lead 1–3 값 (verdict_b7.json 에서 분석기가 assert 로 대조)


def gates(rows):
    ok = [r for r in rows if 'error' not in r]
    new = [r for r in ok if r['cell'] in ARMS]
    out = [('G-0 신규 런 수', len(new) == N_PAIRS, f'{len(new)}/{N_PAIRS}'),
           ('G-0b 에러 0', len(rows) == len(ok), f'{len(rows)-len(ok)}')]
    for a in REUSED:
        n = sum(1 for r in ok if r['cell'] == a)
        out.append((f'G-1 재사용 팔 {a} {N_PAIRS}행', n == N_PAIRS, f'{n}/{N_PAIRS}'))
    out.append(('★G-2 V2K 추적 개입 >0 (가지 계수기)',
                all((r.get('track_steps') or 0) > 0 for r in new),
                str(sorted({r.get('track_steps') for r in new}))[:50]))
    out.append(('G-3 V2K 가 성분을 만들었나(음성대조)',
                all((r.get('n_atoms') or 0) > 0 for r in new),
                str(sorted({r.get('n_atoms') for r in new}))[:60]))
    out.append(('G-4 V2K 재적합 0회 (bank 경로 아님)',
                all((r.get('refit_calls') or 0) == 0 for r in new), ''))
    out.append(('G-5 36 창 전건', len({key(r) for r in new}) == N_PAIRS, ''))
    return out


def label(v2k_near, prop_near, pair_vp, ok_intervention):
    if not ok_intervention:
        return ('V2K-NO-INTERVENTION',
                '⛔`V2K` 팔에서 추적 개입이 0회다 ⟹ 이 배치는 가설을 가르지 못했다.')
    w, n = v2k_near['better'], v2k_near['n']
    tail = (f" (기준선 `PROP` {prop_near['better']}/{prop_near['n']} · "
            f"짝 V2K<PROP {pair_vp['better']}/{pair_vp['n']} "
            f"{pair_vp['rel_median_pct']:+.1f}% p={pair_vp['p_one_sided']:.5f})")
    if w >= CONFIRM:
        return ('CHECKPOINT-LIMITED',
                f'★학습을 빼면 실측 기술이 생긴다({w}/{n}, p={v2k_near["p_one_sided"]:.6f}) ⟹ '
                '**표현이 아니라 합성 학습 체크포인트가 한계다.** 처방 = 혼합 커리큘럼 재학습.' + tail)
    if w >= NOMINAL:
        return ('CHECKPOINT-LIMITED (EXPLORATORY)', f'{w}/{n} 명목 통과, 가족단위 미달.' + tail)
    if w <= n - NOMINAL:
        return ('REPRESENTATION-LIMITED',
                f'⛔**학습을 빼도 안 된다**({w}/{n}) ⟹ 체크포인트 가설 기각. `B-7` 의 음성은 '
                '**표현/옵티마이저 수준**이고 재학습으로 풀리지 않는다.' + tail)
    return ('SPLIT-UNDECIDED', f'부호로 갈리지 않는다({w}/{n}).' + tail)
