"""B-3b 단일출처 — **`bank_refit_every_s` 용량-반응**: 2 s 는 조율된 상수인가?

## 왜
`B-3` 이 `refit@2s` 를 36/36 으로 확증했다. 그러나 **`2.0` 이라는 값은 `C`(08-20)가
레코드 5개에서 고른 것**이고, **그 5개는 지금 12개의 부분집합**이다 ⟹ **부분적으로 in-sample**.
저장소는 이 결함(라우터 임계가 스윕된 코퍼스에서 평가됨)에 이미 한 번 물렸다.

## ⛔★두 번째 목적은 **결과를 열기 전에 철회됐다** (2026-08-23, RV-3 지적)
나는 *"용량 곡선의 모양이 누수 가설과 유효수명 가설을 가른다"* 고 적었다:
누수면 **1 s 가 최선이고 단조**, 유효수명이면 **포화**하리라고.
⛔**그 판별자는 무효다.** 주기 τ 는 예측 시점의 **적합 나이를 [0,τ] 로 묶는다** ⟹
***유효수명 가설도 똑같이 단조를 예측한다.*** 두 가설이 같은 모양을 낸다.
✅**대신 누수 가설 자체가 코드로 죽었다**(RV-3 ①②): `_bank_predict_grid` 가 읽는 상태는
`self._bank` 뿐이고 그것을 쓰는 곳은 `_bank_fit_window` 하나, Phase 2 호출부는 `_bank_update`
하나다. predict→update 순서이므로 **채점 시점 적합의 최신 자료는 항상 예측 시각 −1 s 이하**.
⟹ **이 배치의 목적은 in-sample 우려(아래 §왜) 하나로 줄어든다.**
⚠**이 철회는 `results_b3b.jsonl` 이 완성된 뒤, 그러나 분석기를 돌리기 *전에* 이루어졌다**
(커밋 이력이 순서를 증명한다).

## 팔 (기존 `FROZEN`·`REFIT2` 는 `results_b3.jsonl` 에서 재사용)
`REFIT1`(1 s) · `REFIT5`(5 s) · `REFIT10`(10 s). 신규 108런.

## 판정 (결과 전 확정) — 각각 `FROZEN` 대비 짝 부호검정, n=36
- **세 팔 모두 ≥24/36** ⟹ ★★`ANY-REFIT-HELPS` — **2.0 은 조율된 상수가 아니다.**
  ⚠**(RV-3 추가 사실, 결과 전 기재)** `C` 는 `2.0` 을 **지금 12개 중 4개(+동시각 1개)** 에서 골랐다.
  내가 *"5개는 12개의 부분집합"* 이라 쓴 것보다 **더 정확한 수는 4(+1)** 이고, 신선 부분집합은
  **`C` 미노출 BS 7 레코드**다.
  발견은 **값이 아니라 기제**이고 in-sample 우려가 사라진다. **가장 강한 결과.**
- **빠른 것만(1·2) 통과하고 5·10 이 실패** ⟹ `RATE-MATTERS`(주기가 실제 인자)
- **2 s 만 통과** ⟹ ⛔`TUNED-AT-2S` — in-sample 우려가 살아난다. 인용을 좁혀야 한다.
- **모두 실패** ⟹ ⛔`B3-NOT-REPRODUCED`(같은 자료이므로 이 경우는 러너 결함을 뜻한다)

## 부차 (결과 전 기재)
- **최적 주기**: `rel_median_pct` 최소인 팔. ⚠**차이가 3%p 미만이면 "평평" 이라 부르고
  최적점 주장을 하지 않는다**(S17 이 평평한 곳에서 순위를 주장했다가 물렸다).
- **비용**: 팔별 벽시계 중앙. ⚠채점 코드 포함이므로 **절대 예산 주장 금지**.
"""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from gates_b3 import (RECORDS, WINDOWS, SEED, NOISE, FIELD, N_PAIRS,   # noqa: F401
                      sign_test_one_sided, key, paired)

ARMS = {'REFIT1': {'bank_refit_every_s': 1.0},
        'REFIT5': {'bank_refit_every_s': 5.0},
        'REFIT10': {'bank_refit_every_s': 10.0}}
ARM_ORDER = ['REFIT1', 'REFIT5', 'REFIT10']
REUSED = ['FROZEN', 'REFIT2']        # results_b3.jsonl 에서 읽는다
REF = 'FROZEN'
NOMINAL = 24
FLAT_PP = 3.0                        # 결과 전 확정: 3%p 미만 차이는 "평평"


def gates(rows):
    ok = [r for r in rows if 'error' not in r]
    new = [r for r in ok if r['cell'] in ARMS]
    exp = len(ARMS) * N_PAIRS
    out = [('G-0 신규 런 수', len(new) == exp, f'{len(new)}/{exp}'),
           ('G-0b 에러 0', len(rows) == len(ok), f'{len(rows)-len(ok)}')]
    for a in REUSED:
        n = sum(1 for r in ok if r['cell'] == a)
        out.append((f'G-1 재사용 팔 {a} 36행', n == N_PAIRS, f'{n}/{N_PAIRS}'))
    out.append(('G-2 되읽기 일치',
                all(r.get('refit_readback') == ARMS[r['cell']]['bank_refit_every_s'] for r in new), ''))
    # ★가지 계수기 — 주기가 짧을수록 호출이 많아야 한다(단조). 아니면 노브가 안 걸린 것이다.
    med = {}
    for a in ARM_ORDER:
        v = sorted({r.get('refit_calls') for r in ok if r['cell'] == a})
        med[a] = v
    out.append(('G-3 재적합 횟수가 주기와 역단조',
                all(min(med[x]) > max(med[y]) for x, y in zip(ARM_ORDER, ARM_ORDER[1:]) if med[x] and med[y]),
                str(med)[:90]))
    out.append(('G-4 lead 0 전건 비트동일(음성대조)',
                len({r.get('lead0') for r in ok if key(r) == (RECORDS[0], WINDOWS[0])}) == 1, ''))
    out.append(('G-5 36 창 전건', all(len({key(r) for r in ok if r['cell'] == a}) == N_PAIRS
                                      for a in ARM_ORDER), ''))
    return out


def label(res):
    """res = {arm: paired(...) vs FROZEN}, `REFIT2` 포함."""
    order = ['REFIT1', 'REFIT2', 'REFIT5', 'REFIT10']
    passed = [a for a in order if res[a]['better'] >= NOMINAL]
    if len(passed) == len(order):
        rels = [res[a]['rel_median_pct'] for a in order]
        flat = (max(rels) - min(rels)) < FLAT_PP
        return ('ANY-REFIT-HELPS',
                f'네 주기(1·2·5·10 s) **전부** {NOMINAL}/36 문턱을 넘는다 '
                f'({[res[a]["better"] for a in order]}) ⟹ ★**2.0 은 조율된 상수가 아니다** — '
                f'발견은 값이 아니라 기제다. 상대중앙 폭 {max(rels)-min(rels):.1f}%p '
                f'{"= 평평(최적점 주장 금지)" if flat else "= 주기에 반응"}.')
    if passed == ['REFIT1', 'REFIT2'] or passed == ['REFIT1', 'REFIT2', 'REFIT5']:
        return ('RATE-MATTERS',
                f'빠른 주기만 통과한다({passed}) ⟹ 재적합 **주기**가 실제 인자다.')
    if passed == ['REFIT2']:
        return ('TUNED-AT-2S',
                '⛔`2.0` 에서만 통과한다 ⟹ in-sample 조율 우려가 살아난다. 인용을 좁혀야 한다.')
    if not passed:
        return ('B3-NOT-REPRODUCED',
                '⛔어느 주기도 통과하지 못한다 — 같은 자료이므로 이는 러너 결함을 뜻한다.')
    return ('PARTIAL', f'통과 팔 = {passed}. 사전등록한 네 모양 어디에도 안 맞는다 ⟹ 서술로만 읽어라.')
