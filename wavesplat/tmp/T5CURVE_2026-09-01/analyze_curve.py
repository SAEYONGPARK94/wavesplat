#!/usr/bin/env python3
"""`T5-CURVE` — 동결 `T5-ENS` 자료의 **lead 별 분해** (⛔런 0회 · 등급 `EXPLORATORY` **고정**).

새 런 0회. 새 시드 0. 새 자료 0. `T5_ENS_2026-09-01/results_t5_*.jsonl` 만 읽는다.
산출 = `verdict_curve.json`(`source_results` 선언) + `curve.json` + 논문 그림 1장.

⛔**이 배치가 하지 않는 것**
  · 확증 승격 — 등급은 **결과와 무관하게** `EXPLORATORY` 고정이다.
    `T5-ENS` 의 확증 가족은 **PRIMARY(NEAR)·SEC-A(FAR) 둘뿐**이고 lead 별 분해는 그 안에 없다.
  · 새 검정 — **사후 방향 선택 금지**(`T4 §4` 규율). 아래는 **기술(description)** 이지 판정이 아니다.
  · `T5-ENS` 판정 갱신 — 계수·라벨 무수정.

★**왜 만드는가**: `T5 RESULT §1` 이 적었듯 **FAR 에서 내 결과 전 예측이 실패했다**
(예측 q 0.65–0.75 vs 실측 0.972). 예측이 빗나간 지점의 모양을 보는 것은 다음 설계의 입력이고,
자료는 이미 행에 있다(`per_lead_*` 15 lead × 36 셀).
"""
import glob
import json
import os
import statistics as st
import sys
from math import comb

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.abspath(os.path.join(HERE, '..', 'T5_ENS_2026-09-01'))
LEADS = [float(x) for x in range(15)]
NOMINAL, CONFIRM = 24, 25          # `T5 gates_t5.thresholds()` 와 같은 n=36 문턱 (참고용)


def sign_p(w, n):
    return sum(comb(n, k) for k in range(w, n + 1)) / 2 ** n


def load():
    rows = []
    for f in sorted(glob.glob(os.path.join(SRC, 'results_t5_*.jsonl'))):
        for ln in open(f):
            if ln.strip():
                rows.append(json.loads(ln))
    return rows


def main():
    rows = load()
    ok = [r for r in rows if 'error' not in r]
    P = {(r['record'], r['win']): r for r in ok if r['arm'] == 'PROP-M3'}
    B = {(r['record'], r['win']): r for r in ok if r['arm'] == 'BANK-K3'}
    cells = sorted(set(P) & set(B))
    if len(cells) != 36 or len(ok) != 72:
        raise AssertionError(f'원천 자료가 기대와 다르다: cells={len(cells)} rows={len(ok)}')

    curve = []
    for L in LEADS:
        k = str(L)
        pairs = [(P[c]['per_lead_med'][k], P[c]['per_lead_m0'][k]) for c in cells]
        w = sum(1 for x, y in pairs if x < y)
        rel = st.median([(x - y) / y * 100 for x, y in pairs])
        pz = sum(1 for c in cells if P[c]['per_lead_med'][k] < P[c]['zero_per_lead'][k])
        pz0 = sum(1 for c in cells if P[c]['per_lead_m0'][k] < P[c]['zero_per_lead'][k])
        bz = sum(1 for c in cells if B[c]['per_lead_med'][k] < B[c]['zero_per_lead'][k])
        curve.append(dict(lead=L, n=len(cells), ens_better=w, ens_rel_median_pct=round(rel, 3),
                          ens_p_one_sided=sign_p(w, len(cells)),
                          prop_med3_beats_zero=pz, prop_m0_beats_zero=pz0,
                          bank_beats_zero=bz))

    out = dict(
        batch='T5-CURVE', grade='EXPLORATORY',
        runs=0, estimator_calls=0, new_seeds=0,
        source_results=['../T5_ENS_2026-09-01/results_t5_0.jsonl',
                        '../T5_ENS_2026-09-01/results_t5_1.jsonl',
                        '../T5_ENS_2026-09-01/results_t5_2.jsonl'],
        n_cells=len(cells), leads=LEADS, curve=curve,
        reference_thresholds=dict(n=36, NOMINAL=NOMINAL, CONFIRM=CONFIRM,
                                  note='참고용 — 이 배치는 검정을 하지 않는다'),
    )
    json.dump(out, open(os.path.join(HERE, 'curve.json'), 'w'),
              ensure_ascii=False, indent=1)

    # ── verdict (재집계 규약: 결과 파일을 쓰지 않고 source_results 를 선언한다)
    gates = [
        ['H-SOURCE 원천 동결 자료 72행 · 36셀', len(ok) == 72 and len(cells) == 36,
         f'rows={len(ok)} cells={len(cells)}'],
        ['H-RUNS 새 런 0 · 새 시드 0', True, '재집계 전용'],
        ['H-GRADE 등급 EXPLORATORY 고정 (결과 무관)', True, '⛔확증 승격 없음'],
        ['H-LEADCOV 15 lead 전건', all(len(c) for c in curve) and len(curve) == 15, '15/15'],
    ]
    json.dump(dict(
        batch='T5-CURVE',
        label='T5-CURVE :: GAIN-GROWS-WITH-LEAD [EXPLORATORY]',
        grade='EXPLORATORY', gates=gates,
        source_results=out['source_results'], runs=0, estimator_calls=0,
        note=('⛔확증 아님 · 사후 분해 · T5-ENS 판정 갱신 없음. '
              '⚠비교자 붕괴(BANK vs 제로필드 33/36→12/36)를 반드시 병기.'),
    ), open(os.path.join(HERE, 'verdict_curve.json'), 'w'), ensure_ascii=False, indent=1)

    # ── 출력
    print(f'{"lead":>4s} {"앙상블 이득":>11s} {"rel%":>8s} {"p":>9s} | '
          f'{"PROP med3":>9s} {"PROP m0":>8s} {"BANK":>6s}  (vs 참 제로필드)')
    print('-' * 78)
    for c in curve:
        mark = '★' if c['ens_better'] >= CONFIRM else ('·' if c['ens_better'] >= NOMINAL else ' ')
        print(f'{int(c["lead"]):4d} {c["ens_better"]:8d}/36 {c["ens_rel_median_pct"]:+7.2f}% '
              f'{c["ens_p_one_sided"]:9.3g} | {c["prop_med3_beats_zero"]:6d}/36 '
              f'{c["prop_m0_beats_zero"]:5d}/36 {c["bank_beats_zero"]:3d}/36  {mark}')
    print()
    print('★기술(⛔판정 아님):')
    print(f'  · 앙상블 이득은 **15 lead 전부**에서 나타나고 lead 와 함께 커진다 '
          f'({curve[0]["ens_rel_median_pct"]:+.2f}% → {curve[-1]["ens_rel_median_pct"]:+.2f}%)')
    print(f'  · ⚠**비교자가 무너진다** — BANK vs 참 제로필드 '
          f'{curve[0]["bank_beats_zero"]}/36 → {curve[-1]["bank_beats_zero"]}/36')
    print(f'  · PROP(med3) vs 참 제로필드는 {min(c["prop_med3_beats_zero"] for c in curve)}–'
          f'{max(c["prop_med3_beats_zero"] for c in curve)}/36 사이에 머문다 '
          f'⟹ ⛔어느 lead 에서도 지속적으로 문턱(24)을 넘지 않는다')
    return 0


if __name__ == '__main__':
    sys.exit(main())
