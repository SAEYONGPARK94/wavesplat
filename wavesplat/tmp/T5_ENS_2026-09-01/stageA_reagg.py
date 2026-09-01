#!/usr/bin/env python3
"""Stage A — 나우캐스트 앙상블 대칭 비교 재집계 (⛔런 0회 · 등급 `EXPLORATORY` 고정).

동결 `T3_NOWCAST_2026-08-30/results_t3*.jsonl` 만 읽어 `PREREG.md §2.3` 표를 **다시 만든다**.
문서와 어긋나면 `AssertionError` 로 죽는다 — ★손복사 금지 규율(선례 `B3_REFIT/final6.py`).

⛔이 스크립트가 하지 않는 것:
  · 추정기 호출 0회 (임포트조차 하지 않는다)
  · 확증 등급 승격 0건 — `PREREG §2.3`/`§4.7` 이 등급을 **결과와 무관하게** `EXPLORATORY` 로 고정했다
  · `B-2`/`M-8b` 판정 갱신 0건 — 산출은 **공시 한 줄**이지 판정이 아니다

★왜 스크립트인가: 이 표는 내가 사전등록 **전에** 본 값이다(노출 공시 `§2.3`).
  노출된 값일수록 **재도출 가능해야** 나중에 "그때 뭘 봤는지" 를 다툴 수 없다.
"""
import glob
import json
import os
import statistics as st
import sys
from math import comb

HERE = os.path.dirname(os.path.abspath(__file__))
T3 = os.path.abspath(os.path.join(HERE, '..', 'T3_NOWCAST_2026-08-30'))

# ── PREREG §2.3 이 인쇄한 값 (⛔여기서 손으로 고치지 마라 — 문서와 함께만 바뀐다) ──────
CLAIMS = [
    # (라벨, 팔A, 필드A, 팔B, 필드B, better, n, rel_median_pct)
    ('비대칭(역사적 상태)', 'SHIP-K3', 'e_m0', 'PROP-V400M3', 'e_m0', 33, 36, -25.75),
    ('대칭(양쪽 median-of-3)', 'SHIP-K3', 'e_m0', 'PROP-V400M3', 'e_med3', 31, 36, -15.15),
]
# ★`SHIP-K3.e_m0` 이 곧 패키지 median-of-3 이다 — `SHIP-K3` 은 MEM=(0,) 이고 패키지가 내부에서
#   k=3 을 돌린다. `T3 G-MIRROR` 36/36 이 `SHIP-K1M3.e_med3 == SHIP-K3.e_m0` 을 비트로 증명했다.
#   그 항등을 이 스크립트도 독립으로 재확인한다(아래 `check_mirror`).


def load_rows():
    rows = []
    for f in sorted(glob.glob(os.path.join(T3, 'results_t3*.jsonl'))):
        with open(f) as fh:
            for ln in fh:
                ln = ln.strip()
                if ln:
                    rows.append(json.loads(ln))
    if not rows:
        raise SystemExit(f'동결 자료 0행: {T3}/results_t3*.jsonl')
    return rows


def sign_test_one_sided(w, n):
    return sum(comb(n, k) for k in range(w, n + 1)) / 2 ** n


def paired(idx, cells, a, fa, b, fb):
    """`a.fa < b.fb` 인 셀 수, n, 상대차 중앙값(%).

    ⚠**집계 규약은 `gates_t3.paired:375` 의 `st.median` 이다** — 손으로 고른 규약이 아니다.
    ⛔초판(2026-09-01)이 `rels[len//2]`(짝수 n 에서 **위쪽** 중앙원소)를 써서 대칭 비교의
    rel 을 −14.69% 로 적었다. 참 median 은 **−15.15%** 다(계수 31/36 은 불변).
    ★같은 배치 안에 집계 규약이 둘이면 그 배치는 자기 수치를 재현하지 못한다 —
    적대적 심사(2026-09-01)가 동결 전에 잡았다.
    """
    w = n = 0
    rels = []
    for c in cells:
        ra, rb = idx.get((a,) + c), idx.get((b,) + c)
        va = ra.get(fa) if ra else None
        vb = rb.get(fb) if rb else None
        if va is None or vb is None:
            continue
        n += 1
        w += (va < vb)
        rels.append((va - vb) / vb * 100.0)
    return w, n, (st.median(rels) if rels else float('nan'))


def check_mirror(idx, cells):
    """`SHIP-K1M3.e_med3 == SHIP-K3.e_m0` — 하네스 median == 패키지 k=3 (T3 G-MIRROR 상속)."""
    same = tot = 0
    for c in cells:
        ra, rb = idx.get(('SHIP-K1M3',) + c), idx.get(('SHIP-K3',) + c)
        if not ra or not rb:
            continue
        tot += 1
        same += (ra.get('e_med3') == rb.get('e_m0'))
    return same, tot


def main():
    rows = load_rows()
    idx = {(r['arm'], r['record'], r['win']): r for r in rows}
    cells = sorted({(r['record'], r['win']) for r in rows})

    bad = []
    out = {'batch': 'T5-ENS/StageA', 'grade': 'EXPLORATORY',
           'source': os.path.relpath(T3, os.path.join(HERE, '..', '..')),
           'source_rows': len(rows), 'n_cells': len(cells), 'runs': 0,
           'estimator_calls': 0, 'claims': []}

    print(f'동결 원천 {len(rows)}행 · 셀 {len(cells)}개\n')
    print(f'{"비교":34s} {"계수":>8s} {"rel중앙":>10s} {"p(단측)":>11s}')
    print('-' * 68)
    for lab, a, fa, b, fb, w_doc, n_doc, rel_doc in CLAIMS:
        w, n, rel = paired(idx, cells, a, fa, b, fb)
        p = sign_test_one_sided(w, n)
        print(f'{lab:34s} {w:3d}/{n:<4d} {rel:9.2f}% {p:11.4g}')
        out['claims'].append(dict(label=lab, a=f'{a}.{fa}', b=f'{b}.{fb}',
                                  better=w, n=n, rel_median_pct=round(rel, 2),
                                  p_one_sided=p))
        if (w, n) != (w_doc, n_doc):
            bad.append((lab, 'count', f'{w}/{n}', f'{w_doc}/{n_doc}'))
        if abs(rel - rel_doc) > 0.005:
            bad.append((lab, 'rel', f'{rel:.2f}%', f'{rel_doc:.2f}%'))

    same, tot = check_mirror(idx, cells)
    print(f'\nG-MIRROR 상속 확인  SHIP-K1M3.e_med3 == SHIP-K3.e_m0 : {same}/{tot}')
    out['mirror'] = {'same': same, 'n': tot}
    if same != tot or tot == 0:
        bad.append(('G-MIRROR', 'identity', f'{same}/{tot}', f'{tot}/{tot}'))

    with open(os.path.join(HERE, 'stageA_reagg.json'), 'w') as fh:
        json.dump(out, fh, ensure_ascii=False, indent=1)

    if bad:
        raise AssertionError('PREREG §2.3 표기와 동결 자료가 어긋난다: ' + repr(bad))
    print('\n✅ PREREG §2.3 재현 — 전건 일치 (런 0회)')
    print('⛔등급 EXPLORATORY 고정 · ⛔확증 승격 없음 · ⛔B-2/M-8b 판정 갱신 없음')
    return 0


if __name__ == '__main__':
    sys.exit(main())
