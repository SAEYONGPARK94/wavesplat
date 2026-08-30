"""T-2(STAT) 전제 재도출 — ★**PREREG 의 모든 수를 동결 결과 파일에서 낳는다.**

⛔**추정기 0회 · 새 런 0회.** 읽는 것은 동결 jsonl 셋과 `zero_ref` 캐시뿐이다.
⛔손복사 상수 0건: `PREREG.md` 가 인용하는 수는 전부 이 파일이 `premise_t2.json` 에 싣고,
문서와 어긋나면 `AssertionError` 다 (선례 `B2_REAL/depth_recheck.py`·`M10/premise_m10.py`).

산출: `premise_t2.json`
"""
import json
import os
import statistics as st
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import gates_t2 as G                                                    # noqa: E402

OUT = os.path.join(HERE, 'premise_t2.json')


def _rows(path, cell):
    out = []
    for line in open(path):
        r = json.loads(line)
        if 'error' in r or r.get('cell') != cell:
            continue
        r['arm'] = cell
        out.append(r)
    return out


def _vs_zero(rows, lo, hi):
    ks, rel, w = [], [], 0
    for r in rows:
        v = G.fc(r, lo, hi)
        z = G.fc_zero(r['record'], r['win'], lo, hi)
        if v is None or not z:
            continue
        ks.append(G.key(r))
        rel.append(v / z)
        w += (v < z)
    n = len(ks)
    return dict(n=n, better=w, ratio_median=round(st.median(rel), 4) if rel else None,
                p_one_sided=G.sign_test_one_sided(w, n) if n else None)


def _paired(a_rows, b_rows):
    a = {G.key(r): G.fc(r) for r in a_rows}
    b = {G.key(r): G.fc(r) for r in b_rows}
    ks = sorted(set(a) & set(b))
    w = sum(1 for k in ks if a[k] < b[k])
    return dict(n=len(ks), better=w, p_one_sided=G.sign_test_one_sided(w, len(ks)),
                rel_median_pct=round(st.median([(a[k] - b[k]) / b[k] * 100 for k in ks]), 3))


def _wall(path, cell):
    v = [r['wall_s'] for r in _rows(path, cell) if r.get('wall_s')]
    return dict(n=len(v), median_s=round(st.median(v), 2),
                min_s=round(min(v), 2), max_s=round(max(v), 2),
                total_min=round(sum(v) / 60.0, 1))


def main():
    b3 = os.path.join(G.B3, 'results_b3.jsonl')
    b7b = os.path.join(G.B7, 'results_b7b.jsonl')
    m4 = [os.path.join(G.ROOT, 'tmp', 'M4_CAPACITY_2026-08-28', f'results_m4_{i}.jsonl')
          for i in range(4)]

    SF, SR = _rows(b3, 'FROZEN'), _rows(b3, 'REFIT2')
    PF = _rows(b7b, 'PROP-B7B')
    SB = _rows(b7b, 'SHIP-B7B')

    p = dict(
        _what='T-2(STAT) 전제 — 동결 자료 재도출. 추정기 0회.',
        n_records=len(G.RECORDS), n_windows=len(G.WINDOWS), n_pairs=G.N_PAIRS,
        lead_fc=list(G.LEAD_FC), lead_now=list(G.LEAD_NOW),
        thresholds=G.thresholds(), mde=G.mde(), n_refit_expected=G.N_REFIT_EXPECTED,
        # ── ★핵심 전제 셋 (예보 지평, lead 2–14, 참 제로필드 대비)
        ship_frozen_vs_zero=_vs_zero(SF, *G.LEAD_FC),
        prop_frozen_vs_zero=_vs_zero(PF, *G.LEAD_FC),
        ship_refit_vs_zero=_vs_zero(SR, *G.LEAD_FC),
        ship_refit_paired=_paired(SR, SF),
        # ── 서술 전용: 나우캐스트에서는 두 팔이 갈린다(⛔이 배치가 설명하지 않는다)
        nowcast_ship_frozen=_vs_zero(SF, *G.LEAD_NOW),
        nowcast_prop_frozen=_vs_zero(PF, *G.LEAD_NOW),
        # ── 앵커 가능성: `B-7b` 의 SHIP 팔이 `B-3` FROZEN 을 비트 재현했는가
        anchor_b7b_reproduces_b3=all(
            a.get('per_lead') == b.get('per_lead')
            for a, b in zip(sorted(SB, key=G.key), sorted(SF, key=G.key))),
        # ── 비용 실측 (⛔추정 금지)
        cost=dict(ship_frozen=_wall(b3, 'FROZEN'), ship_refit2=_wall(b3, 'REFIT2'),
                  prop_frozen=_wall(b7b, 'PROP-B7B')),
        refit_calls_b3=sorted({r.get('refit_calls') for r in SR}),
    )

    # ── 용량 축 사전점검 (사용자 질문: "용량 × 재적합 상호작용") — ⛔런 0회
    rows = []
    for f in m4:
        if os.path.exists(f):
            for line in open(f):
                r = json.loads(line)
                if 'error' not in r:
                    rows.append(r)
    by = {(r['arm'], r['unit'], r['est_seed']): r['curve'] for r in rows if 'curve' in r}
    cells = sorted({(u, s) for (a, u, s) in by})
    cap = {}
    for A in ('K-8', 'K-15', 'K-60', 'K-128', 'K-128-D0', 'K-MAX'):
        r0, r14, w, n = [], [], 0, 0
        for (u, s) in cells:
            a, b = by.get((A, u, s)), by.get(('SHIP-K', u, s))
            if not a or not b:
                continue
            x0, x14 = a[0] / b[0], a[14] / b[14]
            r0.append(x0)
            r14.append(x14)
            n += 1
            w += (x14 < x0)
        if n:
            cap[A] = dict(n=n, ratio_lead0=round(st.median(r0), 3),
                          ratio_lead14=round(st.median(r14), 3),
                          steeper_with_lead=w,
                          p_one_sided=round(G.sign_test_one_sided(w, n), 5))
    p['capacity_lead_slope'] = cap
    p['capacity_note'] = ('★M-4 동결 곡선. `ratio` = 팔/SHIP-K 의 nRMSE 비. 용량 손해가 '
                          '**lead 와 함께 커지는가**를 본다(= "갱신 없이 더 표류한다" 가설). '
                          '⛔전 계단에서 lead14 비가 lead0 비보다 **작거나 같다** ⟹ 가설 미지지.')

    json.dump(p, open(OUT, 'w'), ensure_ascii=False, indent=1)

    # ── ★PREREG 가 인용하는 수와의 대조 (어긋나면 죽는다)
    assert p['ship_frozen_vs_zero']['better'] == 16, p['ship_frozen_vs_zero']
    assert p['prop_frozen_vs_zero']['better'] == 16, p['prop_frozen_vs_zero']
    assert p['ship_refit_vs_zero']['better'] == 32, p['ship_refit_vs_zero']
    assert p['ship_refit_paired']['better'] == 36, p['ship_refit_paired']
    assert p['nowcast_ship_frozen']['better'] == 33 and p['nowcast_prop_frozen']['better'] == 15
    assert p['anchor_b7b_reproduces_b3'] is True
    assert p['refit_calls_b3'] == [G.N_REFIT_EXPECTED]
    assert all(v['ratio_lead14'] <= v['ratio_lead0'] for v in cap.values()), cap
    assert all(v['p_one_sided'] > G.ALPHA for v in cap.values()), cap
    print(json.dumps({k: p[k] for k in (
        'ship_frozen_vs_zero', 'prop_frozen_vs_zero', 'ship_refit_vs_zero',
        'ship_refit_paired', 'nowcast_ship_frozen', 'nowcast_prop_frozen',
        'thresholds', 'mde', 'cost', 'capacity_lead_slope')}, ensure_ascii=False, indent=1))
    print('premise_t2: PASS ->', OUT)


if __name__ == '__main__':
    main()
