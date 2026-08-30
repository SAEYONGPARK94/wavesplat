"""T3-NOW 전제 재도출 — ★**`PREREG.md` 의 모든 수를 동결 결과 파일에서 낳는다.**

⛔**추정기 0회 · 새 런 0회 · GPU 0회.** 읽는 것은 동결 jsonl 둘과 `zero_ref` 캐시뿐이다:
  · `tmp/T2_STATIONARITY_2026-08-30/results_t2.jsonl`  (144행, 2026-08-30 동결)
  · `tmp/M8_LADDER_2026-08-28/results_m8_*.jsonl`      (432행, 2026-08-28 동결)
  · `tmp/B3_REFIT_2026-08-23/true_zero_perlead.json`   (참 제로필드 캐시)

⛔손복사 상수 0건: `PREREG.md` 가 인용하는 수는 전부 이 파일이 `premise_t3.json` 에 싣고,
문서와 어긋나면 **`AssertionError`** 다 (선례 `T2/premise_t2.py` · `B2_REAL/depth_recheck.py`).

산출: `premise_t3.json`
"""
import glob
import json
import os
import statistics as st
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import gates_t3 as G                                                    # noqa: E402

OUT = os.path.join(HERE, 'premise_t3.json')
T2_RES = G.T2_RESULTS
M8_GLOB = os.path.join(G.M8, 'results_m8_*.jsonl')


# ── 로더 (⛔외부 파일이 비면 조용히 넘어가지 않고 죽는다) ────────────────────
def _t2():
    rows = [json.loads(l) for l in open(T2_RES)]
    rows = [r for r in rows if 'error' not in r]
    assert len(rows) == 144, f'⛔T2 동결 144행이 아니다: {len(rows)}'
    return rows


def _m8():
    files = sorted(glob.glob(M8_GLOB))
    assert files, f'⛔M8 동결 자료 부재: {M8_GLOB}'
    rows = [json.loads(l) for f in files for l in open(f)]
    rows = [r for r in rows if 'error' not in r]
    assert len(rows) == 432, f'⛔M8 동결 432행이 아니다: {len(rows)}'
    return rows


def _win_mean(per_lead, lo, hi):
    v = [x for k, x in per_lead.items() if lo <= float(k) <= hi and x is not None]
    return sum(v) / len(v) if v else None


def _vs_zero_t2(rows, arm, lo, hi):
    """`T2` 행을 참 제로필드에 대고 채점한다 (⛔고정 상수 아님 — `zero_ref.py` 가 낸다)."""
    from zero_ref import zero_ref_perlead
    w, rel, n = 0, [], 0
    for r in rows:
        if r['arm'] != arm:
            continue
        v = _win_mean(r['per_lead'], lo, hi)
        z = _win_mean({str(k): x for k, x in
                       zero_ref_perlead(r['record'], win=r['win'],
                                        noise=G.NOISE).items()}, lo, hi)
        if v is None or not z:
            continue
        n += 1
        rel.append(v / z)
        w += (v < z)
    return dict(n=n, better=w, ratio_median=round(st.median(rel), 4),
                p_one_sided=G.sign_test_one_sided(w, n))


def _paired_t2(rows, arm, ref, lo, hi):
    a = {(r['record'], r['win']): _win_mean(r['per_lead'], lo, hi)
         for r in rows if r['arm'] == arm}
    b = {(r['record'], r['win']): _win_mean(r['per_lead'], lo, hi)
         for r in rows if r['arm'] == ref}
    ks = sorted(set(a) & set(b))
    w = sum(1 for k in ks if a[k] < b[k])
    return dict(n=len(ks), better=w, p_one_sided=G.sign_test_one_sided(w, len(ks)),
                rel_median_pct=round(st.median(
                    [(a[k] - b[k]) / b[k] * 100 for k in ks]), 3))


def _wall(rows, arm, field='arm'):
    v = [r['wall_s'] for r in rows if r.get(field) == arm and r.get('wall_s')]
    return dict(n=len(v), median_s=round(st.median(v), 2), min_s=round(min(v), 2),
                max_s=round(max(v), 2), total_min=round(sum(v) / 60.0, 1))


def main():
    t2, m8 = _t2(), _m8()
    LEAD0 = (0.0, 0.0)
    NOW01 = (0.0, 1.0)          # ⚠`T2 RESULT §5` 가 쓴 구간 — **다른 양이다**

    # ── ★P1. 이 배치가 공격하는 수 (lead 0 단독, 참 제로필드 대비) ───────────
    p = dict(
        _what='T3-NOW 전제 — 동결 자료 재도출. 추정기 0회 · GPU 0회.',
        _sources=dict(t2=os.path.relpath(T2_RES, G.ROOT), m8=os.path.relpath(M8_GLOB, G.ROOT),
                      zero_ref='tmp/B3_REFIT_2026-08-23/true_zero_perlead.json'),
        n_records=len(G.RECORDS), n_windows=len(G.WINDOWS), n_pairs=G.N_PAIRS,
        lead=G.LEAD, seed_base=G.SEED, noise=G.NOISE,
        thresholds=G.thresholds(), mde=G.mde(), mde_single=G.mde_single(),
        expected_runs=G.expected_runs(), expected_rows=G.expected_rows(),
        offsets=list(G.OFFSETS), ens_k=G.ENS_K,
        ship_lead0_vs_zero=_vs_zero_t2(t2, 'SHIP-FROZEN', *LEAD0),
        prop_lead0_vs_zero=_vs_zero_t2(t2, 'PROP-FROZEN', *LEAD0),
        ship_lt_prop_lead0=_paired_t2(t2, 'SHIP-FROZEN', 'PROP-FROZEN', *LEAD0),
        # ⚠**섞지 마라** — `T2 RESULT §5` 의 표는 lead 0–1 **평균**이다.
        t2_nowcast_0_1=dict(
            ship=_vs_zero_t2(t2, 'SHIP-FROZEN', *NOW01),
            prop=_vs_zero_t2(t2, 'PROP-FROZEN', *NOW01),
            note='⛔`T2 RESULT §5`(33/36 · 15/36)는 lead 0–1 평균이다. 이 배치는 '
                 'lead 0 **단독**(33/36 · 17/36)이고 두 수를 섞으면 안 된다.'),
    )

    # ── ★P2. 장치 민감도 — **제안 경로는 환경에 불변이 아니다** (신규 관찰) ──
    t2w3 = {r['record']: r['lead0'] for r in t2
            if r['arm'] == 'PROP-FROZEN' and r['win'] == 3}
    t2b3 = {r['record']: r['lead0'] for r in t2
            if r['arm'] == 'SHIP-FROZEN' and r['win'] == 3}
    m8p = {r['record']: r['nowcast'] for r in m8
           if r['arm'] == 'PROP' and r['est_seed'] == G.SEED}
    m8b = {r['record']: r['nowcast'] for r in m8
           if r['arm'] == 'BANK' and r['est_seed'] == G.SEED}
    recs = sorted(t2w3)
    p['device_sensitivity'] = dict(
        t2_cuda_available=sorted({r['cuda_available'] for r in t2}),
        m8_cuda_available=sorted({r['cuda_available'] for r in m8}),
        bank_bit_equal=sum(1 for r in recs if t2b3[r] == m8b[r]),
        prop_bit_equal=sum(1 for r in recs if t2w3[r] == m8p[r]),
        n=len(recs),
        prop_max_abs_rel_pct=round(100 * max(
            abs(t2w3[r] - m8p[r]) / m8p[r] for r in recs), 1),
        note='★`BANK` 는 두 환경에서 **비트 동일**하고(`_beamform_top` 이 argmax 만 '
             '장치 경계를 넘고 나머지는 CPU float64) `PROP` 은 **한 셀도 일치하지 '
             '않는다**(diffusion 표집이 장치에 민감). ⟹ ①이 배치는 **한 환경**에서 '
             '돌아야 하고 그 환경은 `T2` 앵커가 정한다 ②`M8` 의 사다리 계수는 '
             '이 배치의 팔에 **전이되지 않는다** ⟹ `M8` 은 앵커가 아니다.')

    # ── ★P3. 시드 산포 — 앙상블(분산 축소 장치)의 기제 근거 ─────────────────
    by = {}
    for r in m8:
        by.setdefault((r['arm'], r['record']), {})[r['est_seed']] = r['nowcast']
    def _cv(arm):
        vs = []
        for rec in sorted(G.RECORDS):
            v = list(by[(arm, rec)].values())
            vs.append(st.pstdev(v) / abs(st.mean(v)))     # ⚠모집단 std (== `M8b §1`)
        return dict(cv_median_pct=round(100 * st.median(vs), 2),
                    cv_max_pct=round(100 * max(vs), 1), n=len(vs))
    p['seed_cv'] = {a: _cv(a) for a in
                    ('BANK', 'BANK-K1', 'MMSE-PROP', 'PROP-LS', 'PROP-LS-APG', 'PROP')}
    p['seed_cv_note'] = ('★`M8b RESULT §1` 이 공표한 값의 재도출(모집단 std/평균). '
                         'bank 1.04% vs PROP 13.57% ⟹ **분산 축소 장치의 값어치가 '
                         '약 13배 갈린다**. ⛔이것은 기제 근거이지 결과 예측이 아니다.')

    # ── ★P4/P5. `M8` 의 두 고리 — 이 배치가 n=36 으로 다시 묻는 칸 ───────────
    med = {k: st.median(list(v.values())) for k, v in by.items()}
    def _link(a, b):
        w = sum(1 for r in G.RECORDS if med[(b, r)] > med[(a, r)])
        return dict(n=len(G.RECORDS), worse=w,
                    p_one_sided=round(G.sign_test_one_sided(w, len(G.RECORDS)), 6),
                    rel_median_pct=round(st.median(
                        [(med[(b, r)] - med[(a, r)]) / med[(a, r)] * 100
                         for r in G.RECORDS]), 2))
    p['m8_l1_ensemble'] = _link('BANK', 'BANK-K1')
    p['m8_l6_varpro'] = _link('PROP-LS-APG', 'PROP')
    p['m8_holm_floor_n12'] = dict(
        n=12, family=6, alpha_first=round(0.05 / 6, 6),
        min_wins=next(w for w in range(13) if G.sign_test_one_sided(w, 12) <= 0.05 / 6),
        note='★n=12·6고리 Holm 의 첫 문턱이 11/12 ⟹ ***10/12 이하는 어떤 자료에서도 '
             '기각 불가*** 였다(`M8b/AMENDMENT_2.md`). 이 배치의 n=36·가족3 Holm 첫 '
             '문턱은 25/36 이고 `M8` 이 본 크기(9/12=0.75 · 10/12=0.83)는 그 설계의 '
             'MDE(q80=0.742) **안**에 있다.')

    # ── P6. 비용 — ⛔추정 아님, 동결 로그 실측 ───────────────────────────────
    p['cost'] = dict(
        t2_with_update_loop={a: _wall(t2, a) for a in
                             ('SHIP-FROZEN', 'PROP-FROZEN')},
        m8_init_only={a: _wall(m8, a) for a in
                      ('BANK', 'BANK-K1', 'PROP', 'PROP-LS-APG')},
        note='⚠`T2` 단가는 lead 1–14 `update()` 루프 + 6샤드 경합을 포함하고, `M8` 단가는 '
             '`initialize`+`predict_field` 만이지만 **CPU 전용 환경**이다. 이 배치는 '
             '`update()` 0회 · CUDA 가용 환경 ⟹ 두 값 사이가 기대 범위이고 '
             '★**스모크가 실측한다**(⛔이 표를 최종 비용으로 인용하지 마라).')

    # ── P7. 앵커 성립성 (⛔런 0회로 확인) ────────────────────────────────────
    anc = G.anchor_index()
    p['anchor_cells'] = {a: sum(1 for (x, _, _) in anc if x == a)
                         for a in ('SHIP-FROZEN', 'PROP-FROZEN',
                                   'SHIP-REFIT2', 'PROP-REFIT2')}
    p['anchor_note'] = ('★`T2` 는 `B-3 FROZEN`·`B-7b PROP-B7B` 를 36/36 비트 재현했다'
                        '(`T2 RESULT §4 G-ANCHOR`) ⟹ 이 배치가 `T2` 를 재현하면 '
                        '**세 배치 사슬이 닫힌다**. ⛔`M8` 은 장치 스탬프가 달라 앵커가 '
                        '아니다(P2).')

    # ── P8. ★결과 전에 **보지 않은 것** (= 이 배치의 blind 범위) ─────────────
    p['not_computed'] = [
        '⛔`PROP` 계열의 **장 median-of-3**(= `e_med3`) 을 어떤 형태로도 계산하지 않았다 '
        '— 그것이 이 배치의 신규 측정이다.',
        '⛔`M8` 의 시드-**지표** median 을 `PROP` 단일 런과 비교하지 않았다(C1 의 대용물).',
        '⛔`PROP-V0`(= varpro 0, adaptive_n True) 를 이 환경에서 한 번도 돌리지 않았다.',
        '⛔멤버 간 점별 분산(`spread_rms`)은 동결 자료에 존재하지 않는다(장을 저장한 '
        '배치가 없다) ⟹ E3 은 전적으로 새 측정이다.',
    ]
    p['exposed_before_freeze'] = [
        '✅`PROP-V400M3.e_m0` 는 **완전히 사전 노출**돼 있다 — 그것이 `T2 PROP-FROZEN` '
        'lead0 이고 위 P1 이 그 집계(17/36 · 비 1.0365)를 싣는다. ⟹ C1 의 **기준 팔**은 '
        '알려져 있고 **처치 팔**은 알려져 있지 않다.',
        '✅`SHIP-K3.e_m0` 도 사전 노출(33/36 · 비 0.7288).',
        '✅`M8` 의 사다리 계수(L1 5/12 · L6 9/12 · 시드 CV)를 읽었고, **C2 의 방향은 '
        '거기서 왔다**(⚠다른 장치 환경 = 다른 양).',
    ]

    json.dump(p, open(OUT, 'w'), ensure_ascii=False, indent=1)

    # ── ★`PREREG.md` 가 인용하는 수와의 대조 (어긋나면 죽는다) ───────────────
    assert p['ship_lead0_vs_zero']['better'] == 33, p['ship_lead0_vs_zero']
    assert p['prop_lead0_vs_zero']['better'] == 17, p['prop_lead0_vs_zero']
    assert p['ship_lead0_vs_zero']['ratio_median'] == 0.7288
    assert p['prop_lead0_vs_zero']['ratio_median'] == 1.0365
    assert p['ship_lt_prop_lead0']['better'] == 33
    assert p['ship_lt_prop_lead0']['rel_median_pct'] == -25.751
    assert p['t2_nowcast_0_1']['ship']['better'] == 33
    assert p['t2_nowcast_0_1']['prop']['better'] == 15
    d = p['device_sensitivity']
    assert d['bank_bit_equal'] == 12 and d['prop_bit_equal'] == 0, d
    assert d['t2_cuda_available'] == [True] and d['m8_cuda_available'] == [False], d
    assert d['prop_max_abs_rel_pct'] == 245.1, d
    assert p['seed_cv']['PROP-LS-APG']['cv_median_pct'] == 9.21
    assert p['seed_cv']['BANK']['cv_median_pct'] == 1.04
    assert p['seed_cv']['PROP']['cv_median_pct'] == 13.57
    assert p['seed_cv']['BANK']['cv_max_pct'] == 5.7
    assert p['seed_cv']['PROP']['cv_max_pct'] == 67.3
    assert p['m8_l1_ensemble']['worse'] == 5 and \
        p['m8_l1_ensemble']['rel_median_pct'] == 0.0, p['m8_l1_ensemble']
    assert p['m8_l6_varpro']['worse'] == 9 and \
        p['m8_l6_varpro']['p_one_sided'] == 0.072998, p['m8_l6_varpro']
    assert p['m8_holm_floor_n12']['min_wins'] == 11
    assert p['thresholds']['HOLM_MIN_WINS'] == [25, 25, 24], p['thresholds']
    assert p['thresholds']['NOMINAL'] == 24 and p['thresholds']['NOT_EARN'] == 12
    assert p['mde']['q80'] == 0.742 and p['mde_single']['q80'] == 0.716
    assert p['expected_runs'] == 360 and p['expected_rows'] == 144
    assert p['offsets'] == [0, 1000, 2000] and p['ens_k'] == 3
    assert all(v == 36 for v in p['anchor_cells'].values()), p['anchor_cells']

    print(json.dumps({k: p[k] for k in (
        'ship_lead0_vs_zero', 'prop_lead0_vs_zero', 'ship_lt_prop_lead0',
        'device_sensitivity', 'seed_cv', 'm8_l1_ensemble', 'm8_l6_varpro',
        'm8_holm_floor_n12', 'thresholds', 'mde', 'cost')},
        ensure_ascii=False, indent=1))
    print('premise_t3: PASS ->', OUT)


if __name__ == '__main__':
    main()
