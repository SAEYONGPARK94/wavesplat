"""`T4-GEOM` RESULT.md **재도출 검증기** — ⛔추정기 0회 · GPU 0회 · 동결 파일만 읽는다.

    python3 tmp/T4_GEOMETRY_2026-08-30/recheck_t4.py

읽는 것은 **동결 자료 둘뿐**이다:
  * `results_t4_0.jsonl` · `results_t4_1.jsonl` · `results_t4_2.jsonl` (합 288행)
  * `verdict_t4.json` (`analyze_t4.analyze()` 출력)

하는 일은 **3중 대조**다 (본보기 = `tmp/T3_NOWCAST_2026-08-30/recheck_t3.py`,
`tmp/STATUS_2026-08-28/f1_table.py` 의 *"jsonl == verdict == 핀"*):

  ① **재도출** — 288행에서 `math.comb` 부호검정·중앙값·문턱을 **처음부터 다시** 계산한다
     (⛔`gates_t4` 를 임포트하지 않는다 — 같은 코드를 두 번 부르면 대조가 아니다).
  ② **verdict 대조** — ①이 `verdict_t4.json` 의 값과 같은가.
  ③ **★문서 대조** — `RESULT.md` 에 그 수가 **실제로 그 글자로** 적혀 있는가.

⛔하나라도 어긋나면 `AssertionError` 다.

⚠**검증 범위(= 주장 범위)**: `RESULT.md` 가 인용하는 수 중 **다른 파일 출처**
(`premise_t4.json` · 동결 `B-12` · `verdict_m10.json` · `W-15` · `VOID_4shard/README.md` ·
`FROZEN_md5.txt` · 러너 로그 · 방벽 출력)는 이 스크립트가 **재도출하지 않는다**.
그 목록을 §UNVERIFIED 로 **찍어서 보여준다** — 침묵하지 않는 것이 요점이다.
"""
import glob
import json
import math
import os
import statistics as st
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
VERDICT_PATH = os.path.join(HERE, 'verdict_t4.json')
RESULT_PATH = os.path.join(HERE, 'RESULT.md')

N_EP = 48
ALPHA = 0.05
N_FAMILY = 2                      # PRIMARY-A(LEADFIELD) · SECONDARY-B(DECAY)
ARMS = ('SHIP', 'SPEC25')
LEADS = (0.0, 0.2, 0.5, 1.0)
PRIMARY_LEAD = 1.0
BAND = (0.5, 1.0)

ROWS = []
for _p in sorted(glob.glob(os.path.join(HERE, 'results_t4_*.jsonl'))):
    with open(_p) as _fh:
        for _l in _fh:
            _l = _l.strip()
            if _l:
                ROWS.append(json.loads(_l))
V = json.load(open(VERDICT_PATH))
DOC = open(RESULT_PATH).read() if os.path.exists(RESULT_PATH) else None

FAIL = []
PINS = []          # (라벨, 재도출값, verdict값 or None, 문서에 있어야 하는 글자 or None)


# ── ① 재도출 (⛔gates_t4 임포트 금지 — 독립 구현) ────────────────────────────
def sign_p(w, n):
    return sum(math.comb(n, k) for k in range(w, n + 1)) / 2 ** n


def min_w(n, alpha):
    return next(w for w in range(n + 1) if sign_p(w, n) <= alpha)


def mde_q80(n, thresh, power=0.80):
    """⛔이분법 — `gates_t4` 와 **다른 구현**(그쪽도 이분법이지만 여기서 새로 쓴다)."""
    lo, hi = 0.5, 1.0
    for _ in range(80):
        mid = (lo + hi) / 2
        pw = sum(math.comb(n, k) * mid ** k * (1 - mid) ** (n - k)
                 for k in range(thresh, n + 1))
        if pw >= power:
            hi = mid
        else:
            lo = mid
    return hi


def med_by_ep(arm, getter):
    """에피소드마다 시드 3개의 **중앙값**(= 짝의 단위)."""
    d = {}
    for r in ROWS:
        if 'error' in r or r.get('arm') != arm:
            continue
        v = getter(r)
        if v is None or v != v:
            continue
        d.setdefault(r['episode'], []).append(float(v))
    return {e: st.median(vs) for e, vs in d.items()}


def lead_get(field, lead_tp):
    key = repr(float(lead_tp))

    def _g(r):
        d = r.get(field) or {}
        return d.get(key, d.get(str(float(lead_tp))))
    return _g


def band_get(field, leads=BAND):
    """밴드 합성 규약(결과 전 고정): rung 들의 **RMS**. ⛔중앙값의 비가 아니다."""
    def _g(r):
        d = r.get(field) or {}
        vs = []
        for L in leads:
            v = d.get(repr(float(L)), d.get(str(float(L))))
            if v is None or v != v:
                return None
            vs.append(float(v))
        return (sum(v * v for v in vs) / len(vs)) ** 0.5
    return _g


def cmp_(arm_a, get_a, arm_b, get_b):
    a, b = med_by_ep(arm_a, get_a), med_by_ep(arm_b, get_b)
    ks = sorted(set(a) & set(b))
    w = sum(1 for k in ks if a[k] < b[k])
    rel = [(a[k] - b[k]) / b[k] for k in ks if b[k]]
    return dict(better=w, n=len(ks), ties=sum(1 for k in ks if a[k] == b[k]),
                p=sign_p(w, len(ks)),
                rel_pct=round(100.0 * st.median(rel), 3) if rel else None,
                med_a=round(st.median([a[k] for k in ks]), 5),
                med_b=round(st.median([b[k] for k in ks]), 5),
                losers=[k for k in ks if not a[k] < b[k]])


def pin(label, got, want_verdict=None, want_doc=None, tol=0.0):
    """①재도출 == ②verdict == ③문서. ⛔셋 중 하나라도 어긋나면 기록한다."""
    PINS.append((label, got, want_verdict, want_doc))
    if want_verdict is not None:
        ok = (abs(got - want_verdict) <= tol
              if isinstance(got, float) or isinstance(want_verdict, float)
              else got == want_verdict)
        if not ok:
            FAIL.append(f'[verdict] {label}: 재도출 {got!r} != verdict {want_verdict!r}')
    if want_doc is not None and DOC is not None and want_doc not in DOC:
        FAIL.append(f'[문서] {label}: RESULT.md 에 {want_doc!r} 이(가) 없다')


# ── 0. 자료 무결성 ──────────────────────────────────────────────────────────
pin('행 수', len(ROWS), V['n_rows'], '288행')
pin('유효 행 수', sum(1 for r in ROWS if 'error' not in r), V['n_good'], None)
pin('에러 0', sum(1 for r in ROWS if 'error' in r), 0, '에러 **0**')
pin('팔별 행수', {a: sum(1 for r in ROWS if r['arm'] == a) for a in ARMS},
    {a: 144 for a in ARMS}, '`SHIP` 144 · `SPEC25` 144')
pin('에피소드 수', len({r['episode'] for r in ROWS}), N_EP, 'ep 48')
pin('추정기 시드', sorted({r['seed'] for r in ROWS}), [42, 43, 44], '42·43·44')
pin('층 × 성분수', Counter((r['bf'], r['ncomp']) for r in ROWS),
    Counter({('BF4', 15): 72, ('BF4', 30): 72, ('BF5', 15): 72, ('BF5', 30): 72}),
    '15성분 24 + 30성분 24')
pin('게이트 수', len(V['gates']), 17, '17/17')
pin('게이트 PASS 수', sum(1 for g in V['gates'] if g['pass']), len(V['gates']), '17/17')
pin('차단 게이트 수', sum(1 for g in V['gates'] if g['blocking']), 14,
    '차단 14 + 범위축소 3')
pin('범위축소 게이트 수', sum(1 for g in V['gates'] if not g['blocking']), 3, None)
pin('차단 실패', V['blocked_by'], [], None)
pin('비교자 유효', V['comparator_ok'], True, None)

# ── 1. 문턱 · MDE (⛔손복사 아님 — math.comb 이 낸다) ────────────────────────
nominal = min_w(N_EP, ALPHA)
confirm = min_w(N_EP, ALPHA / N_FAMILY)
pin('NOMINAL', nominal, V['thresholds']['NOMINAL'], 'NOMINAL 31')
pin('CONFIRM (가족 α=0.025)', confirm, V['thresholds']['CONFIRM'], 'CONFIRM 32')
pin('NOT_EARN', N_EP - nominal, V['thresholds']['NOT_EARN'], 'NOT_EARN 17')
pin('MDE q80', round(mde_q80(N_EP, confirm), 4), round(V['thresholds']['q80'], 4),
    '0.7109', tol=1e-9)
pin('바닥 p', float('%.3e' % sign_p(N_EP, N_EP)),
    float('%.3e' % V['thresholds']['floor_p']), '3.553e-15', tol=0.0)

# ── 2. PRIMARY-A · G-CMP · S-1 · 밴드 민감도 ────────────────────────────────
gp = lambda L: lead_get('lead_rmse', L)            # noqa: E731
gz = lambda L: lead_get('lead_zero', L)            # noqa: E731

pr = cmp_('SHIP', gp(PRIMARY_LEAD), 'SPEC25', gp(PRIMARY_LEAD))
pin('PRIMARY better', pr['better'], V['primary']['better'], '**47/48**')
pin('PRIMARY n', pr['n'], V['primary']['n'], None)
pin('PRIMARY p', pr['p'], V['primary']['p_one_sided'], '1.74e-13', tol=1e-20)
pin('PRIMARY rel%', pr['rel_pct'], V['primary']['rel_median_pct'], '−46.299%', tol=1e-9)
pin('PRIMARY 중앙 SHIP', pr['med_a'], V['ladder']['1.0']['ship_median'], '**0.13012**',
    tol=1e-12)
pin('PRIMARY 중앙 SPEC25', pr['med_b'], V['ladder']['1.0']['spec_median'],
    '**0.22352**', tol=1e-12)
pin('PRIMARY 패배 에피소드', pr['losers'], ['BF5_15comp_seed1635_uniform_ra_rf_ds'],
    '`BF5_15comp_seed1635`')

cm = cmp_('SPEC25', gp(PRIMARY_LEAD), 'SPEC25', gz(PRIMARY_LEAD))
pin('G-CMP better', cm['better'], V['comparator']['better'], '**34/48**')
pin('G-CMP p', cm['p'], V['comparator']['p_one_sided'], '0.00276', tol=1e-15)
pin('G-CMP rel%', cm['rel_pct'], V['comparator']['rel_median_pct'], '−8.017%', tol=1e-9)
pin('G-CMP ≥ CONFIRM', cm['better'] >= confirm, True, None)

ex = cmp_('SHIP', gp(PRIMARY_LEAD), 'SHIP', gz(PRIMARY_LEAD))
pin('S-1 better', ex['better'], V['existence']['better'], '**46/48**')
pin('S-1 p', ex['p'], V['existence']['p_one_sided'], '4.18e-12', tol=1e-18)
pin('S-1 rel%', ex['rel_pct'], V['existence']['rel_median_pct'], '−49.249%', tol=1e-9)

bd = cmp_('SHIP', band_get('lead_rmse'), 'SPEC25', band_get('lead_rmse'))
pin('밴드 RMS better', bd['better'], V['band_sensitivity']['better'], '**46/48**')
pin('밴드 RMS rel%', bd['rel_pct'], V['band_sensitivity']['rel_median_pct'],
    '−45.524%', tol=1e-9)
pin('밴드 RMS p', bd['p'], V['band_sensitivity']['p_one_sided'], None, tol=1e-18)

# ── 3. SECONDARY-B `DECAY` (⛔반대 방향에 p 를 붙이지 않는다) ────────────────
ratios = {}
for L in (0.0, PRIMARY_LEAD):
    s, b = med_by_ep('SHIP', gp(L)), med_by_ep('SPEC25', gp(L))
    ratios[L] = {e: s[e] / b[e] for e in (set(s) & set(b)) if b[e]}
ks = sorted(set(ratios[0.0]) & set(ratios[PRIMARY_LEAD]))
wd = sum(1 for e in ks if ratios[PRIMARY_LEAD][e] > ratios[0.0][e])
pin('DECAY more (등록 방향)', wd, V['secondary_b_decay']['more'], '**12/48**')
pin('DECAY n', len(ks), V['secondary_b_decay']['n'], None)
pin('DECAY p (등록 방향)', sign_p(wd, len(ks)),
    V['secondary_b_decay']['p_one_sided'], '0.9999', tol=1e-12)
pin('DECAY r 중앙 @lead0', round(st.median([ratios[0.0][e] for e in ks]), 5),
    V['secondary_b_decay']['r_lead0_median'], '0.58554', tol=1e-12)
pin('DECAY r 중앙 @lead1.0',
    round(st.median([ratios[PRIMARY_LEAD][e] for e in ks]), 5),
    V['secondary_b_decay']['r_primary_median'], '0.53701', tol=1e-12)
# ⛔반대 방향(36/48)의 p 는 **계산하지 않는다** — 사후 방향 선택이다(RESULT §4).

# ── 4. 사다리 (⚠서술 전용 · ⛔rung 순위 지목 금지) ──────────────────────────
LADDER_DOC = {                     # (SHIP<SPEC25, rel%, SHIP중앙, SPEC중앙, ZERO중앙,
    0.0: ('43/48', '−41.446%', '0.12159', '0.20666', '0.24733', '47/48', '46/48'),
    0.2: ('44/48', '−46.095%', '0.12045', '0.21127', '0.2473', '47/48', '46/48'),
    0.5: ('46/48', '−46.511%', '0.12425', '0.21395', '0.24615', '43/48', '46/48'),
    1.0: ('47/48', '−46.299%', '0.13012', '0.22352', '0.24812', '34/48', '46/48'),
}
zero_meds = []
for L in LEADS:
    key = repr(float(L))
    lv = V['ladder'][key]
    d_ss, d_rel, d_sm, d_pm, d_zm, d_sz, d_shz = LADDER_DOC[L]
    ss = cmp_('SHIP', gp(L), 'SPEC25', gp(L))
    sz = cmp_('SPEC25', gp(L), 'SPEC25', gz(L))
    shz = cmp_('SHIP', gp(L), 'SHIP', gz(L))
    pin(f'사다리 {L} SHIP<SPEC25', ss['better'], lv['ship_vs_spec']['better'], d_ss)
    pin(f'사다리 {L} rel%', ss['rel_pct'], lv['ship_vs_spec']['rel_median_pct'],
        d_rel, tol=1e-9)
    pin(f'사다리 {L} p', ss['p'], lv['ship_vs_spec']['p_one_sided'], None, tol=1e-18)
    pin(f'사다리 {L} SHIP 중앙', ss['med_a'], lv['ship_median'], d_sm, tol=1e-12)
    pin(f'사다리 {L} SPEC25 중앙', ss['med_b'], lv['spec_median'], d_pm, tol=1e-12)
    pin(f'사다리 {L} ZERO 중앙', sz['med_b'], lv['zero_median'], d_zm, tol=1e-12)
    pin(f'사다리 {L} SPEC25<ZERO', sz['better'], lv['spec_vs_zero']['better'], d_sz)
    pin(f'사다리 {L} SHIP<ZERO', shz['better'], lv['ship_vs_zero']['better'], d_shz)
    zero_meds.append(sz['med_b'])
# ★F1 부류 오독 위험의 근거 두 수: 비교자 붕괴 47→34 와 ZERO 의 평탄성
pin('★비교자 SPEC25<ZERO 0.0→1.0', (V['ladder']['0.0']['spec_vs_zero']['better'],
                                    V['ladder']['1.0']['spec_vs_zero']['better']),
    (47, 34), '47 → 34')
pin('★ZERO 중앙 범위(네 rung)', (min(zero_meds), max(zero_meds)), (0.24615, 0.24812),
    '0.24615–0.24812')

# ── 5. 라벨 · 접기 (⛔결과 전 규칙을 다시 적용해 본다) ───────────────────────
if not V['comparator_ok']:
    lab = 'LEADFIELD-VOID'
elif pr['better'] >= confirm:
    lab = 'LEADFIELD-HOLDS'
elif pr['better'] >= nominal:
    lab = 'LEADFIELD-HOLDS-NOMINAL'
elif pr['better'] <= N_EP - nominal:
    lab = 'LEADFIELD-BREAKS'
else:
    lab = 'LEADFIELD-UNDECIDED'
pin('라벨(재적용)', lab, V['label'], '`T4-GEOM :: LEADFIELD-HOLDS`')
pin('접기', 'hold' if lab == 'LEADFIELD-HOLDS' else
    ('break' if lab == 'LEADFIELD-BREAKS' else 'undecided'), V['fold'], '`hold`')
pin('PRIMARY lead (Tp)', PRIMARY_LEAD, V['primary_lead_tp'], 'lead 1.0 Tp')
pin('rung 목록', list(LEADS), V['leads_tp'], '0.0/0.2/0.5/1.0 Tp')
pin('밴드 rung', list(BAND), V['band_lead_tp'], '밴드 RMS(0.5, 1.0)')
pin('선별 GO', V['screen']['go'], ['LEADTIME'], None)
pin('선별 NO-GO 수', len(V['screen']['no_go']), 9, None)

# ── 6. ★대조군·개입 계수기·되읽기 (행 필드에서 직접) ────────────────────────
pin('G-6 음성대조 비트동일', sum(1 for r in ROWS if r['neg_lead0_bitmatch'] is True),
    len(ROWS), '288/288')
pin('G-7 양성대조 셀 수', sum(1 for r in ROWS if r['pos_compose_relerr'] is not None),
    144, '0/144')
pin('G-7 최대 rel', float('%.1e' % max(r['pos_compose_relerr'] for r in ROWS
                                       if r['pos_compose_relerr'] is not None)),
    9.2e-15, '9.2e-15', tol=0.0)
pin('G-12 개입 계수(팔별)',
    sorted({(r['arm'], r['cap_forecast_steps'], r['cap_predict_future'])
            for r in ROWS}),
    [('SHIP', 0, 24), ('SHIP', 0, 72), ('SPEC25', 324000, 0), ('SPEC25', 756000, 0)],
    '756000')
pin('G-12 SPEC25 BF5 가지', 324000, 324000, '324000')
pin('G-9 lead 프레임 BF4',
    sorted({tuple(sorted(r['lead_frames'].items())) for r in ROWS if r['bf'] == 'BF4'}),
    [(('0.0', 0), ('0.2', 14), ('0.5', 35), ('1.0', 70))], 'BF4 0/14/35/70')
pin('G-9 lead 프레임 BF5',
    sorted({tuple(sorted(r['lead_frames'].items())) for r in ROWS if r['bf'] == 'BF5'}),
    [(('0.0', 0), ('0.2', 18), ('0.5', 45), ('1.0', 90))], 'BF5 0/18/45/90')
pin('G-13 원점 집합',
    sorted({(r['bf'], r['n_origins'], r['origin_first'], r['origin_last'],
             r['origin_stride_s']) for r in ROWS}),
    [('BF4', 18, 141, 226, 0.5), ('BF5', 6, 181, 206, 0.5)], 'BF4 18개 · BF5 6개')
pin('G-16 층별 원점', {b: sorted({r['n_origins'] for r in ROWS if r['bf'] == b})
                       for b in ('BF4', 'BF5')},
    {'BF4': [18], 'BF5': [6]}, '**BF4 18 / BF5 6**')
pin('G-5 채점 창',
    sorted({(r['bf'], r['n_scored'], r['t_first'], r['t_last']) for r in ROWS}),
    [('BF4', 160, 14.100000000000001, 30.0), ('BF5', 120, 18.1, 30.0)], None)
pin('G-15 mesh 분할', dict(Counter(r['mesh_verdict'] for r in ROWS)),
    {'DEGRADED': 102, 'ADEQUATE': 186}, '**DEGRADED 102 / ADEQUATE 186**')
pin('G-10 스탬프',
    sorted({(r['device'], r['cuda_available'], r['omp_threads']) for r in ROWS}),
    [('cpu', True, '1')], "`device='cpu'` · `cuda_available=True` · `omp_threads='1'`")
pin('G-10 code_md5 단일', sorted({r['code_md5'] for r in ROWS}),
    ['a876258d9d8e6084e546f2496331864f'], 'a876258d')
pin('prior_device 분포', dict(Counter(r['prior_device'] for r in ROWS)),
    {'cuda': 287, None: 1}, '287행이')
pin('prior_device None 행',
    [(r['arm'], r['episode'], r['seed']) for r in ROWS if r['prior_device'] is None],
    [('SPEC25', 'BF4_15comp_seed1581_uniform_ra_rf_ds', 42)], '`None`')

# ── 7. ★학습 경로 되읽기 (RESULT §6) ────────────────────────────────────────
ship = [r for r in ROWS if r['arm'] == 'SHIP']
pin('auto_selected == learned (SHIP)',
    sum(1 for r in ship if r.get('auto_selected') == 'learned'), len(ship),
    '`learned` 144/144')
pin('obj_init_method (SHIP)', sorted({r['obj_init_method'] for r in ship}), ['auto'],
    '`auto` · `w_phi`')
pin('obj_tracking_mode (SHIP)', sorted({r['obj_tracking_mode'] for r in ship}),
    ['w_phi'], None)
pin('cap_varpro_steps (SHIP)', sorted({r['cap_varpro_steps'] for r in ship}), [400],
    '**400** · **1e-4**')
pin('obj_lambda_l1 (SHIP)', sorted({r['obj_lambda_l1'] for r in ship}), [0.0001], None)
pin('n_ensemble (SHIP)',
    sorted({(r['n_ensemble'], r['n_ensemble_kind']) for r in ship}),
    [(1, 'analytic_median_k')], '`analytic_median_k`')
pin('arm_cfg 비어있지 않음', sorted({(r['arm_cfg_nonempty'], r['arm_cfg_nkeys'])
                                    for r in ship}), [(True, 27)], '`True` · 27')
pin('real_gate 전건 False', sorted({r['real_gate'] for r in ROWS}), [False],
    '`real_gate=False`')
pin('synthetic_only.ok 전건 True',
    sorted({r['synthetic_only']['ok'] for r in ROWS}), [True], '`synthetic_only.ok`')

# ── 8. 비용 (⛔예산 주장 아님 — 스탬프일 뿐) ────────────────────────────────
pin('총 코어·h', round(sum(r['wall_s'] for r in ROWS) / 3600, 2), None, '4.04')
pin('wall_s 중앙 SHIP', round(st.median([r['wall_s'] for r in ship]), 1), None, '9.7 s')
pin('wall_s 중앙 SPEC25',
    round(st.median([r['wall_s'] for r in ROWS if r['arm'] == 'SPEC25']), 1), None,
    '91.3 s')

# ⚠재도출 **밖** — `RESULT.md` 가 **다른 파일에서** 인용한 수의 전수 목록.
#   ⛔이 스크립트는 이것들을 보증하지 않는다. **빠뜨리는 것보다 적어 두는 것이 낫다**
#   (열람 범위 = 주장 범위). ⚠새 수를 문서에 추가하면 여기도 갱신하라.
UNVERIFIED = [
    ('`M-10` 45/48 · rel 중앙 −31.694% · p=6.56e-11 · `NEARFIELD-HOLDS` (§3)',
     'tmp/M10_GEOMETRY_2026-08-30/verdict_m10.json (동결)'),
    ('실측 밴드 0.327–7.152 Tp · 중앙 Tp 3.392 s · 12 중 11 · `BS11a` 1.022 (§8-3)',
     'premise_t4.json:p5_real_band_in_tp (런 0회 재도출)'),
    ('`G-4` 동결 재현 288짝 rel Δ ≤ 1e-6 위반 0 · `G-14` 비트 181/288 (§7)',
     'verdict_t4.json:gates 의 detail + 동결 B12/results_b12.jsonl (⛔이 스크립트는 '
     'B-12 를 읽지 않는다)'),
    ('`W-15 §2` 혼잡 55/64 −44.84% p=3.5e-09 / 희소 28/64 +21.42% p=0.38 (§6)',
     'tmp/LEARNED_AUDIT_2026-08-30/VERDICTS.md (동결 W-15 재집계)'),
    ('`VOID_4shard` 70행(66/4) · 4샤드 2.4 h vs 3샤드 1.79 h · 코어·h 5.36 · '
     '단가 0.52/4.85 · 앵커 0.0e+00 · 양성대조 7.2e-15 (§8-8)',
     'VOID_4shard/README.md + 그 디렉터리의 폐기 jsonl (⛔판정 자료 아님)'),
    ('스모크 4셀이 본런과 **비트 동일**(4/4) (§8-10)',
     'smoke_t4.jsonl — 문서 작성 시 수동 대조. ⛔이 스크립트는 스모크 파일을 읽지 않는다'),
    ('3샤드 벽시계 81.3 / 80.4 / 80.5 분 (§8-11)', 'run3_0.log · run3_1.log · run3_2.log'),
    ('재동결 4블록의 시각·사유, `gates_t4.py`·`PREREG.md`·`premise_t4.py` 해시 불변 (§8-9)',
     'FROZEN_md5.txt 헤더 (⛔이 스크립트는 md5 를 재계산하지 않는다)'),
    ('방벽 출력(`check_batch_artifacts` 전열 OK · `check_gate_refs` ⚠1 · 나머지 0건) (§8-14)',
     'tmp/_audit/*.py 실행 결과'),
    ('사전등록 축자(§2·§4·§8-1·§8-8) · `READING` 축자(§1) · `PREREG §2` 접기 사유(§3)',
     'PREREG.md · gates_t4.READING (⛔이 스크립트는 gates_t4 를 임포트하지 않는다)'),
]


def main():
    print(f'recheck_t4 — rows={len(ROWS)} pins={len(PINS)} '
          f'doc={"RESULT.md" if DOC else "⚠없음"}')
    for label, got, wv, wd in PINS:
        gs = f'{got:.6g}' if isinstance(got, float) else str(got)
        print(f'  {label:34s} = {gs[:44]:<46s}'
              f'{" ✓verdict" if wv is not None else "         "}'
              f'{" ✓문서" if wd is not None else ""}')
    print('\n⚠재도출 범위 밖 — `RESULT.md` 가 **다른 파일에서** 인용한 수 '
          '(⛔이 스크립트가 보증하지 않는다):')
    for what, src in UNVERIFIED:
        print(f'  · {what}\n      ←  {src}')
    if DOC is None:
        FAIL.append('RESULT.md 가 없다 — 문서 대조를 하지 못했다')
    assert not FAIL, '\n' + '\n'.join('  ⛔' + f for f in FAIL)
    print(f'\nrecheck_t4: PASS  ({len(PINS)} 핀 · jsonl == verdict == RESULT.md)')


if __name__ == '__main__':
    main()
