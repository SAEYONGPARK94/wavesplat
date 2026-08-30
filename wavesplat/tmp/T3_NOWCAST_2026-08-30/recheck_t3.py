"""`T3-NOW` RESULT.md **재도출 검증기** — ⛔추정기 0회 · GPU 0회 · 파일 2개만 읽는다.

    python3 tmp/T3_NOWCAST_2026-08-30/recheck_t3.py

읽는 것은 **동결 자료 둘뿐**이다:
  * `results_t3.jsonl` (144행)
  * `verdict_t3now.json` (`gates_t3.analyze()` 출력)

하는 일은 **3중 대조**다 (본보기 = `tmp/STATUS_2026-08-28/f1_table.py` 의
*"jsonl == verdict == 핀"*, `tmp/B3_REFIT_2026-08-23/final6.py`):

  ① **재도출** — 144행에서 `math.comb` 부호검정·중앙값을 **처음부터 다시** 계산한다
     (⛔`gates_t3` 를 임포트하지 않는다 — 같은 코드를 두 번 부르면 대조가 아니다).
  ② **verdict 대조** — ①이 `verdict_t3now.json` 의 값과 같은가.
  ③ **★문서 대조** — `RESULT.md` 에 그 수가 **실제로 그 글자로** 적혀 있는가.

⛔하나라도 어긋나면 `AssertionError` 다.

⚠**검증 범위(= 주장 범위)**: `RESULT.md` 가 인용하는 수 중 **`premise_t3.json` 출처**
(동결 `M8`·`T2` 재집계 = 이 배치의 측정이 아닌 것)는 이 스크립트가 **재도출하지 않는다**.
그 목록을 §UNVERIFIED 로 **찍어서 보여준다** — 침묵하지 않는 것이 요점이다.
"""
import json
import math
import os
import statistics as st
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
ROWS_PATH = os.path.join(HERE, 'results_t3.jsonl')
VERDICT_PATH = os.path.join(HERE, 'verdict_t3now.json')
RESULT_PATH = os.path.join(HERE, 'RESULT.md')

N = 36
ALPHA = 0.05
N_FAMILY = 3
ARMS = ('SHIP-K3', 'SHIP-K1M3', 'PROP-V400M3', 'PROP-V0M3')

ROWS = [json.loads(l) for l in open(ROWS_PATH)]
V = json.load(open(VERDICT_PATH))
DOC = open(RESULT_PATH).read() if os.path.exists(RESULT_PATH) else None

FAIL = []
PINS = []          # (라벨, 재도출값, verdict값 or None, 문서에 있어야 하는 글자 or None)


# ── ① 재도출 (⛔gates_t3 임포트 금지 — 독립 구현) ────────────────────────────
def sign_p(w, n):
    return sum(math.comb(n, k) for k in range(w, n + 1)) / 2 ** n


def min_w(n, alpha):
    return next(w for w in range(n + 1) if sign_p(w, n) <= alpha)


def mde_q80(n, thresh, power=0.80):
    q = 0.5
    while sum(math.comb(n, k) * q ** k * (1 - q) ** (n - k)
              for k in range(thresh, n + 1)) < power:
        q += 0.001
    return round(q, 3)


def cell(r):
    return (r['record'], r['win'])


def field(arm, f):
    return {cell(r): r[f] for r in ROWS
            if r['arm'] == arm and 'error' not in r and r.get(f) is not None}


def paired(arm_a, fa, arm_b, fb):
    a, b = field(arm_a, fa), field(arm_b, fb)
    ks = sorted(set(a) & set(b))
    w = sum(1 for k in ks if a[k] < b[k])
    return dict(n=len(ks), better=w, ties=sum(1 for k in ks if a[k] == b[k]),
                p=sign_p(w, len(ks)),
                rel=st.median([(a[k] - b[k]) / b[k] * 100 for k in ks]),
                med_a=st.median([a[k] for k in ks]),
                med_b=st.median([b[k] for k in ks]))


def vs_zero(arm, f):
    sub = [r for r in ROWS if r['arm'] == arm and 'error' not in r
           and r.get(f) is not None and r.get('zero_lead0')]
    w = sum(1 for r in sub if r[f] < r['zero_lead0'])
    return dict(n=len(sub), better=w, p=sign_p(w, len(sub)),
                ratio=st.median([r[f] / r['zero_lead0'] for r in sub]),
                med=st.median([r[f] for r in sub]))


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
pin('행 수', len(ROWS), V['n_rows'], '144행')
pin('에러 0', sum(1 for r in ROWS if 'error' in r), 0, None)
pin('팔별 셀 수', {a: sum(1 for r in ROWS if r['arm'] == a) for a in ARMS},
    {a: N for a in ARMS}, None)
pin('추정기 런 수(= Σ n_members)', sum(r['n_members'] for r in ROWS), 360, '360')
pin('게이트 PASS 수', sum(1 for g in V['gates'] if g['pass']), len(V['gates']), '47/47')
pin('레코드 수', len({r['record'] for r in ROWS}), 12, '12 레코드')
pin('창', sorted({r['win'] for r in ROWS}), [1, 3, 5], '{1,3,5}')
pin('장치 스탬프',
    sorted({(r['device'], r['cuda_available'], str(r['omp_threads'])) for r in ROWS}),
    [('cpu', True, '1')], "`cuda_available=True` · `device='cpu'`")

# ── 1. 문턱 · MDE (⛔손복사 아님 — math.comb 이 낸다) ────────────────────────
nominal = min_w(N, ALPHA)
holm_steps = [min_w(N, ALPHA / (N_FAMILY - r)) for r in range(N_FAMILY)]
pin('NOMINAL', nominal, V['thresholds']['NOMINAL'], 'NOMINAL **24**')
pin('HOLM_MIN_WINS', holm_steps, V['thresholds']['HOLM_MIN_WINS'], '**[25, 25, 24]**')
pin('NOT_EARN', N - nominal, V['thresholds']['NOT_EARN'], '`NOT_EARN` **12**')
pin('MDE 가족 q80', mde_q80(N, holm_steps[0]), V['mde']['q80'], '0.742')
pin('MDE 단일검정 q80', mde_q80(N, nominal), V['mde_single']['q80'], '0.716')

# ── 2. 확증 셋 (C1 PRIMARY · C2 · C3) ───────────────────────────────────────
c1 = paired('PROP-V400M3', 'e_med3', 'PROP-V400M3', 'e_m0')
pin('C1 better', c1['better'], V['primary']['better'], '**29/36**')
pin('C1 n', c1['n'], V['primary']['n'], None)
pin('C1 ties', c1['ties'], V['primary']['ties'], None)
pin('C1 p', c1['p'], V['primary']['p_one_sided'], 'p=0.00016', tol=1e-12)
pin('C1 rel%', c1['rel'], V['primary']['rel_median_pct'], '−8.65%', tol=1e-9)
pin('C1 median(e_med3)', c1['med_a'], V['primary']['median_arm'], '0.8279', tol=1e-12)
pin('C1 median(e_m0)', c1['med_b'], V['primary']['median_ref'], '0.8937', tol=1e-12)

c2 = paired('PROP-V0M3', 'e_m0', 'PROP-V400M3', 'e_m0')
pin('C2 better', c2['better'], V['c2']['better'], '**23/36**')
pin('C2 p', c2['p'], V['c2']['p_one_sided'], 'p=0.0662', tol=1e-12)
pin('C2 rel%', c2['rel'], V['c2']['rel_median_pct'], '−5.43%', tol=1e-9)

c3 = vs_zero('PROP-V400M3', 'e_med3')
pin('C3 better', c3['better'], V['c3']['better'], '**21/36**')
pin('C3 p', c3['p'], V['c3']['p_one_sided'], 'p=0.2025', tol=1e-12)
pin('C3 비(중앙)', c3['ratio'], V['c3']['ratio_median'], '0.848', tol=1e-12)

# Holm — ⛔단측 p 를 오름차순으로 걸어 **끊기는 지점**까지만 생존
ps = [c1['p'], c2['p'], c3['p']]
order = sorted(range(3), key=lambda i: ps[i])
surv = [False] * 3
for rank, i in enumerate(order):
    if ps[i] <= ALPHA / (3 - rank):
        surv[i] = True
    else:
        break
pin('Holm 생존', dict(zip(('C1', 'C2', 'C3'), surv)), V['holm_survivors'], None)
pin('라벨', V['label'], 'T3-NOW :: PROP-ENSEMBLE-EARNS (CONFIRMATORY)',
    '`T3-NOW :: PROP-ENSEMBLE-EARNS (CONFIRMATORY)`')
pin('라벨 C2', V['label_c2'], 'T3-NOW :: VARPRO-UNDECIDED',
    '`T3-NOW :: VARPRO-UNDECIDED`')
pin('라벨 C3', V['label_c3'], 'T3-NOW :: PROP-NOWCAST-UNDECIDED',
    '`T3-NOW :: PROP-NOWCAST-UNDECIDED`')
pin('차단 게이트', V['blocked_by'], [], None)

# ── 3. 서술 — 참 제로필드 대비 (⛔고정 상수 금지: 행의 `zero_lead0` 를 쓴다) ──
d = V['descriptive']['vs_zero']
for arm, fld, key_, w_doc, r_doc in (
        ('SHIP-K3', 'e_m0', 'SHIP-K3', '33/36', '0.729'),
        ('SHIP-K1M3', 'e_med3', 'SHIP-K1M3', '33/36', '0.729'),
        ('PROP-V400M3', 'e_med3', 'PROP-V400M3', '**21/36**', '0.848'),
        ('PROP-V0M3', 'e_med3', 'PROP-V0M3', '**27/36**', '0.838')):
    z = vs_zero(arm, fld)
    pin(f'{arm}.{fld} vs 제로 better', z['better'], d[key_]['better'], w_doc)
    pin(f'{arm}.{fld} vs 제로 비', z['ratio'], d[key_]['ratio_median'], r_doc, tol=1e-12)

# ★C1 의 크기를 「제로필드 기준」으로 옮긴 두 수 (= 이 배치가 낸 것)
z400_m0 = vs_zero('PROP-V400M3', 'e_m0')
pin('PROP-V400M3.e_m0 vs 제로 better (= G-PC 17)', z400_m0['better'], 17, '**17/36**')
pin('PROP-V400M3.e_m0 vs 제로 비 (= 동결 1.0365)', round(z400_m0['ratio'], 4), 1.0365,
    '1.0365', tol=1e-12)
z0_m0 = vs_zero('PROP-V0M3', 'e_m0')
pin('PROP-V0M3.e_m0 vs 제로 better (탐색적)', z0_m0['better'], 24, '24/36')
pin('PROP-V0M3.e_m0 vs 제로 비 (탐색적)', round(z0_m0['ratio'], 4), 0.886, '0.886',
    tol=1e-12)

# 짝 SHIP < PROP (= G-PC 33)
sp = paired('SHIP-K3', 'e_m0', 'PROP-V400M3', 'e_m0')
pin('짝 SHIP-K3 < PROP-V400M3 (= G-PC 33)', sp['better'], 33, '33/36')

# ── 4. 탐색적 E1 · E2 · E3 (⛔확증 아님) ────────────────────────────────────
bad_pos = [(a, k) for a in ('PROP-V400M3', 'PROP-V0M3')
           for k, v in field(a, 'e_med3').items()
           if not (v > 0 and field(a, 'e_m0').get(k, 0) > 0)]
if bad_pos:                    # ⛔조용히 넘어가지 않는다(로그의 정의역 밖 = 자료 결함)
    FAIL.append(f'[자료] 비양수 오차 {len(bad_pos)}셀 — E1 로그비 계산 불가: {bad_pos[:3]}')
g = {a: {k: math.log(field(a, 'e_med3')[k] / field(a, 'e_m0')[k])
         for k, v in field(a, 'e_med3').items()
         if v > 0 and field(a, 'e_m0').get(k, 0) > 0}
     for a in ('PROP-V400M3', 'PROP-V0M3')}
ks = sorted(set(g['PROP-V400M3']) & set(g['PROP-V0M3']))
e1 = sum(1 for k in ks if g['PROP-V400M3'][k] < g['PROP-V0M3'][k])
pin('E1 상호작용', e1, V['exploratory']['interaction']['v400_gains_more'], '20/36')
pin('E1 p', sign_p(e1, len(ks)),
    V['exploratory']['interaction']['p_one_sided'], '0.3089', tol=1e-12)

e2 = paired('SHIP-K3', 'e_m0', 'SHIP-K1M3', 'e_m0')
pin('E2 bank 앙상블 better', e2['better'],
    V['exploratory']['bank_ensemble']['better'], '**15/36**')
pin('E2 동점', e2['ties'], V['exploratory']['bank_ensemble']['ties'], '동점 **17**')
pin('E2 rel%', e2['rel'], V['exploratory']['bank_ensemble']['rel_median_pct'],
    'rel 중앙 **0.00%**', tol=1e-12)

for arm, doc in (('SHIP-K1M3', '**0.043**'), ('PROP-V400M3', '**0.559**'),
                 ('PROP-V0M3', '0.407')):
    sub = [r for r in ROWS if r['arm'] == arm and r.get('spread_rms') is not None]
    pin(f'E3 R 중앙 {arm}', st.median([r['spread_rms'] / r['e_m0'] for r in sub]),
        V['exploratory']['spread'][arm]['R_median'], doc, tol=1e-12)

# ── 5. ★거울 · 앵커 · bank 멤버 퇴화 (기제) ─────────────────────────────────
k1, k3 = field('SHIP-K1M3', 'e_med3'), field('SHIP-K3', 'e_m0')
mir_val = sum(1 for k in k3 if k1.get(k) == k3[k])
md5_k1 = {cell(r): r['field_md5_med'] for r in ROWS if r['arm'] == 'SHIP-K1M3'}
md5_k3 = {cell(r): r['field_md5_members'][0] for r in ROWS if r['arm'] == 'SHIP-K3'}
pin('G-MIRROR 값 비트 일치', mir_val, N, '36/36')
pin('G-MIRROR 해시 일치', sum(1 for k in md5_k3 if md5_k1.get(k) == md5_k3[k]), N, None)

deg = Counter(len(set(r['field_md5_members']))
              for r in ROWS if r['arm'] == 'SHIP-K1M3')
pin('bank 멤버 완전 퇴화(1 distinct)', deg[1], 9, '완전 퇴화 **9**')
pin('bank 멤버 부분 퇴화(2 distinct)', deg[2], 11, '부분 퇴화 **11**')
pin('bank 멤버 전-distinct(3)', deg[3], 16, '16/36')
for arm in ('PROP-V400M3', 'PROP-V0M3'):
    dp = Counter(len(set(r['field_md5_members'])) for r in ROWS if r['arm'] == arm)
    pin(f'{arm} 멤버 전-distinct', dp[3], N, None)

anc = {a: Counter(r['anchor_bitmatch'] for r in ROWS if r['arm'] == a) for a in ARMS}
pin('G-ANCHOR SHIP-K3', anc['SHIP-K3'][True], N, '36/36')
pin('G-ANCHOR PROP-V400M3', anc['PROP-V400M3'][True], N, '36/36')
pin('앵커 없는 팔은 None', [anc[a][None] for a in ('SHIP-K1M3', 'PROP-V0M3')], [N, N],
    None)

# ── 6. 비용 (⛔예산 주장 아님 — 스탬프일 뿐) ────────────────────────────────
pin('wall_s 중앙(전 144행)', st.median([r['wall_s'] for r in ROWS]),
    V['descriptive']['wall_s_median'], '13.97', tol=1e-9)
for arm, doc in (('SHIP-K3', '3.2'), ('SHIP-K1M3', '3.0'),
                 ('PROP-V400M3', '30.3'), ('PROP-V0M3', '29.0')):
    pin(f'wall_s 중앙 {arm}',
        round(st.median([r['wall_s'] for r in ROWS if r['arm'] == arm]), 1),
        None, doc)
pin('총 벽시계(분)', round(sum(r['wall_s'] for r in ROWS) / 60, 1), None, '39.2')

# ⚠재도출 **밖** — `RESULT.md` 가 **다른 파일에서** 인용한 수의 전수 목록.
#   ⛔이 스크립트는 이것들을 보증하지 않는다. **빠뜨리는 것보다 적어 두는 것이 낫다**
#   (열람 범위 = 주장 범위). ⚠새 수를 문서에 추가하면 여기도 갱신하라.
UNVERIFIED = [
    ('`T2` 재적합 표 — bank 36/36 · −23.96% · 제안 13/36 · +4.04% (§4)',
     'tmp/T2_STATIONARITY_2026-08-30/RESULT.md §3 (동결 results_t2.jsonl)'),
    ('장치 민감도 — 제안 경로 `M8`↔`T2` **0/12** 비트 일치 (§10.7)',
     'premise_t3.json:device_sensitivity'),
    ('탐침 36셀 distinct {3:16, 2:11, 1:9} · mirror_ok 36/36 (§4)',
     'probe_mirror_b.json ✅**본 배치 144행이 재현했다** — 그 재현은 위에서 핀으로 검사된다'),
    ('실측 아크 짝-가족 Holm 생존 **0개** (§10.1)',
     'tmp/B2_REAL_2026-08-23/MULTIPLICITY.md 08-26 배너'),
    ('패키지 줄번호 인용 `sequential_update.py:1139/1338/1941/2028-2032` (§5)',
     '패키지 소스 (⛔이 스크립트는 코드를 읽지 않는다)'),
    ('승인 축자 (§10.6) · 개정 사유 (§10.8) · 사전등록 축자 (§2·§8·§9)',
     'APPROVAL_T3_2026-08-30.md · AMENDMENT_0_PREFREEZE.md · PREREG.md / gates_t3.READING'),
]


def main():
    print(f'recheck_t3 — rows={len(ROWS)} pins={len(PINS)} '
          f'doc={"RESULT.md" if DOC else "⚠없음"}')
    for label, got, wv, wd in PINS:
        gs = f'{got:.6g}' if isinstance(got, float) else str(got)
        print(f'  {label:46s} = {gs:<28s}'
              f'{" ✓verdict" if wv is not None else "         "}'
              f'{" ✓문서" if wd is not None else ""}')
    print('\n⚠재도출 범위 밖 — `RESULT.md` 가 **다른 파일에서** 인용한 수 '
          '(⛔이 스크립트가 보증하지 않는다):')
    for what, src in UNVERIFIED:
        print(f'  · {what}   ←  {src}')
    if DOC is None:
        FAIL.append('RESULT.md 가 없다 — 문서 대조를 하지 못했다')
    assert not FAIL, '\n' + '\n'.join('  ⛔' + f for f in FAIL)
    print(f'\nrecheck_t3: PASS  ({len(PINS)} 핀 · jsonl == verdict == RESULT.md)')


if __name__ == '__main__':
    main()
