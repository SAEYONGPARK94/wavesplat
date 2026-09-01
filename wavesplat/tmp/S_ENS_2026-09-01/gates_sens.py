#!/usr/bin/env python3
"""`S-ENS` 게이트·문턱·라벨 — **단일출처**. 러너·분석기는 여기를 호출만 한다.

사전등록 = `PREREG.md`. ⛔수를 문서에서 베끼지 마라 — 전부 재도출이다.

★질문: *"실측에서 확증된 앙상블 이득이 **합성에서도**, 그리고 **배포 코드 경로 자체에서도**
나타나는가?"*
"""
import json
import os
import statistics as st
import sys
from math import comb

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
W15 = os.path.abspath(os.path.join(HERE, '..', 'W15_DEPLOY_VERIFY_2026-08-22'))
T5 = os.path.abspath(os.path.join(HERE, '..', 'T5_ENS_2026-09-01'))
for _p in (HERE, W15, ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# ★코퍼스·층은 **임포트한다**(⛔재선언 금지, PREREG §4.3).
from gates_w15 import EPISODES, STRATUM, SEEDS  # noqa: E402

BATCH = 'S-ENS'
SEED = SEEDS[0]                       # 고정 — ★새 시드 청구 0
FIELD = 'avg_nRMSE'
STRATA = ('crowded', 'sparse')

# ─────────────────────────────────────────────────────────────────────────────
# 1. 팔 — ⛔전 키 명시. 두 팔은 `learned_ensemble_k` **하나만** 다르다.
#    ★둘 다 배포 라우터를 그대로 탄다(auto + 임계 0 ⟹ 라우터가 학습으로 해소).
# ─────────────────────────────────────────────────────────────────────────────
_BASE = dict(
    init_method='auto', auto_switch_thresh=0,
    varpro_steps=400, learned_init_iters=0, lambda_l1=1e-4,
    k_snap_dk=None, adaptive_n=True, max_n=30,
    tracking_mode='w_phi', tracking_steps=10, tracking_lr=0.005, tracking_window=3,
    use_confidence=True,
    analytic_ensemble_k=3,      # 배포값. ⚠이 배치에서는 무연산(라우터가 학습으로 간다)
    real_domain_gate=True, seed=SEED,
)

ARMS = {
    'LEARN-M1': dict(CFG=dict(_BASE, learned_ensemble_k=1), EXPECT_K=1),
    'LEARN-M3': dict(CFG=dict(_BASE, learned_ensemble_k=3), EXPECT_K=3),
}
ARM_ORDER = ['LEARN-M1', 'LEARN-M3']
TREATMENT_KEY = 'learned_ensemble_k'


def arm_config(arm):
    return dict(ARMS[arm]['CFG'])


def cells():
    return list(EPISODES)


def expected_rows():
    return len(ARM_ORDER) * len(EPISODES)


def expected_runs():
    return len(EPISODES) * sum(a['EXPECT_K'] for a in ARMS.values())


# ─────────────────────────────────────────────────────────────────────────────
# 2. 문턱 — 재도출 (PREREG §4.4)
# ─────────────────────────────────────────────────────────────────────────────
ALPHA = 0.05
N_FAMILY = 2                          # PRIMARY 혼잡 · SEC-A 희소
N_PAIRS = 64                          # 층당 셀 수


def sign_test_one_sided(w, n):
    return sum(comb(n, k) for k in range(w, n + 1)) / 2 ** n


def _min_w(n, alpha):
    return next(w for w in range(n + 1) if sign_test_one_sided(w, n) <= alpha)


def thresholds(n=N_PAIRS):
    nominal = _min_w(n, ALPHA)
    holm = [_min_w(n, ALPHA / (N_FAMILY - r)) for r in range(N_FAMILY)]
    return dict(n=n, NOMINAL=nominal, HOLM_MIN_WINS=holm, HOLM_FIRST=holm[0],
                NOT_EARN=n - holm[0],
                p_nominal=sign_test_one_sided(nominal, n),
                p_holm_first=sign_test_one_sided(holm[0], n),
                alpha=ALPHA, n_family=N_FAMILY, family_alpha_first=ALPHA / N_FAMILY)


def mde(n=N_PAIRS, power=0.80, thresh=None):
    t = thresholds(n)['HOLM_FIRST'] if thresh is None else thresh
    q = 0.5
    while sum(comb(n, k) * q ** k * (1 - q) ** (n - k) for k in range(t, n + 1)) < power:
        q += 0.001
    return dict(n=n, thresh=t, power=power, q80=round(q, 3))


def power_at(q, n=N_PAIRS, thresh=None):
    """★결과 전 예측 하의 검정력 (PREREG §3). ⛔이것 없이 설계를 동결하지 마라."""
    t = thresholds(n)['HOLM_FIRST'] if thresh is None else thresh
    return sum(comb(n, k) * q ** k * (1 - q) ** (n - k) for k in range(t, n + 1))


def holm_survivors(pvals):
    order = sorted(pvals.items(), key=lambda kv: (kv[1] is None, kv[1]))
    out, still = {}, True
    for r, (name, p) in enumerate(order):
        still = still and (p is not None and p <= ALPHA / (len(order) - r))
        out[name] = still
    return out


# ─────────────────────────────────────────────────────────────────────────────
# 3. 집계 — 규약은 `st.median` 하나뿐(`gates_t3.paired:375` 상속)
# ─────────────────────────────────────────────────────────────────────────────
def _empty(arm, ref):
    return dict(arm=arm, ref=ref, n=0, better=None, ties=None,
                rel_median_pct=None, p_one_sided=None)


def paired(rows, arm_a, arm_b, stratum=None, field=FIELD):
    def m(arm):
        return {r['episode']: r.get(field) for r in rows
                if r.get('arm') == arm and r.get(field) is not None
                and (stratum is None or STRATUM.get(r['episode']) == stratum)}
    a, b = m(arm_a), m(arm_b)
    ks = sorted(set(a) & set(b))
    n = len(ks)
    if n == 0:
        return _empty(arm_a, arm_b)
    w = sum(1 for k in ks if a[k] < b[k])
    return dict(arm=arm_a, ref=arm_b, stratum=stratum, n=n, better=w,
                ties=sum(1 for k in ks if a[k] == b[k]),
                rel_median_pct=st.median([(a[k] - b[k]) / b[k] * 100 for k in ks]),
                p_one_sided=sign_test_one_sided(w, n))


# ─────────────────────────────────────────────────────────────────────────────
# 4. 라벨 — 결과 전 확정 · 공역 전건
# ─────────────────────────────────────────────────────────────────────────────
CLAIMS = {
    'PRIMARY': dict(stratum='crowded', pos='SYNTH-ENSEMBLE-EARNS',
                    neg='SYNTH-ENSEMBLE-REVERSED', und='SYNTH-ENSEMBLE-UNDECIDED'),
    'SEC-A': dict(stratum='sparse', pos='SYNTH-ENSEMBLE-EARNS-SPARSE',
                  neg='SYNTH-ENSEMBLE-REVERSED-SPARSE',
                  und='SYNTH-ENSEMBLE-UNDECIDED-SPARSE'),
}


def grade(v, surv):
    t = thresholds(v['n'] or N_PAIRS)
    w = v['better']
    if w is None:
        return 'NOROWS', t
    if surv:
        return 'CONFIRMATORY', t
    if w >= t['NOMINAL']:
        return 'EXPLORATORY', t
    if w <= t['NOT_EARN']:
        return 'REVERSE', t
    return 'UNDECIDED', t


def label_for(claim, v, surv):
    c = CLAIMS[claim]
    g, _ = grade(v, surv)
    if g == 'NOROWS':
        return None, g
    if g == 'CONFIRMATORY':
        return f'{BATCH} :: {c["pos"]} (CONFIRMATORY)', g
    if g == 'EXPLORATORY':
        return f'{BATCH} :: {c["pos"]} [EXPLORATORY]', g
    if g == 'REVERSE':
        return f'{BATCH} :: {c["neg"]} [EXPLORATORY]', g
    return f'{BATCH} :: {c["und"]}', g


READING = {
    'UNDECIDED': ('실측에서 확증된 앙상블 이득이 합성에서는 **검출되지 않았다**. '
                  '⛔"차이 없다" 금지. ★귀결: 이득이 실측 특이일 가능성이 커지고 '
                  'D3 의 배포 축은 근거를 잃는다.'),
    'CONFIRMATORY': ('합성에서도 이득이 있다 ⟹ 배포 기본값 전환의 **필요조건 하나**가 '
                     '충족된다. ⛔충분조건 아님 — 단가 3배·프레임 예산은 별도 측정이고 '
                     '전환은 별도 사전등록이다.'),
    'EXPLORATORY': '명목 문턱은 넘었으나 가족 보정에서 탈락 ⟹ 확증 아님.',
    'REVERSE': '⚠합성에서 앙상블이 **해친다**(실측과 방향 반대). ⛔확증 승격 금지.',
}

DISCLOSURE = [
    'blind 아님 — 실측 방향·크기를 알고 있다(PREREG §2). 단 합성 방향은 어떤 동결 자료에도 없다',
    '⛔배포 기본값 전환 권고 아님 — 필요조건 하나일 뿐이다',
    '⛔실측 판정 갱신 아님 · T3/T5 크기 전이 금지',
    '⛔층 간 대비에 p 를 붙이지 않는다(B-17 교훈)',
    '⚠체크포인트가 평가와 같은 분포라는 단서는 그대로다',
    f'MDE q80 = {mde()["q80"]} 병기 의무',
]

# ─────────────────────────────────────────────────────────────────────────────
# 5. 게이트
# ─────────────────────────────────────────────────────────────────────────────
BLOCKING = ['G-ENSK', 'G-ANCHOR-M1', 'G-ROUTE', 'G-ARM', 'G-DEPLOY-PIN', 'G-ROWS']


def _anchor_w15():
    """동결 `W-15 POST` 의 (episode, seed) → avg_nRMSE."""
    out = {}
    p = os.path.join(W15, 'results.jsonl')
    if os.path.exists(p):
        for ln in open(p):
            if ln.strip():
                r = json.loads(ln)
                if r.get('cell') == 'POST' and int(r.get('seed', -1)) == SEED:
                    out[r['episode']] = r.get('avg_nRMSE')
    return out


def gates(rows, premise=None):
    G = []

    def A(name, ok, detail=''):
        G.append((name, bool(ok), detail))

    ok = [r for r in rows if 'error' not in r]
    have = bool(ok)

    # ── ★G-ENSK : 패키지가 실제로 앙상블을 만들었는가
    #    ⛔T3/T5 에서는 불가능했던 검사다 — 그때는 하네스가 앙상블을 소유했다.
    e_ok, e_bad = have, []
    for r in ok:
        want = ARMS.get(r.get('arm'), {}).get('EXPECT_K')
        got = r.get('ensemble_k')
        if want is None or got != want:
            e_ok = False
            e_bad.append(f'{r.get("arm")}/{r.get("episode")}: {got}!={want}')
    A('G-ENSK 패키지 ensemble_k == 팔의 기대값', e_ok, ';'.join(e_bad[:3]))

    # ── ★G-ANCHOR-M1 : 배포 기본값 불변의 실행 증명
    anc = _anchor_w15()
    M1 = [r for r in ok if r.get('arm') == 'LEARN-M1']
    a_ok, a_n, a_bad = bool(M1), 0, []
    for r in M1:
        ref = anc.get(r['episode'])
        if ref is None:
            continue
        a_n += 1
        if ref != r.get(FIELD):
            a_ok = False
            a_bad.append(f'{r["episode"]}: {r.get(FIELD)} != {ref}')
    A('G-ANCHOR-M1 LEARN-M1 == 동결 W-15 POST (비트)', a_ok and a_n > 0,
      f'compared={a_n} bad={a_bad[:2]}')

    # ── G-ROUTE : 두 팔 전건 학습으로 해소됐는가
    A('G-ROUTE auto_selected == learned 전건',
      have and all(r.get('auto_selected') == 'learned' for r in ok),
      f'{sorted({r.get("auto_selected") for r in ok})}')

    # ── G-ARM : 두 팔이 처치 키 하나만 다른가 (의도표 자기대조가 아니라 diff)
    ca, cb = ARMS['LEARN-M1']['CFG'], ARMS['LEARN-M3']['CFG']
    diff = {k for k in set(ca) | set(cb) if ca.get(k) != cb.get(k)}
    A(f'G-ARM 두 팔이 {TREATMENT_KEY} 하나만 다르다',
      diff == {TREATMENT_KEY} and bool(ca) and bool(cb), f'diff={sorted(diff)}')

    # ── G-DEPLOY-PIN
    A('G-DEPLOY-PIN 출하 기본값 트립와이어',
      have and all(r.get('deploy_pin_ok') is True for r in ok))

    # ── G-ROWS
    cov = {(a, s): len({r['episode'] for r in ok
                        if r.get('arm') == a and STRATUM.get(r['episode']) == s})
           for a in ARM_ORDER for s in STRATA}
    A('G-ROWS 팔별·층별 커버리지 64/64',
      have and all(v == N_PAIRS for v in cov.values()) and len(ok) == expected_rows(),
      f'{cov} rows={len(ok)}/{expected_rows()}')

    # ── 범위 축소 (⛔blocked_by 에 넣지 않는다)
    w1 = [r.get('wall_s') for r in ok if r.get('arm') == 'LEARN-M1' and r.get('wall_s')]
    w3 = [r.get('wall_s') for r in ok if r.get('arm') == 'LEARN-M3' and r.get('wall_s')]
    A('H-COST 단가 공시', True,
      f'M1 중앙 {st.median(w1):.1f}s · M3 중앙 {st.median(w3):.1f}s' if w1 and w3 else 'n/a')
    A('H-SEED 새 시드 청구 0 (설계상)', True, f'SEED={SEED} 고정 · 동결 코퍼스')
    return G


def gate_id(name):
    return name.split()[0]


def blocked_by(G):
    return [gate_id(n) for n, o, _ in G if not o and gate_id(n) in BLOCKING]


# ─────────────────────────────────────────────────────────────────────────────
# 6. 분석
# ─────────────────────────────────────────────────────────────────────────────
def analyze(rows):
    ok = [r for r in rows if 'error' not in r]
    v = {k: paired(ok, 'LEARN-M3', 'LEARN-M1', stratum=c['stratum'])
         for k, c in CLAIMS.items()}
    surv = holm_survivors({k: x['p_one_sided'] for k, x in v.items()})
    labels, grades = {}, {}
    for k in CLAIMS:
        labels[k], grades[k] = label_for(k, v[k], surv.get(k, False))
    return dict(batch=BATCH, contrasts=v, holm=surv, labels=labels, grades=grades,
                thresholds=thresholds(), mde=mde(),
                reading={k: READING.get(g, '') for k, g in grades.items()},
                disclosure=DISCLOSURE, n_rows=len(ok))


# ─────────────────────────────────────────────────────────────────────────────
# 7. ★자가진단 (동결 **전**) — `T5 §7.3` 의 7항 상속
# ─────────────────────────────────────────────────────────────────────────────
def _fixture():
    """동결 `W-15 POST` 에서 픽스처를 만든다 — 게이트가 통과 **가능**함을 먼저 증명한다."""
    anc = _anchor_w15()
    rows = []
    for ep in EPISODES:
        base = anc.get(ep)
        if base is None:
            continue
        common = dict(batch=BATCH, episode=ep, stratum=STRATUM.get(ep), seed=SEED,
                      auto_selected='learned', deploy_pin_ok=True, approved=True)
        rows.append(dict(common, arm='LEARN-M1', ensemble_k=1,
                         avg_nRMSE=base, wall_s=13.8))
        rows.append(dict(common, arm='LEARN-M3', ensemble_k=3,
                         avg_nRMSE=base * 0.93, wall_s=41.3))
    return rows


def _forge(rows, what):
    R = [dict(r) for r in rows]
    if what == 'G-ENSK':
        for r in R:
            if r['arm'] == 'LEARN-M3':
                r['ensemble_k'] = 1          # ★패키지가 앙상블을 안 만든 시나리오
                break
    elif what == 'G-ANCHOR-M1':
        for r in R:
            if r['arm'] == 'LEARN-M1':
                r['avg_nRMSE'] = r['avg_nRMSE'] * 1.000001
                break
    elif what == 'G-ROUTE':
        R[0]['auto_selected'] = 'v2_konly'
    elif what == 'G-DEPLOY-PIN':
        R[0]['deploy_pin_ok'] = False
    elif what == 'G-ROWS':
        R = R[:-2]
    else:
        return None
    return R


# ★행 위조로 시험할 수 없는 게이트 + 대체 시험. ⛔사문화 방지(selftest 가 단언).
FORGERY_EXEMPT = {
    'G-ARM': '팔 정의는 코드 상수이지 행이 아니다. 대체 시험 = selftest 8항'
             '(처치 키를 하나 더 넣으면 diff 가 커져 FAIL 하는지 확인).',
}


def selftest(verbose=True):
    fails = []

    def chk(name, ok, detail=''):
        if verbose:
            print(f'  [{"OK " if ok else "FAIL"}] {name}  {detail}')
        if not ok:
            fails.append(name)

    print('── §6.3 게이트 자가진단 ──────────────────────────────────────────')
    rows = _fixture()
    chk('1 픽스처 생성', len(rows) == expected_rows(), f'{len(rows)}/{expected_rows()}')
    bad = [n for n, o, d in gates(rows) if not o]
    chk('1 픽스처에서 전 게이트 PASS', not bad, f'fail={bad}')

    uncovered = []
    for g in BLOCKING:
        F = _forge(rows, g)
        if F is None:
            if g not in FORGERY_EXEMPT:
                uncovered.append(g)
            continue
        got = dict((gate_id(n), o) for n, o, _ in gates(F))
        chk(f'2 위조탐침 {g} → FAIL', got.get(g) is False, f'got={got.get(g)}')
    chk('3 위조 없는 차단 게이트 0개', not uncovered, f'{uncovered}')
    chk('3b 면제 게이트가 차단 목록에 실재 · 사유 기재',
        all(g in BLOCKING and FORGERY_EXEMPT[g].strip() for g in FORGERY_EXEMPT),
        f'{list(FORGERY_EXEMPT)}')

    for a in ARM_ORDER:
        D = [r for r in rows if r['arm'] != a]
        chk(f'4 팔 삭제 민감도 ({a})', bool(blocked_by(gates(D))))

    G0, A0 = gates([]), analyze([])
    chk('5 0행: 차단 게이트가 FAIL', bool(blocked_by(G0)))
    chk('5 0행: 라벨 None', all(v is None for v in A0['labels'].values()))
    chk('5 0행: 카운터 None', all(A0['contrasts'][k]['better'] is None for k in CLAIMS))

    holes = []
    for w in range(N_PAIRS + 1):
        for surv in (True, False):
            v = dict(n=N_PAIRS, better=w, p_one_sided=sign_test_one_sided(w, N_PAIRS))
            lab, g = label_for('PRIMARY', v, surv)
            if lab is None or g not in READING:
                holes.append((w, surv))
    chk(f'6 라벨 공역 전건 정의 ({2*(N_PAIRS+1)} 조합)', not holes, f'holes={holes[:3]}')

    # 7. 처치가 실제로 단일 요인인가 (코드 상수 검사)
    ca, cb = ARMS['LEARN-M1']['CFG'], ARMS['LEARN-M3']['CFG']
    d = {k for k in set(ca) | set(cb) if ca.get(k) != cb.get(k)}
    chk('7 두 팔이 처치 키 하나만 다르다', d == {TREATMENT_KEY}, f'{sorted(d)}')

    # 8. G-ARM 의 대체 시험 — 두 번째 차이를 넣으면 반드시 FAIL
    _save = ARMS['LEARN-M3']['CFG']
    try:
        ARMS['LEARN-M3']['CFG'] = dict(_save, max_n=99)
        got = dict((gate_id(n), o) for n, o, _ in gates(rows))
        chk('8 G-ARM 대체 시험 (처치 키 2개 → FAIL)', got.get('G-ARM') is False,
            f'got={got.get("G-ARM")}')
    finally:
        ARMS['LEARN-M3']['CFG'] = _save

    print('──────────────────────────────────────────────────────────────────')
    if fails:
        print(f'⛔자가진단 실패 {len(fails)}건: {fails}')
        return 1
    t, m = thresholds(), mde()
    print(f'✅자가진단 전건 통과 · 차단 게이트 {len(BLOCKING)}')
    print(f'   문턱 NOMINAL={t["NOMINAL"]} Holm={t["HOLM_MIN_WINS"]} '
          f'NOT_EARN={t["NOT_EARN"]} · MDE q80={m["q80"]}')
    print('   검정력: ' + ' · '.join(f'q={q}→{power_at(q):.3f}'
                                   for q in (0.75, 0.70, 0.65, 0.60)))
    print(f'   기대 행/런 = {expected_rows()}/{expected_runs()}')
    return 0


if __name__ == '__main__':
    sys.exit(selftest())
