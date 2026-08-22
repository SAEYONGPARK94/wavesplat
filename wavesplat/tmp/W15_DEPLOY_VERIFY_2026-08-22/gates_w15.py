"""W15 단일출처 — **배포 전/후 ITT 검증** (통일 적용 확인).

`W11`·`W13`·`W14` 는 팔을 **강제**해서 두 경로를 비교했다. 이 배치는 다르다:
**배포 구성 그대로** 돌려서 *"통일 적용이 실제로 무엇을 바꿨는가"* 를 잰다(ITT).
"""
import json, os, re
HERE = os.path.dirname(os.path.abspath(__file__))
_C = json.load(open(os.path.join(HERE, 'corpus.json')))
EPISODES = _C['crowded'] + _C['sparse']          # 128
STRATUM = {e: 'crowded' for e in _C['crowded']}
STRATUM.update({e: 'sparse' for e in _C['sparse']})
SEEDS = [1626, 1627, 1628]                        # 신선 시드

# ★개정1(2026-08-22): 세 팔 **전부 `init_method='auto'`** 로 두고 임계만 바꾼다.
# 사유: 명시적 `init_method` 는 median-of-3 앙상블을 **못 받는다**(`make_updater` 독스트링 축자).
# 그래서 W13/W14 의 강제 해석적 팔은 배포 해석적이 아니었다 = 비교가 편향돼 있었다.
ARMS = {
    'POST':         {},                              # 배포 그대로 (thresh 0 = 항상 학습)
    'PRE':          {'auto_switch_thresh': 8},       # 통일 이전 = 혼합 라우팅
    'ALL-ANALYTIC': {'auto_switch_thresh': 999999},  # ★항상 해석적 + 앙상블 = 진짜 배포 해석적
}
ARM_ORDER = ['PRE', 'POST', 'ALL-ANALYTIC']
FIELD = 'avg_nRMSE'
ZERO_REF = None   # ★상수 금지 — 에피소드별 실측 zero_nRMSE 를 쓴다(개정1)
THRESHOLD = 79    # /128, 단측 부호검정 p<0.05 (결과 전 확정)


def sign_test_one_sided(w, n):
    from math import comb
    return sum(comb(n, k) for k in range(w, n + 1)) / 2 ** n


def med_by_episode(rows, arm, field=FIELD):
    import statistics as st
    d = {}
    for r in rows:
        if r['cell'] == arm and r.get(field) is not None:
            d.setdefault(r['episode'], []).append(r[field])
    return {e: st.median(v) for e, v in d.items()}


def primary(rows, stratum=None, field=FIELD):
    """H1(통일이 낫다) = POST 가 PRE 보다 **낮다**."""
    import statistics as st
    a, b = med_by_episode(rows, 'POST', field), med_by_episode(rows, 'PRE', field)
    ks = sorted(set(a) & set(b))
    if stratum:
        ks = [k for k in ks if STRATUM.get(k) == stratum]
    n = len(ks)
    w = sum(1 for k in ks if a[k] < b[k])
    return {'n': n, 'post_better': w, 'ties': sum(1 for k in ks if a[k] == b[k]),
            'rel_median_pct': st.median([(a[k] - b[k]) / b[k] * 100 for k in ks]) if ks else None,
            'p_one_sided': sign_test_one_sided(w, n) if n else None,
            'post_beats_zero': None, 'pre_beats_zero': None}


def zero_ref():
    """★에피소드별 실측 제로필드 — 손복사 상수(0.25)는 틀렸다(실측 0.180–0.292)."""
    import json, os
    z = {}
    src = os.path.join(HERE, '..', 'B_DEPLOY_2026-08-19', 'results.jsonl')
    for l in open(src):
        r = json.loads(l)
        if 'error' not in r and r.get('zero_nRMSE') is not None:
            z[r['episode']] = r['zero_nRMSE']
    return z


def beats_zero(rows, arm):
    z = zero_ref()
    m = med_by_episode(rows, arm)
    ks = [k for k in m if k in z]
    return sum(1 for k in ks if m[k] < z[k]), len(ks)


def gates(rows):
    ok = [r for r in rows if 'error' not in r]
    exp = len(ARMS) * len(EPISODES) * len(SEEDS)
    out = [('G-0 런 수', len(ok) == exp, f'{len(ok)}/{exp}'),
           ('G-0b 에러 0', len(rows) == len(ok), f'{len(rows)-len(ok)}')]
    post = [r for r in ok if r['cell'] == 'POST']
    pre = [r for r in ok if r['cell'] == 'PRE']
    # ★G-1 통일이 실제로 먹었나 — POST 는 전건 learned 여야 한다
    out.append(('G-1 POST 전건 learned (통일 확인)',
                all(r.get('auto_selected') == 'learned' for r in post),
                f'{sum(1 for r in post if r.get("auto_selected")=="learned")}/{len(post)}'))
    # ★G-2 PRE 는 두 갈래여야 한다 (대조가 실재하는가)
    sel = {}
    for r in pre: sel[r.get('auto_selected')] = sel.get(r.get('auto_selected'), 0) + 1
    out.append(('G-2 PRE 는 두 갈래 (대조 실재)', len(sel) >= 2, str(sel)))
    out.append(('G-3 POST 전건 VARPRO 발화',
                all(r.get('varpro_fired') for r in post),
                f'{sum(1 for r in post if r.get("varpro_fired"))}/{len(post)}'))
    # ★G-4(개정1) ALL-ANALYTIC 이 전건 해석적이고 앙상블을 받았나
    aa = [r for r in ok if r['cell'] == 'ALL-ANALYTIC']
    out.append(('G-4 ALL-ANALYTIC 전건 v2_konly',
                all(r.get('auto_selected') == 'v2_konly' for r in aa),
                f'{sum(1 for r in aa if r.get("auto_selected")=="v2_konly")}/{len(aa)}'))
    out.append(('G-4b ALL-ANALYTIC 앙상블 k=3 작동',
                all((r.get('ensemble_k') or 1) == 3 for r in aa),
                f'{sorted({r.get("ensemble_k") for r in aa})}'))
    # ★G-5(개정1) 셸 분해가 기록됐나 — 네 배치가 이걸 버렸다
    out.append(('G-5 셸 분해 기록됨',
                all(r.get('avg_shell_rmse') for r in ok),
                f'{sum(1 for r in ok if r.get("avg_shell_rmse"))}/{len(ok)}'))
    return out


def label(p):
    if p['post_better'] >= THRESHOLD:
        return ('UNIFY-IMPROVES', f'통일이 배포 수준에서 개선한다({p["post_better"]}/{p["n"]}, '
                                  f'중앙 {p["rel_median_pct"]:+.2f}%).')
    if p['post_better'] <= p['n'] - THRESHOLD:
        return ('UNIFY-REGRESSES', f'통일이 배포 수준에서 악화한다({p["n"]-p["post_better"]}/{p["n"]}).')
    return ('UNIFY-NEUTRAL-SIGN',
            f'부호로는 안 갈린다({p["post_better"]}/{p["n"]}). '
            f'⚠크기는 별도로 읽어라: 중앙 {p["rel_median_pct"]:+.2f}%, '
            f'제로필드 이김 POST {p["post_beats_zero"]}/{p["n"]} vs PRE {p["pre_beats_zero"]}/{p["n"]}.')
