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

ARMS = {
    'POST': {},                                   # 배포 그대로 (thresh 0 = 통일)
    'PRE':  {'auto_switch_thresh': 8},            # 통일 이전 거동
}
ARM_ORDER = ['PRE', 'POST']
FIELD = 'avg_nRMSE'
ZERO_REF = 0.25
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
            'post_beats_zero': sum(1 for k in ks if a[k] < ZERO_REF),
            'pre_beats_zero': sum(1 for k in ks if b[k] < ZERO_REF)}


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
