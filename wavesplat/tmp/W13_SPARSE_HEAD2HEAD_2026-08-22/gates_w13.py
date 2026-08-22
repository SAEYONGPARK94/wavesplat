"""W13 단일출처 — **희소 코퍼스에서 학습 vs 해석적 정면 비교**.

혼잡은 이미 답이 있다(`C` 재집계, 런 0회): 학습이 **59/64, p=9.0e-13**,
네 기하 전부에서 상대 −21~−45%. `C` 의 `SHIP` 팔은 **192/192 전부 학습** 이므로
그 대조는 처음부터 정면 비교였다.

⟹ 남은 구멍은 **희소**다. W11 의 꼬리가 *"희소 단순 해상에서는 해석적이 낫다"* 고 말한다
(`2comp` 에서 통일 비용 최대 +1184%). 그것을 **강제 두 팔**로 확정한다.
"""
import json, os
HERE = os.path.dirname(os.path.abspath(__file__))
EPISODES = json.load(open(os.path.join(HERE, 'corpus.json')))['sparse']   # 64 (2comp 32 · 5comp 32)
SEEDS = [1620, 1621, 1622]        # 신선 시드 (잔여 [1619–1818], W12 가 1619 미사용)

ARMS = {
    'LEARNED':  {'init_method': 'learned'},    # 배포 기본값(VARPRO 400 포함) + 라우터 제거
    'ANALYTIC': {'init_method': 'v2_konly'},   # 해석적 경로 강제
}
ARM_ORDER = ['LEARNED', 'ANALYTIC']
FIELD = 'avg_nRMSE'
ZERO_REF = 0.25        # 희소 제로필드 실측 중앙 (B 96셀에서 0.2500)
THRESHOLD = 41         # /64, 단측 부호검정 p<0.05 (결과 전 확정)


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


def primary(rows, ncomp=None):
    """H1(학습이 낫다) = LEARNED 가 ANALYTIC 보다 **낮다**."""
    import re, statistics as st
    a, b = med_by_episode(rows, 'LEARNED'), med_by_episode(rows, 'ANALYTIC')
    ks = sorted(set(a) & set(b))
    if ncomp:
        ks = [k for k in ks if re.search(r'_(\d+)comp', k).group(1) == str(ncomp)]
    n = len(ks)
    w = sum(1 for k in ks if a[k] < b[k])
    return {'n': n, 'learned_better': w,
            'rel_median_pct': st.median([(a[k] - b[k]) / b[k] * 100 for k in ks]) if ks else None,
            'p_one_sided': sign_test_one_sided(w, n) if n else None,
            'learned_beats_zero': sum(1 for k in ks if a[k] < ZERO_REF),
            'analytic_beats_zero': sum(1 for k in ks if b[k] < ZERO_REF)}


def gates(rows):
    ok = [r for r in rows if 'error' not in r]
    exp = len(ARMS) * len(EPISODES) * len(SEEDS)
    out = [('G-0 런 수', len(ok) == exp, f'{len(ok)}/{exp}'),
           ('G-0b 에러 0', len(rows) == len(ok), f'{len(rows)-len(ok)}')]
    # ★G-1 두 팔이 정말 서로 다른 경로인가 — 성분 수로 확인(해석적/학습이 같을 리 없다)
    l = [r for r in ok if r['cell'] == 'LEARNED']
    a = [r for r in ok if r['cell'] == 'ANALYTIC']
    out.append(('G-1 LEARNED 팔에서 VARPRO 발화',
                all(r.get('varpro_fired') for r in l),
                f'{sum(1 for r in l if r.get("varpro_fired"))}/{len(l)}'))
    out.append(('G-1b ANALYTIC 팔에서 VARPRO 미발화',
                not any(r.get('varpro_fired') for r in a),
                f'{sum(1 for r in a if r.get("varpro_fired"))}/{len(a)}'))
    # ★G-2 라우터가 돌지 않았다(양 팔 모두 명시 지정)
    out.append(('G-2 라우터 미작동(양 팔)',
                all(r.get('auto_selected') is None for r in ok),
                f'{sum(1 for r in ok if r.get("auto_selected") is not None)}'))
    return out


def label(p, p2, p5):
    if p['learned_better'] >= THRESHOLD:
        return ('LEARNED-WINS-SPARSE',
                f'희소에서도 학습이 낫다({p["learned_better"]}/{p["n"]}) ⟹ 통일에 층별 대가가 없다.')
    if p['learned_better'] <= p['n'] - THRESHOLD:
        return ('ANALYTIC-WINS-SPARSE',
                f'희소에서는 해석적이 낫다({p["n"]-p["learned_better"]}/{p["n"]}) '
                f'⟹ 통일은 희소를 판다. 대가 = 중앙 {p["rel_median_pct"]:+.2f}% '
                f'(2comp {p2["rel_median_pct"]:+.1f}% · 5comp {p5["rel_median_pct"]:+.1f}%).')
    return ('SPARSE-NULL',
            f'희소에서 두 경로가 갈리지 않는다({p["learned_better"]}/{p["n"]}) ⟹ 통일 비용 없음.')
