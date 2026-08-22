"""W11 단일출처 — 라우터를 없애고 **학습 경로 하나로 통일**할 수 있는가.

라우터가 존재하는 유일한 명분은 "학습 경로가 단순/희소 해상에서 진다" 이다.
★그 명분을 아무도 잰 적이 없다. 이 배치가 잰다.

표적 = `B` 배치(1,152런)에서 **라우터가 실제로 작동한 44 에피소드**
(= 시드 3개 중 한 번이라도 `v2_konly` 를 고른 것; 혼잡 11 · 희소 33).
⚠**선택이 in-sample 이다** — 같은 시드의 라우팅으로 표적을 골랐다. 범위에 명시한다.
"""
import json, os
HERE = os.path.dirname(os.path.abspath(__file__))
_T = json.load(open(os.path.join(HERE, '..', 'corpus_router_targets.json')))
TARGETS, SEEDS = _T['targets'], _T['seeds']
STRATUM = _T['stratum']

ARMS = {
    'ROUTED':        {},                            # 배포 기본값 = 라우터가 고른다
    'FORCE-LEARNED': {'init_method': 'learned'},    # 통일안 = 라우터 제거
}
ARM_ORDER = ['ROUTED', 'FORCE-LEARNED']
FIELD = 'avg_nRMSE'
THRESHOLD = 28      # /44, 단측 부호검정 p<0.05 (결과 전 확정)


def sign_test_one_sided(w, n):
    from math import comb
    return sum(comb(n, k) for k in range(w, n + 1)) / 2 ** n


def med_by_episode(rows, arm, field=FIELD):
    import statistics as st
    d = {}
    for r in rows:
        if r['cell'] == arm and field in r and r[field] is not None:
            d.setdefault(r['episode'], []).append(r[field])
    return {e: st.median(v) for e, v in d.items()}


def primary(rows, stratum=None):
    """H1 = 라우터가 제 값을 한다 = ROUTED 가 FORCE-LEARNED 보다 **낮다**(좋다)."""
    import statistics as st
    a, b = med_by_episode(rows, 'ROUTED'), med_by_episode(rows, 'FORCE-LEARNED')
    ks = sorted(set(a) & set(b))
    if stratum:
        ks = [k for k in ks if STRATUM.get(k) == stratum]
    n = len(ks)
    routed_better = sum(1 for k in ks if a[k] < b[k])
    rel = st.median([(b[k] - a[k]) / a[k] * 100 for k in ks]) if ks else float('nan')
    return {'n': n, 'routed_better': routed_better,
            'force_cost_pct_median': rel,          # +면 통일이 손해
            'p_one_sided': sign_test_one_sided(routed_better, n) if n else None}


def gates(rows):
    ok = [r for r in rows if 'error' not in r]
    exp = len(ARMS) * len(TARGETS) * len(SEEDS)
    out = [('G-0 런 수', len(ok) == exp, f'{len(ok)}/{exp}'),
           ('G-0b 에러 0', len(rows) == len(ok), f'{len(rows)-len(ok)}')]
    # ★G-1 강제가 실제로 먹었나 — 라우터 산출을 되읽는다
    f = [r for r in ok if r['cell'] == 'FORCE-LEARNED']
    bad = [r['job_id'] for r in f if r.get('auto_selected') is not None]
    out.append(('G-1 FORCE 팔에서 라우터 미작동', not bad, f'라우터가 돈 런 {len(bad)}'))
    # ★G-2 ROUTED 팔에서 라우터가 실제로 두 갈래를 냈나 (표적 선택이 유효한가)
    r_ = [r for r in ok if r['cell'] == 'ROUTED']
    sel = {}
    for r in r_: sel.setdefault(r.get('auto_selected'), 0); sel[r.get('auto_selected')] += 1
    out.append(('G-2 ROUTED 팔이 v2_konly 를 실제로 고름', sel.get('v2_konly', 0) > 0, str(sel)))
    # G-3 학습으로 간 런에서 VARPRO 가 발화했나
    lr = [r for r in ok if r.get('auto_selected') == 'learned' or r['cell'] == 'FORCE-LEARNED']
    out.append(('G-3 학습 경로 런에서 VARPRO 발화',
                all(r.get('varpro_fired') for r in lr),
                f'{sum(1 for r in lr if r.get("varpro_fired"))}/{len(lr)}'))
    return out


def label(p):
    if p['routed_better'] >= THRESHOLD:
        return ('ROUTER-EARNS',
                f'라우터가 제 값을 한다({p["routed_better"]}/{p["n"]}). '
                f'통일 비용 = 중앙 {p["force_cost_pct_median"]:+.2f}%.')
    return ('ROUTER-UNNECESSARY',
            f'라우터가 제 값을 못 한다({p["routed_better"]}/{p["n"]}, 문턱 {THRESHOLD}). '
            f'학습 경로로 통일해도 손해가 문턱 아래다(중앙 {p["force_cost_pct_median"]:+.2f}%).')
