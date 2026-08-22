"""W16 단일출처 — **새 배포 구성에서 헤드라인 재확립** (P1).

2026-08-22 에 출하 기본값이 셋 바뀌었다(`varpro_steps` 400 · `learned_init_iters` 0 ·
`auto_switch_thresh` 0). ⟹ 과거 헤드라인(S13·S13b·HOSGATE·B)은 **구 구성의 수**이고
새 구성과 **비교 불가**다(검수 V3 census). 이 배치가 **살아있는 측정 #1·#2 를 새 구성에서 다시 세운다.**

★`ZERO` 팔은 **런이 필요 없다** — 제로필드 예측은 추정기 설정과 무관하므로
`B` 배치의 에피소드별 `zero_nRMSE` 를 그대로 쓴다(구성 불변량).
"""
import json, os
HERE = os.path.dirname(os.path.abspath(__file__))
_C = json.load(open(os.path.join(HERE, 'corpus.json')))
EPISODES = _C['crowded'] + _C['sparse']
STRATUM = {e: 'crowded' for e in _C['crowded']}
STRATUM.update({e: 'sparse' for e in _C['sparse']})
SEEDS = [1629, 1630, 1631]
ARMS = {'SHIP': {}}                     # 새 배포 기본값 그대로
ARM_ORDER = ['SHIP']
FIELD = 'avg_nRMSE'
THRESHOLD_CROWDED = 41                  # /64, 단측 p<0.05 (결과 전 확정)
THRESHOLD_SPARSE = 41


def sign_test_one_sided(w, n):
    from math import comb
    return sum(comb(n, k) for k in range(w, n + 1)) / 2 ** n


def zero_ref():
    z = {}
    for l in open(os.path.join(HERE, '..', 'B_DEPLOY_2026-08-19', 'results.jsonl')):
        r = json.loads(l)
        if 'error' not in r and r.get('zero_nRMSE') is not None:
            z[r['episode']] = r['zero_nRMSE']
    return z


def med_by_episode(rows, arm='SHIP', field=FIELD):
    import statistics as st
    d = {}
    for r in rows:
        if r['cell'] == arm and r.get(field) is not None:
            d.setdefault(r['episode'], []).append(r[field])
    return {e: st.median(v) for e, v in d.items()}


def primary(rows, stratum=None):
    """H1 = SHIP 이 제로필드보다 **낮다**(= 기술이 있다)."""
    import statistics as st
    a, z = med_by_episode(rows), zero_ref()
    ks = sorted(set(a) & set(z))
    if stratum:
        ks = [k for k in ks if STRATUM.get(k) == stratum]
    n = len(ks)
    w = sum(1 for k in ks if a[k] < z[k])
    return {'n': n, 'ship_better': w,
            'rel_median_pct': st.median([(a[k] - z[k]) / z[k] * 100 for k in ks]) if ks else None,
            'p_one_sided': sign_test_one_sided(w, n) if n else None}


def gates(rows):
    ok = [r for r in rows if 'error' not in r]
    exp = len(EPISODES) * len(SEEDS)
    out = [('G-0 런 수', len(ok) == exp, f'{len(ok)}/{exp}'),
           ('G-0b 에러 0', len(rows) == len(ok), f'{len(rows)-len(ok)}')]
    # ★G-1 새 배포 구성이 실제로 걸렸나 — 세 키 전부 되읽기
    bad = [r['job_id'] for r in ok
           if r.get('varpro_steps') != 400 or r.get('init_iters') != 0
           or r.get('auto_selected') != 'learned']
    out.append(('G-1 새 배포 구성 전건 적용', not bad, f'이탈 {len(bad)}'))
    out.append(('G-2 셸 분해 기록됨',
                all(r.get('avg_shell_rmse') for r in ok),
                f'{sum(1 for r in ok if r.get("avg_shell_rmse"))}/{len(ok)}'))
    out.append(('G-3 제로필드 기준 128 에피소드 확보',
                len(set(zero_ref()) & set(EPISODES)) == len(EPISODES), ''))
    return out


def label(pc, ps):
    ok_c = pc['ship_better'] >= THRESHOLD_CROWDED
    ok_s = ps['ship_better'] >= THRESHOLD_SPARSE
    if ok_c and ok_s:
        return ('HEADLINE-HOLDS-BOTH', '새 배포 구성에서 두 층 모두 제로필드 우위가 재확립된다.')
    if ok_c:
        return ('HEADLINE-HOLDS-CROWDED', '혼잡만 재확립. 희소는 문턱 미달.')
    if ok_s:
        return ('HEADLINE-HOLDS-SPARSE', '희소만 재확립. 혼잡은 문턱 미달.')
    return ('HEADLINE-LOST', '⛔새 배포 구성에서 제로필드 우위가 두 층 모두 재확립되지 않는다.')
