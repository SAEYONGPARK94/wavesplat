"""W10 단일출처 — 창-안 벽의 **남은 절반**을 2×2 로 가른다.

W9: 척도불변(λ_l1 만)을 켜니 벽이 1.5163 → 1.2752 (초과분 47% 제거). 나머지는?
첫 후보 = `lambda_smooth` 가 **여전히 [m] 단위**다 (`sequential_update.py` 의 척도불변
분기가 `lambda_l1` 에만 붙는다). 이 배치는 그 요인을 **직접 절제**한다.

★설계: W9 의 두 셀(smooth=1e-4 × SI off/on)을 그대로 재사용하고
**smooth=0 두 셀만 새로 돌린다** ⟹ 32런으로 완전한 2×2.
"""
ARMS = {   # 새로 도는 팔만
    'ORA-WPHI-NOSM':    dict(extra={'tracking_lr': 0.0, 'lambda_l1': 1e-4,
                                    'lambda_smooth': 0.0}, scale_invariant=False),
    'ORA-WPHI-SI-NOSM': dict(extra={'tracking_lr': 0.0, 'lambda_l1': 1e-4,
                                    'lambda_smooth': 0.0}, scale_invariant=True),
}
ARM_ORDER = ['ORA-WPHI-NOSM', 'ORA-WPHI-SI-NOSM']
# W9 에서 그대로 빌려오는 셀 (같은 시드 1619, 같은 에피소드, 같은 코드)
W9_DIR = '../W9_PENALTY_BUDGET_2026-08-22'
W9_CELLS = {'ORA-WPHI-0STEP': 'ref', 'ORA-WPHI-SHIP': 'smooth_on_si_off',
            'ORA-WPHI-SI': 'smooth_on_si_on'}
REF = 'ORA-WPHI-0STEP'
REGIME, N_EPISODES, SEEDS = 'crowded', 16, [1619]
FIELD = 'window_resid'

# ★결과 전 확정 — W9 실측값 (인용 아니라 이 배치의 기준점)
R_BASE = 1.5163     # smooth on, SI off  (= 출하)
R_SI   = 1.2752     # smooth on, SI on
# 초과분 = ratio - 1. 요인이 설명하는 몫 = 제거된 초과분 / 기준 초과분.
BASE_EXCESS = R_BASE - 1.0


def sign_test_one_sided(w, n):
    from math import comb
    return sum(comb(n, k) for k in range(w, n + 1)) / 2 ** n


def ratio_vs_ref(rows, arm, field=FIELD):
    import statistics as st
    a = {r['episode']: r[field] for r in rows if r['cell'] == arm and field in r}
    b = {r['episode']: r[field] for r in rows if r['cell'] == REF and field in r}
    ks = sorted(set(a) & set(b)); n = len(ks)
    w = sum(1 for k in ks if a[k] > b[k])
    return {'arm': arm, 'n': n, 'above': w,
            'ratio': st.median([a[k] / b[k] for k in ks if b[k]]) if ks else float('nan'),
            'p_one_sided': sign_test_one_sided(w, n) if n else None}


def share(ratio):
    """이 팔이 설명하는 기준 초과분의 몫."""
    return (BASE_EXCESS - (ratio - 1.0)) / BASE_EXCESS


def gates(rows):
    ok = [r for r in rows if 'error' not in r]
    exp = len(ARMS) * N_EPISODES * len(SEEDS)
    new = [r for r in ok if r['cell'] in ARMS]
    out = [('G-0 신규 런 수', len(new) == exp, f'{len(new)}/{exp}'),
           ('G-0b 에러 0', len(rows) == len(ok), f'{len(rows)-len(ok)}')]
    bad = [r['job_id'] for r in new
           if r.get('scale_invariant_readback') is not ARMS[r['cell']]['scale_invariant']]
    out.append(('G-1 척도불변 되읽기', not bad, f'불일치 {len(bad)}'))
    bad2 = [r['job_id'] for r in new
            if abs(r.get('lambda_smooth_readback', -1)) > 1e-12]
    out.append(('G-1b λ_smooth=0 객체 되읽기', not bad2, f'불일치 {len(bad2)}'))
    # ★G-2 W9 재사용 셀이 같은 자료인가 — 기준 팔의 값이 바뀌면 비교가 깨진다
    ref = [r for r in ok if r['cell'] == REF]
    out.append(('G-2 W9 기준 팔 16행 존재', len(ref) == N_EPISODES, f'{len(ref)}/{N_EPISODES}'))
    v = ratio_vs_ref(ok, 'ORA-WPHI-SHIP')
    out.append(('G-3 양성대조 = W9 재현 (ABORT)',
                abs(v['ratio'] - R_BASE) < 1e-6, f'{v["ratio"]:.4f} vs {R_BASE}'))
    return out


def label(nosm, si_nosm):
    """PRIMARY = smooth 절제가 남은 절반을 설명하는가."""
    s_nosm, s_both = share(nosm['ratio']), share(si_nosm['ratio'])
    if s_both >= 0.95:
        return ('SMOOTH-COMPLETES', f'두 벌점을 다 고치면 벽이 사실상 사라진다(설명 {s_both:.0%}) '
                                    '⟹ 창-안 벽 = 벌점 단위 문제 전부.')
    if s_nosm > 0.10:
        return ('SMOOTH-PARTIAL', f'λ_smooth 도 기여한다(단독 설명 {s_nosm:.0%}, 합계 {s_both:.0%}) '
                                  '그러나 벽이 남는다 ⟹ 제3의 원인이 있다.')
    return ('SMOOTH-NULL', f'λ_smooth 는 무관하다(단독 설명 {s_nosm:.0%}) '
                           '⟹ 남은 절반은 벌점이 아니라 옵티마이저/제약 쪽이다.')
