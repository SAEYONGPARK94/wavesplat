"""W17 단일출처 — **창-안 벽의 남은 28% 는 무엇인가**.

W9·W10 이 벽을 회계했다(혼잡 n=16, 진실 0-스텝 기준, 창 안 비):
  · `λ_l1` 차원 불일치      47%
  · `λ_smooth` 단독          9%
  · 두 벌점의 상호작용       16%
  · ⛔**미지**              **28%**   ← 이 배치가 잰다

두 벌점을 다 고친 팔(`SI-NOSM`)이 여전히 **1.1457** 이다. 남은 후보는 **옵티마이저**다:
유한 스텝(미수렴) · 학습률 · 급경사 클램프. **한 번에 하나씩 절제한다.**
"""
import os
HERE = os.path.dirname(os.path.abspath(__file__))
W10_DIR = '../W10_SMOOTH_2026-08-22'
W9_DIR = '../W9_PENALTY_BUDGET_2026-08-22'
BORROWED = {'ORA-WPHI-0STEP': W9_DIR, 'ORA-WPHI-SI-NOSM': W10_DIR}
REF = 'ORA-WPHI-0STEP'

_SI = {'tracking_lr': 0.0, 'lambda_l1': 1e-4, 'lambda_smooth': 0.0}
ARMS = {   # 신규만. 전부 척도불변 ON + smooth 0 (= W10 최선 팔) 위에 한 요인씩
    'SI-NOSM-STEPS10': dict(extra=dict(_SI, max_iters_per_stage=[2000, 2000, 5000]),
                            scale_invariant=True),
    'SI-NOSM-LR10X':   dict(extra=dict(_SI, lr=1e-3), scale_invariant=True),
    'SI-NOSM-NOCLAMP': dict(extra=dict(_SI, max_steepness=1e9), scale_invariant=True),
}
ARM_ORDER = ['SI-NOSM-STEPS10', 'SI-NOSM-LR10X', 'SI-NOSM-NOCLAMP']
REGIME, N_EPISODES, SEEDS = 'crowded', 16, [1632]
FIELD = 'window_resid'
R_BASE = None      # ★손복사 금지 — W9/W10 원본에서 재계산한다


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


def ratio_vs_ref(rows, arm, field=FIELD):
    import statistics as st
    a, b = med_by_episode(rows, arm, field), med_by_episode(rows, REF, field)
    ks = sorted(set(a) & set(b)); n = len(ks)
    w = sum(1 for k in ks if a[k] > b[k])
    return {'arm': arm, 'n': n, 'above': w,
            'ratio': st.median([a[k] / b[k] for k in ks if b[k]]) if ks else float('nan'),
            'p_one_sided': sign_test_one_sided(w, n) if n else None}


def share(ratio, base):
    """이 팔이 없앤, `SI-NOSM` 잔여 초과분의 비율."""
    return (base - 1.0 - (ratio - 1.0)) / (base - 1.0) if base > 1.0 else float('nan')


def gates(rows):
    ok = [r for r in rows if 'error' not in r]
    new = [r for r in ok if r['cell'] in ARMS]
    exp = len(ARMS) * N_EPISODES * len(SEEDS)
    out = [('G-0 신규 런 수', len(new) == exp, f'{len(new)}/{exp}'),
           ('G-0b 에러 0', len(rows) == len(ok), f'{len(rows)-len(ok)}')]
    out.append(('G-1 척도불변 되읽기', all(r.get('scale_invariant_readback') for r in new),
                f'{sum(1 for r in new if r.get("scale_invariant_readback"))}/{len(new)}'))
    out.append(('G-1b λ_smooth=0 되읽기',
                all(abs(r.get('lambda_smooth_readback', 1)) < 1e-12 for r in new), ''))
    for a in BORROWED:
        n = sum(1 for r in ok if r['cell'] == a)
        out.append((f'G-2 차용 팔 {a} 16행', n == N_EPISODES, f'{n}/{N_EPISODES}'))
    return out


def label(res, base):
    """어느 요인이 잔여를 없애는가."""
    best = min(res, key=lambda a: res[a]['ratio'])
    s = share(res[best]['ratio'], base)
    if s >= 0.80:
        return (f'RESIDUAL-IS-{best}', f'잔여의 {s:.0%}가 `{best}` 로 사라진다 ⟹ 남은 28%는 그 요인이다.')
    if max(share(res[a]['ratio'], base) for a in res) < 0.20:
        return ('RESIDUAL-UNEXPLAINED',
                '세 요인 어느 것도 잔여를 20% 이상 못 없앤다 ⟹ 옵티마이저 절제로는 설명되지 않는다.')
    return ('RESIDUAL-PARTIAL',
            f'최선 `{best}` 가 잔여의 {s:.0%}만 없앤다 ⟹ 부분 설명, 복합 원인.')
