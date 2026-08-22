"""W12 단일출처 — **실측까지 포함한 통일**이 가능한가.

배포는 실측 입력을 라우터 밖으로 빼내 `bank_mmse` 로 보낸다. 그 게이트의 사유가
코드에 축자로 적혀 있다(`sequential_update.py`, REAL-DOMAIN GATE):
  · 실측 30창 전부에서 제안의 `n_selected >= 8` ⟹ th=8 라우터라면 **30/30 학습 경로**
  · 그 팔은 *"no better and sometimes worse on real seas"* 로 측정됐다(2026-07)
  · 탈출구가 이미 있다: `real_domain_gate=False`
⟹ ★**이 배치는 그 "sometimes worse" 를 숫자로 만든다.**

⚠**사용자 지시로 상시지시(*"실측에 learned 강제 = 비승인 정책"*)를 이번 건에 한해 해제.**
"""
RECORDS = ['AA15a', 'AA15b', 'AA14', 'YS17', 'LJ18g']
WIN = 3
SEED = 42

ARMS = {
    'DEPLOYED':      {},                                                  # 실측 게이트 → bank_mmse
    'FORCE-LEARNED': {'real_domain_gate': False, 'init_method': 'learned'},
    'FORCE-V2K':     {'real_domain_gate': False, 'init_method': 'v2_konly'},
}
ARM_ORDER = ['DEPLOYED', 'FORCE-LEARNED', 'FORCE-V2K']
ZERO_FIELD = 1.0      # σ 정규화 ⟹ 제로필드 = 1.0 (정의상)

# ★n=5 ⟹ α=0.05 는 원리적으로 불가(부호검정 최소 양측 p = 2^-4 = 0.0625).
# 그래서 이 배치의 판정은 **기술적(DESCRIPTIVE)** 이고, 그렇게 결과 전에 못 박는다.
INFERENTIAL = False


def summarize(rows, arm, field='nowcast'):
    import statistics as st
    v = {r['record']: r[field] for r in rows
         if r['cell'] == arm and r.get(field) is not None}
    return {'arm': arm, 'n': len(v), 'median': st.median(v.values()) if v else None,
            'beats_zero': sum(1 for x in v.values() if x < ZERO_FIELD), 'per_record': v}


def paired(rows, arm, ref='DEPLOYED', field='nowcast'):
    import statistics as st
    a = {r['record']: r[field] for r in rows if r['cell'] == arm and r.get(field) is not None}
    b = {r['record']: r[field] for r in rows if r['cell'] == ref and r.get(field) is not None}
    ks = sorted(set(a) & set(b))
    return {'arm': arm, 'n': len(ks),
            'better_than_ref': sum(1 for k in ks if a[k] < b[k]),
            'rel_median_pct': st.median([(a[k] - b[k]) / b[k] * 100 for k in ks]) if ks else None}


def gates(rows):
    ok = [r for r in rows if 'error' not in r]
    exp = len(ARMS) * len(RECORDS)
    out = [('G-0 런 수', len(ok) == exp, f'{len(ok)}/{exp}'),
           ('G-0b 에러 0', len(rows) == len(ok), f'{len(rows)-len(ok)}')]
    # ★G-1 게이트 우회가 실제로 먹었나 — 라우터 산출 되읽기
    f = [r for r in ok if r['cell'] == 'FORCE-LEARNED']
    out.append(('G-1 FORCE-LEARNED 가 실제로 learned 로 갔나',
                all(r.get('route') == 'learned' for r in f),
                str({r['record']: r.get('route') for r in f})))
    d = [r for r in ok if r['cell'] == 'DEPLOYED']
    out.append(('G-2 DEPLOYED 는 bank_mmse 로 갔나',
                all(r.get('route') == 'bank_mmse' for r in d),
                str({r['record']: r.get('route') for r in d})))
    out.append(('G-3 학습 팔에서 VARPRO 발화',
                all(r.get('varpro_fired') for r in f),
                f'{sum(1 for r in f if r.get("varpro_fired"))}/{len(f)}'))
    return out


def label(fl, dep):
    """DESCRIPTIVE. 통일(학습 강제)이 실측에서 어떤 상태인가."""
    if fl['beats_zero'] < fl['n']:
        return ('REAL-UNIFY-LOSES-SKILL',
                f'학습 강제가 제로필드를 못 이기는 레코드가 있다 '
                f'({fl["beats_zero"]}/{fl["n"]} 만 통과) ⟹ 실측을 포함한 통일은 기술을 잃는다.')
    if fl['median'] > dep['median']:
        return ('REAL-UNIFY-COSTS',
                f'학습 강제가 전 레코드에서 제로필드는 이기나 배포보다 나쁘다 '
                f'(중앙 {fl["median"]:.4f} vs {dep["median"]:.4f}) ⟹ 통일 비용이 있다.')
    return ('REAL-UNIFY-OK',
            f'학습 강제가 배포에 뒤지지 않는다(중앙 {fl["median"]:.4f} vs {dep["median"]:.4f}).')
