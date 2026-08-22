"""B-2 단일출처 — **실측에서 제안 표현이 무엇을 하는가** (학위 구멍 B).

⚠**축소 실행이다.** 원 설계(D1)는 Wang 경쟁자 팔을 포함했으나 선행 자기검정 `G-8` 에서
**떨어졌다**(`tmp/B2_WANGFIX_2026-08-22/RESULT.md`: 수리해도 제로필드 0.2502→0.2510).
⟹ ⛔**경쟁자 비교는 뺀다.** 남은 질문 = ***"우리 제안 표현이 실측에서 기술을 갖는가."***

★**전방팬을 뺐다** ⟹ 레코드 **5 → 12**. 실측에서 **처음으로 α=0.05 가 가능**하다
(부호검정 바닥 p: n=5 는 0.0625(불가), **n=12 는 0.0005**).
사유는 자료가 아니라 관측 모형이었다 — `RECORDS` 주석 축자
*"only records whose patch can host a (possibly scaled) forward fan"*,
그리고 **보유 실측 13개는 전부 고정 관측소**다.
"""
import os
RECORDS = ['AA14', 'AA15a', 'AA15b', 'BS11a', 'BS11b', 'BS11c', 'BS11d',
           'BS13a', 'BS13b', 'BS13c', 'LJ18a', 'YS17']
# ★결과 전 선언: 박스 크기로 층을 나눈다(추정기 성능과 무관한 기하 속성).
LARGE_BOX = ['AA14', 'AA15a', 'AA15b', 'LJ18a', 'YS17']      # 40 m 이상
SMALL_BOX = ['BS11a', 'BS11b', 'BS11c', 'BS11d', 'BS13a', 'BS13b', 'BS13c']  # 13–19 m, 개구/λ≈1
WIN = 3
SEED = 42
ZERO_REF = 1.0          # σ 정규화 ⟹ 제로필드 = 1.0 (정의상, 손복사 아님)

ARMS = {
    'SHIP': {},                                                   # 배포 = 실측 게이트 → bank_mmse
    'PROP': {'real_domain_gate': False, 'init_method': 'learned'},  # ★제안 표현 강제
}
ARM_ORDER = ['SHIP', 'PROP']
FIELD = 'nowcast'
THRESHOLD = 10          # /12, 단측 부호검정 p=0.0193 (결과 전 확정)

# ⚠결과 전에 기록하는 자료 특성 (사후 배제 금지의 근거)
NOTES = {
    'LJ18a': '진폭 분포가 비대칭(범위 -10.4~4.2 m, 극값 질량 0%). 클램프 아님. 포함한다.',
    'LJ18b': '⛔제외 — 극값 질량 31.0% = 문서화된 클램프 인공물(2026-08-13 감사). '
             '결과 전 배제이며 성능과 무관한 자료 결함이다.',
    'BS*':   '박스 13–19 m 로 개구/λ≈1. **훨씬 어렵다.** ⛔사후 배제 금지.',
}


def sign_test_one_sided(w, n):
    from math import comb
    return sum(comb(n, k) for k in range(w, n + 1)) / 2 ** n


def summarize(rows, arm, field=FIELD, subset=None):
    import statistics as st
    v = {r['record']: r[field] for r in rows
         if r['cell'] == arm and r.get(field) is not None
         and (subset is None or r['record'] in subset)}
    n = len(v)
    w = sum(1 for x in v.values() if x < ZERO_REF)
    return {'arm': arm, 'n': n, 'beats_zero': w,
            'median': st.median(v.values()) if v else None,
            'p_one_sided': sign_test_one_sided(w, n) if n else None}


def paired(rows, arm, ref, field=FIELD, subset=None):
    import statistics as st
    a = {r['record']: r[field] for r in rows if r['cell'] == arm and r.get(field) is not None}
    b = {r['record']: r[field] for r in rows if r['cell'] == ref and r.get(field) is not None}
    ks = sorted(set(a) & set(b))
    if subset: ks = [k for k in ks if k in subset]
    n = len(ks)
    w = sum(1 for k in ks if a[k] < b[k])
    return {'arm': arm, 'ref': ref, 'n': n, 'better': w,
            'rel_median_pct': st.median([(a[k] - b[k]) / b[k] * 100 for k in ks]) if ks else None,
            'p_one_sided': sign_test_one_sided(w, n) if n else None}


def gates(rows):
    ok = [r for r in rows if 'error' not in r]
    exp = len(ARMS) * len(RECORDS)
    out = [('G-0 런 수', len(ok) == exp, f'{len(ok)}/{exp}'),
           ('G-0b 에러 0', len(rows) == len(ok), f'{len(rows)-len(ok)}')]
    prop = [r for r in ok if r['cell'] == 'PROP']
    ship = [r for r in ok if r['cell'] == 'SHIP']
    # ★G-1 제안 표현이 실제로 돌았나 — 성분 수와 VARPRO 로 확인(라우터 산출 아님)
    out.append(('G-1 PROP 이 Gaussian 성분을 만들었나',
                all((r.get('n_atoms') or 0) > 0 for r in prop),
                str({r['record']: r.get('n_atoms') for r in prop})[:110]))
    out.append(('G-1b PROP 전건 VARPRO 발화',
                all(r.get('varpro_fired') for r in prop),
                f'{sum(1 for r in prop if r.get("varpro_fired"))}/{len(prop)}'))
    # ★G-2 배포 팔은 bank 경로여야 한다 (음성 대조)
    out.append(('G-2 SHIP 은 성분 0개(bank 경로)',
                all((r.get('n_atoms') or 0) == 0 for r in ship),
                str({r['record']: r.get('n_atoms') for r in ship})[:110]))
    # ★G-3 양 팔이 **같은 관측·같은 질의**를 받았나
    out.append(('G-3 팔 간 관측·질의 해시 일치',
                len({(r['record'], r.get('obs_hash'), r.get('q_hash')) for r in ok})
                == len({r['record'] for r in ok}),
                ''))
    out.append(('G-4 12 레코드 전건', len({r['record'] for r in ok}) == len(RECORDS), ''))
    return out


def label(prop, ship, prop_vs_ship):
    """PRIMARY = PROP 이 실측에서 제로필드를 이기는가 (n=12, 문턱 10)."""
    if prop['beats_zero'] >= THRESHOLD:
        return ('REAL-SKILL-CONFIRMED',
                f'제안 표현이 실측에서 기술을 갖는다({prop["beats_zero"]}/{prop["n"]}, '
                f'p={prop["p_one_sided"]:.4f}). ★학위 구멍 B 가 닫힌다.')
    if prop['beats_zero'] <= prop['n'] - THRESHOLD:
        return ('REAL-NO-SKILL',
                f'제안 표현이 실측에서 무기술이다({prop["beats_zero"]}/{prop["n"]}). '
                '⟹ 구멍 B 는 "측정된 한계" 로 확정된다(정직한 음성).')
    return ('REAL-UNDECIDED',
            f'부호로 갈리지 않는다({prop["beats_zero"]}/{prop["n"]}). 크기를 따로 읽어라.')
