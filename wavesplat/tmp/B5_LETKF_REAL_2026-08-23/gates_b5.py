"""B-5 단일출처 — ★★★**저장소 최초의 실측 외부 기준선 행**.

## 왜 이것이 오늘 마지막이자 가장 중요한 배치인가
저장소의 **9주째 미해결 결함**은 하나다: ***실측에서 외부 기준선 행이 0개다.***
`B-3`·`B-3b`·`B-3c`·`B-4` 는 전부 **우리 팔 안의 노브**이고, 저장소는 `SUPPORT-TRADEOFF` 축에서
그 부류를 열 번 물렸다. `B-2` 의 Wang 경쟁자는 자기검정 `G-8` 에서 떨어졌다.
⟹ **우리 자신의 합성 기준선(`LETKFGrid`)을 실측에 걸어본다.** 그것은 다른 방법 부류다
(격자 기반 앙상블 칼만 vs 격자 비의존 스펙트럼).

## ★기준선에 유리하게 설계한다 (S16 의 교훈: 기준선을 일부러 강화하라)
- **관측잡음을 참값으로 준다**(`obs_noise = 0.02` = 우리가 주입한 값). 우리 팔은 이 값을 모른다.
- **국소화를 두 개 돌리고 레코드별로 더 좋은 쪽을 준다**(`loc=50 m` · `loc=박스 대각/2`).
  ⚠**이것은 기준선에 오라클 선택을 주는 것**이고 **우리에게 불리한 방향**이다. 그렇게 적어 둔다.
- **격자를 채점 격자보다 촘촘하게 준다**(≈1 m, 64×64 상한). 채점은 이중선형 보간으로 되돌린다.
- **앙상블 100**, 기억창 전 프레임(319–405 사이클)으로 스핀업.

## ⚠자기검정 게이트 `G-8` (결과 전 확정, **필수**)
경쟁자가 **무기술이면 비교 자체가 무의미**하다(`B2_WANGFIX` 의 교훈).
⟹ `LETKF-BEST-DA` 가 **참 제로필드**(레코드별, `zero_ref.py`)를 **lead 0 에서 ≥8/12 레코드** 이겨야 한다.
- 실패 시 판정은 **`NO-VALID-COMPETITOR`** 이고 ⛔**어떤 우열 비교도 보고하지 않는다.**
- ⚠우리 팔은 같은 자리에서 **11/12** 다(33/36 창). **8/12 는 관대한 문턱**이다.

## 팔 (36 = 12 레코드 × 창 {1,3,5}, 잡음 0.02, 시드 42)
- `LETKF-DA50` · `LETKF-DABOX` — 미래 관측을 **동화**한다(우리 `REFIT2` 와 같은 스트림·같은 순서).
- `LETKF-FC50` — t0 이후 **관측 없이 예보만**(우리 `FROZEN` 과 같은 과제).
- `FROZEN`·`REFIT2` 는 `../B3_REFIT_2026-08-23/results_b3.jsonl` 에서 재사용.

## 판정 (결과 전 확정, `G-8` 통과 시에만)
**PRIMARY** = `REFIT2` < `LETKF-BEST-DA` 인 창 수(`fc_mean`), n=36:
- **≥26/36** ⟹ ★★★`SHIP-BEATS-LETKF-REAL (CONFIRMATORY)` — **저장소 최초의 실측 외부 기준선 우위.**
- **≥24/36** ⟹ ⚠`SHIP-BEATS-LETKF-REAL (EXPLORATORY)`
- **19~23** ⟹ `UNDECIDED-REAL`
- **≤18/36** ⟹ ⛔`LETKF-BEATS-SHIP-REAL` — **정직한 음성. 그대로 보고한다.**

**SECONDARY** = `FROZEN` vs `LETKF-FC50`(둘 다 순수 예보, 같은 문턱).
★**부차(서술 전용)**: lead 별 분해, 벽시계.
"""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
B2 = os.path.abspath(os.path.join(HERE, '..', 'B2_REAL_2026-08-23'))
B3 = os.path.abspath(os.path.join(HERE, '..', 'B3_REFIT_2026-08-23'))
for p in (HERE, B2, B3):
    if p not in sys.path: sys.path.insert(0, p)
from gates_b3 import (RECORDS, WINDOWS, SEED, NOISE, FIELD,           # noqa: F401
                      N_PAIRS, sign_test_one_sided, key, paired)

ARMS = {'LETKF-DA50':  dict(loc='fixed50', da=True),
        'LETKF-DABOX': dict(loc='box',     da=True),
        'LETKF-FC50':  dict(loc='fixed50', da=False)}
ARM_ORDER = ['LETKF-DA50', 'LETKF-DABOX', 'LETKF-FC50']
REUSED = ['FROZEN', 'REFIT2']
N_ENS, OBS_NOISE, GRID_MAX, GRID_DX = 100, 0.02, 64, 1.0
SELFTEST_MIN = 8          # /12 레코드, lead 0
NOMINAL, CONFIRM = 24, 26


def best_da(rows):
    """레코드-창별로 두 국소화 중 **더 좋은 쪽**을 고른 가상 팔 (기준선에 유리)."""
    out = {}
    for r in rows:
        if r.get('cell') not in ('LETKF-DA50', 'LETKF-DABOX') or 'error' in r:
            continue
        k = key(r)
        if k not in out or (r.get(FIELD) is not None and r[FIELD] < out[k][FIELD]):
            out[k] = r
    return [dict(v, cell='LETKF-BEST-DA') for v in out.values()]


def selftest(rows, zero_lead0):
    """G-8: LETKF-BEST-DA 가 lead 0 에서 참 제로필드를 이기는 **레코드** 수 (창 중앙으로 접음)."""
    import statistics as st
    per = {}
    for r in best_da(rows):
        if r.get('lead0') is not None:
            per.setdefault(r['record'], []).append(r['lead0'])
    w = sum(1 for rec, v in per.items() if st.median(v) < zero_lead0[rec])
    return w, len(per)


def gates(rows, zero_lead0=None):
    ok = [r for r in rows if 'error' not in r]
    new = [r for r in ok if r['cell'] in ARMS]
    exp = len(ARMS) * N_PAIRS
    out = [('G-0 신규 런 수', len(new) == exp, f'{len(new)}/{exp}'),
           ('G-0b 에러 0', len(rows) == len(ok), f'{len(rows)-len(ok)}')]
    for a in REUSED:
        n = sum(1 for r in ok if r['cell'] == a)
        out.append((f'G-1 재사용 팔 {a} {N_PAIRS}행', n == N_PAIRS, f'{n}/{N_PAIRS}'))
    out.append(('G-2 되읽기: 앙상블·잡음·국소화',
                all(r.get('n_ens') == N_ENS and abs(r.get('obs_noise', -1) - OBS_NOISE) < 1e-12
                    and r.get('loc_radius') for r in new), ''))
    # ★G-3 동화 팔은 미래 관측을 실제로 먹었나 / 예보 팔은 안 먹었나 (가지 단위 계수기)
    da = [r for r in new if r['cell'] in ('LETKF-DA50', 'LETKF-DABOX')]
    fc = [r for r in new if r['cell'] == 'LETKF-FC50']
    out.append(('G-3 DA 팔 미래 동화 >0', all((r.get('future_cycles') or 0) > 0 for r in da),
                str(sorted({r.get('future_cycles') for r in da}))[:50]))
    out.append(('G-3b FC 팔 미래 동화 0', all(r.get('future_cycles') == 0 for r in fc),
                str(sorted({r.get('future_cycles') for r in fc}))[:50]))
    out.append(('G-4 예측장 전건 유한', all(r.get('finite') for r in new), ''))
    out.append(('G-5 두 국소화가 실제로 다르다',
                len({(r['record'], r['loc_radius']) for r in new
                     if r['cell'] in ('LETKF-DA50', 'LETKF-DABOX')}) > len(RECORDS), ''))
    if zero_lead0 is not None:
        w, n = selftest(rows, zero_lead0)
        out.append((f'★G-8 자기검정 (LETKF 가 참 제로필드를 lead0 에서 이김, 문턱 {SELFTEST_MIN})',
                    w >= SELFTEST_MIN, f'{w}/{n} 레코드'))
    return out


def label(passed_selftest, p_primary, p_secondary, st_w, st_n):
    if not passed_selftest:
        return ('NO-VALID-COMPETITOR',
                f'⛔LETKF 가 실측에서 참 제로필드를 못 이긴다({st_w}/{st_n} 레코드, 문턱 '
                f'{SELFTEST_MIN}) ⟹ **우열 비교를 보고하지 않는다.** `B2_WANGFIX` 와 같은 결말이고, '
                '이는 **우리 재현에 대한 판정**이지 LETKF 방법에 대한 판정이 아니다.')
    w, n = p_primary['better'], p_primary['n']
    tail = (f' (부차: 순수 예보 `FROZEN` vs `LETKF-FC50` = '
            f'{p_secondary["better"]}/{p_secondary["n"]}, p={p_secondary["p_one_sided"]:.5f})')
    if w >= CONFIRM:
        return ('SHIP-BEATS-LETKF-REAL (CONFIRMATORY)',
                f'★★★배포 실측 경로(+2 s 재적합)가 LETKF 를 {w}/{n} 이긴다'
                f'(p={p_primary["p_one_sided"]:.6f}) — **저장소 최초의 실측 외부 기준선 우위.**' + tail)
    if w >= NOMINAL:
        return ('SHIP-BEATS-LETKF-REAL (EXPLORATORY)',
                f'{w}/{n} 명목 통과이나 가족단위 미달. ⚠확증 아님.' + tail)
    if w <= n - NOMINAL:
        return ('LETKF-BEATS-SHIP-REAL',
                f'⛔LETKF 가 우리를 {n-w}/{n} 이긴다 — **정직한 음성.**' + tail)
    return ('UNDECIDED-REAL', f'부호로 갈리지 않는다({w}/{n}).' + tail)
