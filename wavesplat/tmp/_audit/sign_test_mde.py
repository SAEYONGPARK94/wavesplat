#!/usr/bin/env python3
"""★부호검정 MDE 의 **유일 구현** — `tmp/_audit/sign_test_mde.py`.

⛔**이 파일을 복사하지 마라.** 새 배치의 `gates_*.py`·`consts_*.py` 는 이렇게 부른다:

    import os, sys
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '_audit')))
    from sign_test_mde import mde_q, k_one_sided, k_two_sided   # noqa: E402

    THRESH = k_one_sided(N, 0.05)
    Q80    = mde_q(N, THRESH, 0.80)          # ⟹ 음성 결과 인용 시 **병기 의무**

──────────────────────────────────────────────────────────────────────────────
★왜 이 파일이 생겼나 (2026-09-05 수치 정합 감사)
──────────────────────────────────────────────────────────────────────────────
`T11`~`T16` 의 MDE 상수를 독립 재도출한 결과 **두 부류의 결함**이 나왔다.

  (a) ⛔**내림(round-to-nearest)이 검정력을 문턱 아래로 떨군다.**
      `T12 sec_q80 = 0.8332` 의 실제 검정력은 **0.79995** 이고
      `T14 primary_q80 = 0.7630` 은 **0.79986** 이다. 「검정력 0.80」 이라 광고하는데
      0.80 이 **아니다**. ★크기는 작지만 방향이 틀렸다 — MDE 를 낮게 적는 것은
      ***설계가 실제보다 예민하다고 말하는 것***이라 음성 결과의 면죄부가 된다.

  (b) ⛔**같은 양이 배치마다 다르게 인쇄됐다.** `n=28 · crit 19 · 검정력 .80` 이
      `T12`·`T14` 에서는 `0.7304`, `T13` 에서는 `0.7305` 다.
      ★이것이 이 저장소의 이름 붙은 누적 결함 **「손복사 기준 상수」**이고,
      하필 ***모든 음성 결과가 병기해야 하는 상수 안에서*** 재발했다.

  ✅좋은 선례도 이미 있었다: `T11 q90 = 0.7955` 는 주석이 스스로
     *"0.795 의 검정력은 0.8998 < 0.90"* 이라 적어 **정직하게** 처리했다.

──────────────────────────────────────────────────────────────────────────────
★반올림 규약 (이 파일이 그 유일출처다)
──────────────────────────────────────────────────────────────────────────────
**MDE 는 참값보다 낮게 보고되어서는 안 된다.** 낮은 MDE = *"이만큼 작은 효과도
잡아낸다"* 이므로, 내림은 **설계의 검출력을 과대주장**한다. ⟹ 규약은

    보고값 = 4자리 격자 위에서 **실제로 목표 검정력을 달성하는** 가장 가까운 값

  · 이김 다리(`side='upper'`, 기각 `W>=crit`) : q > .5 ⟹ **올림**
        mde_q = min{ q ∈ 10^-nd 격자 : P(W>=crit | n,q) >= power }
  · 해침 다리(`side='lower'`, 기각 `W<=crit`) : q < .5 ⟹ **내림**
        mde_q = max{ q ∈ 10^-nd 격자 : P(W<=crit | n,q) >= power }

★두 규칙은 **하나**다: *효과 크기 `|q − 0.5|` 를 언제나 0.5 에서 **멀어지는 쪽으로**
반올림한다.* ⟹ 어느 다리에서도 보고값은 **보수적**이다.

★정의가 「연속해의 반올림」이 아니라 **「격자 위의 달성 가능한 최솟값」**인 것이 요점이다
— 그래서 반환값 자신이 항상 `power_at(반환값) >= power` 를 만족한다(부동소수 사정에
의존하지 않는다). 이분법은 **정수 격자** 위에서 돌고 검정력은 `Fraction` **정확 산술**로
평가하므로 결과는 재현적이고 반올림 오차가 없다.

⚠**보수적인 쪽으로 어긋난 옛 값은 결함이 아니다** — 격자 최솟값보다 큰(이김) / 작은(해침)
값은 검정력이 목표 이상이므로 안전하다. 검사기가 그것을 `보수적` 으로 따로 센다.

⛔이 함수는 **판정에 쓰이지 않는다.** MDE 는 라벨을 만들지 않고 **공시**에만 들어간다
(문턱 `crit` 이 라벨을 만든다). 이 사실이 이 감사의 파급을 가둔다.

자가진단: `python3 tmp/_audit/sign_test_mde.py --selftest`
"""
from fractions import Fraction
from math import comb

__all__ = ['p_ge', 'p_le', 'power_at', 'k_one_sided', 'k_two_sided',
           'mde_q', 'mde_exact', 'ND_DEFAULT', 'ROUNDING_RULE']

ND_DEFAULT = 4
ROUNDING_RULE = ('MDE 는 참값보다 낮게 보고하지 않는다 — 효과 크기 |q-0.5| 를 '
                 '0.5 에서 멀어지는 쪽으로 반올림하고, 반환값 자신이 목표 검정력을 '
                 '달성한다(격자 위 달성 가능 최솟값).')


def _q(q):
    """`float`·`int`·`Fraction` 를 **정확한** `Fraction` 으로 (부동소수 값 그대로)."""
    return q if isinstance(q, Fraction) else Fraction(q)


def p_ge(k, n, q=Fraction(1, 2)):
    """`P(W >= k | n, q)` — 정확 산술."""
    q = _q(q)
    return sum(comb(n, i) * q ** i * (1 - q) ** (n - i) for i in range(k, n + 1))


def p_le(k, n, q=Fraction(1, 2)):
    """`P(W <= k | n, q)` — 정확 산술."""
    q = _q(q)
    return sum(comb(n, i) * q ** i * (1 - q) ** (n - i) for i in range(0, k + 1))


def power_at(q, n, crit, side='upper'):
    """주어진 참 승률 `q` 에서의 **검정력** (float). ⚠계산 자체는 정확 산술이다."""
    _check_side(side)
    return float(p_ge(crit, n, q) if side == 'upper' else p_le(crit, n, q))


def _check_side(side):
    if side not in ('upper', 'lower'):
        raise ValueError(f"side 는 'upper'|'lower' — 받은 값 {side!r}")


def k_one_sided(n, alpha=0.05):
    """단측 문턱 `min{k : P(W>=k | n,.5) <= alpha}` — 없으면 `None`(기각 원리적 불가)."""
    a = _q(alpha)
    c = [k for k in range(n + 1) if p_ge(k, n) <= a]
    return min(c) if c else None


def k_two_sided(n, alpha=0.05):
    """양측 문턱 `k⁺ = min{k : P(W>=k | n,.5) <= alpha/2}` — 없으면 `None`.

    ⚠`alpha` 는 **양측 전체**의 유의수준이다(Bonferroni 보정은 호출자가 미리 나눈다).
    """
    a = _q(alpha) / 2
    c = [k for k in range(n + 1) if p_ge(k, n) <= a]
    return min(c) if c else None


def mde_q(n, crit, power, side='upper', nd=ND_DEFAULT, alpha=None):
    """★**최소 검출 효과** — 목표 검정력을 실제로 달성하는 `10^-nd` 격자 위의 값.

    인자
      n     : 블록 수(부호검정의 시행 수)
      crit  : 기각 문턱. `None` 이면 `alpha` 로부터 유도한다
              (`side='upper'` → `k_one_sided`, `side='lower'` → 대칭 `n - k_one_sided`).
      power : 목표 검정력(예: 0.80)
      side  : `'upper'` = 기각 `W>=crit` (q>.5) · `'lower'` = 기각 `W<=crit` (q<.5)
      nd    : 보고 자릿수(이 저장소 규약 = 4)
      alpha : `crit` 이 `None` 일 때만 쓴다(단측 α). ⚠Bonferroni 는 호출자가 나눈다.

    반환  : float. **문턱이 존재하지 않으면 `None`**(그 `n` 에서는 어떤 자료로도 기각 불가).

    ⛔반환값을 다시 반올림하지 마라 — 이미 규약대로 반올림된 값이다(모듈 독스트링).
    """
    _check_side(side)
    if crit is None:
        if alpha is None:
            raise ValueError('crit 이 None 이면 alpha 가 필요하다')
        k = k_one_sided(n, alpha)
        if k is None:
            return None
        crit = k if side == 'upper' else n - k
    scale = 10 ** nd
    tgt = _q(power)
    if side == 'upper':
        # 검정력은 q 에 대해 **증가** ⟹ 조건을 만족하는 최소 격자점.
        lo, hi = scale // 2, scale                      # q ∈ [0.5, 1.0]
        if p_ge(crit, n, Fraction(hi, scale)) < tgt:
            return None                                 # q=1 에서도 미달(crit>n 등)
        while lo < hi:
            mid = (lo + hi) // 2
            if p_ge(crit, n, Fraction(mid, scale)) >= tgt:
                hi = mid
            else:
                lo = mid + 1
        return lo / scale
    # 해침 다리 — 검정력은 q 에 대해 **감소** ⟹ 조건을 만족하는 최대 격자점.
    lo, hi = 0, scale // 2                              # q ∈ [0.0, 0.5]
    if p_le(crit, n, Fraction(lo, scale)) < tgt:
        return None
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if p_le(crit, n, Fraction(mid, scale)) >= tgt:
            lo = mid
        else:
            hi = mid - 1
    return lo / scale


def mde_exact(n, crit, power, side='upper', iters=60):
    """연속해(반올림 **전**) — ⛔보고하지 마라. 진단·감사 출력 전용이다.

    ⚠`iters=60` 이면 구간 폭이 `0.5/2^60 ~ 4e-19` 로 float 해상도 아래다(200 은 낭비였다).
    """
    _check_side(side)
    tgt = float(power)
    if side == 'upper':
        lo, hi = 0.5, 1.0
        for _ in range(iters):
            mid = (lo + hi) / 2.0
            if power_at(mid, n, crit, side) < tgt:
                lo = mid
            else:
                hi = mid
    else:
        lo, hi = 0.0, 0.5
        for _ in range(iters):
            mid = (lo + hi) / 2.0
            if power_at(mid, n, crit, side) < tgt:
                hi = mid
            else:
                lo = mid
    return (lo + hi) / 2.0


# ── 자가진단 ────────────────────────────────────────────────────────────────
def _selftest():
    """★규약이 **규약인가** — 항진이 아니라 성질을 검사한다."""
    bad = []

    def chk(cond, msg):
        if not cond:
            bad.append(msg)

    # 1. 반환값 자신이 목표 검정력을 **달성**한다 (규약의 전부).
    for n in range(8, 65, 7):
        for a in (0.05, 0.025, 0.00625):
            k = k_one_sided(n, a)
            if k is None:
                continue
            for pw in (0.50, 0.80, 0.90, 0.95):
                q = mde_q(n, k, pw)
                chk(q is not None and power_at(q, n, k) >= pw,
                    f'상측 달성 실패 n={n} k={k} pw={pw} q={q}')
                # 2. **최소성** — 한 눈금 아래는 미달이어야 한다(과대보고 금지).
                chk(power_at(round(q - 1e-4, 4), n, k) < pw,
                    f'상측 최소성 실패 n={n} k={k} pw={pw} q={q}')

    # 3. 해침 다리 — 대칭.
    for n in (16, 28, 64):
        k = n - k_one_sided(n, 0.025)
        for pw in (0.50, 0.80, 0.90):
            q = mde_q(n, k, pw, side='lower')
            chk(q is not None and power_at(q, n, k, 'lower') >= pw,
                f'하측 달성 실패 n={n} k={k} pw={pw} q={q}')
            chk(power_at(round(q + 1e-4, 4), n, k, 'lower') < pw,
                f'하측 최대성 실패 n={n} k={k} pw={pw} q={q}')

    # 4. ★**안전 방향** — 규약값은 언제나 연속해보다 0.5 에서 멀거나 같다.
    for n, k in ((24, 17), (28, 19), (28, 20), (52, 33), (60, 39), (60, 40)):
        q, x = mde_q(n, k, 0.80), mde_exact(n, k, 0.80)
        chk(q >= x - 1e-12, f'안전 방향 위반(상측) n={n} k={k}: {q} < {x}')
    q, x = mde_q(64, 23, 0.80, side='lower'), mde_exact(64, 23, 0.80, side='lower')
    chk(q <= x + 1e-12, f'안전 방향 위반(하측): {q} > {x}')

    # 5. ⛔**항진 방지** — 규약을 어긴 값(내림)은 반드시 「미달」로 판정돼야 한다.
    #    `T14 primary_q80` 이 실제로 그 값이다.
    chk(power_at(0.7630, 28, 20) < 0.80, '항진: 0.7630@n28k20 이 0.80 이상으로 계산됐다')
    chk(mde_q(28, 20, 0.80) == 0.7631, f'T14 규약값 불일치: {mde_q(28, 20, 0.80)}')
    chk(mde_q(28, 19, 0.80) == 0.7305, f'crit19@n28 규약값 불일치: {mde_q(28, 19, 0.80)}')

    # 6. 도달 불가 문턱 — `None` 을 조용한 기본값으로 바꾸지 않는다.
    #    ⚠`alpha` 는 **양측 전체** 수준이다 — `T16` 의 *"n>=7"* 은 `ALPHA_FAM=0.025` 기준이다.
    chk(k_two_sided(6, 0.025) is None, 'n=6 양측 문턱이 존재하는 것으로 계산됐다')
    chk(k_two_sided(7, 0.025) == 7, 'n=7 양측 문턱이 7 이 아니다')
    chk(k_two_sided(60, 0.025) == 40, 'T16 THRESH_D 재현 실패')

    if bad:
        print('⛔자가진단 실패')
        for m in bad:
            print('   ' + m)
        return 1
    print('✅sign_test_mde 자가진단 통과 (달성·최소·대칭·안전방향·항진·도달불가)')
    return 0


if __name__ == '__main__':
    import sys
    if '--selftest' in sys.argv:
        sys.exit(_selftest())
    print(__doc__)
    print('예: mde_q(28, 20, 0.80) =', mde_q(28, 20, 0.80),
          ' · 검정력', f'{power_at(mde_q(28, 20, 0.80), 28, 20):.8f}')
