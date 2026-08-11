"""S9 규약 상수 — **단일 출처**. 모든 n-의존 상수를 여기서 **유도**한다.

## 왜 이 파일이 존재하는가
같은 부류의 결함이 이 아크에서 **세 번** 났다:
1. S8 `AMENDMENT_2` — 상속된 시드 상수(`24<=seed<=31`)가 낡아 게이트가 FAIL, **배치 3.4h 지연**.
2. S8 `AMENDMENT_1` (A1) — 상속된 `regime()` 이 다른 디렉터리를 하드코딩해 **오라벨 직전**.
3. ★S9 사전감사 — `analyze_s8.py` 의 `MDE_SIGN80_C=0.7234` · `HARM_K=21` · `BEST_K=23` · `/32`
   가 전부 **n=32 하드코딩**이다. 희소 n=64 에 그대로 상속하면 **43/64 개선 셸을 "악화 셸"로
   찍고**, 근거리 "무해" 문턱이 **1.533배 느슨해진다**.
   ⚠비대칭 n(희소 64 / 혼잡 32) 때문에 **혼잡에선 맞고 희소에서만 틀려 스팟체크를 통과한다.**

S8 이 스스로 내린 처방은 **"상수를 두 번 적지 말고 한 곳에서 유도하라"** 였다.
그 처방은 그때 `gate.py` 에만 적용됐고 **채점기에는 적용된 적이 없다.** 이 파일이 그것이다.

## 자기시험 (import 시 실행)
유도식이 S8 의 하드코딩 값을 **재현**하는지 확인한다. 재현하지 못하면 **import 가 실패한다** —
"새 코드가 옛 결과를 바꿔버리는" 사고를 조용히 지나가게 두지 않는다.
"""
from math import comb, erf, sqrt

# ── 승계 상수 (rx2 → S1 → S7 → S8). ⛔재선택 금지 ────────────────────────────
D2_N3 = 1.693                          # 3표본 범위 -> sigma
MDE_MULT = 2.1279493432578005          # rx2 승계
MEDIAN_SE = 1.2533141373155001         # sqrt(pi/2)
ALPHA = 0.05                           # PRIMARY 양측
HARM_ALPHA = 0.1102                    # 안전 게이트 (무보정, S8 §6 승계)
POWER = 0.80                           # mde_sign80 의 검정력 정의


def Phi(z):
    return 0.5 * (1.0 + erf(z / sqrt(2.0)))


def Phi_inv(p, lo=-10.0, hi=10.0, iters=200):
    for _ in range(iters):
        m = 0.5 * (lo + hi)
        if Phi(m) < p:
            lo = m
        else:
            hi = m
    return 0.5 * (lo + hi)


def p_two_sided(k, n):
    """부호 k/n 의 양측 정확 이항 p (q=0.5)."""
    return sum(comb(n, i) for i in range(n + 1)
               if abs(i - n / 2.0) >= abs(k - n / 2.0)) / 2.0 ** n


def binom_ge(n, k, p):
    return sum(comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(k, n + 1))


def min_k(n, alpha):
    """양측 p <= alpha 를 만족하는 최소 부호수."""
    for k in range(n // 2, n + 1):
        if p_two_sided(k, n) <= alpha:
            return k
    return None


def p_for_power(n, k, power=POWER):
    """P(Bin(n,p) >= k) = power 를 만족하는 참 부호율."""
    lo, hi = 0.5, 1.0
    for _ in range(200):
        m = 0.5 * (lo + hi)
        if binom_ge(n, k, m) < power:
            lo = m
        else:
            hi = m
    return 0.5 * (lo + hi)


def mde_sign80_c(n, k):
    """부호검정 MDE 계수 = 검정력 80% 에 필요한 부호율의 z 환산.

    ⚠**n 과 k 양쪽에 의존한다.** S8 은 이 값을 상수 0.7234 로 적어 두었고 그것은
    (n=32, k=23) 전용이다. n=64, k=41 에서는 **0.4720** 이다.
    """
    return Phi_inv(p_for_power(n, k, POWER))


def primary_k(n):
    """PRIMARY 통과 최소 부호수 (양측 alpha=0.05, **단일** PRIMARY)."""
    return min_k(n, ALPHA)


def harm_k(n):
    """안전 게이트 발화 최소 악화 부호수 (양측 p <= 0.1102, 무보정)."""
    return min_k(n, HARM_ALPHA)


def holm_k(n, m):
    """Holm-m 최악순위에서 통과하는 최소 부호수."""
    return min_k(n, ALPHA / m)


def block_all_worse_p(block_n):
    """꼬리 규칙 (d) '한 블록 전부 악화' 의 우연 확률. ⚠블록 크기에 지수적으로 의존한다.

    n=32 는 블록 8 (p=3.9e-3), n=64 는 블록 16 (p=1.5e-5) ⟹ **254배 비대칭**.
    같은 규칙 문구를 두 레짐에 쓰면 희소에서 사실상 도달 불가가 된다.
    """
    return 0.5 ** block_n


def block_worse_k(block_n, ref_p=None):
    """블록 규칙을 **확률로 매칭**한 문턱. 기준 = n=32 의 8/8 (3.9e-3)."""
    ref = block_all_worse_p(8) if ref_p is None else ref_p
    for k in range(block_n, block_n // 2, -1):
        p = sum(comb(block_n, i) for i in range(k, block_n + 1)) / 2.0 ** block_n
        if p >= ref:
            return k
    return block_n


# ── 이 배치의 확정 파라미터 ──────────────────────────────────────────────────
N = {'sparse': 64, 'crowded': 32}
BLOCK_N = {'sparse': 16, 'crowded': 8}      # 레짐당 4블록
SEEDS = {'sparse': range(48, 64), 'crowded': range(48, 56)}
DATA_DIR = {'sparse': 'data_s9_sparse_2026-08-11', 'crowded': 'data_s9_crowded_2026-08-11'}


def table(regime):
    n = N[regime]
    k = primary_k(n)
    return dict(
        regime=regime, n=n,
        primary_k=k, primary_p=p_two_sided(k, n),
        mde_sign80_c=mde_sign80_c(n, k),
        harm_k=harm_k(n), harm_p=p_two_sided(harm_k(n), n),
        block_n=BLOCK_N[regime], block_worse_k=block_worse_k(BLOCK_N[regime]),
        block_all_worse_p=block_all_worse_p(BLOCK_N[regime]),
    )


# ── ★자기시험: 유도식이 S8 하드코딩을 재현하는가 ─────────────────────────────
def _self_test():
    assert primary_k(32) == 23, primary_k(32)
    assert abs(p_two_sided(23, 32) - 0.02006) < 1e-5
    assert harm_k(32) == 21, harm_k(32)
    # ★S8 `mde_prespec_s8.py:36` 의 하드코딩 0.7234 를 유도식이 재현해야 한다
    c32 = mde_sign80_c(32, 23)
    assert abs(c32 - 0.7234) < 5e-4, f'S8 하드코딩 0.7234 재현 실패: {c32}'
    # n=64 에서 실제로 달라지는지 (상속하면 안 되는 이유의 증거)
    c64 = mde_sign80_c(64, 41)
    assert abs(c64 - 0.4720) < 5e-4, c64
    assert abs(c32 / c64 - 1.533) < 0.01, c32 / c64
    assert primary_k(64) == 41 and harm_k(64) == 39


_self_test()

if __name__ == '__main__':
    import json
    print('★S8 하드코딩 재현 자기시험 PASS (0.7234 = 유도값)')
    print()
    for rg in ('crowded', 'sparse'):
        t = table(rg)
        print(f'--- {rg} (n={t["n"]}) ---')
        print(json.dumps(t, indent=2, default=str))
    print()
    print('★상속 금지 대조표 (S8 상수를 그대로 쓰면 희소에서 어떻게 틀리는가)')
    print(f'  mde_sign80_c : S8 0.7234 (n=32,k=23)  vs  S9 희소 {mde_sign80_c(64, 41):.4f} '
          f'(n=64,k=41)  = {0.7234 / mde_sign80_c(64, 41):.3f} 배 과대')
    print(f'  harm_k       : S8 21/32 ({21/32:.1%})  vs  S9 희소 {harm_k(64)}/64 '
          f'({harm_k(64)/64:.1%})  — 21 을 그대로 쓰면 널에서도 항상 발화')
    print(f'  block (d)    : 8/8 p={block_all_worse_p(8):.2e}  vs  16/16 p='
          f'{block_all_worse_p(16):.2e} = {block_all_worse_p(8)/block_all_worse_p(16):.0f}배 비대칭'
          f'  ⟹ 희소 매칭 문턱 {block_worse_k(16)}/16')
