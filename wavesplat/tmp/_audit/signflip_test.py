#!/usr/bin/env python3
r"""★**부호뒤집기 치환검정의 유일 구현** — `tmp/_audit/signflip_test.py`.

⛔**이 파일을 복사하지 마라.** 새 배치의 `gates_*.py`·`analyze_*.py` 는 이렇게 부른다:

    import os, sys
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '_audit')))
    from signflip_test import signflip_p, signflip_power, signflip_mde   # noqa: E402

    p = signflip_p(d, alternative='less')     # d = 레코드별 짝 차 (음수 = 처치가 낫다)

형제 = `sign_test_mde.py`(부호검정). ⛔**둘 중 무엇을 쓸지는 결과 전에 고정하라** — 아래 §3.

──────────────────────────────────────────────────────────────────────────────
★왜 이 파일이 생겼나 (2026-09-06)
──────────────────────────────────────────────────────────────────────────────
원장의 상시 문장 — *"첫 Holm α=0.05/6 아래로 `n=12` 가 낼 수 있는 값은 11/12·12/12 뿐
⟹ 10/12 는 어떤 자료에서도 기각 불가"* — 은 ★**부호검정에 대해서만 참**이다.

동결 `T17` 행에 정확 2¹² 치환을 돌린 실측(`tmp/_audit/ENDPOINT_SIGNFLIP.md`):

    PROP-BANDR < PROP   9/12   부호 p = 0.073 (널)   →   치환 p = 0.00195  ✅Holm α/6 통과
    (음성대조) PROP-FIX2 1/12   부호 p = 0.99976      →   치환 p = 0.99658   ← 항진 아님

⟹ ***실측 축(레코드 12)의 검정력 벽은 표본이 아니라 끝점 선택이었다.***

──────────────────────────────────────────────────────────────────────────────
§3 ⛔이 검정을 쓰기 전에 반드시 읽을 것
──────────────────────────────────────────────────────────────────────────────
1. ★**다른 추정량을 잰다.** 부호검정은 **방향의 일관성**, 치환은 **크기와 산포**.
   두 검정은 **양방향으로 어긋난다** — 실측 반례: `T12 C-MINDK` 는 부호 2/24 p=2.0e-5
   (강한 유의)인데 치환 p=0.53(완전 널; 평균 log비 −0.0016).
   ⟹ **끝점을 바꾸면 판정 대상이 바뀐다. 결과 전에 하나로 고정하라.**
2. **가정이 더 강하다** — 귀무에서 짝 차의 **부호 대칭성**을 요구한다(부호검정은 중앙값 0 만).
3. ⛔**꼬리가 무거우면 진다.** 실측(2026-09-06 감사): 홈 코퍼스의 짝 log비는
   `sd/robSD` 1.45–1.98 이고 거기서 치환은 부호검정보다 **0.04–0.15배**로 진다.
   ✅제3자는 0.98–1.44 이고 **3.4–8.6배 이긴다**. ⟹ **코퍼스마다 먼저 꼬리를 재라**(`tail_ratio`).
4. **최소 도달 p 는 부호검정과 같다** (`1/2^n`). 이득은 바닥이 아니라 **도달 빈도**다.

──────────────────────────────────────────────────────────────────────────────
§4 ★1종오류율 실측 (2026-09-06, 이 도구를 쓰기 전에 재 봤다)
──────────────────────────────────────────────────────────────────────────────
⛔**검정을 만들었으면 귀무에서 얼마나 자주 틀리는지 먼저 재라** — 이 저장소는
*"항상 통과하는 검사는 장식"* 과 *"검사기가 스스로 항진한다"* 로 여러 번 물렸다.

귀무(대칭) 자료에서 **양측** p ≤ α 의 실측 비율:

    n   분포              R      α=.05        α=.01
    8   정규            4000    0.0503       0.0100
    8   t3(무거운꼬리)   4000    0.0465       0.0090
    8   라플라스         4000    0.0462       0.0088
   12   정규            4000    0.0517       0.0112
   12   t3(무거운꼬리)   4000    0.0510       0.0118
   12   라플라스         4000    0.0447       0.0103
   16   정규            2000    0.0560       0.0100
   16   t3(무거운꼬리)   2000    0.0440       0.0090
   16   라플라스         2000    0.0395       0.0090

⟹ ✅**9/9 가 명목 안**(±2se). **무거운 꼬리에서도 부풀지 않는다** — 치환검정이라
당연하지만, 재지 않고 쓰는 것과 재고 쓰는 것은 다르다.
⚠`n=12` 양측에서 도달 가능한 p 는 `1/4096` 의 배수라 **이산적**이다 — 정확히 0.05 를 못 맞춘다.
⟹ 아래 자가진단은 **빠른 판(n=8, R=600)** 만 돌린다. 전수는 이 표가 정본이다.
"""
import itertools
import math

import numpy as np

EXACT_MAX_N = 22          # 2^22 = 4.2M — 그 위는 몬테카를로


_SIGN_CACHE = {}


def _signs(n):
    """(2^n, n) 의 ±1 행렬 — 캐시된다."""
    S = _SIGN_CACHE.get(n)
    if S is None:
        idx = np.arange(2 ** n, dtype=np.int64)[:, None]
        S = 1.0 - 2.0 * ((idx >> np.arange(n)) & 1).astype(np.float64)
        _SIGN_CACHE[n] = S
    return S


def _stud(x):
    """studentized 평균. sd=0 이면 0 (퇴화 표본에서 유의를 만들지 않는다)."""
    n = len(x)
    s = float(np.std(x, ddof=1)) if n > 1 else 0.0
    return float(np.mean(x) / (s / math.sqrt(n))) if s > 0 else 0.0


def signflip_p(d, alternative='less', n_mc=200_000, seed=0, statistic=_stud):
    """부호뒤집기 치환 p. `d` = 짝 차 배열.

    `alternative='less'`  : 통계량이 **작을수록** 처치가 낫다(음수 방향).
    `alternative='greater'`: 반대.
    `alternative='two-sided'`: |t| 기준.
    n ≤ EXACT_MAX_N 이면 **정확 열거**, 그 위는 MC(관측 포함, `(cnt+1)/(N+1)`).
    """
    d = np.asarray(d, float)
    d = d[np.isfinite(d)]
    n = len(d)
    if n < 2:
        return float('nan')
    obs = statistic(d)
    if alternative == 'less':
        cmp = lambda t: t <= obs
    elif alternative == 'greater':
        cmp = lambda t: t >= obs
    elif alternative == 'two-sided':
        cmp = lambda t: abs(t) >= abs(obs)
    else:
        raise ValueError(f'alternative={alternative!r}')

    if n <= EXACT_MAX_N:
        if statistic is _stud:
            # ★벡터화 정확 열거 — 2^n 개 부호패턴을 한 번에 (순수 파이썬 루프보다 ~1000배).
            #   `_signs(n)` 은 캐시된다. 스칼라 경로와 **비트 동일**임을 자가진단이 확인한다.
            S = _signs(n)                      # (2^n, n) ±1
            X = S * d                          # (2^n, n)
            m = X.mean(axis=1)
            sd = X.std(axis=1, ddof=1)
            with np.errstate(divide='ignore', invalid='ignore'):
                t = np.where(sd > 0, m / (sd / math.sqrt(n)), 0.0)
            if alternative == 'less':
                return float((t <= obs).mean())
            if alternative == 'greater':
                return float((t >= obs).mean())
            return float((np.abs(t) >= abs(obs)).mean())
        cnt = 0
        for bits in itertools.product((1.0, -1.0), repeat=n):
            if cmp(statistic(d * np.array(bits))):
                cnt += 1
        return cnt / 2 ** n
    rng = np.random.default_rng(seed)
    cnt = 0
    for _ in range(n_mc):
        if cmp(statistic(d * rng.choice((1.0, -1.0), n))):
            cnt += 1
    return (cnt + 1) / (n_mc + 1)


def tail_ratio(d):
    """`sd / robustSD` — 1 에 가까울수록 가벼운 꼬리. §3-3 의 판별자."""
    d = np.asarray(d, float)
    d = d[np.isfinite(d)]
    q1, q3 = np.percentile(d, [25, 75])
    rob = (q3 - q1) / 1.349
    return float(np.std(d, ddof=1) / rob) if rob > 0 else float('inf')


def signflip_power(shape, n, alpha=0.05, dz=0.0, n_rep=2000, seed=0,
                   alternative='less', n_mc=4000):
    """관측된 **모양**(중앙정렬한 짝 차 표본)을 `dz`(sd 단위)만큼 이동시켜 재표집한 검정력.

    ⚠`shape` 는 실제로 측정된 짝 차라야 한다 — 정규 가정을 쓰지 않는 것이 이 함수의 요점이다.
    """
    s = np.asarray(shape, float)
    s = s[np.isfinite(s)] - np.median(np.asarray(shape, float))
    sd = float(np.std(s, ddof=1))
    rng = np.random.default_rng(seed)
    hit = 0
    for _ in range(n_rep):
        x = rng.choice(s, n, replace=True) + dz * sd
        if signflip_p(x, alternative=alternative, n_mc=n_mc, seed=int(rng.integers(1 << 30))) <= alpha:
            hit += 1
    return hit / n_rep


def signflip_mde(shape, n, alpha=0.05, target=0.80, alternative='less',
                 lo=0.0, hi=2.0, iters=12, **kw):
    """검정력 `target` 을 주는 최소 |dz| (sd 단위). ⛔**음성 결과 인용 시 병기 의무.**"""
    for _ in range(iters):
        mid = (lo + hi) / 2
        if signflip_power(shape, n, alpha, -mid if alternative == 'less' else mid,
                          alternative=alternative, **kw) >= target:
            hi = mid
        else:
            lo = mid
    return hi


# ── 자가진단 ────────────────────────────────────────────────────────────────
def _selftest():
    ok = True

    def chk(name, cond, detail=''):
        nonlocal ok
        print(f"  [{'OK  ' if cond else 'FAIL'}] {name}{('  ' + detail) if detail else ''}")
        ok &= bool(cond)

    # ① 최소 도달 p 는 1/2^n (부호검정과 같다)
    d = -np.arange(1.0, 13.0)
    chk('n=12 전승의 정확 p == 1/4096',
        abs(signflip_p(d, 'less') - 1 / 4096) < 1e-12, f'{signflip_p(d,"less"):.6g}')

    # ② ★항진 검사 — 대칭 잡음에서 유의가 나오면 안 된다
    rng = np.random.default_rng(7)
    sig = sum(1 for _ in range(200)
              if signflip_p(rng.normal(size=12), 'two-sided') <= 0.05)
    chk('★항진: 대칭 잡음 200회 중 양측 유의', sig <= 20, f'{sig}/200 (기대 ≈10)')

    # ③ ★음성대조 — 반대 방향 자료에서 'less' 는 크게 나와야 한다
    chk('★반대 방향 자료의 less p > 0.9', signflip_p(np.arange(1.0, 13.0), 'less') > 0.9)

    # ④ 퇴화 표본이 유의를 만들지 않는다
    chk('상수 표본은 유의 아님', signflip_p(np.full(12, -3.0), 'less') > 0.05,
        f'{signflip_p(np.full(12,-3.0),"less"):.4g}')

    # ④b ★벡터화 경로가 스칼라 경로와 **비트 동일**한가 (내가 오늘 넣은 최적화의 검증)
    rng2 = np.random.default_rng(3)
    worst = 0.0
    for _ in range(30):
        y = rng2.normal(size=10)
        for alt in ('less', 'greater', 'two-sided'):
            a = signflip_p(y, alt)
            b = signflip_p(y, alt, statistic=lambda v: _stud(v))   # 스칼라 경로 강제
            worst = max(worst, abs(a - b))
    chk('★벡터화 == 스칼라 (30표본 × 3방향)', worst == 0.0, f'max|Δ|={worst:.3g}')

    # ④c ★1종오류율 빠른 판 (n=8, R=600) — 전수는 §4 표가 정본
    rng3 = np.random.default_rng(11)
    R = 600
    hit = sum(1 for _ in range(R)
              if signflip_p(rng3.standard_t(3, size=8), 'two-sided') <= 0.05)
    se = (0.05 * 0.95 / R) ** 0.5
    chk('★1종오류 (n=8, t3, R=600, α=.05)', hit / R <= 0.05 + 3 * se,
        f'{hit/R:.4f} (명목 0.05, +3se = {0.05+3*se:.4f})')

    # ⑤ 양측 == 두 단측의 대칭성
    x = np.array([-3, -1, -2, 5, -4, -1, -2, -3, 1, -6, -2, -1], float)
    chk('two-sided ≥ less', signflip_p(x, 'two-sided') >= signflip_p(x, 'less'))

    # ⑥ ★동결 `T17` 값 재현 (ENDPOINT_SIGNFLIP.md §2 의 핀)
    #    ⚠파일이 없으면 건너뛴다(이 도구는 그 배치에 의존하지 않는다).
    import json, os, collections
    f = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                     '..', 'T17_REALSCALE_2026-09-05', 'results_t17.jsonl')
    if os.path.exists(f):
        rows = [json.loads(l) for l in open(f, encoding='utf-8')]
        by = collections.defaultdict(dict)
        for r in rows:
            by[(r['record'], int(r['win']))][r['arm']] = r
        recs = sorted({k[0] for k in by})
        def fold(a, b):
            out = []
            for rec in recs:
                v = [math.log(d[a]['leadmean']) - math.log(d[b]['leadmean'])
                     for (r, w), d in by.items()
                     if r == rec and a in d and b in d
                     and d[a].get('leadmean', 0) > 0 and d[b].get('leadmean', 0) > 0]
                if v:
                    out.append(float(np.mean(v)))
            return np.array(out)
        p = signflip_p(fold('PROP-BANDR', 'PROP'), 'less')
        chk('★T17 PROP-BANDR 재현 (핀 0.001953)', abs(p - 0.001953125) < 1e-9, f'{p:.6f}')
        p2 = signflip_p(fold('PROP-FIX2', 'PROP'), 'less')
        chk('★T17 음성대조 PROP-FIX2 (핀 0.996582)', abs(p2 - 0.99658203125) < 1e-9, f'{p2:.6f}')
    else:
        print('  [skip] T17 재현 — 동결 파일 없음')

    print('\n' + ('✅자가진단 전건 통과' if ok else '⛔자가진단 실패'))
    return 0 if ok else 1


if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1 and sys.argv[1] not in ('--selftest',):
        print(f'사용법: {sys.argv[0]} [--selftest]')
        sys.exit(2)
    print('── signflip_test 자가진단 ' + '─' * 40)
    sys.exit(_selftest())
