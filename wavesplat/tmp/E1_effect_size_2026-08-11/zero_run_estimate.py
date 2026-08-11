"""E1 대체 — **런 0회**로 효과 크기에 답한다.

3에이전트가 독립적으로 같은 결론에 도달했다: E1 을 돌려도 목표 폭에 못 간다(P=0.08~0.23).
대신 **이미 있는 세 배치**(S7·S8·S9, 전부 출하 팔·전부 신선 표본)로 답할 수 있는 만큼 답한다.

## 구간 방법 — ⛔BCa 를 쓰지 않는다
중앙값의 잭나이프는 값이 2개뿐이라 가속도가 퇴화한다(a≈1.6e-19) ⟹ BCa ≡ 백분위수이고
커버리지가 0.93~0.94 로 명목 미달. **부호검정을 뒤집은 정확 CI**(순서통계량)를 쓴다 —
사전등록 통계량(부호검정)과 **같은 함수**라 검정-구간 불일치가 원천 차단된다.
"""
import json, statistics as st
from math import comb
from collections import defaultdict

def paired(path, treat, base, rg, rel=False):
    R = defaultdict(dict)
    for line in open(path):
        r = json.loads(line)
        if 'error' in r: continue
        R[(r['cell'], r['episode'])][r['seed']] = r
    eps = {ep for (c, ep), d in R.items() if next(iter(d.values()))['regime'] == rg}
    out = []
    for ep in sorted(eps):
        a, b = R.get((treat, ep)), R.get((base, ep))
        if not a or not b: continue
        va = st.median([x['nRMSE'] for x in a.values()])
        vb = st.median([x['nRMSE'] for x in b.values()])
        out.append((va - vb) / vb if rel else va - vb)
    return sorted(out)

def median_ci(xs, alpha=0.05):
    """부호검정 반전 정확 CI (분포무관, 순서통계량)."""
    n = len(xs)
    k = 0
    for i in range(n // 2 + 1):
        p = 2 * sum(comb(n, j) for j in range(i + 1)) / 2 ** n
        if p > alpha: break
        k = i
    return xs[k], xs[n - 1 - k], 2 * sum(comb(n, j) for j in range(k + 1)) / 2 ** n

def lower_bound(xs, alpha=0.025):
    """단측 97.5% 상한(=효과가 '적어도 이만큼')."""
    n = len(xs); k = 0
    for i in range(n // 2 + 1):
        if sum(comb(n, j) for j in range(i + 1)) / 2 ** n > alpha: break
        k = i
    return xs[n - 1 - k]

B = {'S7': ('tmp/S7_confirm_n32_2026-08-10/results.jsonl', 'AUTO-L1-1e4', 'AUTO'),
     'S8': ('tmp/S8_deploy_confirm_2026-08-10/results.jsonl', 'AUTO-L1-1e4', 'AUTO'),
     'S9': ('tmp/S9_lr_x_l1_2026-08-11/results.jsonl', 'AUTO-L1', 'AUTO')}

for rg in ('sparse', 'crowded'):
    print(f'\n{"="*74}\n{rg.upper()}  —  출하 팔 λ 1e-3 → 1e-4, 세 신선 표본\n{"="*74}')
    for scale, rel in (('상대 %', True), ('절대 m', False)):
        print(f'\n  [{scale}]')
        allx = []
        for name, (p, t, b) in B.items():
            xs = paired(p, t, b, rg, rel=rel)
            allx += xs
            lo, hi, cov = median_ci(xs)
            f = (lambda v: f'{v*100:+7.2f}%') if rel else (lambda v: f'{v:+9.5f}')
            print(f'    {name} n={len(xs):3d}  중앙 {f(st.median(xs))}  '
                  f'95%CI [{f(lo)}, {f(hi)}]  (실효 {1-cov:.3f})')
        allx.sort()
        lo, hi, cov = median_ci(allx)
        f = (lambda v: f'{v*100:+7.2f}%') if rel else (lambda v: f'{v:+9.5f}')
        print(f'    ── 합침 n={len(allx):3d}  중앙 {f(st.median(allx))}  '
              f'95%CI [{f(lo)}, {f(hi)}]')
        print(f'    ★단측 97.5%: 효과는 **적어도** {f(lower_bound(allx))}')
