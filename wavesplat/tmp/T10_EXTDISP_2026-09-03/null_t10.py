"""`T10` 자 검증 3차 — **비 통계량의 귀무를 잰다**. ⛔제3자 자료 미접촉.

§5–§6 이 세운 것: 점추정은 못 믿고 **얕다/아니다 판별은 된다**. 그러면 판정 통계량은
    ratio = (수심 적합 잔차) / (심해 가정 잔차)
이고, 필요한 것은 ***"참으로 심해일 때 ratio 가 얼마나 내려갈 수 있는가"*** 다.
`d=200 m` 는 관측 대역에서 심해와 구별 불가(§4)이므로 **그것으로 귀무를 만든다**.
"""
import json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import control_t10 as CT                                             # noqa: E402
from ridge2_t10 import ridge2, fit                                   # noqa: E402

N_NULL, N_COMP = 10, 200
DEEP = 400.0                     # ⛔관측 대역에서 심해와 구별 불가한 수심


def one(depth, seed):
    E = CT.synth(depth, n_comp=N_COMP, seed=seed)
    k, om, wt = ridge2(E, CT.DX, CT.DY, CT.DT)
    d, rel, rel_deep = fit(k, om, wt)
    return dict(depth=depth, seed=seed, fit=d, rel=rel, rel_deep=rel_deep,
                ratio=rel / rel_deep if rel_deep > 0 else np.nan)


def main():
    null = [one(DEEP, 9000 + i) for i in range(N_NULL)]
    r = np.array([x['ratio'] for x in null])
    q05 = float(np.sort(r)[0])                     # 최소 = 가장 보수적 문턱
    print(f'★귀무(d={DEEP:.0f} m, n={N_NULL}): ratio 최소 {r.min():.4f} · '
          f'중앙 {np.median(r):.4f} · 최대 {r.max():.4f}')
    treat = [one(d, 100 + int(d)) for d in (10.0, 20.0, 30.0, 50.0, 100.0)]
    print('\n처치(참 수심 알려진 합성):')
    for x in treat:
        print(f'  d={x["depth"]:6.1f}  ratio {x["ratio"]:.4f}  '
              f'{"⟸ 귀무 최소 아래 = 얕다 검출" if x["ratio"] < q05 else ""}')
    json.dump(dict(null=null, treat=treat, null_min=q05,
                   null_median=float(np.median(r))),
              open(os.path.join(HERE, 'null_t10.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
