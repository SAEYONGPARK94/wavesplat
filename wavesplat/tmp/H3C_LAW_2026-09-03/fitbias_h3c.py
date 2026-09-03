"""`H3c` 개정 1 의 측정 — **중앙값 정합이 두 부류에 대칭이 아니다**. 런 0회.

⛔감사 지적: 개선 13.0% 중 일부는 **규칙이 A 를 자기 최적에서 더 멀리 놓은** 산물이다.
⚠라벨은 불변 — C 의 **자기 최적조차** 문턱 0.20 에 한참 못 미친다.
"""
import json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import gates_h3c as G                                                # noqa: E402
import run_h3c as R                                                  # noqa: E402


def main():
    z = np.load(os.path.join(G.H3, 'cells_h3.npz'))
    P, rate = z['P'], z['rate']
    sA = R.fit_A(P, rate); mA = R.mre(sA * P, rate)
    TC = R.fit_C(P, rate); mC = R.mre(np.exp(-TC / P), rate)
    oA = min(R.mre(g * P, rate) for g in np.linspace(0.2 * sA, 5 * sA, 1200))
    oC = min(R.mre(np.exp(-t / P), rate)
             for t in np.geomspace(max(TC * 0.1, 1e-9), TC * 10, 1200))
    out = dict(mre_A=mA, mre_C=mC, opt_A=oA, opt_C=oC,
               bias_A_pct=100 * (mA / oA - 1), bias_C_pct=100 * (mC / oC - 1),
               rel_matched_pct=100 * (1 - mC / mA), rel_fair_pct=100 * (1 - oC / oA),
               threshold=0.20)
    json.dump(out, open(os.path.join(HERE, 'fitbias_h3c.json'), 'w'), indent=1)
    print(f"정합  A {mA:.4f}  C {mC:.4f}  개선 {out['rel_matched_pct']:+.1f}%")
    print(f"최적  A {oA:.4f}  C {oC:.4f}  개선 {out['rel_fair_pct']:+.1f}%")
    print(f"편향  A +{out['bias_A_pct']:.2f}%  C +{out['bias_C_pct']:.2f}%")
    print(f"⟹ C 의 자기 최적 {oC:.4f} 도 문턱 {out['threshold']} 에 미달 ⟹ 라벨 불변")


if __name__ == '__main__':
    main()
