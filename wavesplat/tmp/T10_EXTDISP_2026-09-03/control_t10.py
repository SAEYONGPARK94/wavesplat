"""`T10` 자 검증 2차 — **합성 양성 대조**. 참 수심을 내가 정하고 복원되는지 본다.

⛔1차(`ruler_t10.py`)는 우리 코퍼스에서 실패했고, 진단 결과 **자의 결함이 아니었다**:
  · 우리 기록은 **30.1 s**(301프레임 × 0.1 s) ⟹ ω 분해능이 ω 의 **10–30%**
  · 우리 수심 50 m 에서 관측 대역의 `tanh(kd)` 가 **0.987–1.0000** ⟹ **수심이 식별 불가**
⟹ ★***우리 코퍼스는 이 자의 양성 대조가 될 수 없다.*** 그래서 **제3자와 같은 격자**
  (128², dx 4.879 m, 502 프레임, dt 1.0 s)에서 참 수심을 아는 장을 합성해 검증한다.
"""
import json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from ruler_t10 import ridge, fit_depth, G_ACC                        # noqa: E402

NX = NY = 128
DX = DY = 4.879
NT, DT = 502, 1.0


def synth(depth, n_comp=60, seed=0, hs=2.0):
    """참 분산관계 위의 성분 합. ⛔선형(구속 고조파 없음) — 자는 능선만 본다."""
    rng = np.random.default_rng(seed)
    x = np.arange(NX) * DX; y = np.arange(NY) * DY
    X, Y = np.meshgrid(x, y, indexing='xy')
    t = np.arange(NT) * DT
    kmag = rng.uniform(0.012, 0.30, n_comp)
    ang = rng.normal(0.0, 0.5, n_comp)
    kx, ky = kmag * np.cos(ang), kmag * np.sin(ang)
    om = np.sqrt(G_ACC * kmag * np.tanh(kmag * depth))
    a = rng.uniform(0.5, 1.5, n_comp) * kmag ** -1.5
    a *= hs / 4.0 / np.sqrt(np.sum(a ** 2) / 2)
    ph = rng.uniform(0, 2 * np.pi, n_comp)
    E = np.empty((NT, NY, NX))
    th0 = kx[None, None, :] * X[..., None] + ky[None, None, :] * Y[..., None]
    for j, tt in enumerate(t):
        E[j] = (a * np.cos(th0 - om * tt + ph)).sum(-1)
    return E


def main():
    out = []
    for d in (10.0, 20.0, 50.0, 200.0):
        E = synth(d, seed=int(d))
        k, om, wt = ridge(E, DX, DY, DT, kmin=0.012, kmax=0.30)
        fd, rel, rel_deep = fit_depth(k, om, wt)
        deep_ok = abs(rel_deep - rel) < 0.05 * max(rel, 1e-12)
        out.append(dict(true=d, fit=fd, rel=rel, rel_deep=rel_deep, n=len(k)))
        print(f'  참 {d:6.1f} m → 적합 {fd:8.1f} m   잔차 {rel:.5f}  '
              f'(심해가정 {rel_deep:.5f})  능선 {len(k)}점'
              + ('   ⟸ 심해와 구별 불가' if deep_ok else ''))
    print(f'\n★ω 분해능 Δω = {2*np.pi/(NT*DT):.5f} rad/s  '
          f'(우리 코퍼스는 {2*np.pi/30.1:.4f} = {2*np.pi/30.1/(2*np.pi/(NT*DT)):.0f}배 거칠다)')
    json.dump(out, open(os.path.join(HERE, 'control_t10.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
