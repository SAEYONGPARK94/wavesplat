"""PROBE (사전등록 이전, 기술 타당성 전용 — 추정기 0회 호출, 비교 0회).

목적 3가지:
  P-a  `initialize_from_components` 가 t=0 에서 선형 GT 와 비트단위로 맞는가
  P-b  `evaluate_at` 의 **분리가능(separable)** 재작성이 원본과 수치적으로 같은가
       (원본은 O(n_modes x n_pts); 비선형 진화 후 128x128=16384 모드가 전부 켜져서
        프레임당 수십 초가 된다 -> 그대로면 배치가 불가능하다)
  P-c  step(dt) 실측 비용 + Hs 안정성 + 3차 구속고조파의 Nyquist 초과 분율

여기서 나오는 것은 **비용과 배선**뿐이다. 어떤 팔의 정확도도 계산하지 않는다.
"""
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from wavesim.config import WaveSimConfig            # noqa: E402
from wavesim.gt_generator import make_grid, evaluate_field   # noqa: E402
from wavesim.hos_simulator import HOSConfig, HOSSimulator    # noqa: E402
from wavesim.wave_generator import jonswap_components        # noqa: E402

G = 9.81


def snap(components, dkx, dky, Nx, Ny, depth):
    """성분을 HOS 스펙트럼 격자에 스냅하고 omega 를 분산관계로 재도출.

    두 팔이 **완전히 같은 (kx,ky,A,phi)** 를 쓰게 하려고 선형 팔도 이 성분으로 만든다.
    안 그러면 짝 차이에 '파수 양자화' 가 섞여 비선형성과 교락된다."""
    kx1 = np.fft.fftfreq(Nx, d=1.0 / (Nx * dkx))
    ky1 = np.fft.fftfreq(Ny, d=1.0 / (Ny * dky))
    out = []
    for c in components:
        ix = int(round(c.kx / dkx)) % Nx
        iy = int(round(c.ky / dky)) % Ny
        kx, ky = float(kx1[ix]), float(ky1[iy])
        k = float(np.hypot(kx, ky))
        w = float(np.sqrt(G * k * np.tanh(np.clip(k * depth, 0, 20)))) if k > 0 else 0.0
        out.append(type(c)(A=c.A, omega=w, kx=kx, ky=ky, phi=c.phi, Q=c.Q))
    return out


def separable_evaluate_at(sim, X, Y):
    """evaluate_at 와 **같은 합**을 분리가능 형태로 계산.

    eta(x_i,y_j) = Re[ sum_{p,q} a_pq e^{i kx_p x_i} e^{i ky_q y_j} ]
                 = Re[ (Ex @ a @ Ey^T)_{ij} ],  Ex[i,p]=e^{i kx_p x_i}
    X,Y 가 meshgrid(indexing='ij') 로 만들어진 **정규격자**일 때만 유효하며,
    호출부에서 그 전제를 검사한다."""
    x = X[:, 0]
    y = Y[0, :]
    kx1 = sim.KX[:, 0]
    ky1 = sim.KY[0, :]
    Ex = np.exp(1j * np.outer(x, kx1))          # (nx, Nx)
    Ey = np.exp(1j * np.outer(y, ky1))          # (ny, Ny)
    a = sim.eta_hat
    core = Ex @ a @ Ey.T                        # (nx, ny)
    Z = np.real(core)
    Zx = np.real(Ex @ (1j * sim.KX * a) @ Ey.T)
    Zy = np.real(Ex @ (1j * sim.KY * a) @ Ey.T)
    mag = np.sqrt(Zx ** 2 + Zy ** 2 + 1.0)
    return Z, -Zx / mag, -Zy / mag, 1.0 / mag


def main():
    cfg = WaveSimConfig()
    depth, Hs, Tp = 50.0, 1.2, 7.0
    Nx = Ny = 128
    Lx, Ly, M, hdt = 600.0, 400.0, 3, 0.02

    comps = jonswap_components(Hs=Hs, Tp=Tp, n_components=15, depth=depth, seed=1581,
                               direction_deg=0.0, direction_spread_deg=30.0,
                               dirs_per_freq=1, random_amplitudes=True,
                               random_frequencies=True, stream_tag='BF4_15')
    dkx, dky = 2 * np.pi / Lx, 2 * np.pi / Ly
    sc = snap(comps, dkx, dky, Nx, Ny, depth)
    dk = [np.hypot(a.kx - b.kx, a.ky - b.ky) for a, b in zip(comps, sc)]
    print(f'P-0 snapping: max |dk| = {max(dk):.5f} rad/m  (bin dkx={dkx:.5f} dky={dky:.5f})')
    kk = np.array([np.hypot(c.kx, c.ky) for c in sc])
    AA = np.array([c.A for c in sc])
    k_nyq = min(Nx // 2 * dkx, Ny // 2 * dky)
    frac3 = float((AA[3 * kk > k_nyq] ** 2).sum() / (AA ** 2).sum())
    print(f'    k range {kk.min():.4f}-{kk.max():.4f}, k_nyq={k_nyq:.4f}; '
          f'variance with 3k>k_nyq = {100*frac3:.3f}%')
    bins = {(int(round(c.kx / dkx)) % Nx, int(round(c.ky / dky)) % Ny) for c in sc}
    print(f'    distinct bins {len(bins)} / {len(sc)} components (collisions add coherently)')

    hos = HOSSimulator(HOSConfig(Lx=Lx, Ly=Ly, Nx=Nx, Ny=Ny, depth=depth, M=M))
    hos.initialize_from_components(sc)

    X, Y = make_grid(cfg.gt_grid.x_range, cfg.gt_grid.y_range,
                     nx=cfg.gt_grid.nx, ny=cfg.gt_grid.ny)
    print(f'    grid X shape {X.shape}  X[:,0][:3]={X[:, 0][:3]} Y[0,:][:3]={Y[0, :][:3]}')

    # P-a: t=0 identity vs linear GT
    Z_h, NXh, NYh, NZh = hos.evaluate_at(X, Y)
    Z_l, NXl, NYl, NZl = evaluate_field(sc, X, Y, 0.0)
    print(f'P-a t=0 |HOS - linear|max = {np.abs(Z_h - Z_l).max():.3e} m '
          f'(rms_gt {np.sqrt((Z_l**2).mean()):.4f} m); normal max d = '
          f'{max(np.abs(NXh-NXl).max(), np.abs(NYh-NYl).max(), np.abs(NZh-NZl).max()):.3e}')

    # P-b: separable == original, on a FULL spectrum (after one step)
    hos.step(hdt)
    nnz = int((np.abs(hos.eta_hat) > 1e-15).sum())
    t0 = time.time(); Z0, A0, B0, C0 = hos.evaluate_at(X, Y); t_orig = time.time() - t0
    t0 = time.time(); Z1, A1, B1, C1 = separable_evaluate_at(hos, X, Y); t_sep = time.time() - t0
    print(f'P-b after 1 step: nonzero modes {nnz}/{Nx*Ny}')
    print(f'    gt grid {X.shape}: orig {t_orig*1000:.1f} ms, separable {t_sep*1000:.1f} ms, '
          f'|dZ|max={np.abs(Z0-Z1).max():.3e}, |dN|max='
          f'{max(np.abs(A0-A1).max(), np.abs(B0-B1).max(), np.abs(C0-C1).max()):.3e}')
    Xm, Ym = make_grid(cfg.gt_grid.x_range, cfg.gt_grid.y_range, nx=200, ny=167)
    t0 = time.time(); Zm0, *_ = hos.evaluate_at(Xm, Ym); t_orig_m = time.time() - t0
    t0 = time.time(); Zm1, *_ = separable_evaluate_at(hos, Xm, Ym); t_sep_m = time.time() - t0
    print(f'    mesh 200x167: orig {t_orig_m*1000:.1f} ms, separable {t_sep_m*1000:.1f} ms, '
          f'|dZ|max={np.abs(Zm0-Zm1).max():.3e}')

    # P-c: step cost + Hs stability over the full 30 s episode
    print(f'P-c step cost (M={M}, {Nx}x{Ny}), Hs_init={hos.get_Hs():.4f} m')
    t0 = time.time()
    for _ in range(50):
        hos.step(hdt)
    per = (time.time() - t0) / 50
    print(f'    {per*1000:.1f} ms/step -> 30 s episode @dt={hdt} = {1500*per:.0f} s of stepping')
    for target in (5.0, 10.0, 20.0, 30.0):
        while hos.t < target - 1e-9:
            hos.step(min(hdt, target - hos.t))
        print(f'    t={hos.t:5.2f}s  Hs={hos.get_Hs():.4f} m  '
              f'max|eta|={np.abs(hos.get_eta()).max():.3f} m')
    print(f'    total wall for 30 s of HOS: {time.time()-t0:.0f} s')


if __name__ == '__main__':
    main()
