"""PROBE 2 (사전등록 이전, 기술 타당성 전용 — 추정기 0회, 비교 0회).

Q-a  분리가능 evaluate_at 을 **축 수정 후** 원본과 대조 (PROBE1 의 내 버그 수정)
Q-b  HOS 스펙트럼 격자 후보별로 "3차 구속고조파가 절단되는 분산 분율" 을 8셀 전수 측정
Q-c  후보 격자의 step 비용
"""
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from wavesim.config import WaveSimConfig                      # noqa: E402
from wavesim.gt_generator import make_grid                    # noqa: E402
from wavesim.hos_simulator import HOSConfig, HOSSimulator     # noqa: E402
from wavesim.wave_generator import jonswap_components         # noqa: E402

G = 9.81
SEA = {'BF4': (1.2, 7.0), 'BF5': (2.0, 9.0)}


def snap(components, dkx, dky, Nx, Ny, depth):
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


def sep_eval(sim, X, Y):
    """make_grid 규약: X[i,j]=x[j], Y[i,j]=y[i] (배열 shape (ny,nx))."""
    x = X[0, :]
    y = Y[:, 0]
    Ex = np.exp(1j * np.outer(x, sim.KX[:, 0]))       # (nx, Nx)
    Ey = np.exp(1j * np.outer(y, sim.KY[0, :]))       # (ny, Ny)
    a = sim.eta_hat
    Z = np.real(Ey @ a.T @ Ex.T)
    Zx = np.real(Ey @ (1j * sim.KX * a).T @ Ex.T)
    Zy = np.real(Ey @ (1j * sim.KY * a).T @ Ex.T)
    mag = np.sqrt(Zx ** 2 + Zy ** 2 + 1.0)
    return Z, -Zx / mag, -Zy / mag, 1.0 / mag


CAND = [(128, 128, 600.0, 400.0), (192, 128, 600.0, 400.0), (192, 160, 600.0, 400.0)]


def main():
    cfg = WaveSimConfig()
    depth = 50.0
    print('Q-b  3차 구속고조파 절단 분산 분율 (3|kx|>kx_nyq or 3|ky|>ky_nyq)')
    for Nx, Ny, Lx, Ly in CAND:
        dkx, dky = 2 * np.pi / Lx, 2 * np.pi / Ly
        kxn, kyn = np.pi * Nx / Lx, np.pi * Ny / Ly
        worst = []
        for bf in ('BF4', 'BF5'):
            Hs, Tp = SEA[bf]
            for nc in (2, 5, 15, 30):
                fr = []
                for seed in (1581, 1590, 1600):
                    c = jonswap_components(Hs=Hs, Tp=Tp, n_components=nc, depth=depth,
                                           seed=seed, direction_deg=0.0,
                                           direction_spread_deg=30.0, dirs_per_freq=1,
                                           random_amplitudes=True, random_frequencies=True,
                                           stream_tag=f'{bf}_{nc}')
                    sc = snap(c, dkx, dky, Nx, Ny, depth)
                    A = np.array([x.A for x in sc])
                    bad = np.array([(3 * abs(x.kx) > kxn) or (3 * abs(x.ky) > kyn) for x in sc])
                    fr.append((A[bad] ** 2).sum() / (A ** 2).sum())
                worst.append((f'{bf}_{nc}', max(fr)))
        s = '  '.join(f'{k}:{100*v:.1f}%' for k, v in worst)
        print(f'  Nx={Nx} Ny={Ny} L=({Lx:.0f},{Ly:.0f}) kx_nyq={kxn:.3f} ky_nyq={kyn:.3f}')
        print(f'      {s}')

    print('\nQ-c  step 비용 + Q-a 분리가능 검증')
    X, Y = make_grid(cfg.gt_grid.x_range, cfg.gt_grid.y_range,
                     nx=cfg.gt_grid.nx, ny=cfg.gt_grid.ny)
    Xm, Ym = make_grid(cfg.gt_grid.x_range, cfg.gt_grid.y_range, nx=200, ny=167)
    c = jonswap_components(Hs=1.2, Tp=7.0, n_components=15, depth=depth, seed=1581,
                           direction_deg=0.0, direction_spread_deg=30.0, dirs_per_freq=1,
                           random_amplitudes=True, random_frequencies=True,
                           stream_tag='BF4_15')
    for Nx, Ny, Lx, Ly in CAND:
        sim = HOSSimulator(HOSConfig(Lx=Lx, Ly=Ly, Nx=Nx, Ny=Ny, depth=depth, M=3))
        sim.initialize_from_components(snap(c, 2 * np.pi / Lx, 2 * np.pi / Ly, Nx, Ny, depth))
        sim.step(0.02)
        Z0, A0, B0, C0 = sim.evaluate_at(X, Y)
        Z1, A1, B1, C1 = sep_eval(sim, X, Y)
        dz = np.abs(Z0 - Z1).max()
        dn = max(np.abs(A0 - A1).max(), np.abs(B0 - B1).max(), np.abs(C0 - C1).max())
        Zm0, *_ = sim.evaluate_at(Xm, Ym)
        Zm1, *_ = sep_eval(sim, Xm, Ym)
        t0 = time.time()
        for _ in range(20):
            sim.step(0.02)
        per = (time.time() - t0) / 20
        t0 = time.time()
        for _ in range(10):
            sep_eval(sim, X, Y)
            sep_eval(sim, Xm, Ym)
        per_fr = (time.time() - t0) / 10
        print(f'  Nx={Nx} Ny={Ny}: step {per*1000:5.1f} ms -> 1500 steps {1500*per:5.0f} s | '
              f'sep/frame {per_fr*1000:.1f} ms -> 301 fr {301*per_fr:.0f} s | '
              f'Q-a |dZ|={dz:.2e} |dN|={dn:.2e} |dZ_mesh|={np.abs(Zm0-Zm1).max():.2e}')


if __name__ == '__main__':
    main()
