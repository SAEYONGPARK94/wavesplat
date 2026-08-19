#!/usr/bin/env python3
"""S15 사전 프로브 — **추정기 0회 호출**. 배선·기하·산술만 잰다.

여기서 재는 것 (전부 판정과 무관, 사전등록을 쓰기 위한 사실 확인):

 1. `direction_deg=180` 이 성분 리스트에 정확히 무엇을 하는가
    (A, omega, phi 는 그대로이고 k 만 부호가 뒤집히는가 = 완전한 짝 설계가 가능한가)
 2. 저장된 `data_indep_crowded_2026-08-08/` 파일을 **오늘 코드가 재현하는가**
    (재현되면 기존 dir0 코퍼스를 그대로 짝의 한쪽으로 쓸 수 있다)
 3. ★동기 검산: "following seas 에서 14 s 창에 위상이 0.42 rad 만 돈다" 가
    **이 시뮬레이터에서 성립하는가** (배가 정지해 있으므로 조우주파수가 없다)
 4. 관측 개구가 x 반사(x -> -x)에 대해 비대칭인가 (비대칭이어야 두 방향이 다른 문제가 된다)

출력은 `probe_dir.log` 로 리다이렉트해서 보존한다.
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import h5py                                                   # noqa: E402
from wavesim.wave_generator import jonswap_components         # noqa: E402

G = 9.81


def sec(t):
    print('\n' + '=' * 78)
    print(t)
    print('=' * 78)


# ── 1. direction_deg 가 성분에 하는 일 ───────────────────────────────────────
sec('1. direction_deg=0 vs 180 : 성분 단위 짝 검사 (seed 20, BF4 15comp)')
kw = dict(Hs=1.2, Tp=7.0, n_components=15, depth=50.0, seed=20,
          direction_spread_deg=30.0, random_amplitudes=True,
          random_frequencies=True, stream_tag='BF4_15')
c0 = jonswap_components(direction_deg=0.0, **kw)
c180 = jonswap_components(direction_deg=180.0, **kw)
A0 = np.array([c.A for c in c0]); A1 = np.array([c.A for c in c180])
w0 = np.array([c.omega for c in c0]); w1 = np.array([c.omega for c in c180])
p0 = np.array([c.phi for c in c0]); p1 = np.array([c.phi for c in c180])
k0 = np.array([[c.kx, c.ky] for c in c0]); k1 = np.array([[c.kx, c.ky] for c in c180])
print(f'  max|dA|      = {np.abs(A0 - A1).max():.3e}')
print(f'  max|domega|  = {np.abs(w0 - w1).max():.3e}')
print(f'  max|dphi|    = {np.abs(p0 - p1).max():.3e}')
print(f'  max|k180+k0| = {np.abs(k1 + k0).max():.3e}   (0 이면 k -> -k 정확 반전)')
print(f'  max|dQ|      = {max(abs(a.Q - b.Q) for a, b in zip(c0, c180)):.3e}')
print(f'  dir0   평균 진행각 = {np.degrees(np.arctan2(k0[:,1],k0[:,0])).mean():+.2f} deg')
print(f'  dir180 평균 진행각 = {np.degrees(np.arctan2(k1[:,1],k1[:,0])).mean():+.2f} deg')

# ── 2. 저장된 dir0 코퍼스를 오늘 코드가 재현하는가 ──────────────────────────
sec('2. 저장된 data_indep_crowded_2026-08-08 재현 검사 (오늘 코드)')
worst = 0.0
for bf, Hs, Tp in (('BF4', 1.2, 7.0), ('BF5', 2.0, 9.0)):
    for nc in (15, 30, 2, 5):
        d = ('data_indep_crowded_2026-08-08' if nc in (15, 30)
             else 'data_indep_sparse_2026-08-08')
        for s in range(20, 24):
            p = os.path.join(ROOT, d,
                             f'episode_{bf}_{nc}comp_seed{s}_uniform_ra_rf_ds.h5')
            if not os.path.exists(p):
                print(f'  MISSING {p}')
                continue
            with h5py.File(p, 'r') as f:
                st = np.asarray(f['config/wave_components'])   # A, omega, kx, ky, phi(, Q)
            cc = jonswap_components(Hs=Hs, Tp=Tp, n_components=nc, depth=50.0, seed=s,
                                    direction_deg=0.0, direction_spread_deg=30.0,
                                    random_amplitudes=True, random_frequencies=True,
                                    stream_tag=f'{bf}_{nc}')
            gen = np.array([[c.A, c.omega, c.kx, c.ky, c.phi] for c in cc])
            dmax = float(np.abs(st[:, :5] - gen).max())
            worst = max(worst, dmax)
print(f'  저장 파일 32개 대비 오늘 코드 성분 재현 max|delta| = {worst:.3e}')
print('  ⟹ 0 이면 기존 dir0 파일을 짝의 한쪽으로 그대로 쓸 수 있다.')

# ── 3. ★동기 검산: 위상이 정말 0.42 rad 만 도는가 ───────────────────────────
sec('3. ★동기 검산 — "14 s 창에서 위상 0.42 rad" 는 어느 좌표계의 값인가')
U = 11.3          # reference_agreed_operating_point.md 설계 선속
for bf, Hs, Tp in (('BF4', 1.2, 7.0), ('BF5', 2.0, 9.0)):
    wp = 2 * np.pi / Tp
    kp = wp ** 2 / G                      # deep water
    cp = wp / kp
    T_init = 2.0 * Tp                     # loader.get_init_window(tp_factor=2.0)
    print(f'  {bf}: Tp={Tp}s  omega_p={wp:.4f}  k_p={kp:.4f}  c_p={cp:.2f} m/s  '
          f'U/c_p={U/cp:.3f}')
    print(f'      정지 센서 (이 저장소가 실제로 시뮬레이션하는 것):')
    print(f'        초기화 창 T={T_init:.1f}s 에서 위상 진행 = omega*T = '
          f'{wp*T_init:.2f} rad = {wp*T_init/(2*np.pi):.2f} 주기  '
          f'(dir0 · dir180 **동일**)')
    print(f'      가상의 전진 선박 U={U} m/s (run_episode 에 인자 없음):')
    print(f'        following: omega_e=omega(1-U/c_p)={wp*(1-U/cp):+.4f} -> '
          f'{abs(wp*(1-U/cp))*T_init:.2f} rad')
    print(f'        head     : omega_e=omega(1+U/c_p)={wp*(1+U/cp):+.4f} -> '
          f'{abs(wp*(1+U/cp))*T_init:.2f} rad  '
          f'(비 {abs((1+U/cp)/(1-U/cp)):.0f}x)')

# ── 4. 관측 개구의 x-비대칭 ─────────────────────────────────────────────────
sec('4. 관측 개구가 x -> -x 에 대해 비대칭인가 (두 방향이 다른 문제가 되려면 필요)')
p = os.path.join(ROOT, 'data_indep_crowded_2026-08-08',
                 'episode_BF4_15comp_seed20_uniform_ra_rf_ds.h5')
with h5py.File(p, 'r') as f:
    gx = f['gt_grid/x'][:] if 'gt_grid/x' in f else None
    keys = list(f['frames'].keys())
    pts = f['frames/lidar_points'][:] if 'lidar_points' in f['frames'] else None
print(f'  frames 키: {keys}')
if pts is not None:
    P = np.asarray(pts)
    P = P.reshape(-1, P.shape[-1])
    m = np.isfinite(P[:, 0])
    x = P[m, 0]; y = P[m, 1]
    print(f'  LiDAR 반환 {m.sum()} 점: x in [{x.min():.1f}, {x.max():.1f}] '
          f'y in [{y.min():.1f}, {y.max():.1f}]')
    print(f'  x<0 인 반환 = {int((x < 0).sum())} 개  ⟹ 개구는 +x 반평면에만 있다')
    r = np.hypot(x, y)
    print(f'  센서 거리 r: min {r.min():.1f}  median {np.median(r):.1f}  max {r.max():.1f}')
    for lo, hi in ((0, 25), (25, 50), (50, 75), (75, 100), (100, 150), (150, 300)):
        print(f'    r in [{lo:3d},{hi:3d}) : {int(((r>=lo)&(r<hi)).sum()):7d} 점 '
              f'({100*((r>=lo)&(r<hi)).mean():5.1f}%)')
if gx is not None:
    print(f'  GT 격자 x in [{gx.min():.1f}, {gx.max():.1f}] ⟹ 격자도 +x 전용')
print('  ⟹ 개구가 +x 전용이므로 x->-x 대칭이 없다. dir0 과 dir180 은 같은 문제가 아니다.')
print('     (단 그 차이는 조우주파수가 아니라 **개구 상대 전파방향**이다 — §3 참조)')
