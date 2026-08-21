"""K6 실행 전 게이트 (PREREG §3 / §6.1).

`data_indep_2026-08-07/` 의 혼잡16 이 **진짜로 16 독립 추출**인지 확인한다.
규약은 `tmp/audit_independence_2026-08-07/audit_independence.py` 와 동일:
난수 단위(draw)를 허용오차로 세고, 방향은 kx/ky 바이트가 아니라
theta = atan2(ky, kx) 로 센다.

⛔4축(진폭·omega·위상·방향) + 진실면이 **전부 16종**이 아니면 K6 을 돌리지 않는다.
비교 대조로 기존 `data_finemesh_2026-07-23/` 의 같은 16 셀도 함께 출력한다.
결과 파일은 읽지 않는다.
"""
import os
import sys

import h5py
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..'))
TOL = 1e-4

CROWDED16 = [f'episode_BF{bf}_{nc}comp_seed{s}_uniform.h5'
             for bf in (4, 5) for nc in (15, 30) for s in range(4)]


def n_draws(vecs):
    seen = []
    for v in vecs:
        if not any(len(v) == len(s) and np.allclose(v, s, atol=TOL, rtol=0) for s in seen):
            seen.append(v)
    return len(seen)


def report(data_dir, eps, label):
    comps, flds = [], []
    for ep in eps:
        p = os.path.join(REPO, data_dir, ep)
        if not os.path.exists(p):
            print(f'  !! 없음: {ep}')
            return None
        with h5py.File(p, 'r') as f:
            comps.append(np.array(f['config/wave_components']))
            flds.append(np.array(f['frames/gt_wave_field']))
    axes = {
        '진폭 A':    n_draws([c[:, 0] for c in comps]),
        'omega':     n_draws([c[:, 1] for c in comps]),
        '위상 phi':  n_draws([c[:, 4] for c in comps]),
        '방향 theta': n_draws([np.arctan2(c[:, 3], c[:, 2]) for c in comps]),
        '진실면':     n_draws([f.ravel() for f in flds]),
    }
    print(f'\n### {label}  ({data_dir}, {len(eps)} 에피소드)')
    for k, v in axes.items():
        mark = '  ✅' if v == len(eps) else f'  ⛔ ({len(eps)} 이어야 함)'
        print(f'  {k:12s} draw 종수 = {v:3d}{mark}')
    lo = min(axes.values())
    floor = min(1.0, 2.0 / 2 ** lo)
    print(f'  ⟹ 가장 좁은 축 = {lo}  ⟹ 전승 시 바닥 p = {floor:.2e}'
          + ('  ⛔ 0.05 도달 불가' if floor > 0.05 else '  ✅ α=0.05 도달 가능'))
    return axes, lo, floor


def main():
    print('K6 실행 전 독립성 게이트 (PREREG §3)')
    indep = [e.replace('.h5', '_ra_rf_ds.h5') for e in CROWDED16]
    new = report('data_indep2_2026-08-07', indep, '★신규 독립 표본 v2 (ra+rf+ds)')
    old = report('data_finemesh_2026-07-23', CROWDED16, '(대조) 기존 표본')
    print('\n' + '=' * 70)
    if new is None:
        print('GATE: FAIL — 신규 표본을 읽을 수 없다'); return 1
    axes, lo, floor = new
    ok = lo == len(indep)
    print(f'GATE: {"PASS ✅ — K6 실행 허가" if ok else "FAIL ⛔ — K6 실행 금지"}')
    if old:
        print(f'  (기존 표본의 가장 좁은 축 = {old[1]}, 바닥 p = {old[2]:.4f})')
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
