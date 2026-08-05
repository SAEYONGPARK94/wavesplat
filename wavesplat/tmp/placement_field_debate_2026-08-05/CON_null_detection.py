"""CON: is `detection_rate` distinguishable from chance once the search box shrinks?

Re-analysis only. Reads GT `config/wave_components` metadata out of the existing episode
files (no simulation, no estimator run) and re-runs THE SAME matcher used by
tmp/e1_grid_2026-08-05/run_e1.py::match_components against atoms drawn uniformly at
random inside each cell's search box.

Uniform atoms have ZERO resolution by construction. Whatever detection rate they achieve
is the part of the measured detection rate that is bought by COVERAGE alone.
"""
import os
import sys

import h5py
import numpy as np
from scipy.optimize import linear_sum_assignment

REPO = '/home/vexit/wavesplat/wavesplat'
DATA = os.path.join(REPO, 'data_finemesh_2026-07-23')
EPS = ['episode_BF4_15comp_seed0_uniform.h5', 'episode_BF4_30comp_seed0_uniform.h5',
       'episode_BF5_15comp_seed0_uniform.h5', 'episode_BF5_30comp_seed0_uniform.h5',
       'episode_BF3_5comp_seed0_uniform.h5', 'episode_BF3_2comp_seed0_uniform.h5']
NDRAW = 400
WIDE, NARROW = 4.0, 0.8


def det_rate(ekx, eky, gkx, gky, k_scale):
    cost = np.hypot(ekx[:, None] - gkx[None, :], eky[:, None] - gky[None, :]) / k_scale
    r, c = linear_sum_assignment(cost)
    return float(np.sum(cost[r, c] < 0.5)) / len(gkx)


rng = np.random.default_rng(20260805)
print(f'{"episode":34s} {"Ngt":>3s} {"kscale":>7s} {"tol[rad/m]":>10s} '
      f'{"GTnn_med":>8s} {"tol/GTnn":>8s}')
gts = {}
for ep in EPS:
    with h5py.File(os.path.join(DATA, ep), 'r') as f:
        gt = np.asarray(f['config/wave_components'])
    gkx, gky = gt[:, 2], gt[:, 3]
    k_scale = max(np.max(np.hypot(gkx, gky)), 0.01)
    D = np.hypot(gkx[:, None] - gkx[None, :], gky[:, None] - gky[None, :])
    np.fill_diagonal(D, np.inf)
    nn = float(np.median(D.min(1))) if len(gkx) > 1 else float('nan')
    tol = 0.5 * k_scale
    gts[ep] = (gkx, gky, k_scale)
    print(f'{ep:34s} {len(gkx):3d} {k_scale:7.3f} {tol:10.3f} {nn:8.3f} {tol/nn:8.2f}')

print('\n=== chance detection rate, atoms UNIFORM in box (no resolution at all) ===')
print(f'{"episode":34s} {"n_est":>5s} | {"WIDE(-4,4)":>18s} | {"NARROW(-.8,.8)":>18s}')
for ep in EPS:
    gkx, gky, ks = gts[ep]
    for n_est in (3, 15, 30, 40):
        out = []
        for half in (WIDE, NARROW):
            v = [det_rate(rng.uniform(-half, half, n_est), rng.uniform(-half, half, n_est),
                          gkx, gky, ks) for _ in range(NDRAW)]
            out.append(f'{np.mean(v):.3f}+-{np.std(v):.3f}')
        print(f'{ep:34s} {n_est:5d} | {out[0]:>18s} | {out[1]:>18s}')
    print()
