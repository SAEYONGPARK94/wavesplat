#!/usr/bin/env python3
"""C smoke: one moving-sensor episode.  Measures cost and runs the movement gate.

Constants are the ones every recent accuracy batch used (HOSGATE gen_hos.py:41-43,
S15 gen_s15.py:47), so the only thing that differs from a normal episode is the pose.
"""
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.abspath(os.path.join(HERE, '..', '..')))

import moving_sensor as MS                            # noqa: E402
from wavesim.config import WaveSimConfig              # noqa: E402
from wavesim.pipeline import run_episode              # noqa: E402

MESH_NX, MESH_NY = 200, 167          # gen_hos.py:42 / gen_s15.py
DURATION, DT, DEPTH, N_PTS = 30.0, 0.1, 50.0, 50
U_DESIGN = 11.3
OUT = os.path.join(HERE, 'smoke_data')


def build_cfg(U):
    cfg = WaveSimConfig()
    cfg.mesh.nx, cfg.mesh.ny = MESH_NX, MESH_NY
    assert cfg.hos.enabled is False, 'HOS must stay off (as in every accuracy batch)'
    x0, x1 = cfg.gt_grid.x_range
    dx_gt = (x1 - x0) / (cfg.gt_grid.nx - 1)
    dx_mesh = (x1 - x0) / (cfg.mesh.nx - 1)
    new_x1 = x1 + U * DURATION + 20.0
    # preserve SPACING, not node count -- mesh_adequacy.py:67-68 judges on spacing
    cfg.gt_grid.nx = int(round((new_x1 - x0) / dx_gt)) + 1
    cfg.mesh.nx = int(round((new_x1 - x0) / dx_mesh)) + 1
    cfg.gt_grid.x_range = (x0, new_x1)
    return cfg, dx_gt, dx_mesh


def one(U, direction, seed, nc, tag):
    cfg, dx_gt, dx_mesh = build_cfg(U)
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()
    with MS.MovingSensor(U, DT, DURATION):
        path = run_episode(
            sea_state='BF4', n_components=nc, n_pts=N_PTS, seed=seed,
            raydrop_model='uniform', duration=DURATION, dt=DT, depth=DEPTH,
            output_dir=OUT, verbose=False, cfg=cfg, direction_deg=direction,
            direction_spread_deg=30.0, peak_locked=False, dirs_per_freq=1,
            random_amplitudes=True, random_frequencies=True, decorrelate_stream=True)
    wall = time.time() - t0
    ok, msg = MS.check(U, DURATION, DT)
    s = MS.stats()
    print('  %-22s U=%5.1f dir=%3d  %6.1f s  frames=%3d  gate=%s  %s'
          % (tag, U, direction, wall, s.get('calls', 0),
             'PASS' if ok else 'FAIL', msg))
    print('        grid x=[%.0f,%.0f] gt_nx=%d mesh_nx=%d  spacing gt %.4f / mesh %.4f m'
          % (cfg.gt_grid.x_range[0], cfg.gt_grid.x_range[1], cfg.gt_grid.nx,
             cfg.mesh.nx, dx_gt, dx_mesh))
    return wall, ok, path


def main():
    print('=' * 88)
    print('C SMOKE -- moving sensor, one episode per cell   (constants = accuracy batches)')
    print('=' * 88)
    res = []
    for U, d, tag in ((0.0, 0.0, 'STATIC control'),
                      (U_DESIGN, 0.0, 'MOVING following'),
                      (U_DESIGN, 180.0, 'MOVING head')):
        res.append(one(U, d, 1598, 15, tag))
    walls = [r[0] for r in res if r[1]]
    if walls:
        print('\n  cost: median %.1f s/ep  ->  64 episodes ~ %.1f h'
              % (float(np.median(walls)), 64 * float(np.median(walls)) / 3600.0))


if __name__ == '__main__':
    main()
