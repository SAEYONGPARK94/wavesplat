#!/usr/bin/env python3
"""C corpus generation -- 2x2 (U x direction), n=16 crowded, PREREG §3 + AMENDMENT_1.

Constants are the accuracy-batch standard (HOSGATE gen_hos.py:41-43): BF4, 15 components,
30.0 s @ 0.1 s, depth 50, n_pts 50, uniform raydrop, spread 30 deg, and the three
independence flags that the 2026-08-07 generator fix requires.

Each cell writes to its OWN directory.  run_episode's filename does not encode U, so
(U=0, dir 0) and (U=11.3, dir 0) collide -- AMENDMENT_1 A-3, caught in the smoke where it
silently destroyed one of three episodes.
"""
import json
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
from wavesim.wave_generator import jonswap_components  # noqa: E402

MESH_NX, MESH_NY = 200, 167
DURATION, DT, DEPTH, N_PTS = 30.0, 0.1, 50.0, 50
SEA, NCOMP, SPREAD = 'BF4', 15, 30.0
U_DESIGN = 11.3
SEEDS = list(range(1598, 1614))                       # PREREG §3: 16 episode seeds

CELLS = [
    ('A_static_following', 0.0, 0.0),
    ('B_static_head', 0.0, 180.0),
    ('C_moving_following', U_DESIGN, 0.0),            # <- PRIMARY cell
    ('D_moving_head', U_DESIGN, 180.0),
]


def build_cfg(U):
    cfg = WaveSimConfig()
    cfg.mesh.nx, cfg.mesh.ny = MESH_NX, MESH_NY
    assert cfg.hos.enabled is False, 'HOS must stay off'
    x0, x1 = cfg.gt_grid.x_range
    dx_gt = (x1 - x0) / (cfg.gt_grid.nx - 1)
    dx_mesh = (x1 - x0) / (cfg.mesh.nx - 1)
    new_x1 = x1 + U * DURATION + 20.0
    cfg.gt_grid.nx = int(round((new_x1 - x0) / dx_gt)) + 1
    cfg.mesh.nx = int(round((new_x1 - x0) / dx_mesh)) + 1
    cfg.gt_grid.x_range = (x0, new_x1)
    return cfg, dx_gt, dx_mesh


def encounter_spectrum(seed, direction, U):
    """G-2a/G-2c evidence, computed from the component table the episode will use."""
    comps = jonswap_components(
        Hs=1.2, Tp=7.0, n_components=NCOMP, depth=DEPTH, direction_deg=direction,
        direction_spread_deg=SPREAD, seed=seed, peak_locked=False, dirs_per_freq=1,
        random_amplitudes=True, random_frequencies=True, stream_tag='ds')
    kx = np.array([c.kx for c in comps])
    om = np.array([c.omega for c in comps])
    A = np.array([c.A for c in comps])
    we = om - kx * U
    w = A ** 2 / np.sum(A ** 2)
    return dict(we_peak=float(we[int(np.argmax(A))]),
                we_energy_mean=float(np.sum(w * np.abs(we))),
                frac_energy_slow=float(np.sum(w[np.abs(we) < 0.1])),
                cycles_energy=float(np.sum(w * np.abs(we)) * DURATION / (2 * np.pi)))


def main():
    root = os.path.join(HERE, 'data')
    manifest = []
    t_start = time.time()
    for cell, U, direction in CELLS:
        out = os.path.join(root, cell)
        os.makedirs(out, exist_ok=True)
        cfg, dx_gt, dx_mesh = build_cfg(U)
        for seed in SEEDS:
            done = [f for f in os.listdir(out) if f.endswith('.h5')
                    and ('seed%d_' % seed) in f]
            if done:
                continue
            t0 = time.time()
            with MS.MovingSensor(U, DT, DURATION):
                path = run_episode(
                    sea_state=SEA, n_components=NCOMP, n_pts=N_PTS, seed=seed,
                    raydrop_model='uniform', duration=DURATION, dt=DT, depth=DEPTH,
                    output_dir=out, verbose=False, cfg=build_cfg(U)[0],
                    direction_deg=direction, direction_spread_deg=SPREAD,
                    peak_locked=False, dirs_per_freq=1, random_amplitudes=True,
                    random_frequencies=True, decorrelate_stream=True,
                    use_gpu_raycast=True)
            ok, msg = MS.check(U, DURATION, DT)
            row = dict(cell=cell, U=U, direction_deg=direction, seed=seed,
                       path=os.path.relpath(path, HERE), gate_G1=bool(ok),
                       gate_G1_msg=msg, wall_s=round(time.time() - t0, 1),
                       grid_x=list(cfg.gt_grid.x_range), gt_nx=cfg.gt_grid.nx,
                       mesh_nx=cfg.mesh.nx, dx_gt=round(dx_gt, 4),
                       dx_mesh=round(dx_mesh, 4),
                       **encounter_spectrum(seed, direction, U))
            manifest.append(row)
            with open(os.path.join(HERE, 'gen_manifest.json'), 'w') as fh:
                json.dump(manifest, fh, indent=1)
            print('[%5.1f min] %-20s seed %d  %5.1f s  G1=%s  we_peak=%+.4f '
                  'cycles=%.2f' % ((time.time() - t_start) / 60.0, cell, seed,
                                   row['wall_s'], 'OK' if ok else 'FAIL',
                                   row['we_peak'], row['cycles_energy']),
                  flush=True)
            if not ok:
                print('  !! G-1 FAILED -- PREREG §7 requires ABORT'); return 1
    n_by_cell = {c: sum(1 for r in manifest if r['cell'] == c) for c, _, _ in CELLS}
    print('\nDONE %.1f min | per-cell counts: %s' % ((time.time() - t_start) / 60.0,
                                                     n_by_cell))
    # G-0b: the collision guard AMENDMENT_1 A-3 exists for
    for cell, _, _ in CELLS:
        d = os.path.join(root, cell)
        n = len([f for f in os.listdir(d) if f.endswith('.h5')])
        print('  %-20s %d files %s' % (cell, n, 'OK' if n == len(SEEDS) else '!! MISMATCH'))
    return 0


if __name__ == '__main__':
    sys.exit(main())
