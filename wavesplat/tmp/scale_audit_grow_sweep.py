"""
Read-only diagnostic (SCALE_AUDIT, 2026-07-23): Phase-1-only (no tracking loop) sweep across
one representative episode per (sea_state, n_components) combo in the shipped dataset, to
measure how `_adaptive_prune_grow`'s `residual_rms` (sequential_update.py:482) compares to the
fixed `grow_residual_threshold=0.05` [m] (sequential_update.py:86) across the Hs range.

Does NOT modify any production file. Does NOT touch coarse_to_fine.SCALE_INVARIANT_ADVANCE or
tmp/scale_invariant_advance*. Calls the shipped `SequentialUpdater.initialize()` unmodified
(same code path `python -m wavesplat.run` uses) and reads its own verbose prints / model state
after the call returns -- adds no new randomness (seed fixed to the run.py default, 42).
"""
import glob
import re
import time
import numpy as np
import torch

from wavesplat.data_loader import EpisodeDataLoader
from wavesplat.optimizer.sequential_update import SequentialUpdater

DEFAULT_CONFIG = {
    'n_gaussians': [3, 8, 15], 'depth': 50.0, 'lr': 0.01,
    'lambda_l1': 0.001, 'lambda_smooth': 0.0001, 'tracking_steps': 10,
    'tracking_mode': 'w_phi', 'tracking_lr': 0.005, 'init_method': 'v2_konly',
    'use_dispersion': True, 'use_confidence': True, 'adaptive_n': True,
    'prune_threshold': 0.10, 'grow_residual_threshold': 0.05, 'max_n': 30,
    'tracking_window': 3, 'device': 'cpu', 'seed': 42,
}

files = sorted(glob.glob('data/*.h5'))
seen = {}
for fp in files:
    m = re.match(r'data/episode_(BF\d)_(\d+comp)_seed(\d+)_(\w+)\.h5', fp)
    if not m:
        continue
    bf, comp, seed, kind = m.groups()
    tag = (bf, comp)
    # prefer uniform/seed0 when available, else take first found
    if tag not in seen or (kind == 'uniform' and seed == '0'):
        seen[tag] = fp

combos = sorted(seen.items(), key=lambda kv: (kv[0][0], int(kv[0][1].replace('comp', ''))))

print(f"{'sea':5s} {'comp':6s} {'Hs':>6s} {'N_after_sched':>14s} {'grow?':>6s} "
      f"{'rollback?':>10s} {'residual_rms':>13s} {'ratio_to_Hs':>12s} {'N_final':>8s}")

results = []
for (bf, comp), fp in combos:
    t0 = time.time()
    torch.manual_seed(42)
    np.random.seed(42)
    loader = EpisodeDataLoader(fp, device='cpu')
    cfg = dict(DEFAULT_CONFIG)
    cfg['depth'] = loader.depth
    updater = SequentialUpdater(cfg)
    init_frames = loader.get_init_window(tp_factor=2.0)

    import io
    import contextlib
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        res = updater.initialize(
            init_frames,
            x_range=(float(loader.gt_grid_x[0]), float(loader.gt_grid_x[-1])),
            y_range=(float(loader.gt_grid_y[0]), float(loader.gt_grid_y[-1])),
            verbose=True,
        )
    log = buf.getvalue()

    grew = '[Adaptive] Grew' in log
    rollback = 'rolled back' in log
    # parse residual RMS from either "no growth needed" or "Grew" line
    rrmatch = re.search(r'Residual RMS=([\d.]+)', log)
    if rrmatch is None:
        rrmatch = re.search(r'residual RMS=([\d.]+)\)', log)
    residual_rms = float(rrmatch.group(1)) if rrmatch else float('nan')

    Hs = loader.Hs_true
    ratio = residual_rms / Hs if Hs else float('nan')
    n_final = updater.model.n_gaussians

    print(f"{bf:5s} {comp:6s} {Hs:6.2f} {15:14d} {str(grew):>6s} "
          f"{str(rollback):>10s} {residual_rms:13.4f} {ratio:12.4f} {n_final:8d}  "
          f"[{time.time()-t0:.1f}s] {fp}")
    results.append(dict(bf=bf, comp=comp, Hs=Hs, grew=grew, rollback=rollback,
                         residual_rms=residual_rms, ratio=ratio, n_final=n_final, file=fp))

import json
with open('tmp/scale_audit_grow_sweep_results.json', 'w') as f:
    json.dump(results, f, indent=2, default=str)
print("\nSaved tmp/scale_audit_grow_sweep_results.json")
