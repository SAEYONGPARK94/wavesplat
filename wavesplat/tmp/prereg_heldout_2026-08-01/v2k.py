"""Arm-C unit: median-of-3 `v2_konly` fit on one observation window, rendered
at a list of times.  Separated into its own module so the seeds can be run in
spawned worker processes (the parent holds a CUDA context for the bank arm's
beamform; fork would poison it).

Each member is `set_seed(s)` immediately followed by a fresh
`SequentialUpdater(init_method='v2_konly')` init, so a member's result depends
only on its own seed -- process placement cannot change a value
(asserted by test_parallel.py).
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))


def _setup():
    os.environ.setdefault('OMP_NUM_THREADS', '4')
    if ROOT not in sys.path:
        sys.path.insert(0, ROOT)
    sp = os.path.join(ROOT, 'scripts')
    if sp not in sys.path:
        sys.path.insert(0, sp)
    try:
        os.chdir(ROOT)
    except OSError:
        pass
    import warnings
    warnings.filterwarnings('ignore')


def v2_member(args):
    """One v2_konly member.  args = (P, z, depth, gxq, gyq, t_shift, rel_times, seed).
    Returns (n_rel, ny, nx) float64 fields."""
    _setup()
    import numpy as np
    import torch
    from wavesplat.optimizer.sequential_update import SequentialUpdater
    from wavesplat.data_loader import FrameData
    from wavesplat.run import DEFAULT_CONFIG, set_seed

    P, z, depth, gxq, gyq, t_shift, rel_times, seed = args
    P = np.asarray(P, float)
    z = np.asarray(z, float)
    frames = []
    for tt in np.unique(P[:, 2]):
        m = P[:, 2] == tt
        n = int(m.sum())
        frames.append(FrameData(
            t=float(tt - t_shift),
            x_obs=torch.tensor(P[m, 0], dtype=torch.float32),
            y_obs=torch.tensor(P[m, 1], dtype=torch.float32),
            z_obs=torch.tensor(z[m], dtype=torch.float32),
            confidence=torch.ones(n, dtype=torch.float32),
            incidence_angles=torch.zeros(n, dtype=torch.float32)))
    cfg = dict(DEFAULT_CONFIG)
    # init_method EXPLICIT (prereg 3.6): never inherit the shipped 'auto' default.
    cfg.update(depth=float(depth), seed=int(seed), device='cpu',
               init_method='v2_konly')
    set_seed(int(seed))
    upd = SequentialUpdater(cfg)
    upd.initialize(frames,
                   x_range=(float(np.min(gxq)), float(np.max(gxq))),
                   y_range=(float(np.min(gyq)), float(np.max(gyq))),
                   verbose=False)
    tx = torch.tensor(np.asarray(gxq, float), dtype=torch.float32)
    ty = torch.tensor(np.asarray(gyq, float), dtype=torch.float32)
    out = [upd.predict_field(tx, ty, float(tr)).cpu().numpy().astype(float)
           for tr in rel_times]
    return np.stack(out, 0)
