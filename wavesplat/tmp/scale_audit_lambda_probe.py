"""
Read-only diagnostic (SCALE_AUDIT, 2026-07-23): decompose the Phase-1 training loss into its
data-fit term vs the L1 term, AFTER the shipped `initialize()` has converged, across a few
episodes spanning the Hs range. Does NOT modify any production file, does NOT touch
coarse_to_fine.SCALE_INVARIANT_ADVANCE or tmp/scale_invariant_advance*.

Reproduces the exact formula in sequential_update.py:_compute_loss (read, not copied-and-edited
behaviour -- same weights/residual construction) but prints the two additive terms separately
instead of only their (already-computed, unchanged) sum.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from wavesplat.data_loader import EpisodeDataLoader
from wavesplat.optimizer.sequential_update import SequentialUpdater

DEFAULT_CONFIG = {
    'n_gaussians': [3, 8, 15], 'depth': 50.0, 'lr': 0.01,
    'lambda_l1': 0.001, 'lambda_smooth': 0.0001,
    'tracking_steps': 10, 'tracking_mode': 'w_phi', 'tracking_lr': 0.005,
    'init_method': 'v2_konly', 'use_dispersion': True, 'use_confidence': True,
    'adaptive_n': True, 'prune_threshold': 0.10, 'grow_residual_threshold': 0.05,
    'max_n': 30, 'tracking_window': 3, 'device': 'cpu', 'seed': 42,
}

import argparse
_p = argparse.ArgumentParser()
_p.add_argument('--episodes', nargs='+', default=[
    'data/episode_BF2_1comp_seed0_uniform.h5',
    'data/episode_BF3_2comp_seed0_uniform.h5',
])
_args = _p.parse_args()
episodes = _args.episodes

for ep in episodes:
    torch.manual_seed(42); np.random.seed(42)
    loader = EpisodeDataLoader(ep, device='cpu')
    cfg = dict(DEFAULT_CONFIG); cfg['depth'] = loader.depth
    updater = SequentialUpdater(cfg)
    init_frames = loader.get_init_window(tp_factor=2.0)
    res = updater.initialize(
        init_frames,
        x_range=(float(loader.gt_grid_x[0]), float(loader.gt_grid_x[-1])),
        y_range=(float(loader.gt_grid_y[0]), float(loader.gt_grid_y[-1])),
        verbose=False,
    )
    m = updater.model
    with torch.no_grad():
        # Reproduce the data-fit term exactly (same construction as _compute_loss)
        total_data = torch.tensor(0.0); n_obs = 0
        for frame in init_frames:
            if len(frame.x_obs) == 0:
                continue
            z_pred = m(frame.x_obs, frame.y_obs, frame.t)
            residual = z_pred - frame.z_obs
            weights = frame.confidence if cfg['use_confidence'] else torch.ones_like(frame.confidence)
            total_data = total_data + (weights * residual ** 2).sum()
            n_obs += len(frame.x_obs)
        data_term = (total_data / n_obs).item() if n_obs > 0 else float('nan')
        l1_term = (cfg['lambda_l1'] * torch.sum(torch.abs(m.w))).item()
        w_np = m.w.cpu().numpy()
        sum_abs_w = float(np.sum(np.abs(w_np)))
    print(f"{ep}: Hs={loader.Hs_true:.2f} N={m.n_gaussians} "
          f"data_term={data_term:.6f} l1_term={l1_term:.6f} "
          f"l1/data_ratio={l1_term/data_term if data_term>0 else float('nan'):.4f} "
          f"sum|w|={sum_abs_w:.4f} final_loss_reported={res['loss_history'][-1]:.6f}")
