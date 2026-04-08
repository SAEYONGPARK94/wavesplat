"""
run_noise_ablation.py

Noise robustness ablation: inject additive Gaussian noise into LiDAR z_obs
and measure WaveSplat nRMSE degradation.

Reviewer defense: quantify method sensitivity to realistic sensor noise
beyond the baseline simulator noise model.
"""

import sys
import os
import json
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from wavesplat.data_loader import EpisodeDataLoader, FrameData
from wavesplat.run import set_seed
from wavesplat.optimizer.sequential_update import SequentialUpdater
from wavesplat.evaluation.metrics import compute_all_metrics


# Representative episodes for noise sweep (available in repo)
NOISE_EPISODES = [
    'episode_BF3_5comp_seed0_uniform.h5',
    'episode_BF4_5comp_seed0_uniform.h5',
    'episode_BF4_15comp_seed0_uniform.h5',
]

# Additive Gaussian noise σ (meters) relative to baseline simulator noise (~0.02 m)
# These correspond to 0x, 1x, 2.5x, 5x baseline σ
NOISE_LEVELS = [0.0, 0.02, 0.05, 0.10]


def add_noise_to_frame(frame, sigma, rng):
    """Return a new FrameData with additive Gaussian noise on z_obs."""
    if sigma <= 0 or len(frame.z_obs) == 0:
        return frame
    noise = torch.from_numpy(
        rng.normal(0.0, sigma, size=frame.z_obs.shape).astype(np.float32)
    )
    return FrameData(
        t=frame.t,
        x_obs=frame.x_obs,
        y_obs=frame.y_obs,
        z_obs=frame.z_obs + noise,
        confidence=frame.confidence,
        incidence_angles=frame.incidence_angles,
        gt_field=frame.gt_field,
    )


class NoisyLoader:
    """Wraps EpisodeDataLoader injecting additive Gaussian noise on z_obs."""

    def __init__(self, loader, sigma, seed=1234):
        self.loader = loader
        self.sigma = sigma
        self.rng = np.random.default_rng(seed)
        self.gt_grid_x = loader.gt_grid_x
        self.gt_grid_y = loader.gt_grid_y
        self.depth = loader.depth
        self.Hs_true = loader.Hs_true
        self.Tp_true = loader.Tp_true
        self.sea_state = loader.sea_state
        self.n_frames = loader.n_frames

    def get_init_window(self, tp_factor=2.0):
        return [add_noise_to_frame(f, self.sigma, self.rng)
                for f in self.loader.get_init_window(tp_factor)]

    def get_tracking_frames(self, tp_factor=2.0):
        return [add_noise_to_frame(f, self.sigma, self.rng)
                for f in self.loader.get_tracking_frames(tp_factor)]

    def __iter__(self):
        for frame in self.loader:
            yield add_noise_to_frame(frame, self.sigma, self.rng)


def run_ws_with_noise(h5_path, sigma):
    set_seed(42)
    loader = EpisodeDataLoader(h5_path)
    noisy = NoisyLoader(loader, sigma, seed=1234)

    cfg = {
        'n_gaussians': [3, 8, 15],
        'depth': noisy.depth,
        'lr': 0.01,
        'lambda_l1': 0.001,
        'lambda_smooth': 0.0001,
        'tracking_steps': 10,
        'tracking_mode': 'w_phi',
        'tracking_lr': 0.005,
        'init_method': 'v2_konly',
        'use_dispersion': True,
        'use_confidence': True,
        'adaptive_n': True,
        'prune_threshold': 0.10,
        'grow_residual_threshold': 0.05,
        'max_n': 30,
        'tracking_window': 3,
        'device': 'cpu',
        'seed': 42,
    }

    updater = SequentialUpdater(cfg)
    init_frames = noisy.get_init_window(tp_factor=2.0)
    updater.initialize(
        init_frames,
        x_range=(float(noisy.gt_grid_x[0]), float(noisy.gt_grid_x[-1])),
        y_range=(float(noisy.gt_grid_y[0]), float(noisy.gt_grid_y[-1])),
    )

    grid_x = noisy.gt_grid_x
    grid_y = noisy.gt_grid_y
    tracking_frames = noisy.get_tracking_frames(tp_factor=2.0)

    frame_results = []
    for frame in tracking_frames:
        updater.update(frame)
        pred = updater.predict_field(grid_x, grid_y, frame.t)
        pred_np = pred.cpu().numpy()
        gt_np = frame.gt_field.cpu().numpy()
        metrics = compute_all_metrics(
            pred_np, gt_np,
            grid_x.cpu().numpy(), grid_y.cpu().numpy(),
            noisy.Hs_true)
        frame_results.append(metrics)

    return {
        'avg_nRMSE': float(np.mean([r['nRMSE'] for r in frame_results])),
        'final_n': updater.model.n_gaussians,
    }


def main():
    data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
    results = {}

    print("=" * 80)
    print("NOISE ROBUSTNESS ABLATION")
    print("=" * 80)
    print(f"Noise σ levels (m): {NOISE_LEVELS}")

    for ep_name in NOISE_EPISODES:
        h5_path = os.path.join(data_dir, ep_name)
        if not os.path.exists(h5_path):
            print(f"  SKIP (not found): {ep_name}")
            continue

        short = ep_name.replace('episode_', '').replace('.h5', '').replace('_seed0_uniform', '')
        results[short] = {}
        print(f"\n[{short}]")

        for sigma in NOISE_LEVELS:
            try:
                r = run_ws_with_noise(h5_path, sigma)
                results[short][f"sigma_{sigma:.2f}"] = r
                print(f"  σ={sigma:.2f} m: nRMSE={r['avg_nRMSE']:.4f} "
                      f"(N={r['final_n']})", flush=True)
            except Exception as e:
                print(f"  σ={sigma:.2f} m: FAILED ({e})")
                results[short][f"sigma_{sigma:.2f}"] = {'error': str(e)}

    out_path = os.path.join(os.path.dirname(__file__), '..', 'results',
                            'noise_ablation.json')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump({'noise_levels': NOISE_LEVELS, 'results': results}, f, indent=2)

    print(f"\nSaved: {out_path}")


if __name__ == '__main__':
    main()
