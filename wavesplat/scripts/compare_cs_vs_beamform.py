#!/usr/bin/env python3
"""
Compare CS init vs beamforming init on existing Gerstner GT episodes.

Tests init-only quality: given sparse LiDAR observations, which method
produces better initial (kx, ky, w, phi) estimates?

Evaluation:
  - Reconstruct field on GT grid at a test frame
  - Measure nRMSE against GT field
  - Compare: beamforming_v2 vs CS_LASSO

Usage:
    python3 -m scripts.compare_cs_vs_beamform
"""

import numpy as np
import torch
import time
import json
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from wavesplat.data_loader import EpisodeDataLoader
from wavesplat.optimizer.initializer import initialize_from_frames_v2
from wavesplat.optimizer.cs_initializer import initialize_cs
from wavesplat.model.dispersion import dispersion_omega
from wavesplat.evaluation.metrics import nrmse, pearson_correlation


def predict_field(kx, ky, w, phi, depth, grid_x, grid_y, t):
    """Predict field from spectral components."""
    k_mag = torch.sqrt(kx**2 + ky**2 + 1e-12)
    omega = dispersion_omega(k_mag, depth)

    Xg, Yg = torch.meshgrid(grid_x, grid_y, indexing='xy')
    x_flat = Xg.reshape(-1)
    y_flat = Yg.reshape(-1)

    phase = (kx[None, :] * x_flat[:, None]
             + ky[None, :] * y_flat[:, None]
             - omega[None, :] * t
             + phi[None, :])
    z = torch.sum(w[None, :] * torch.cos(phase), dim=1)
    return z.reshape(len(grid_y), len(grid_x)).detach().numpy()


def evaluate_episode(h5_path, verbose=True):
    """Evaluate beamforming vs CS on one episode."""
    loader = EpisodeDataLoader(h5_path, device='cpu')
    depth = loader.depth if loader.depth else 50.0
    Hs = loader.Hs_true

    init_frames = loader.get_init_window(tp_factor=2.0)
    if len(init_frames) < 5:
        return None

    # Use frame just after init window for evaluation
    tracking_frames = loader.get_tracking_frames(tp_factor=2.0)
    if len(tracking_frames) < 2:
        return None

    eval_frame = tracking_frames[1]  # 2nd tracking frame
    gt_field = eval_frame.gt_field
    if gt_field is None:
        return None
    gt_np = gt_field.numpy()
    t_eval = eval_frame.t
    grid_x = loader.gt_grid_x
    grid_y = loader.gt_grid_y

    episode_name = os.path.basename(h5_path)
    results = {'episode': episode_name, 'Hs': float(Hs)}

    np.random.seed(42)
    torch.manual_seed(42)

    # --- Beamforming v2 (current default) ---
    for K in [15, 30]:
        t0 = time.time()
        kx, ky, w, phi = initialize_from_frames_v2(
            init_frames, n_gaussians=K, depth=depth, seed=42)
        elapsed = time.time() - t0

        with torch.no_grad():
            pred = predict_field(kx, ky, w, phi, depth, grid_x, grid_y, t_eval)

        nm = nrmse(pred, gt_np, Hs)
        rho = pearson_correlation(pred, gt_np)

        label = f"BF_K{K}"
        results[label] = {
            'nrmse': float(nm), 'rho': float(rho),
            'time_s': float(elapsed), 'K': K,
        }
        if verbose:
            print(f"    {label:12s}: nRMSE={nm:.4f}, ρ={rho:.3f}, {elapsed:.2f}s")

    # --- CS LASSO ---
    for K in [15, 30, 50]:
        for alpha in [0.01, 0.001, 0.0001]:
            # Grid resolution scales with K
            n_grid = max(21, int(np.sqrt(K) * 7))
            t0 = time.time()
            kx, ky, w, phi = initialize_cs(
                init_frames, n_gaussians=K, depth=depth,
                n_kx=n_grid, n_ky=n_grid,
                alpha=alpha, max_components=K, seed=42)
            elapsed = time.time() - t0

            n_active = int(torch.sum(w > 0.001).item())

            with torch.no_grad():
                pred = predict_field(kx, ky, w, phi, depth, grid_x, grid_y, t_eval)

            nm = nrmse(pred, gt_np, Hs)
            rho = pearson_correlation(pred, gt_np)

            label = f"CS_K{K}_a{alpha}"
            results[label] = {
                'nrmse': float(nm), 'rho': float(rho),
                'time_s': float(elapsed), 'K': K,
                'alpha': alpha, 'n_active': n_active,
            }
            if verbose:
                print(f"    {label:20s}: nRMSE={nm:.4f}, ρ={rho:.3f}, "
                      f"active={n_active}, {elapsed:.2f}s")

    return results


def main():
    import glob

    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            'data')

    # Select representative episodes
    targets = [
        'episode_BF3_5comp_seed0_slope.h5',
        'episode_BF4_5comp_seed0_slope.h5',
        'episode_BF4_15comp_seed0_slope.h5',
        'episode_BF4_30comp_seed0_slope.h5',
        'episode_BF5_15comp_seed0_slope.h5',
        'episode_BF5_30comp_seed0_slope.h5',
    ]

    all_results = []
    for target in targets:
        path = os.path.join(data_dir, target)
        if not os.path.exists(path):
            # Try uniform variant
            path = path.replace('_slope', '_uniform')
        if not os.path.exists(path):
            print(f"  Skipping {target} (not found)")
            continue

        print(f"\n[{target}]")
        result = evaluate_episode(path, verbose=True)
        if result is not None:
            all_results.append(result)

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY: Init-only nRMSE (lower = better)")
    print("=" * 80)

    # Find best CS config per episode
    for r in all_results:
        bf_best = min(v['nrmse'] for k, v in r.items()
                      if isinstance(v, dict) and k.startswith('BF'))
        cs_configs = [(k, v) for k, v in r.items()
                      if isinstance(v, dict) and k.startswith('CS')]
        if cs_configs:
            cs_best_label, cs_best = min(cs_configs, key=lambda x: x[1]['nrmse'])
            imp = (bf_best - cs_best['nrmse']) / bf_best * 100
            print(f"  {r['episode']:45s}: BF_best={bf_best:.4f}, "
                  f"CS_best={cs_best['nrmse']:.4f} ({cs_best_label}), "
                  f"improvement={imp:+.1f}%")

    # Save
    output_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                               'results', 'cs_vs_beamform.json')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved to {output_path}")


if __name__ == '__main__':
    main()
