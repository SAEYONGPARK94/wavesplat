#!/usr/bin/env python3
"""
Evaluate WaveSplat prediction accuracy at different time horizons.

Protocol:
  1. Run WaveSplat init phase → get estimated spectrum (kx, ky, w, phi)
  2. Freeze spectrum at t_freeze (end of init)
  3. Predict forward at horizons [5, 10, 20, 30, 60]s
  4. Compare: linear prediction vs Stokes2 prediction vs GT
  5. Also compare: tracking (with observations) as upper bound

This measures how well the estimated spectrum predicts future wave states
without new observations — relevant for LiDAR occlusion or forecast.

Usage:
    python3 -m scripts.evaluate_prediction
    python3 -m scripts.evaluate_prediction --episodes 12 --horizons 5,10,20,30
"""

import argparse
import glob
import json
import os
import sys
import time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from wavesplat.data_loader import EpisodeDataLoader
from wavesplat.optimizer.sequential_update import SequentialUpdater
from wavesplat.model.spectral_gaussian import SpectralGaussianModel
from wavesplat.model.stokes2 import Stokes2Transfer
from wavesplat.model.dispersion import dispersion_omega
from wavesplat.evaluation.metrics import nrmse, pearson_correlation, variance_ratio


DEFAULT_CONFIG = {
    'n_gaussians': [3, 8, 15],
    'depth': 50.0,
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


def predict_field_linear(kx, ky, w, phi, depth, grid_x, grid_y, t):
    """Linear prediction: z = Σ w_i cos(kx_i x + ky_i y - ω_i t + φ_i)."""
    k_mag = torch.sqrt(kx ** 2 + ky ** 2 + 1e-12)
    omega = dispersion_omega(k_mag, depth)

    Xg, Yg = torch.meshgrid(grid_x, grid_y, indexing='xy')
    x_flat = Xg.reshape(-1)
    y_flat = Yg.reshape(-1)

    phase = (kx[None, :] * x_flat[:, None]
             + ky[None, :] * y_flat[:, None]
             - omega[None, :] * t
             + phi[None, :])
    z = torch.sum(w[None, :] * torch.cos(phase), dim=1)
    return z.reshape(len(grid_y), len(grid_x)).numpy()


def predict_field_stokes2(kx, ky, w, phi, depth, grid_x, grid_y, t):
    """Linear + Stokes 2nd order correction."""
    k_mag = torch.sqrt(kx ** 2 + ky ** 2 + 1e-12)
    omega = dispersion_omega(k_mag, depth)

    Xg, Yg = torch.meshgrid(grid_x, grid_y, indexing='xy')
    x_flat = Xg.reshape(-1)
    y_flat = Yg.reshape(-1)

    # Linear part
    phase = (kx[None, :] * x_flat[:, None]
             + ky[None, :] * y_flat[:, None]
             - omega[None, :] * t
             + phi[None, :])
    z_linear = torch.sum(w[None, :] * torch.cos(phase), dim=1)

    # Stokes2 correction
    try:
        st = Stokes2Transfer(kx.numpy(), ky.numpy(), depth)
        z2 = st.correction(x_flat, y_flat, t, w, kx, ky, phi, omega)
        z_total = z_linear + z2
    except Exception:
        z_total = z_linear

    return z_total.reshape(len(grid_y), len(grid_x)).detach().numpy()


def evaluate_episode(h5_path, config, horizons, verbose=False):
    """Evaluate prediction on one episode.

    Returns dict with per-horizon metrics for linear and stokes2 predictions.
    """
    loader = EpisodeDataLoader(h5_path, device='cpu')
    depth = loader.depth if loader.depth else config['depth']
    Hs = loader.Hs_true

    # Get init and tracking frames
    init_frames = loader.get_init_window(tp_factor=2.0)
    tracking_frames = loader.get_tracking_frames(tp_factor=2.0)

    if len(init_frames) < 5 or len(tracking_frames) < 5:
        return None

    # Run WaveSplat initialization
    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])

    updater = SequentialUpdater(config)
    x_range = (float(loader.gt_grid_x[0]), float(loader.gt_grid_x[-1]))
    y_range = (float(loader.gt_grid_y[0]), float(loader.gt_grid_y[-1]))
    updater.initialize(init_frames, x_range=x_range, y_range=y_range)

    # Extract frozen spectrum at end of init
    with torch.no_grad():
        frozen_kx = updater.model.kx.clone()
        frozen_ky = updater.model.ky.clone()
        frozen_w = updater.model.w.clone()
        frozen_phi = updater.model.phi.clone()

    t_freeze = init_frames[-1].t
    n_comp = len(frozen_kx)

    grid_x = loader.gt_grid_x
    grid_y = loader.gt_grid_y

    if verbose:
        print(f"  Frozen at t={t_freeze:.1f}s, N={n_comp}, Hs={Hs:.2f}m")

    # Evaluate at each horizon
    results = {}
    for horizon in horizons:
        t_target = t_freeze + horizon

        # Find closest tracking frame to t_target
        best_frame = None
        best_dt = float('inf')
        for frame in tracking_frames:
            dt = abs(frame.t - t_target)
            if dt < best_dt:
                best_dt = dt
                best_frame = frame
        if best_frame is None or best_dt > 1.0:
            continue

        # GT field
        gt_field = best_frame.gt_field
        if gt_field is None:
            continue
        gt_np = gt_field.numpy()

        # Linear prediction (frozen spectrum propagated to t_target)
        with torch.no_grad():
            z_linear = predict_field_linear(
                frozen_kx, frozen_ky, frozen_w, frozen_phi,
                depth, grid_x, grid_y, best_frame.t)

            z_stokes2 = predict_field_stokes2(
                frozen_kx, frozen_ky, frozen_w, frozen_phi,
                depth, grid_x, grid_y, best_frame.t)

        # Tracking prediction (WaveSplat with observations at each frame)
        # Run tracking up to this frame to get the "with observations" baseline
        # (We skip this for now - just compare frozen predictions)

        nrmse_linear = nrmse(z_linear, gt_np, Hs)
        nrmse_stokes2 = nrmse(z_stokes2, gt_np, Hs)
        rho_linear = pearson_correlation(z_linear, gt_np)
        rho_stokes2 = pearson_correlation(z_stokes2, gt_np)

        results[horizon] = {
            'nrmse_linear': float(nrmse_linear),
            'nrmse_stokes2': float(nrmse_stokes2),
            'rho_linear': float(rho_linear),
            'rho_stokes2': float(rho_stokes2),
            'stokes2_improvement': float(nrmse_linear - nrmse_stokes2),
            'stokes2_improv_pct': float(
                (nrmse_linear - nrmse_stokes2) / nrmse_linear * 100
            ) if nrmse_linear > 0 else 0.0,
            't_actual': float(best_frame.t),
            'dt_from_target': float(best_dt),
        }

        if verbose:
            imp = results[horizon]['stokes2_improv_pct']
            print(f"    Horizon {horizon:3d}s: linear={nrmse_linear:.4f}, "
                  f"stokes2={nrmse_stokes2:.4f}, "
                  f"improvement={imp:+.1f}%, "
                  f"ρ_lin={rho_linear:.3f}, ρ_s2={rho_stokes2:.3f}")

    return {
        'episode': os.path.basename(h5_path),
        'Hs': float(Hs),
        'n_comp_estimated': n_comp,
        't_freeze': float(t_freeze),
        'horizons': results,
    }


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate WaveSplat prediction accuracy')
    parser.add_argument('--data_dir', type=str, default='data')
    parser.add_argument('--episodes', type=int, default=12,
                        help='Max episodes to evaluate')
    parser.add_argument('--horizons', type=str, default='5,10,20,30',
                        help='Comma-separated prediction horizons in seconds')
    parser.add_argument('--output', type=str,
                        default='results/prediction_eval.json')
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()

    horizons = [int(h) for h in args.horizons.split(',')]

    # Select representative episodes (mix of sea states and components)
    pattern = os.path.join(args.data_dir, 'episode_BF*.h5')
    all_episodes = sorted(glob.glob(pattern))

    # Stratified selection: pick from different BF/comp combinations
    selected = []
    seen_types = set()
    for ep in all_episodes:
        name = os.path.basename(ep)
        # Extract BF and comp type
        parts = name.replace('episode_', '').replace('.h5', '').split('_')
        ep_type = f"{parts[0]}_{parts[1]}"  # e.g., "BF4_5comp"
        if ep_type not in seen_types:
            seen_types.add(ep_type)
            selected.append(ep)
        if len(selected) >= args.episodes:
            break

    print(f"Evaluating {len(selected)} episodes, horizons={horizons}s")

    all_results = []
    t0 = time.time()

    for i, ep_path in enumerate(selected):
        ep_name = os.path.basename(ep_path)
        print(f"\n[{i+1}/{len(selected)}] {ep_name}")

        result = evaluate_episode(
            ep_path, DEFAULT_CONFIG, horizons, verbose=args.verbose)
        if result is not None:
            all_results.append(result)

    elapsed = time.time() - t0
    print(f"\n{'='*70}")
    print(f"Completed {len(all_results)} episodes in {elapsed:.1f}s")

    # Aggregate statistics
    print(f"\n{'Horizon':>8} {'nRMSE_lin':>10} {'nRMSE_s2':>10} "
          f"{'Improv':>8} {'ρ_lin':>7} {'ρ_s2':>7}")
    print("-" * 55)

    summary = {}
    for h in horizons:
        lin_vals = [r['horizons'][h]['nrmse_linear']
                    for r in all_results if h in r['horizons']]
        s2_vals = [r['horizons'][h]['nrmse_stokes2']
                   for r in all_results if h in r['horizons']]
        rho_lin = [r['horizons'][h]['rho_linear']
                   for r in all_results if h in r['horizons']]
        rho_s2 = [r['horizons'][h]['rho_stokes2']
                  for r in all_results if h in r['horizons']]

        if not lin_vals:
            continue

        avg_lin = np.mean(lin_vals)
        avg_s2 = np.mean(s2_vals)
        avg_rho_lin = np.mean(rho_lin)
        avg_rho_s2 = np.mean(rho_s2)
        imp = (avg_lin - avg_s2) / avg_lin * 100

        summary[str(h)] = {
            'n_episodes': len(lin_vals),
            'nrmse_linear_mean': float(avg_lin),
            'nrmse_linear_std': float(np.std(lin_vals)),
            'nrmse_stokes2_mean': float(avg_s2),
            'nrmse_stokes2_std': float(np.std(s2_vals)),
            'improvement_pct': float(imp),
            'rho_linear_mean': float(avg_rho_lin),
            'rho_stokes2_mean': float(avg_rho_s2),
        }

        print(f"{h:>6}s {avg_lin:10.4f} {avg_s2:10.4f} "
              f"{imp:+7.1f}% {avg_rho_lin:7.3f} {avg_rho_s2:7.3f}")

    # Save results
    output = {
        'summary': summary,
        'per_episode': all_results,
        'horizons': horizons,
        'n_episodes': len(all_results),
        'elapsed_s': elapsed,
    }

    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {args.output}")


if __name__ == '__main__':
    main()
