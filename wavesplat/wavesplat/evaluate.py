"""
evaluate.py
Batch evaluation: run WaveSplat + baselines on episodes and generate comparison tables.

Usage:
    python -m wavesplat.evaluate --data_dir data/ --output results/
    python -m wavesplat.evaluate --episode data/episode_BF4_30comp_seed42_uniform.h5 --verbose
"""

import argparse
import glob
import json
import os
import time
import numpy as np

from wavesplat.data_loader import EpisodeDataLoader
from wavesplat.run import run_estimation, DEFAULT_CONFIG
from wavesplat.evaluation.metrics import compute_all_metrics, rmse, nrmse
from wavesplat.baselines.gpr_baseline import GPRBaseline
from wavesplat.baselines.etkf_grid import ETKFGrid
from wavesplat.baselines.oracle_baseline import OracleBaseline


def run_gpr_baseline(h5_path: str, window: int = 1,
                     verbose: bool = False) -> dict:
    """Run GPR baseline on an episode."""
    loader = EpisodeDataLoader(h5_path)
    gpr = GPRBaseline(window_frames=window, length_scale=20.0)

    grid_x = loader.gt_grid_x.cpu().numpy()
    grid_y = loader.gt_grid_y.cpu().numpy()

    frame_results = []
    for i, frame in enumerate(loader):
        x = frame.x_obs.cpu().numpy()
        y = frame.y_obs.cpu().numpy()
        z = frame.z_obs.cpu().numpy()

        elapsed = gpr.update(x, y, z)
        pred = gpr.predict_grid(grid_x, grid_y)
        gt = frame.gt_field.cpu().numpy()

        metrics = compute_all_metrics(
            pred, gt, grid_x, grid_y, loader.Hs_true,
            obs_x=x if len(x) > 0 else None,
            obs_y=y if len(y) > 0 else None,
        )
        metrics['elapsed_ms'] = elapsed
        metrics['t'] = frame.t
        frame_results.append(metrics)

        if verbose and i % 50 == 0:
            print(f"  [GPR-{window}] t={frame.t:.1f}s RMSE={metrics['RMSE_z']:.4f}m")

    avg_rmse = np.mean([r['RMSE_z'] for r in frame_results])
    avg_nrmse = np.mean([r['nRMSE'] for r in frame_results])
    avg_ms = np.mean([r['elapsed_ms'] for r in frame_results])

    return {
        'method': f'GPR(window={window})',
        'avg_RMSE_z': avg_rmse,
        'avg_nRMSE': avg_nrmse,
        'avg_inference_ms': avg_ms,
        'frame_results': frame_results,
    }


def run_etkf_baseline(h5_path: str, grid_resolution: str = 'medium',
                      verbose: bool = False) -> dict:
    """Run ETKF-grid baseline on an episode."""
    loader = EpisodeDataLoader(h5_path)
    gt_grid_x = loader.gt_grid_x.cpu().numpy()
    gt_grid_y = loader.gt_grid_y.cpu().numpy()

    # ETKF grid resolution
    res_map = {'coarse': (20, 15), 'medium': (40, 30), 'fine': (80, 60)}
    etkf_nx, etkf_ny = res_map.get(grid_resolution, (40, 30))
    etkf_gx = np.linspace(gt_grid_x[0], gt_grid_x[-1], etkf_nx)
    etkf_gy = np.linspace(gt_grid_y[0], gt_grid_y[-1], etkf_ny)

    etkf = ETKFGrid(etkf_gx, etkf_gy, depth=loader.depth, n_ensemble=50)

    frame_results = []
    for i, frame in enumerate(loader):
        x = frame.x_obs.cpu().numpy()
        y = frame.y_obs.cpu().numpy()
        z = frame.z_obs.cpu().numpy()

        elapsed = etkf.update(x, y, z)

        # Interpolate ETKF grid to GT grid for comparison
        etkf_pred = etkf.predict_grid()
        from scipy.interpolate import RegularGridInterpolator
        interp = RegularGridInterpolator(
            (etkf_gy, etkf_gx), etkf_pred,
            method='linear', bounds_error=False, fill_value=0.0
        )
        Xg, Yg = np.meshgrid(gt_grid_x, gt_grid_y)
        pred = interp(np.column_stack([Yg.ravel(), Xg.ravel()])).reshape(
            len(gt_grid_y), len(gt_grid_x)
        )

        gt = frame.gt_field.cpu().numpy()
        metrics = compute_all_metrics(
            pred, gt, gt_grid_x, gt_grid_y, loader.Hs_true,
            obs_x=x if len(x) > 0 else None,
            obs_y=y if len(y) > 0 else None,
        )
        metrics['elapsed_ms'] = elapsed
        metrics['t'] = frame.t
        frame_results.append(metrics)

        if verbose and i % 50 == 0:
            print(f"  [ETKF-{grid_resolution}] t={frame.t:.1f}s "
                  f"RMSE={metrics['RMSE_z']:.4f}m")

    avg_rmse = np.mean([r['RMSE_z'] for r in frame_results])
    avg_nrmse = np.mean([r['nRMSE'] for r in frame_results])
    avg_ms = np.mean([r['elapsed_ms'] for r in frame_results])

    return {
        'method': f'ETKF-grid({grid_resolution})',
        'avg_RMSE_z': avg_rmse,
        'avg_nRMSE': avg_nrmse,
        'avg_inference_ms': avg_ms,
        'frame_results': frame_results,
    }


def run_oracle_baseline(h5_path: str, n_gaussians: int = 15,
                        verbose: bool = False) -> dict:
    """Run WaveSplat-oracle baseline: true (kx, ky, A), optimize only φ."""
    import torch

    loader = EpisodeDataLoader(h5_path)
    grid_x = loader.gt_grid_x
    grid_y = loader.gt_grid_y

    if loader.wave_components is None:
        raise ValueError("Episode missing wave_components for oracle baseline")

    oracle = OracleBaseline(
        wave_components=loader.wave_components,
        depth=loader.depth,
        n_gaussians=n_gaussians,
    )

    # Phase 1: Initialize phases
    init_frames = loader.get_init_window(tp_factor=2.0)
    oracle.initialize(init_frames, max_iters=200, verbose=verbose)

    # Phase 2: Track
    tracking_frames = loader.get_tracking_frames(tp_factor=2.0)
    frame_results = []

    for i, frame in enumerate(tracking_frames):
        update_result = oracle.update(frame)

        pred = oracle.predict_grid(grid_x, grid_y, frame.t)
        pred_np = pred.cpu().numpy()
        gt_np = frame.gt_field.cpu().numpy()

        metrics = compute_all_metrics(
            pred_np, gt_np,
            grid_x.cpu().numpy(), grid_y.cpu().numpy(),
            loader.Hs_true,
            obs_x=frame.x_obs.cpu().numpy() if len(frame.x_obs) > 0 else None,
            obs_y=frame.y_obs.cpu().numpy() if len(frame.y_obs) > 0 else None,
        )
        metrics['elapsed_ms'] = update_result['elapsed_ms']
        metrics['t'] = frame.t
        frame_results.append(metrics)

        if verbose and i % 50 == 0:
            print(f"  [Oracle] t={frame.t:.1f}s RMSE={metrics['RMSE_z']:.4f}m")

    avg_rmse = np.mean([r['RMSE_z'] for r in frame_results])
    avg_nrmse = np.mean([r['nRMSE'] for r in frame_results])
    avg_ms = np.mean([r['elapsed_ms'] for r in frame_results])

    return {
        'method': 'WaveSplat-oracle',
        'avg_RMSE_z': avg_rmse,
        'avg_nRMSE': avg_nrmse,
        'avg_inference_ms': avg_ms,
        'frame_results': frame_results,
    }


def evaluate_episode(h5_path: str, verbose: bool = False) -> dict:
    """Run all methods on a single episode and compare."""
    results = {}

    if verbose:
        print(f"\n{'='*60}")
        print(f"Evaluating: {h5_path}")

    # WaveSplat
    if verbose:
        print("\n[WaveSplat]")
    ws_result = run_estimation(h5_path, verbose=verbose)
    results['WaveSplat'] = {
        'avg_RMSE_z': ws_result['avg_RMSE_z'],
        'avg_nRMSE': ws_result['avg_nRMSE'],
        'avg_Hs_error': ws_result['avg_Hs_error'],
        'avg_inference_ms': ws_result['avg_inference_ms'],
        'T_converge': ws_result['T_converge'],
    }

    # GPR baselines
    for window in [1, 5]:
        if verbose:
            print(f"\n[GPR window={window}]")
        gpr_result = run_gpr_baseline(h5_path, window=window, verbose=verbose)
        results[f'GPR(w={window})'] = {
            'avg_RMSE_z': gpr_result['avg_RMSE_z'],
            'avg_nRMSE': gpr_result['avg_nRMSE'],
            'avg_inference_ms': gpr_result['avg_inference_ms'],
        }

    # ETKF baseline
    if verbose:
        print("\n[ETKF-grid]")
    etkf_result = run_etkf_baseline(h5_path, grid_resolution='medium',
                                     verbose=verbose)
    results['ETKF-grid'] = {
        'avg_RMSE_z': etkf_result['avg_RMSE_z'],
        'avg_nRMSE': etkf_result['avg_nRMSE'],
        'avg_inference_ms': etkf_result['avg_inference_ms'],
    }

    # WaveSplat-oracle (upper bound)
    if verbose:
        print("\n[WaveSplat-oracle]")
    oracle_result = run_oracle_baseline(h5_path, verbose=verbose)
    results['WaveSplat-oracle'] = {
        'avg_RMSE_z': oracle_result['avg_RMSE_z'],
        'avg_nRMSE': oracle_result['avg_nRMSE'],
        'avg_inference_ms': oracle_result['avg_inference_ms'],
    }

    # Print comparison table
    if verbose:
        print(f"\n{'='*60}")
        print(f"{'Method':<20} {'RMSE_z [m]':>12} {'nRMSE':>10} {'ms/frame':>10}")
        print(f"{'-'*52}")
        for name, r in results.items():
            print(f"{name:<20} {r['avg_RMSE_z']:>12.4f} "
                  f"{r['avg_nRMSE']:>10.4f} "
                  f"{r.get('avg_inference_ms', 0):>10.1f}")

    return results


def main():
    parser = argparse.ArgumentParser(description='WaveSplat evaluation')
    parser.add_argument('--episode', type=str, default=None,
                        help='Single episode to evaluate')
    parser.add_argument('--data_dir', type=str, default=None,
                        help='Directory of episodes to batch evaluate')
    parser.add_argument('--output', type=str, default='results/',
                        help='Output directory')
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    if args.episode:
        results = evaluate_episode(args.episode, verbose=args.verbose)
        out_path = os.path.join(args.output, 'eval_single.json')
        with open(out_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\nResults saved to {out_path}")

    elif args.data_dir:
        episodes = sorted(glob.glob(os.path.join(args.data_dir, '*.h5')))
        print(f"Found {len(episodes)} episodes")

        all_results = {}
        for ep in episodes:
            name = os.path.basename(ep)
            try:
                results = evaluate_episode(ep, verbose=args.verbose)
                all_results[name] = results
            except Exception as e:
                print(f"  FAILED: {name}: {e}")
                all_results[name] = {'error': str(e)}

        out_path = os.path.join(args.output, 'eval_batch.json')
        with open(out_path, 'w') as f:
            json.dump(all_results, f, indent=2, default=str)
        print(f"\nBatch results saved to {out_path}")


if __name__ == '__main__':
    main()
