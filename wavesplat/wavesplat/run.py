"""
run.py
Main inference script for WaveSplat wave field estimation.

Usage:
    python -m wavesplat.run --episode data/episode_BF4_30comp_seed42_uniform.h5
    python -m wavesplat.run --episode data/episode_BF4_30comp_seed42_uniform.h5 --verbose
"""

import argparse
import json
import os
import time
import numpy as np
import torch

from wavesplat.data_loader import EpisodeDataLoader
from wavesplat.optimizer.sequential_update import SequentialUpdater
from wavesplat.evaluation.metrics import compute_all_metrics, rmse, nrmse


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


def set_seed(seed: int):
    """Set global random seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_yaml_config(yaml_path: str) -> dict:
    """Load estimation config from YAML and flatten into optimizer-compatible dict."""
    import yaml
    with open(yaml_path) as f:
        raw = yaml.safe_load(f)

    cfg = {}
    if 'model' in raw:
        cfg['n_gaussians'] = raw['model'].get('n_gaussians', [3, 8, 15])
        cfg['depth'] = raw['model'].get('depth', 50.0)
    if 'optimizer' in raw:
        cfg['lr'] = raw['optimizer'].get('lr', 0.01)
        cfg['lambda_l1'] = raw['optimizer'].get('lambda_l1', 0.001)
        cfg['lambda_smooth'] = raw['optimizer'].get('lambda_smooth', 0.0001)
        cfg['tracking_steps'] = raw['optimizer'].get('tracking_steps', 10)
        cfg['tracking_mode'] = raw['optimizer'].get('tracking_mode', 'w_phi')
        cfg['tracking_lr'] = raw['optimizer'].get('tracking_lr', 0.005)
        cfg['init_method'] = raw['optimizer'].get('init_method', 'v2')
    if 'physics' in raw:
        cfg['use_dispersion'] = raw['physics'].get('use_dispersion', True)
        cfg['use_confidence'] = raw['physics'].get('use_confidence', True)
    if 'adaptive' in raw:
        cfg['adaptive_n'] = raw['adaptive'].get('adaptive_n', True)
        cfg['prune_threshold'] = raw['adaptive'].get('prune_threshold', 0.10)
        cfg['grow_residual_threshold'] = raw['adaptive'].get('grow_residual_threshold', 0.05)
        cfg['max_n'] = raw['adaptive'].get('max_n', 30)
    if 'tracking' in raw:
        cfg['tracking_window'] = raw['tracking'].get('tracking_window', 3)
    if 'schedule' in raw:
        cfg['loss_thresholds'] = raw['schedule'].get('loss_thresholds')
        cfg['max_iters_per_stage'] = raw['schedule'].get('max_iters_per_stage')
    if 'device' in raw:
        cfg['device'] = raw['device']
    if 'seed' in raw:
        cfg['seed'] = raw['seed']
    return cfg


def run_estimation(h5_path: str, config: dict = None,
                   verbose: bool = False) -> dict:
    """
    Run full WaveSplat estimation on an episode.

    Args:
        h5_path: path to HDF5 episode file
        config: algorithm configuration dict
        verbose: print progress

    Returns:
        dict with per-frame metrics, timing, and parameters
    """
    cfg = DEFAULT_CONFIG.copy()
    if config:
        cfg.update(config)
    cfg['depth'] = cfg.get('depth', 50.0)

    # Set random seed for reproducibility
    seed = cfg.get('seed', 42)
    if seed is not None:
        set_seed(seed)

    # Load episode
    loader = EpisodeDataLoader(h5_path, device=cfg['device'])
    cfg['depth'] = loader.depth

    if verbose:
        print(f"Episode: {h5_path}")
        print(f"  Sea state: {loader.sea_state}, Hs={loader.Hs_true:.2f}m, "
              f"Tp={loader.Tp_true:.1f}s")
        print(f"  Frames: {loader.n_frames}, "
              f"GT grid: {len(loader.gt_grid_y)}×{len(loader.gt_grid_x)}")

    # Phase 1: Initialization
    init_frames = loader.get_init_window(tp_factor=2.0)
    if verbose:
        print(f"\n[Phase 1] Initialization: {len(init_frames)} frames "
              f"({init_frames[-1].t:.1f}s)")

    updater = SequentialUpdater(cfg)
    init_result = updater.initialize(
        init_frames,
        x_range=(float(loader.gt_grid_x[0]), float(loader.gt_grid_x[-1])),
        y_range=(float(loader.gt_grid_y[0]), float(loader.gt_grid_y[-1])),
        verbose=verbose,
    )

    # Evaluate initialization on last init frame
    grid_x = loader.gt_grid_x
    grid_y = loader.gt_grid_y
    last_init = init_frames[-1]

    # Detect real data (no GT available): gt_field is None or all zeros
    has_gt = (last_init.gt_field is not None and
              float(torch.abs(last_init.gt_field).max()) > 1e-10)

    pred_init = updater.predict_field(grid_x, grid_y, last_init.t)
    if has_gt:
        init_rmse = rmse(pred_init.cpu().numpy(), last_init.gt_field.cpu().numpy())
        init_nrmse = nrmse(pred_init.cpu().numpy(), last_init.gt_field.cpu().numpy(),
                           loader.Hs_true)
    else:
        init_rmse = float('nan')
        init_nrmse = float('nan')

    if verbose:
        print(f"  Init RMSE={init_rmse:.4f}m, nRMSE={init_nrmse:.4f}")

    # Phase 2: Tracking
    tracking_frames = loader.get_tracking_frames(tp_factor=2.0)
    if verbose:
        print(f"\n[Phase 2] Tracking: {len(tracking_frames)} frames")

    frame_results = []
    t_converge = None
    converge_threshold = 0.05  # nRMSE threshold

    for i, frame in enumerate(tracking_frames):
        update_result = updater.update(frame)

        # Evaluate on GT grid
        pred = updater.predict_field(grid_x, grid_y, frame.t)
        pred_np = pred.cpu().numpy()

        if has_gt:
            gt_np = frame.gt_field.cpu().numpy()
            frame_metrics = compute_all_metrics(
                pred_np, gt_np,
                grid_x.cpu().numpy(), grid_y.cpu().numpy(),
                loader.Hs_true,
                obs_x=frame.x_obs.cpu().numpy() if len(frame.x_obs) > 0 else None,
                obs_y=frame.y_obs.cpu().numpy() if len(frame.y_obs) > 0 else None,
            )
        else:
            # Real data mode: no GT metrics, report prediction statistics
            Hs_pred = float(4.0 * np.std(pred_np))
            frame_metrics = {
                'Hs_pred': Hs_pred,
                'pred_std': float(np.std(pred_np)),
                'pred_max': float(np.max(pred_np)),
                'pred_min': float(np.min(pred_np)),
            }

        frame_metrics['elapsed_ms'] = update_result['elapsed_ms']
        frame_metrics['n_pts'] = update_result['n_pts']
        frame_metrics['t'] = frame.t
        frame_results.append(frame_metrics)

        # Check convergence (only with GT)
        if has_gt and t_converge is None and frame_metrics.get('nRMSE', 1.0) < converge_threshold:
            t_converge = frame.t

        if verbose and i % 20 == 0:
            if has_gt:
                print(f"  [t={frame.t:.1f}s] RMSE={frame_metrics['RMSE_z']:.4f}m, "
                      f"nRMSE={frame_metrics['nRMSE']:.4f}, "
                      f"{update_result['elapsed_ms']:.1f}ms")
            else:
                print(f"  [t={frame.t:.1f}s] Hs_pred={frame_metrics['Hs_pred']:.4f}m, "
                      f"n_pts={frame_metrics['n_pts']}, "
                      f"{update_result['elapsed_ms']:.1f}ms")

    # Aggregate results
    if frame_results:
        avg_ms = np.mean([r['elapsed_ms'] for r in frame_results])
        if has_gt:
            avg_rmse = np.mean([r['RMSE_z'] for r in frame_results])
            avg_nrmse = np.mean([r['nRMSE'] for r in frame_results])
            avg_hs_err = np.mean([r['Hs_error'] for r in frame_results])
        else:
            avg_rmse = avg_nrmse = avg_hs_err = float('nan')
    else:
        avg_rmse = avg_nrmse = avg_ms = avg_hs_err = float('nan')

    summary = {
        'episode': h5_path,
        'sea_state': loader.sea_state,
        'Hs_true': loader.Hs_true,
        'Tp_true': loader.Tp_true,
        'n_gaussians': updater.model.n_gaussians,
        'init_iters': init_result['n_iters'],
        'init_time_s': init_result['elapsed_s'],
        'init_rmse': init_rmse,
        'init_nrmse': init_nrmse,
        'tracking_frames': len(frame_results),
        'avg_RMSE_z': avg_rmse,
        'avg_nRMSE': avg_nrmse,
        'avg_Hs_error': avg_hs_err,
        'avg_inference_ms': avg_ms,
        'T_converge': t_converge,
        'frame_results': frame_results,
        'spectrum_params': updater.model.get_spectrum_params(),
    }

    if verbose:
        print(f"\n[Summary]")
        print(f"  Avg RMSE_z = {avg_rmse:.4f}m")
        print(f"  Avg nRMSE  = {avg_nrmse:.4f}")
        print(f"  Avg Hs_err = {avg_hs_err:.4f}")
        print(f"  Avg speed  = {avg_ms:.1f}ms/frame")
        if t_converge:
            print(f"  T_converge = {t_converge:.1f}s")

    return summary


def main():
    parser = argparse.ArgumentParser(description='WaveSplat inference')
    parser.add_argument('--episode', type=str, required=True,
                        help='Path to HDF5 episode file')
    parser.add_argument('--config', type=str, default=None,
                        help='Path to estimation YAML config')
    parser.add_argument('--n_gaussians', type=int, nargs='+', default=None)
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--lambda_l1', type=float, default=None)
    parser.add_argument('--lambda_smooth', type=float, default=None)
    parser.add_argument('--tracking_steps', type=int, default=None)
    parser.add_argument('--tracking_mode', type=str, default=None,
                        choices=['phi_only', 'w_phi', 'all'],
                        help='Which params to update during tracking')
    parser.add_argument('--tracking_lr', type=float, default=None,
                        help='Learning rate for tracking phase')
    parser.add_argument('--init_method', type=str, default=None,
                        choices=['v1', 'v2', 'v2_konly'],
                        help='Init method: v1 (spatial FFT), v2 (beamform+proj), v2_konly (beamform kx/ky only)')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed for reproducibility (default: 42)')
    parser.add_argument('--adaptive_n', type=str, default=None,
                        choices=['true', 'false'],
                        help='Enable adaptive N (prune weak, grow if needed)')
    parser.add_argument('--max_n', type=int, default=None,
                        help='Maximum number of Gaussians for adaptive N')
    parser.add_argument('--tracking_window', type=int, default=None,
                        help='Number of recent frames for multi-frame tracking')
    parser.add_argument('--output', type=str, default=None,
                        help='Output JSON path for results')
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()

    # Build config: YAML base → CLI overrides
    config = {}
    if args.config:
        config = load_yaml_config(args.config)
    # CLI flags override YAML
    if args.n_gaussians is not None:
        config['n_gaussians'] = args.n_gaussians
    if args.lr is not None:
        config['lr'] = args.lr
    if args.lambda_l1 is not None:
        config['lambda_l1'] = args.lambda_l1
    if args.lambda_smooth is not None:
        config['lambda_smooth'] = args.lambda_smooth
    if args.tracking_steps is not None:
        config['tracking_steps'] = args.tracking_steps
    if args.tracking_mode is not None:
        config['tracking_mode'] = args.tracking_mode
    if args.tracking_lr is not None:
        config['tracking_lr'] = args.tracking_lr
    if args.init_method is not None:
        config['init_method'] = args.init_method
    if args.seed is not None:
        config['seed'] = args.seed
    if args.adaptive_n is not None:
        config['adaptive_n'] = args.adaptive_n == 'true'
    if args.max_n is not None:
        config['max_n'] = args.max_n
    if args.tracking_window is not None:
        config['tracking_window'] = args.tracking_window

    result = run_estimation(args.episode, config, verbose=args.verbose)

    if args.output:
        # Convert numpy arrays for JSON serialization
        save_result = {k: v for k, v in result.items()
                       if k not in ('frame_results', 'spectrum_params')}
        os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
        with open(args.output, 'w') as f:
            json.dump(save_result, f, indent=2, default=str)
        print(f"Results saved to {args.output}")


if __name__ == '__main__':
    main()
