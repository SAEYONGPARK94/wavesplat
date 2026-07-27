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
from wavesplat.optimizer.sequential_update import SequentialUpdater, make_updater
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
    # 'auto' since 2026-07-26 (user decision): GT-free regime router (learned proposal
    # n_selected, threshold 8) -> 'learned' or 'v2_konly'.
    # Falls back to 'v2_konly' if the diffusion checkpoint is unavailable, and is FORCED
    # to single-run 'v2_konly' on real (GT-free) input by the real-domain gate.
    # Deployed configuration `auto_v1_m3`, authorised on the fresh reserve-seed 40
    # (tmp/confirm2_2026-07-27/RESULT_CONFIRM2.md §4): mixed_v1 checkpoint + router th=8
    # + guard RETIRED + median-of-3 analytic fallback. Slice means on that set:
    # all 0.0877 / sparse 0.0175 / 5comp 0.0315 / crowded 0.1860 / 2comp 0.0160.
    'init_method': 'auto',
    # Median-of-k field ensemble on the ANALYTIC fallback of the router only (see
    # sequential_update.MedianEnsembleUpdater). 3 = the pre-registered, confirmed value;
    # 1 disables it. Explicit init_method='v2_konly' is unaffected either way.
    'analytic_ensemble_k': 3,
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
        if 'analytic_ensemble_k' in raw['optimizer']:
            cfg['analytic_ensemble_k'] = int(raw['optimizer']['analytic_ensemble_k'])
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

    # `auto` (shipped default) -> median-of-k analytic-fallback ensemble wrapper; every
    # explicit init_method -> the plain SequentialUpdater, bit-identical to before.
    updater = make_updater(cfg)
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
        # observation-fit loss: the only quality signal available WITHOUT ground truth,
        # so it is what a deployed multi-restart policy would have to select on.
        frame_metrics['obs_loss'] = update_result['loss']
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

    # Phase 3 (optional): Amplitude calibration (LS refit)
    refit_result = {}
    if cfg.get('amplitude_refit', False):
        refit_frames = tracking_frames[-5:] if len(tracking_frames) >= 5 \
            else tracking_frames
        if refit_frames:
            refit_result = updater.amplitude_refit(
                refit_frames, verbose=verbose)

            # Re-evaluate last frames with refitted amplitudes
            for fr_result in frame_results[-len(refit_frames):]:
                t_val = fr_result['t']
                matching = [f for f in refit_frames
                            if abs(f.t - t_val) < 1e-6]
                if matching and has_gt:
                    pred = updater.predict_field(grid_x, grid_y, t_val)
                    pred_np = pred.cpu().numpy()
                    gt_np = matching[0].gt_field.cpu().numpy()
                    new_metrics = compute_all_metrics(
                        pred_np, gt_np,
                        grid_x.cpu().numpy(), grid_y.cpu().numpy(),
                        loader.Hs_true,
                        obs_x=matching[0].x_obs.cpu().numpy() if len(matching[0].x_obs) > 0 else None,
                        obs_y=matching[0].y_obs.cpu().numpy() if len(matching[0].y_obs) > 0 else None,
                    )
                    fr_result.update(new_metrics)

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
        # WavePrior UNIFIED: which arm the init_method='auto' router chose, and the
        # GT-free signal it used. None for every other init_method (no behaviour change).
        'auto_selected': getattr(updater, '_auto_selected', None),
        'auto_n_selected': getattr(updater, '_auto_n_selected', None),
        # True when the real-domain gate bypassed the router (GT-free input); None on
        # every non-'auto' path. See sequential_update.SequentialUpdater.initialize.
        'auto_real_gate': getattr(updater, '_auto_real_gate', None),
        # Members actually run by the median-of-k analytic ensemble (1 = no ensemble).
        # NOTE: 'n_gaussians' and 'spectrum_params' above describe MEMBER 0 only -- the
        # ensemble has no single component set, its output is defined pointwise in the
        # field. Per-member counts are reported here instead of a meaningless aggregate.
        'ensemble_k': getattr(updater, 'ensemble_k', 1),
        'ensemble_member_n': init_result.get('member_n'),
        'init_iters': init_result['n_iters'],
        'init_time_s': init_result['elapsed_s'],
        'init_rmse': init_rmse,
        'init_nrmse': init_nrmse,
        'tracking_frames': len(frame_results),
        'avg_RMSE_z': avg_rmse,
        'avg_nRMSE': avg_nrmse,
        'avg_Hs_error': avg_hs_err,
        'avg_inference_ms': avg_ms,
        # GT-free run-quality signal, for multi-restart selection (see
        # tmp/ESTIMATOR_VARIANCE_RESULT_2026-07-23.md): mean observation-fit loss.
        'avg_obs_loss': float(np.mean([r['obs_loss'] for r in frame_results
                                       if np.isfinite(r.get('obs_loss', np.nan))]))
        if frame_results else float('nan'),
        'T_converge': t_converge,
        'frame_results': frame_results,
        'spectrum_params': updater.model.get_spectrum_params(),
        'refit': refit_result,
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


def run_estimation_multi(episode_path, config=None, seeds=(42, 1, 2, 3, 4), verbose=False):
    """Run `run_estimation` once per seed and report median + spread.

    WHY THIS EXISTS. A fixed episode with a fixed config still gives a **bimodal** spread of
    outcomes, with coefficient of variation up to 53% (measured; see
    tmp/ESTIMATOR_VARIANCE_RESULT_2026-07-23.md), because component phases are initialised at
    random in two places. The spread is *larger* on the Nyquist-adequate regenerated dataset,
    not smaller (tmp/MIN_DK_RESULT_2026-07-23.md). A single-seed number is therefore one draw
    from a wide, two-cluster distribution -- not an estimate of the method's performance.

    Selecting the *best* run is not an option: a pre-registered test of GT-free selection on
    the observation-fit loss FAILED (tmp/RESTART_SELECTION_RESULT_2026-07-23.md), and picking
    on ground truth is not deployable. So the honest summary is the **median with spread**.

    Returns the per-seed results plus median/mean/sd/min/max of avg_nRMSE. Report the median
    and the spread together; a median quoted without its spread is not citable.
    """
    import numpy as _np
    runs = []
    for s in seeds:
        cfg = dict(config or {})
        cfg['seed'] = s
        r = run_estimation(episode_path, cfg, verbose=verbose)
        runs.append({'seed': s, 'avg_nRMSE': float(r['avg_nRMSE']),
                     'avg_obs_loss': float(r.get('avg_obs_loss', float('nan')))})
    a = _np.array([r['avg_nRMSE'] for r in runs], float)
    return {
        'runs': runs,
        'n_seeds': len(runs),
        'nRMSE_median': float(_np.median(a)),
        'nRMSE_mean': float(a.mean()),
        'nRMSE_sd': float(a.std(ddof=1)) if len(a) > 1 else 0.0,
        'nRMSE_min': float(a.min()),
        'nRMSE_max': float(a.max()),
        'nRMSE_cv_pct': float(100 * a.std(ddof=1) / a.mean()) if len(a) > 1 and a.mean() else 0.0,
    }


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
                        choices=['v1', 'v2', 'v2_konly', 'relax', 'relax_konly',
                                 'learned', 'auto'],
                        help='Init method: v1/v2/v2_konly/relax/relax_konly/learned/auto '
                             '(learned = cond-diffusion set-proposal, WavePrior Stage-1; '
                             'auto = GT-free regime router that picks learned vs v2_konly '
                             'from the proposal n_selected, WavePrior UNIFIED)')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed for reproducibility (default: 42)')
    parser.add_argument('--n_seeds', type=int, default=None,
                        help='Run this many seeds and report median+spread instead of a '
                             'single run. Required for citable per-episode numbers: '
                             'run-to-run CV reaches 53%% (see '
                             'tmp/ESTIMATOR_VARIANCE_RESULT_2026-07-23.md).')
    parser.add_argument('--adaptive_n', type=str, default=None,
                        choices=['true', 'false'],
                        help='Enable adaptive N (prune weak, grow if needed)')
    parser.add_argument('--max_n', type=int, default=None,
                        help='Maximum number of Gaussians for adaptive N')
    parser.add_argument('--tracking_window', type=int, default=None,
                        help='Number of recent frames for multi-frame tracking')
    parser.add_argument('--neural_corrector', type=str, default=None,
                        help='Path to neural corrector checkpoint (.pt)')
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
    if args.neural_corrector is not None:
        config['neural_corrector_path'] = args.neural_corrector

    if args.n_seeds:
        seeds = [42, 1, 2, 3, 4, 5, 6, 7][:args.n_seeds] if args.n_seeds <= 8 \
            else [42] + list(range(1, args.n_seeds))
        result = run_estimation_multi(args.episode, config, seeds=seeds,
                                      verbose=args.verbose)
        print(f"\n=== median-of-{result['n_seeds']} ===")
        print(f"  nRMSE median {result['nRMSE_median']:.4f}  "
              f"(sd {result['nRMSE_sd']:.4f}, CV {result['nRMSE_cv_pct']:.1f}%, "
              f"range {result['nRMSE_min']:.4f}-{result['nRMSE_max']:.4f})")
        print("  report the median WITH the spread; a median alone is not citable.")
    else:
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
