"""
run_three_contributions.py
Three additional contribution experiments for WaveSplat:

Experiment 1: Spectral Parameter Estimation Accuracy
  - Compare estimated (kx, ky, ω, A) with ground truth components
  - Hungarian matching for component correspondence
  - LETKF cannot do this → WaveSplat-unique contribution

Experiment 2: Sparse Observation Robustness (n_pts sweep)
  - n_pts = [10, 20, 30, 50, 100] on selected episodes
  - WaveSplat vs LETKF under extreme sparsity

Experiment 3: Deterministic Forecast
  - Use estimated spectral params to predict wave field at t+Δt
  - Compare forward-model extrapolation vs GT at future frames
  - LETKF provides only current snapshot → WaveSplat-unique
"""

import sys
import os
import json
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from wavesplat.run import run_estimation, DEFAULT_CONFIG
from wavesplat.data_loader import EpisodeDataLoader
from wavesplat.baselines.letkf_grid import LETKFGrid
from wavesplat.evaluation.metrics import compute_all_metrics
from scipy.interpolate import RegularGridInterpolator

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')

# Episodes for spectral estimation + forecast (varying complexity)
SPECTRAL_EPISODES = [
    'episode_BF3_1comp_seed0_uniform.h5',
    'episode_BF3_2comp_seed0_uniform.h5',
    'episode_BF4_1comp_seed0_uniform.h5',
    'episode_BF4_5comp_seed0_uniform.h5',
    'episode_BF4_15comp_seed0_uniform.h5',
    'episode_BF4_30comp_seed0_uniform.h5',
    'episode_BF5_15comp_seed0_uniform.h5',
    'episode_BF5_30comp_seed0_uniform.h5',
]

# Episodes for sparse robustness (subset)
SPARSE_EPISODES = [
    'episode_BF3_2comp_seed0_uniform.h5',
    'episode_BF4_1comp_seed0_uniform.h5',
    'episode_BF4_5comp_seed0_uniform.h5',
    'episode_BF4_15comp_seed0_uniform.h5',
]

METRIC_KEYS = ['nRMSE', 'Variance_Ratio', 'Pearson_rho', 'Skill_Score',
               'Directional_R2', 'Hs_error']

WS_CONFIG = {
    'init_method': 'v2_konly',
    'use_dispersion': True,
    'use_confidence': True,
    'adaptive_n': True,
    'tracking_window': 3,
    'seed': 42,
}


# ═══════════════════════════════════════════════════════════════════════════
# Experiment 1: Spectral Parameter Estimation Accuracy
# ═══════════════════════════════════════════════════════════════════════════

def match_components(est_kx, est_ky, est_w, gt_kx, gt_ky, gt_A):
    """
    Hungarian matching between estimated and GT components.
    Cost = normalized wavenumber distance.
    Returns: matched pairs, unmatched GT, unmatched Est.
    """
    N_est = len(est_kx)
    N_gt = len(gt_kx)

    if N_est == 0 or N_gt == 0:
        return [], list(range(N_gt)), list(range(N_est))

    # Cost matrix: wavenumber distance (normalized by max |k|)
    k_scale = max(np.max(np.sqrt(gt_kx**2 + gt_ky**2)), 0.01)
    cost = np.zeros((N_est, N_gt))
    for i in range(N_est):
        for j in range(N_gt):
            dk = np.sqrt((est_kx[i] - gt_kx[j])**2 + (est_ky[i] - gt_ky[j])**2)
            cost[i, j] = dk / k_scale

    row_ind, col_ind = linear_sum_assignment(cost)

    # Filter out poor matches (cost > 0.5 = 50% of max k)
    matched = []
    for r, c in zip(row_ind, col_ind):
        if cost[r, c] < 0.5:
            matched.append((r, c, cost[r, c]))

    matched_est = {m[0] for m in matched}
    matched_gt = {m[1] for m in matched}
    unmatched_gt = [j for j in range(N_gt) if j not in matched_gt]
    unmatched_est = [i for i in range(N_est) if i not in matched_est]

    return matched, unmatched_gt, unmatched_est


def spectral_accuracy(est_params, gt_components):
    """
    Compute spectral parameter estimation accuracy.

    gt_components: (N_gt, 6) array [A, omega, kx, ky, phi, Q]
    est_params: dict {w, kx, ky, phi, omega, k_mag}
    """
    gt_A = gt_components[:, 0]
    gt_omega = gt_components[:, 1]
    gt_kx = gt_components[:, 2]
    gt_ky = gt_components[:, 3]

    est_kx = est_params['kx']
    est_ky = est_params['ky']
    est_w = np.abs(est_params['w'])
    est_omega = est_params['omega']

    matched, unmatched_gt, unmatched_est = match_components(
        est_kx, est_ky, est_w, gt_kx, gt_ky, gt_A)

    result = {
        'n_gt': len(gt_A),
        'n_est': len(est_kx),
        'n_matched': len(matched),
        'n_unmatched_gt': len(unmatched_gt),
        'n_unmatched_est': len(unmatched_est),
        'detection_rate': len(matched) / max(len(gt_A), 1),
    }

    if matched:
        # Per-matched-pair errors
        kx_errors = []
        ky_errors = []
        k_mag_errors = []
        omega_errors = []
        amp_errors = []
        direction_errors = []

        for est_idx, gt_idx, cost in matched:
            # Wavenumber errors
            dkx = est_kx[est_idx] - gt_kx[gt_idx]
            dky = est_ky[est_idx] - gt_ky[gt_idx]
            kx_errors.append(dkx)
            ky_errors.append(dky)

            gt_kmag = np.sqrt(gt_kx[gt_idx]**2 + gt_ky[gt_idx]**2)
            est_kmag = np.sqrt(est_kx[est_idx]**2 + est_ky[est_idx]**2)
            k_mag_errors.append((est_kmag - gt_kmag) / max(gt_kmag, 1e-6))

            # Frequency error
            omega_errors.append(
                (est_omega[est_idx] - gt_omega[gt_idx]) / max(gt_omega[gt_idx], 1e-6))

            # Amplitude error (relative)
            amp_errors.append(
                (est_w[est_idx] - gt_A[gt_idx]) / max(gt_A[gt_idx], 1e-6))

            # Direction error (degrees)
            gt_dir = np.degrees(np.arctan2(gt_ky[gt_idx], gt_kx[gt_idx]))
            est_dir = np.degrees(np.arctan2(est_ky[est_idx], est_kx[est_idx]))
            d_dir = abs(est_dir - gt_dir)
            if d_dir > 180:
                d_dir = 360 - d_dir
            direction_errors.append(d_dir)

        result['k_mag_rel_error_mean'] = float(np.mean(np.abs(k_mag_errors)))
        result['k_mag_rel_error_std'] = float(np.std(k_mag_errors))
        result['omega_rel_error_mean'] = float(np.mean(np.abs(omega_errors)))
        result['omega_rel_error_std'] = float(np.std(omega_errors))
        result['amp_rel_error_mean'] = float(np.mean(np.abs(amp_errors)))
        result['amp_rel_error_std'] = float(np.std(amp_errors))
        result['direction_error_deg_mean'] = float(np.mean(direction_errors))
        result['direction_error_deg_std'] = float(np.std(direction_errors))

        # Energy-weighted detection rate (fraction of total energy captured)
        matched_gt_indices = [m[1] for m in matched]
        matched_energy = np.sum(gt_A[matched_gt_indices]**2)
        total_energy = np.sum(gt_A**2)
        result['energy_detection_rate'] = float(matched_energy / max(total_energy, 1e-12))
    else:
        result['k_mag_rel_error_mean'] = float('nan')
        result['omega_rel_error_mean'] = float('nan')
        result['amp_rel_error_mean'] = float('nan')
        result['direction_error_deg_mean'] = float('nan')
        result['energy_detection_rate'] = 0.0

    return result


def run_experiment1():
    """Spectral Parameter Estimation Accuracy."""
    print("=" * 80)
    print("EXPERIMENT 1: Spectral Parameter Estimation Accuracy")
    print("=" * 80)

    results = {}
    for ep_name in SPECTRAL_EPISODES:
        h5_path = os.path.join(DATA_DIR, ep_name)
        if not os.path.exists(h5_path):
            print(f"  SKIP: {ep_name}")
            continue

        short = ep_name.replace('episode_', '').replace('.h5', '')
        print(f"\n  [{short}]", flush=True)

        # Load GT components
        loader = EpisodeDataLoader(h5_path)
        gt_comp = loader.wave_components  # (N, 6): [A, omega, kx, ky, phi, Q]

        # Run WaveSplat
        result = run_estimation(h5_path, config=WS_CONFIG)
        est_params = result['spectrum_params']

        # Compare
        acc = spectral_accuracy(est_params, gt_comp)
        results[short] = acc

        print(f"    GT components: {acc['n_gt']}, "
              f"Estimated: {acc['n_est']}, "
              f"Matched: {acc['n_matched']}")
        print(f"    Detection rate: {acc['detection_rate']:.1%} "
              f"(energy: {acc['energy_detection_rate']:.1%})")
        if acc['n_matched'] > 0:
            print(f"    |k| rel error: {acc['k_mag_rel_error_mean']:.3f}")
            print(f"    ω rel error:   {acc['omega_rel_error_mean']:.3f}")
            print(f"    A rel error:   {acc['amp_rel_error_mean']:.3f}")
            print(f"    Dir error:     {acc['direction_error_deg_mean']:.1f}°")

    # Summary by complexity
    print(f"\n  {'─'*60}")
    print(f"  Summary by complexity:")
    import re
    for ncomp in [1, 2, 5, 15, 30]:
        vals = [v for k, v in results.items()
                if f'_{ncomp}comp_' in k]
        if vals:
            det = np.mean([v['detection_rate'] for v in vals])
            edet = np.mean([v['energy_detection_rate'] for v in vals])
            k_err = np.mean([v['k_mag_rel_error_mean'] for v in vals
                           if not np.isnan(v['k_mag_rel_error_mean'])])
            dir_err = np.mean([v['direction_error_deg_mean'] for v in vals
                             if not np.isnan(v['direction_error_deg_mean'])])
            print(f"    {ncomp:>2d} comp: det={det:.1%}  "
                  f"energy_det={edet:.1%}  "
                  f"|k|_err={k_err:.3f}  "
                  f"dir_err={dir_err:.1f}°")

    return results


# ═══════════════════════════════════════════════════════════════════════════
# Experiment 2: Sparse Observation Robustness
# ═══════════════════════════════════════════════════════════════════════════

def subsample_frame(frame, n_pts_target, rng):
    """Create a copy of FrameData with subsampled observations."""
    import copy
    n = len(frame.x_obs)
    if n <= n_pts_target:
        return frame
    idx = rng.choice(n, n_pts_target, replace=False)
    idx_t = torch.tensor(idx, dtype=torch.long)
    new_frame = copy.copy(frame)
    new_frame.x_obs = frame.x_obs[idx_t]
    new_frame.y_obs = frame.y_obs[idx_t]
    new_frame.z_obs = frame.z_obs[idx_t]
    new_frame.confidence = frame.confidence[idx_t]
    new_frame.incidence_angles = frame.incidence_angles[idx_t]
    return new_frame


def run_wavesplat_with_npts(h5_path, n_pts_target):
    """Run WaveSplat with subsampled observations via SequentialUpdater."""
    import torch
    from wavesplat.optimizer.sequential_update import SequentialUpdater
    from wavesplat.evaluation.metrics import rmse, nrmse

    cfg = dict(DEFAULT_CONFIG)
    cfg.update(WS_CONFIG)

    torch.manual_seed(cfg.get('seed', 42))
    np.random.seed(cfg.get('seed', 42))

    loader = EpisodeDataLoader(h5_path, device=cfg['device'])
    cfg['depth'] = loader.depth

    rng = np.random.default_rng(42)

    # Subsample init frames
    init_frames = loader.get_init_window(tp_factor=2.0)
    init_frames_sub = [subsample_frame(f, n_pts_target, rng) for f in init_frames]

    updater = SequentialUpdater(cfg)
    updater.initialize(
        init_frames_sub,
        x_range=(float(loader.gt_grid_x[0]), float(loader.gt_grid_x[-1])),
        y_range=(float(loader.gt_grid_y[0]), float(loader.gt_grid_y[-1])),
    )

    grid_x = loader.gt_grid_x
    grid_y = loader.gt_grid_y
    tracking_frames = loader.get_tracking_frames(tp_factor=2.0)

    frame_metrics_list = []
    for frame in tracking_frames:
        frame_sub = subsample_frame(frame, n_pts_target, rng)
        update_result = updater.update(frame_sub)

        pred = updater.predict_field(grid_x, grid_y, frame.t)
        pred_np = pred.cpu().numpy()
        gt_np = frame.gt_field.cpu().numpy()

        metrics = compute_all_metrics(
            pred_np, gt_np,
            grid_x.cpu().numpy(), grid_y.cpu().numpy(),
            loader.Hs_true,
            obs_x=frame_sub.x_obs.cpu().numpy() if len(frame_sub.x_obs) > 0 else None,
            obs_y=frame_sub.y_obs.cpu().numpy() if len(frame_sub.y_obs) > 0 else None,
        )
        metrics['elapsed_ms'] = update_result['elapsed_ms']
        frame_metrics_list.append(metrics)

    avg = {}
    for k in METRIC_KEYS + ['elapsed_ms']:
        vals = [m[k] for m in frame_metrics_list
                if not np.isnan(m.get(k, float('nan')))]
        avg[k] = float(np.mean(vals)) if vals else float('nan')
    return avg


def run_letkf_with_npts(h5_path, n_pts_target):
    """Run LETKF with subsampled observations."""
    np.random.seed(42)
    loader = EpisodeDataLoader(h5_path)
    gt_grid_x = loader.gt_grid_x.cpu().numpy()
    gt_grid_y = loader.gt_grid_y.cpu().numpy()

    etkf_nx, etkf_ny = 40, 30
    etkf_gx = np.linspace(gt_grid_x[0], gt_grid_x[-1], etkf_nx)
    etkf_gy = np.linspace(gt_grid_y[0], gt_grid_y[-1], etkf_ny)

    letkf = LETKFGrid(etkf_gx, etkf_gy, depth=loader.depth,
                       n_ensemble=100, mult_inflation=1.0,
                       add_inflation=0.005, loc_radius=50.0)

    rng = np.random.default_rng(42)
    frame_metrics = []
    for frame in loader:
        x = frame.x_obs.cpu().numpy()
        y = frame.y_obs.cpu().numpy()
        z = frame.z_obs.cpu().numpy()

        # Subsample if needed
        if len(x) > n_pts_target:
            idx = rng.choice(len(x), n_pts_target, replace=False)
            x, y, z = x[idx], y[idx], z[idx]

        elapsed = letkf.update(x, y, z)

        pred = letkf.predict_grid()
        interp = RegularGridInterpolator(
            (etkf_gy, etkf_gx), pred,
            method='linear', bounds_error=False, fill_value=0.0)
        Xg, Yg = np.meshgrid(gt_grid_x, gt_grid_y)
        pred_full = interp(np.column_stack([Yg.ravel(), Xg.ravel()])).reshape(
            len(gt_grid_y), len(gt_grid_x))

        gt = frame.gt_field.cpu().numpy()
        metrics = compute_all_metrics(
            pred_full, gt, gt_grid_x, gt_grid_y, loader.Hs_true,
            obs_x=x if len(x) > 0 else None,
            obs_y=y if len(y) > 0 else None)
        metrics['elapsed_ms'] = elapsed
        frame_metrics.append(metrics)

    if frame_metrics:
        avg = {}
        for key in METRIC_KEYS + ['elapsed_ms']:
            vals = [m[key] for m in frame_metrics
                    if not np.isnan(m.get(key, float('nan')))]
            avg[key] = float(np.mean(vals)) if vals else float('nan')
        return avg
    return {}


def run_experiment2():
    """Sparse Observation Robustness."""
    print("\n" + "=" * 80)
    print("EXPERIMENT 2: Sparse Observation Robustness (n_pts sweep)")
    print("=" * 80)

    n_pts_values = [10, 20, 30, 50, 100]
    results = {}

    for ep_name in SPARSE_EPISODES:
        h5_path = os.path.join(DATA_DIR, ep_name)
        if not os.path.exists(h5_path):
            print(f"  SKIP: {ep_name}")
            continue

        short = ep_name.replace('episode_', '').replace('.h5', '')
        print(f"\n  [{short}]", flush=True)
        ep_results = {}

        for n_pts in n_pts_values:
            print(f"    n_pts={n_pts:>3d}: ", end="", flush=True)

            # WaveSplat
            try:
                ws = run_wavesplat_with_npts(h5_path, n_pts)
                ws_nrmse = ws.get('nRMSE', float('nan'))
                ws_ss = ws.get('Skill_Score', float('nan'))
            except Exception as e:
                print(f"WS FAILED ({e})", end="  ")
                ws_nrmse = float('nan')
                ws_ss = float('nan')
                ws = {}

            # LETKF
            try:
                lk = run_letkf_with_npts(h5_path, n_pts)
                lk_nrmse = lk.get('nRMSE', float('nan'))
                lk_ss = lk.get('Skill_Score', float('nan'))
            except Exception as e:
                print(f"LETKF FAILED ({e})")
                lk_nrmse = float('nan')
                lk_ss = float('nan')
                lk = {}

            ep_results[n_pts] = {
                'WaveSplat': {k: ws.get(k, float('nan')) for k in METRIC_KEYS},
                'LETKF': {k: lk.get(k, float('nan')) for k in METRIC_KEYS},
            }
            print(f"WS nRMSE={ws_nrmse:.4f} SS={ws_ss:+.3f}  "
                  f"LETKF nRMSE={lk_nrmse:.4f} SS={lk_ss:+.3f}", flush=True)

        results[short] = ep_results

    # Summary table
    print(f"\n  {'─'*80}")
    print(f"  Average nRMSE by n_pts:")
    print(f"  {'n_pts':>6s}  {'WS nRMSE':>10s}  {'LK nRMSE':>10s}  "
          f"{'WS SS':>8s}  {'LK SS':>8s}  {'WS wins':>8s}")
    print(f"  {'─'*6}  {'─'*10}  {'─'*10}  {'─'*8}  {'─'*8}  {'─'*8}")

    for n_pts in n_pts_values:
        ws_nrmses = []
        lk_nrmses = []
        ws_sss = []
        lk_sss = []
        for ep, ep_res in results.items():
            if n_pts in ep_res:
                ws_n = ep_res[n_pts]['WaveSplat'].get('nRMSE', float('nan'))
                lk_n = ep_res[n_pts]['LETKF'].get('nRMSE', float('nan'))
                ws_s = ep_res[n_pts]['WaveSplat'].get('Skill_Score', float('nan'))
                lk_s = ep_res[n_pts]['LETKF'].get('Skill_Score', float('nan'))
                if not np.isnan(ws_n):
                    ws_nrmses.append(ws_n)
                if not np.isnan(lk_n):
                    lk_nrmses.append(lk_n)
                if not np.isnan(ws_s):
                    ws_sss.append(ws_s)
                if not np.isnan(lk_s):
                    lk_sss.append(lk_s)

        ws_wins = sum(1 for w, l in zip(ws_nrmses, lk_nrmses) if w < l)
        total = min(len(ws_nrmses), len(lk_nrmses))
        print(f"  {n_pts:>6d}  {np.mean(ws_nrmses):>10.4f}  {np.mean(lk_nrmses):>10.4f}  "
              f"{np.mean(ws_sss):>+8.3f}  {np.mean(lk_sss):>+8.3f}  "
              f"{ws_wins}/{total}")

    return results


# ═══════════════════════════════════════════════════════════════════════════
# Experiment 3: Deterministic Forecast
# ═══════════════════════════════════════════════════════════════════════════

def forward_model(params, x, y, t, depth=50.0):
    """
    Evaluate forward model: z(x,y,t) = Σ w_i · cos(kx_i·x + ky_i·y - ω_i·t + φ_i)
    """
    w = params['w']
    kx = params['kx']
    ky = params['ky']
    phi = params['phi']
    omega = params['omega']

    X, Y = np.meshgrid(x, y)
    field = np.zeros_like(X)
    for i in range(len(w)):
        field += w[i] * np.cos(kx[i] * X + ky[i] * Y - omega[i] * t + phi[i])
    return field


def run_experiment3():
    """Deterministic Forecast.

    Strategy: Use only first 20s for estimation (init+tracking),
    then forecast into the remaining 10s and compare with GT.
    """
    print("\n" + "=" * 80)
    print("EXPERIMENT 3: Deterministic Forecast (t+Δt prediction)")
    print("  Strategy: estimate on t=[0,20s], forecast into t=[20,30s]")
    print("=" * 80)

    # Forecast horizons in seconds (relative to estimation cutoff)
    horizons = [0.5, 1.0, 2.0, 3.0, 5.0, 8.0, 10.0]
    CUTOFF_T = 20.0  # Use first 20s for estimation

    forecast_episodes = [
        'episode_BF3_1comp_seed0_uniform.h5',
        'episode_BF3_2comp_seed0_uniform.h5',
        'episode_BF4_1comp_seed0_uniform.h5',
        'episode_BF4_5comp_seed0_uniform.h5',
        'episode_BF4_15comp_seed0_uniform.h5',
    ]

    results = {}

    for ep_name in forecast_episodes:
        h5_path = os.path.join(DATA_DIR, ep_name)
        if not os.path.exists(h5_path):
            print(f"  SKIP: {ep_name}")
            continue

        short = ep_name.replace('episode_', '').replace('.h5', '')
        print(f"\n  [{short}]", flush=True)

        # Run WaveSplat with manual cutoff: use SequentialUpdater directly
        from wavesplat.optimizer.sequential_update import SequentialUpdater

        cfg = dict(DEFAULT_CONFIG)
        cfg.update(WS_CONFIG)
        torch.manual_seed(cfg.get('seed', 42))
        np.random.seed(cfg.get('seed', 42))

        loader = EpisodeDataLoader(h5_path, device=cfg['device'])
        cfg['depth'] = loader.depth
        gt_grid_x = loader.gt_grid_x.cpu().numpy()
        gt_grid_y = loader.gt_grid_y.cpu().numpy()

        # Init phase (same as normal)
        init_frames = loader.get_init_window(tp_factor=2.0)
        updater = SequentialUpdater(cfg)
        updater.initialize(
            init_frames,
            x_range=(float(loader.gt_grid_x[0]), float(loader.gt_grid_x[-1])),
            y_range=(float(loader.gt_grid_y[0]), float(loader.gt_grid_y[-1])),
        )

        # Tracking phase — only up to CUTOFF_T
        tracking_frames = loader.get_tracking_frames(tp_factor=2.0)
        last_est_t = init_frames[-1].t
        for frame in tracking_frames:
            if frame.t > CUTOFF_T:
                break
            updater.update(frame)
            last_est_t = frame.t

        est_params = updater.model.get_spectrum_params()
        print(f"    Estimation ended at t={last_est_t:.1f}s, "
              f"N={len(est_params['w'])} gaussians", flush=True)

        # Forecast into future frames
        ep_results = {}
        for dt_forecast in horizons:
            t_target = last_est_t + dt_forecast

            # Find closest GT frame
            frame_idx = np.argmin(np.abs(loader.timestamps - t_target))
            actual_t = loader.timestamps[frame_idx]

            if abs(actual_t - t_target) > 0.15 or actual_t > loader.timestamps[-1]:
                continue

            # WaveSplat forecast via forward model
            pred_field = forward_model(est_params, gt_grid_x, gt_grid_y, actual_t)
            gt_field = loader._gt_fields[frame_idx]

            metrics = compute_all_metrics(
                pred_field, gt_field, gt_grid_x, gt_grid_y, loader.Hs_true)

            ep_results[dt_forecast] = {
                'nRMSE': float(metrics['nRMSE']),
                'Skill_Score': float(metrics['Skill_Score']),
                'Pearson_rho': float(metrics['Pearson_rho']),
                'Variance_Ratio': float(metrics['Variance_Ratio']),
            }
            print(f"    t+{dt_forecast:.1f}s (t={actual_t:.1f}s): "
                  f"nRMSE={metrics['nRMSE']:.4f}  "
                  f"SS={metrics['Skill_Score']:+.3f}  "
                  f"ρ={metrics['Pearson_rho']:.3f}", flush=True)

        results[short] = ep_results

    # Summary
    print(f"\n  {'─'*70}")
    print(f"  Average forecast metrics by horizon:")
    print(f"  {'Δt':>6s}  {'nRMSE':>8s}  {'SS':>8s}  {'ρ':>8s}  {'VR':>8s}")
    print(f"  {'─'*6}  {'─'*8}  {'─'*8}  {'─'*8}  {'─'*8}")

    for dt in horizons:
        nrmses = []
        sss = []
        rhos = []
        vrs = []
        for ep, ep_res in results.items():
            if dt in ep_res:
                nrmses.append(ep_res[dt]['nRMSE'])
                sss.append(ep_res[dt]['Skill_Score'])
                rhos.append(ep_res[dt]['Pearson_rho'])
                vrs.append(ep_res[dt]['Variance_Ratio'])
        if nrmses:
            print(f"  {dt:>5.1f}s  {np.mean(nrmses):>8.4f}  "
                  f"{np.mean(sss):>+8.3f}  "
                  f"{np.mean(rhos):>8.3f}  "
                  f"{np.mean(vrs):>8.3f}")

    # Forecast by complexity
    print(f"\n  Forecast nRMSE at t+1s by complexity:")
    import re
    for ncomp in [1, 2, 5, 15, 30]:
        vals = [v.get(1.0, {}).get('nRMSE', float('nan'))
                for k, v in results.items() if f'_{ncomp}comp_' in k]
        vals = [v for v in vals if not np.isnan(v)]
        if vals:
            print(f"    {ncomp:>2d} comp: nRMSE={np.mean(vals):.4f}")

    return results


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Experiment 1
    spectral_results = run_experiment1()

    # Experiment 2
    sparse_results = run_experiment2()

    # Experiment 3
    forecast_results = run_experiment3()

    # Save all
    output = {
        'spectral_accuracy': spectral_results,
        'sparse_robustness': {
            ep: {str(k): v for k, v in ep_res.items()}
            for ep, ep_res in sparse_results.items()
        },
        'forecast': {
            ep: {str(k): v for k, v in ep_res.items()}
            for ep, ep_res in forecast_results.items()
        },
    }

    out_path = os.path.join(OUTPUT_DIR, 'three_contributions.json')
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nAll results saved to {out_path}")


if __name__ == '__main__':
    main()
