"""
run_strong_baseline_comparison.py
Compare WaveSplat against stronger baselines:
  1. LETKF + Gaspari-Cohn localization (loc_radius=50m)
  2. Dispersion-aware GPR (3-frame, phase-corrected)
  3. Original ETKF (tuned, Ne=100) for reference
  4. Original GPR-1 (single-frame) for reference

Runs on 6 representative episodes covering BF3-BF5, 1-30 components.
"""

import sys
import os
import json
import time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from wavesplat.data_loader import EpisodeDataLoader
from wavesplat.run import run_estimation
from wavesplat.baselines.etkf_grid import ETKFGrid
from wavesplat.baselines.letkf_grid import LETKFGrid
from wavesplat.baselines.gpr_baseline import GPRBaseline
from wavesplat.baselines.gpr_dispersion import GPRDispersion
from wavesplat.evaluation.metrics import compute_all_metrics
from scipy.interpolate import RegularGridInterpolator


# 6 representative episodes (balanced across sea states)
EPISODES = [
    'episode_BF3_2comp_seed0_uniform.h5',
    'episode_BF3_5comp_seed0_uniform.h5',
    'episode_BF4_1comp_seed0_uniform.h5',
    'episode_BF4_5comp_seed0_uniform.h5',
    'episode_BF4_15comp_seed0_uniform.h5',
    'episode_BF5_15comp_seed0_uniform.h5',
]


def run_wavesplat(h5_path):
    """WaveSplat with full config (adaptive N + tracking)."""
    config = {
        'init_method': 'v2_konly',
        'use_dispersion': True,
        'use_confidence': True,
        'adaptive_n': True,
        'tracking_window': 3,
        'seed': 42,
    }
    result = run_estimation(h5_path, config=config)
    return result


def run_grid_baseline(h5_path, baseline_class, baseline_kwargs, name):
    """Run any grid-based baseline (ETKF or LETKF)."""
    np.random.seed(42)
    loader = EpisodeDataLoader(h5_path)
    gt_grid_x = loader.gt_grid_x.cpu().numpy()
    gt_grid_y = loader.gt_grid_y.cpu().numpy()

    etkf_nx, etkf_ny = 40, 30
    etkf_gx = np.linspace(gt_grid_x[0], gt_grid_x[-1], etkf_nx)
    etkf_gy = np.linspace(gt_grid_y[0], gt_grid_y[-1], etkf_ny)

    baseline = baseline_class(etkf_gx, etkf_gy, depth=loader.depth,
                              **baseline_kwargs)

    frame_metrics = []
    for frame in loader:
        x = frame.x_obs.cpu().numpy()
        y = frame.y_obs.cpu().numpy()
        z = frame.z_obs.cpu().numpy()

        elapsed = baseline.update(x, y, z)

        pred = baseline.predict_grid()
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
        for key in frame_metrics[0]:
            vals = [m[key] for m in frame_metrics if not np.isnan(m.get(key, float('nan')))]
            avg[key] = float(np.mean(vals)) if vals else float('nan')
        return avg
    return {}


def run_gpr_baseline(h5_path, gpr_class, gpr_kwargs, name):
    """Run any GPR-based baseline."""
    np.random.seed(42)
    loader = EpisodeDataLoader(h5_path)
    gt_grid_x = loader.gt_grid_x.cpu().numpy()
    gt_grid_y = loader.gt_grid_y.cpu().numpy()

    gpr = gpr_class(**gpr_kwargs)

    frame_metrics = []
    for frame in loader:
        x = frame.x_obs.cpu().numpy()
        y = frame.y_obs.cpu().numpy()
        z = frame.z_obs.cpu().numpy()

        elapsed = gpr.update(x, y, z)
        pred = gpr.predict_grid(gt_grid_x, gt_grid_y)

        gt = frame.gt_field.cpu().numpy()
        metrics = compute_all_metrics(
            pred, gt, gt_grid_x, gt_grid_y, loader.Hs_true,
            obs_x=x if len(x) > 0 else None,
            obs_y=y if len(y) > 0 else None)
        metrics['elapsed_ms'] = elapsed
        frame_metrics.append(metrics)

    if frame_metrics:
        avg = {}
        for key in frame_metrics[0]:
            vals = [m[key] for m in frame_metrics if not np.isnan(m.get(key, float('nan')))]
            avg[key] = float(np.mean(vals)) if vals else float('nan')
        return avg
    return {}


def main():
    data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'results')
    os.makedirs(output_dir, exist_ok=True)

    methods = {
        'WaveSplat': None,  # handled separately
        'ETKF(tuned)': ('grid', ETKFGrid,
                         {'n_ensemble': 100, 'mult_inflation': 1.0,
                          'add_inflation': 0.005}),
        'LETKF(loc50)': ('grid', LETKFGrid,
                          {'n_ensemble': 100, 'mult_inflation': 1.0,
                           'add_inflation': 0.005, 'loc_radius': 50.0}),
        'GPR-1': ('gpr', GPRBaseline,
                  {'length_scale': 20.0, 'noise_sigma': 0.02,
                   'window_frames': 1}),
        'GPR-disp(3f)': ('gpr', GPRDispersion,
                          {'length_scale': 20.0, 'noise_sigma': 0.02,
                           'window_frames': 3, 'depth': 50.0}),
    }

    all_results = {}

    print(f"Strong Baseline Comparison")
    print(f"Episodes: {len(EPISODES)}")
    print(f"Methods: {list(methods.keys())}")
    print(f"{'='*120}")

    for ep_name in EPISODES:
        h5_path = os.path.join(data_dir, ep_name)
        if not os.path.exists(h5_path):
            print(f"  SKIP: {ep_name}")
            continue

        short = ep_name.replace('episode_', '').replace('.h5', '')
        print(f"\n[{short}]", flush=True)
        ep_results = {}

        # WaveSplat
        try:
            ws = run_wavesplat(h5_path)
            ep_results['WaveSplat'] = {
                'nRMSE': float(ws['avg_nRMSE']),
                'ms': float(ws['avg_inference_ms']),
                'N': ws.get('n_gaussians', -1),
            }
            print(f"  WaveSplat:    nRMSE={ws['avg_nRMSE']:.4f}  "
                  f"({ws['avg_inference_ms']:.1f}ms)", flush=True)
        except Exception as e:
            print(f"  WaveSplat FAILED: {e}")
            ep_results['WaveSplat'] = {'error': str(e)}

        # Grid-based baselines
        for name, spec in methods.items():
            if spec is None:
                continue
            mtype, cls, kwargs = spec

            try:
                if mtype == 'grid':
                    res = run_grid_baseline(h5_path, cls, kwargs, name)
                else:
                    res = run_gpr_baseline(h5_path, cls, kwargs, name)

                ep_results[name] = {
                    'nRMSE': res.get('nRMSE', float('nan')),
                    'VR': res.get('Variance_Ratio', float('nan')),
                    'rho': res.get('Pearson_rho', float('nan')),
                    'SS': res.get('Skill_Score', float('nan')),
                    'Dir_R2': res.get('Directional_R2', float('nan')),
                    'Hs_err': res.get('Hs_error', float('nan')),
                    'ms': res.get('elapsed_ms', float('nan')),
                }
                print(f"  {name:<14s} nRMSE={res.get('nRMSE', float('nan')):.4f}  "
                      f"SS={res.get('Skill_Score', float('nan')):+.3f}  "
                      f"({res.get('elapsed_ms', float('nan')):.1f}ms)", flush=True)
            except Exception as e:
                print(f"  {name} FAILED: {e}")
                ep_results[name] = {'error': str(e)}

        all_results[short] = ep_results

    # Summary table
    print(f"\n{'='*120}")
    print(f"\n{'SUMMARY':^120}")
    print(f"{'='*120}")

    method_names = ['WaveSplat', 'ETKF(tuned)', 'LETKF(loc50)',
                    'GPR-1', 'GPR-disp(3f)']

    header = f"{'Episode':<25}"
    for m in method_names:
        header += f" {m:>14}"
    print(header)
    print("-" * (25 + 15 * len(method_names)))

    method_nrmses = {m: [] for m in method_names}

    for ep, res in all_results.items():
        row = f"{ep:<25}"
        for m in method_names:
            val = res.get(m, {}).get('nRMSE', float('nan'))
            if not np.isnan(val):
                method_nrmses[m].append(val)
            row += f" {val:>14.4f}"
        print(row)

    print("-" * (25 + 15 * len(method_names)))

    avg_row = f"{'AVERAGE':<25}"
    for m in method_names:
        vals = method_nrmses[m]
        avg = np.mean(vals) if vals else float('nan')
        avg_row += f" {avg:>14.4f}"
    print(avg_row)

    # Save
    out_path = os.path.join(output_dir, 'strong_baseline_comparison.json')
    with open(out_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")


if __name__ == '__main__':
    main()
