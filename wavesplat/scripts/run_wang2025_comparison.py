"""
run_wang2025_comparison.py
3-way comparison: WaveSplat vs ETKF (our simplified) vs ETKF-Wang2025 (faithful).

Tests Wang 2025's ETKF with fuzzy logic + interpolated measurements
on our sparse LiDAR setup (12 episodes).
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
from wavesplat.baselines.etkf_wang2025 import ETKFWang2025
from wavesplat.evaluation.metrics import compute_all_metrics
from scipy.interpolate import RegularGridInterpolator


EPISODES = [
    'episode_BF2_1comp_seed0_uniform.h5',
    'episode_BF3_1comp_seed0_uniform.h5',
    'episode_BF3_2comp_seed0_uniform.h5',
    'episode_BF3_2comp_seed0_slope.h5',
    'episode_BF3_5comp_seed0_uniform.h5',
    'episode_BF3_5comp_seed0_slope.h5',
    'episode_BF4_1comp_seed0_uniform.h5',
    'episode_BF4_5comp_seed0_uniform.h5',
    'episode_BF4_15comp_seed0_uniform.h5',
    'episode_BF4_30comp_seed0_uniform.h5',
    'episode_BF5_15comp_seed0_uniform.h5',
    'episode_BF5_30comp_seed0_uniform.h5',
]


def run_wavesplat(h5_path):
    """Full WaveSplat (P1: adaptive N + multi-frame + dispersion)."""
    config = {
        'init_method': 'v2_konly',
        'use_dispersion': True,
        'adaptive_n': True,
        'tracking_window': 3,
        'seed': 42,
    }
    result = run_estimation(h5_path, config=config)
    return {
        'avg_nRMSE': float(result['avg_nRMSE']),
        'avg_inference_ms': float(result['avg_inference_ms']),
    }


def run_etkf_orig(h5_path):
    """Our original ETKF baseline (Ne=100, 40x30 grid)."""
    np.random.seed(42)
    loader = EpisodeDataLoader(h5_path)
    gt_gx = loader.gt_grid_x.cpu().numpy()
    gt_gy = loader.gt_grid_y.cpu().numpy()

    etkf_nx, etkf_ny = 40, 30
    etkf_gx = np.linspace(gt_gx[0], gt_gx[-1], etkf_nx)
    etkf_gy = np.linspace(gt_gy[0], gt_gy[-1], etkf_ny)

    etkf = ETKFGrid(etkf_gx, etkf_gy, depth=loader.depth, n_ensemble=100)

    frame_results = []
    for frame in loader:
        x = frame.x_obs.cpu().numpy()
        y = frame.y_obs.cpu().numpy()
        z = frame.z_obs.cpu().numpy()

        elapsed = etkf.update(x, y, z)

        etkf_pred = etkf.predict_grid()
        interp = RegularGridInterpolator(
            (etkf_gy, etkf_gx), etkf_pred,
            method='linear', bounds_error=False, fill_value=0.0)
        Xg, Yg = np.meshgrid(gt_gx, gt_gy)
        pred = interp(np.column_stack([Yg.ravel(), Xg.ravel()])).reshape(
            len(gt_gy), len(gt_gx))

        gt = frame.gt_field.cpu().numpy()
        metrics = compute_all_metrics(
            pred, gt, gt_gx, gt_gy, loader.Hs_true,
            obs_x=x if len(x) > 0 else None,
            obs_y=y if len(y) > 0 else None)
        metrics['elapsed_ms'] = elapsed
        frame_results.append(metrics)

    if frame_results:
        return {
            'avg_nRMSE': float(np.mean([r['nRMSE'] for r in frame_results])),
            'avg_inference_ms': float(np.mean([r['elapsed_ms'] for r in frame_results])),
        }
    return {'avg_nRMSE': float('nan'), 'avg_inference_ms': float('nan')}


def run_etkf_wang2025(h5_path, use_fuzzy=True):
    """Wang 2025 ETKF with fuzzy logic + interpolated measurements."""
    np.random.seed(42)
    loader = EpisodeDataLoader(h5_path)
    gt_gx = loader.gt_grid_x.cpu().numpy()
    gt_gy = loader.gt_grid_y.cpu().numpy()

    # Wang 2025 uses 128x128 on ~50x50m domain (dx~0.4m).
    # Our domain is 240x200m. Use 80x60 (dx=3m, dy=3.3m) for
    # comparable grid density relative to LiDAR point spacing (~2m).
    etkf_nx, etkf_ny = 80, 60
    etkf_gx = np.linspace(gt_gx[0], gt_gx[-1], etkf_nx)
    etkf_gy = np.linspace(gt_gy[0], gt_gy[-1], etkf_ny)

    etkf = ETKFWang2025(
        etkf_gx, etkf_gy,
        depth=loader.depth,
        n_ensemble=300,      # Wang 2025 optimal (Table 2)
        use_fuzzy=use_fuzzy,
        n_substates=8,        # More sub-states for larger grid
    )

    frame_results = []
    for frame in loader:
        x = frame.x_obs.cpu().numpy()
        y = frame.y_obs.cpu().numpy()
        z = frame.z_obs.cpu().numpy()

        elapsed = etkf.update(x, y, z)

        etkf_pred = etkf.predict_grid()
        interp = RegularGridInterpolator(
            (etkf_gy, etkf_gx), etkf_pred,
            method='linear', bounds_error=False, fill_value=0.0)
        Xg, Yg = np.meshgrid(gt_gx, gt_gy)
        pred = interp(np.column_stack([Yg.ravel(), Xg.ravel()])).reshape(
            len(gt_gy), len(gt_gx))

        gt = frame.gt_field.cpu().numpy()
        metrics = compute_all_metrics(
            pred, gt, gt_gx, gt_gy, loader.Hs_true,
            obs_x=x if len(x) > 0 else None,
            obs_y=y if len(y) > 0 else None)
        metrics['elapsed_ms'] = elapsed
        frame_results.append(metrics)

    if frame_results:
        return {
            'avg_nRMSE': float(np.mean([r['nRMSE'] for r in frame_results])),
            'avg_inference_ms': float(np.mean([r['elapsed_ms'] for r in frame_results])),
        }
    return {'avg_nRMSE': float('nan'), 'avg_inference_ms': float('nan')}


def main():
    data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'results')
    os.makedirs(output_dir, exist_ok=True)

    methods = {
        'WaveSplat': run_wavesplat,
        'ETKF (ours, Ne=100)': run_etkf_orig,
        'ETKF-Wang2025 (Ne=300, fuzzy)': lambda h5: run_etkf_wang2025(h5, use_fuzzy=True),
    }

    print("=" * 110)
    print("3-Way Comparison: WaveSplat vs ETKF-orig vs ETKF-Wang2025")
    print("=" * 110)

    all_results = {}

    for ep_name in EPISODES:
        h5_path = os.path.join(data_dir, ep_name)
        if not os.path.exists(h5_path):
            print(f"  SKIP: {ep_name}")
            continue

        short = ep_name.replace('episode_', '').replace('.h5', '')
        print(f"\n[{short}]")

        ep_results = {}
        for method_name, run_fn in methods.items():
            try:
                t0 = time.time()
                result = run_fn(h5_path)
                total = time.time() - t0
                ep_results[method_name] = result
                print(f"  {method_name:<35s}  nRMSE={result['avg_nRMSE']:.4f}  "
                      f"{result['avg_inference_ms']:.1f}ms/frame  ({total:.1f}s)")
            except Exception as e:
                print(f"  {method_name:<35s}  FAILED: {e}")
                ep_results[method_name] = {'error': str(e)}

        all_results[short] = ep_results

    # Summary
    print(f"\n{'='*110}")
    print("SUMMARY")
    print(f"{'='*110}")

    method_names = list(methods.keys())
    header = f"{'Episode':<30s}"
    for m in method_names:
        tag = m.split('(')[0].strip()[:15]
        header += f"  {'nRMSE-'+tag:>18s}  {'ms-'+tag:>10s}"
    print(header)
    print("-" * len(header))

    avgs = {m: {'nrmse': [], 'ms': []} for m in method_names}
    for ep, res in all_results.items():
        row = f"{ep:<30s}"
        for m in method_names:
            r = res.get(m, {})
            n = r.get('avg_nRMSE', float('nan'))
            ms = r.get('avg_inference_ms', float('nan'))
            row += f"  {n:>18.4f}  {ms:>10.1f}"
            if not np.isnan(n):
                avgs[m]['nrmse'].append(n)
                avgs[m]['ms'].append(ms)
        print(row)

    print("-" * len(header))
    row = f"{'AVERAGE':<30s}"
    for m in method_names:
        n = np.mean(avgs[m]['nrmse']) if avgs[m]['nrmse'] else float('nan')
        ms = np.mean(avgs[m]['ms']) if avgs[m]['ms'] else float('nan')
        row += f"  {n:>18.4f}  {ms:>10.1f}"
    print(row)

    # Save
    out_path = os.path.join(output_dir, 'wang2025_comparison.json')
    with open(out_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")


if __name__ == '__main__':
    main()
