"""
run_gpr_comparison.py
3-way comparison: WaveSplat-P1 vs ETKF vs GPR on 12 representative episodes.

GPR uses Matérn ν=5/2 kernel, single-frame mode (no temporal model).
"""

import sys
import os
import json
import numpy as np
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from wavesplat.data_loader import EpisodeDataLoader
from wavesplat.run import run_estimation, set_seed
from wavesplat.baselines.etkf_grid import ETKFGrid
from wavesplat.baselines.gpr_baseline import GPRBaseline
from wavesplat.evaluation.metrics import compute_all_metrics
from scipy.interpolate import RegularGridInterpolator


EPISODES = [
    'episode_BF2_1comp_seed0_uniform.h5',
    'episode_BF3_1comp_seed0_uniform.h5',
    'episode_BF3_2comp_seed0_uniform.h5',
    'episode_BF3_5comp_seed0_uniform.h5',
    'episode_BF4_1comp_seed0_uniform.h5',
    'episode_BF4_5comp_seed0_uniform.h5',
    'episode_BF4_15comp_seed0_uniform.h5',
    'episode_BF4_30comp_seed0_uniform.h5',
    'episode_BF5_15comp_seed0_uniform.h5',
    'episode_BF5_30comp_seed0_uniform.h5',
]


def run_wavesplat(h5_path):
    """Run WaveSplat-P1."""
    config = {
        'init_method': 'v2_konly',
        'use_dispersion': True,
        'use_confidence': True,
        'adaptive_n': True,
        'tracking_window': 3,
        'seed': 42,
    }
    result = run_estimation(h5_path, config=config)
    return {
        'avg_nRMSE': float(result['avg_nRMSE']),
        'avg_inference_ms': float(result['avg_inference_ms']),
        'n_gaussians': result.get('n_gaussians', 15),
    }


def run_etkf(h5_path):
    """Run ETKF baseline."""
    np.random.seed(42)
    loader = EpisodeDataLoader(h5_path)
    gt_x = loader.gt_grid_x.cpu().numpy()
    gt_y = loader.gt_grid_y.cpu().numpy()

    etkf = ETKFGrid(
        np.linspace(gt_x[0], gt_x[-1], 40),
        np.linspace(gt_y[0], gt_y[-1], 30),
        depth=loader.depth, n_ensemble=50)

    frame_results = []
    for frame in loader:
        x = frame.x_obs.cpu().numpy()
        y = frame.y_obs.cpu().numpy()
        z = frame.z_obs.cpu().numpy()
        elapsed = etkf.update(x, y, z)

        etkf_pred = etkf.predict_grid()
        interp = RegularGridInterpolator(
            (np.linspace(gt_y[0], gt_y[-1], 30),
             np.linspace(gt_x[0], gt_x[-1], 40)),
            etkf_pred, method='linear', bounds_error=False, fill_value=0.0)
        Xg, Yg = np.meshgrid(gt_x, gt_y)
        pred = interp(np.column_stack([Yg.ravel(), Xg.ravel()])).reshape(
            len(gt_y), len(gt_x))

        gt = frame.gt_field.cpu().numpy()
        metrics = compute_all_metrics(pred, gt, gt_x, gt_y, loader.Hs_true)
        metrics['elapsed_ms'] = elapsed
        frame_results.append(metrics)

    return {
        'avg_nRMSE': float(np.mean([r['nRMSE'] for r in frame_results])),
        'avg_inference_ms': float(np.mean([r['elapsed_ms'] for r in frame_results])),
    }


def run_gpr(h5_path):
    """Run GPR baseline (single-frame, Matérn ν=5/2)."""
    loader = EpisodeDataLoader(h5_path)
    gt_x = loader.gt_grid_x.cpu().numpy()
    gt_y = loader.gt_grid_y.cpu().numpy()

    gpr = GPRBaseline(kernel_type='matern', length_scale=20.0,
                      noise_sigma=0.02, window_frames=1)

    frame_results = []
    for frame in loader:
        x = frame.x_obs.cpu().numpy()
        y = frame.y_obs.cpu().numpy()
        z = frame.z_obs.cpu().numpy()

        gpr.reset()  # single-frame mode: reset each frame
        elapsed = gpr.update(x, y, z)

        pred = gpr.predict_grid(gt_x, gt_y)
        gt = frame.gt_field.cpu().numpy()
        metrics = compute_all_metrics(pred, gt, gt_x, gt_y, loader.Hs_true)
        metrics['elapsed_ms'] = elapsed
        frame_results.append(metrics)

    return {
        'avg_nRMSE': float(np.mean([r['nRMSE'] for r in frame_results])),
        'avg_inference_ms': float(np.mean([r['elapsed_ms'] for r in frame_results])),
    }


def main():
    data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'results')
    os.makedirs(output_dir, exist_ok=True)

    results = {}

    print("3-Way Comparison: WaveSplat vs ETKF vs GPR")
    print("=" * 90)
    print(f"{'Episode':<15} {'WS nRMSE':>10} {'WS ms':>7} "
          f"{'ETKF nRMSE':>12} {'ETKF ms':>8} "
          f"{'GPR nRMSE':>11} {'GPR ms':>8}")
    print("-" * 90)

    for ep_name in EPISODES:
        h5_path = os.path.join(data_dir, ep_name)
        if not os.path.exists(h5_path):
            print(f"  SKIP: {ep_name}")
            continue

        short = ep_name.replace('episode_', '').replace('.h5', '').replace('_seed0_uniform', '')
        results[short] = {}

        # WaveSplat
        try:
            set_seed(42)
            ws = run_wavesplat(h5_path)
            results[short]['WS-P1'] = ws
        except Exception as e:
            ws = {'avg_nRMSE': float('nan'), 'avg_inference_ms': float('nan')}
            print(f"  WS FAILED: {e}")

        # ETKF
        try:
            etkf = run_etkf(h5_path)
            results[short]['ETKF'] = etkf
        except Exception as e:
            etkf = {'avg_nRMSE': float('nan'), 'avg_inference_ms': float('nan')}
            print(f"  ETKF FAILED: {e}")

        # GPR
        try:
            gpr = run_gpr(h5_path)
            results[short]['GPR'] = gpr
        except Exception as e:
            gpr = {'avg_nRMSE': float('nan'), 'avg_inference_ms': float('nan')}
            print(f"  GPR FAILED: {e}")

        print(f"{short:<15} {ws['avg_nRMSE']:>10.4f} {ws['avg_inference_ms']:>7.1f} "
              f"{etkf['avg_nRMSE']:>12.4f} {etkf['avg_inference_ms']:>8.1f} "
              f"{gpr['avg_nRMSE']:>11.4f} {gpr['avg_inference_ms']:>8.1f}")

    # Summary
    ws_all = [r['WS-P1']['avg_nRMSE'] for r in results.values() if 'WS-P1' in r]
    et_all = [r['ETKF']['avg_nRMSE'] for r in results.values() if 'ETKF' in r]
    gpr_all = [r['GPR']['avg_nRMSE'] for r in results.values() if 'GPR' in r]

    print("-" * 90)
    if ws_all and et_all and gpr_all:
        print(f"{'AVERAGE':<15} {np.mean(ws_all):>10.4f} "
              f"{'':>7} {np.mean(et_all):>12.4f} "
              f"{'':>8} {np.mean(gpr_all):>11.4f}")
        print(f"\nWS vs ETKF: {(1 - np.mean(ws_all)/np.mean(et_all))*100:.1f}% better")
        print(f"WS vs GPR:  {(1 - np.mean(ws_all)/np.mean(gpr_all))*100:.1f}% better")

    out_path = os.path.join(output_dir, 'gpr_comparison.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")


if __name__ == '__main__':
    main()
