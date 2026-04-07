"""
run_p1_comparison.py
P1: WaveSplat-P0 (baseline) vs WaveSplat-P1 (adaptive N + multi-frame) vs ETKF.

Runs all three methods on 12 representative episodes.
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


def run_wavesplat_p0(h5_path):
    """P0 baseline: no adaptive N, single-frame tracking."""
    config = {
        'init_method': 'v2_konly',
        'use_dispersion': True,
        'use_confidence': True,
        'adaptive_n': False,
        'tracking_window': 1,
        'seed': 42,
    }
    result = run_estimation(h5_path, config=config)
    return {
        'avg_nRMSE': float(result['avg_nRMSE']),
        'avg_inference_ms': float(result['avg_inference_ms']),
        'final_n': result.get('n_gaussians', 15),
    }


def run_wavesplat_p1(h5_path):
    """P1: adaptive N + multi-frame tracking window."""
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
        'final_n': result.get('n_gaussians', 15),
    }


def run_etkf(h5_path, grid_resolution='medium'):
    """Run ETKF-grid baseline."""
    np.random.seed(42)
    loader = EpisodeDataLoader(h5_path)
    gt_grid_x = loader.gt_grid_x.cpu().numpy()
    gt_grid_y = loader.gt_grid_y.cpu().numpy()

    res_map = {'medium': (40, 30), 'fine': (80, 60)}
    etkf_nx, etkf_ny = res_map[grid_resolution]
    etkf_gx = np.linspace(gt_grid_x[0], gt_grid_x[-1], etkf_nx)
    etkf_gy = np.linspace(gt_grid_y[0], gt_grid_y[-1], etkf_ny)

    etkf = ETKFGrid(etkf_gx, etkf_gy, depth=loader.depth, n_ensemble=50)

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
        Xg, Yg = np.meshgrid(gt_grid_x, gt_grid_y)
        pred = interp(np.column_stack([Yg.ravel(), Xg.ravel()])).reshape(
            len(gt_grid_y), len(gt_grid_x))

        gt = frame.gt_field.cpu().numpy()
        metrics = compute_all_metrics(
            pred, gt, gt_grid_x, gt_grid_y, loader.Hs_true,
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

    all_results = {}

    print(f"P1: WaveSplat-P0 vs WaveSplat-P1 vs ETKF-medium")
    print(f"Episodes: {len(EPISODES)}")
    print(f"{'='*90}")

    for ep_name in EPISODES:
        h5_path = os.path.join(data_dir, ep_name)
        if not os.path.exists(h5_path):
            print(f"  SKIP: {ep_name}")
            continue

        short = ep_name.replace('episode_', '').replace('.h5', '')
        print(f"\n[{short}]", flush=True)

        ep_results = {}

        # WaveSplat P0
        try:
            p0 = run_wavesplat_p0(h5_path)
            ep_results['WS-P0'] = p0
            print(f"  P0: nRMSE={p0['avg_nRMSE']:.4f} "
                  f"({p0['avg_inference_ms']:.1f}ms, N={p0['final_n']})", flush=True)
        except Exception as e:
            print(f"  P0 FAILED: {e}")
            ep_results['WS-P0'] = {'error': str(e)}

        # WaveSplat P1
        try:
            p1 = run_wavesplat_p1(h5_path)
            ep_results['WS-P1'] = p1
            print(f"  P1: nRMSE={p1['avg_nRMSE']:.4f} "
                  f"({p1['avg_inference_ms']:.1f}ms, N={p1['final_n']})", flush=True)
        except Exception as e:
            print(f"  P1 FAILED: {e}")
            ep_results['WS-P1'] = {'error': str(e)}

        # ETKF
        try:
            etkf = run_etkf(h5_path, 'medium')
            ep_results['ETKF'] = etkf
            print(f"  ETKF: nRMSE={etkf['avg_nRMSE']:.4f} "
                  f"({etkf['avg_inference_ms']:.1f}ms)", flush=True)
        except Exception as e:
            print(f"  ETKF FAILED: {e}")
            ep_results['ETKF'] = {'error': str(e)}

        all_results[short] = ep_results

    # Summary
    print(f"\n{'='*90}")
    print(f"{'Episode':<30} {'P0 nRMSE':>10} {'P1 nRMSE':>10} {'ETKF':>10} "
          f"{'P0 ms':>8} {'P1 ms':>8} {'P1 N':>6}")
    print(f"{'-'*82}")

    p0_nrmses, p1_nrmses, etkf_nrmses = [], [], []
    p0_speeds, p1_speeds = [], []

    for ep, res in all_results.items():
        p0 = res.get('WS-P0', {})
        p1 = res.get('WS-P1', {})
        et = res.get('ETKF', {})

        p0_n = p0.get('avg_nRMSE', float('nan'))
        p1_n = p1.get('avg_nRMSE', float('nan'))
        et_n = et.get('avg_nRMSE', float('nan'))
        p0_s = p0.get('avg_inference_ms', float('nan'))
        p1_s = p1.get('avg_inference_ms', float('nan'))
        p1_nn = p1.get('final_n', '?')

        print(f"{ep:<30} {p0_n:>10.4f} {p1_n:>10.4f} {et_n:>10.4f} "
              f"{p0_s:>8.1f} {p1_s:>8.1f} {p1_nn:>6}")

        if not np.isnan(p0_n): p0_nrmses.append(p0_n)
        if not np.isnan(p1_n): p1_nrmses.append(p1_n)
        if not np.isnan(et_n): etkf_nrmses.append(et_n)
        if not np.isnan(p0_s): p0_speeds.append(p0_s)
        if not np.isnan(p1_s): p1_speeds.append(p1_s)

    print(f"{'-'*82}")
    if p0_nrmses:
        print(f"{'AVERAGE':<30} {np.mean(p0_nrmses):>10.4f} "
              f"{np.mean(p1_nrmses):>10.4f} {np.mean(etkf_nrmses):>10.4f} "
              f"{np.mean(p0_speeds):>8.1f} {np.mean(p1_speeds):>8.1f}")

        improvement = (np.mean(p0_nrmses) - np.mean(p1_nrmses)) / np.mean(p0_nrmses) * 100
        vs_etkf = (np.mean(etkf_nrmses) - np.mean(p1_nrmses)) / np.mean(etkf_nrmses) * 100
        print(f"\nP1 vs P0: {improvement:+.1f}% nRMSE")
        print(f"P1 vs ETKF: {vs_etkf:+.1f}% nRMSE")

    out_path = os.path.join(output_dir, 'p1_comparison.json')
    with open(out_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")


if __name__ == '__main__':
    main()
