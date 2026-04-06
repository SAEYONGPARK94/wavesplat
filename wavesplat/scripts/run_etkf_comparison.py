"""
run_etkf_comparison.py
P0-3: WaveSplat vs ETKF-grid direct comparison on representative episodes.

Runs both methods on 12 representative episodes covering:
- Sea states: BF2, BF3, BF4, BF5
- Complexities: 1, 2, 5, 15, 30 components
- Raydrop: uniform + slope

Outputs: JSON results + summary table.
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


# Representative episodes: cover BF range × complexity × raydrop
EPISODES = [
    # BF2 — low seas
    'episode_BF2_1comp_seed0_uniform.h5',
    # BF3 — moderate
    'episode_BF3_1comp_seed0_uniform.h5',
    'episode_BF3_2comp_seed0_uniform.h5',
    'episode_BF3_2comp_seed0_slope.h5',
    'episode_BF3_5comp_seed0_uniform.h5',
    'episode_BF3_5comp_seed0_slope.h5',
    # BF4 — rough
    'episode_BF4_1comp_seed0_uniform.h5',
    'episode_BF4_5comp_seed0_uniform.h5',
    'episode_BF4_15comp_seed0_uniform.h5',
    'episode_BF4_30comp_seed0_uniform.h5',
    # BF5 — very rough
    'episode_BF5_15comp_seed0_uniform.h5',
    'episode_BF5_30comp_seed0_uniform.h5',
]


def run_etkf(h5_path, grid_resolution='medium', verbose=False):
    """Run ETKF-grid on episode, return summary dict."""
    np.random.seed(42)
    loader = EpisodeDataLoader(h5_path)
    gt_grid_x = loader.gt_grid_x.cpu().numpy()
    gt_grid_y = loader.gt_grid_y.cpu().numpy()

    res_map = {'coarse': (20, 15), 'medium': (40, 30), 'fine': (80, 60)}
    etkf_nx, etkf_ny = res_map[grid_resolution]
    etkf_gx = np.linspace(gt_grid_x[0], gt_grid_x[-1], etkf_nx)
    etkf_gy = np.linspace(gt_grid_y[0], gt_grid_y[-1], etkf_ny)

    etkf = ETKFGrid(etkf_gx, etkf_gy, depth=loader.depth, n_ensemble=50)

    frame_results = []
    for i, frame in enumerate(loader):
        x = frame.x_obs.cpu().numpy()
        y = frame.y_obs.cpu().numpy()
        z = frame.z_obs.cpu().numpy()

        elapsed = etkf.update(x, y, z)

        # Interpolate to GT grid
        etkf_pred = etkf.predict_grid()
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
            print(f"    [ETKF-{grid_resolution}] t={frame.t:.1f}s "
                  f"nRMSE={metrics['nRMSE']:.4f}")

    if frame_results:
        avg_rmse = np.mean([r['RMSE_z'] for r in frame_results])
        avg_nrmse = np.mean([r['nRMSE'] for r in frame_results])
        avg_ms = np.mean([r['elapsed_ms'] for r in frame_results])
        avg_hs_err = np.mean([r['Hs_error'] for r in frame_results])
        avg_rmse_far = np.mean([r['RMSE_far'] for r in frame_results
                                if not np.isnan(r.get('RMSE_far', float('nan')))])
    else:
        avg_rmse = avg_nrmse = avg_ms = avg_hs_err = avg_rmse_far = float('nan')

    return {
        'avg_RMSE_z': float(avg_rmse),
        'avg_nRMSE': float(avg_nrmse),
        'avg_Hs_error': float(avg_hs_err),
        'avg_inference_ms': float(avg_ms),
        'avg_RMSE_far': float(avg_rmse_far),
        'n_frames': len(frame_results),
    }


def run_wavesplat(h5_path, verbose=False):
    """Run WaveSplat with default config + seed."""
    config = {
        'init_method': 'v2_konly',
        'use_dispersion': True,
        'use_confidence': True,
        'seed': 42,
    }
    result = run_estimation(h5_path, config=config, verbose=verbose)

    # Extract far-field RMSE from frame results
    avg_rmse_far = float('nan')
    if result.get('frame_results'):
        far_vals = [r['RMSE_far'] for r in result['frame_results']
                    if not np.isnan(r.get('RMSE_far', float('nan')))]
        if far_vals:
            avg_rmse_far = float(np.mean(far_vals))

    return {
        'avg_RMSE_z': float(result['avg_RMSE_z']),
        'avg_nRMSE': float(result['avg_nRMSE']),
        'avg_Hs_error': float(result['avg_Hs_error']),
        'avg_inference_ms': float(result['avg_inference_ms']),
        'avg_RMSE_far': avg_rmse_far,
        'T_converge': result.get('T_converge'),
        'init_nrmse': float(result.get('init_nrmse', float('nan'))),
        'n_frames': result.get('tracking_frames', 0),
    }


def main():
    data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'results')
    os.makedirs(output_dir, exist_ok=True)

    verbose = '--verbose' in sys.argv

    all_results = {}

    print(f"P0-3: WaveSplat vs ETKF-grid comparison")
    print(f"Episodes: {len(EPISODES)}")
    print(f"{'='*80}")

    for ep_name in EPISODES:
        h5_path = os.path.join(data_dir, ep_name)
        if not os.path.exists(h5_path):
            print(f"  SKIP (not found): {ep_name}")
            continue

        short = ep_name.replace('episode_', '').replace('.h5', '')
        print(f"\n[{short}]")

        ep_results = {}

        # WaveSplat
        t0 = time.time()
        try:
            ws = run_wavesplat(h5_path, verbose=verbose)
            ws_time = time.time() - t0
            ep_results['WaveSplat'] = ws
            print(f"  WaveSplat: nRMSE={ws['avg_nRMSE']:.4f} "
                  f"({ws['avg_inference_ms']:.1f}ms/f, total {ws_time:.1f}s)")
        except Exception as e:
            print(f"  WaveSplat FAILED: {e}")
            ep_results['WaveSplat'] = {'error': str(e)}

        # ETKF medium
        t0 = time.time()
        try:
            etkf_m = run_etkf(h5_path, 'medium', verbose=verbose)
            etkf_m_time = time.time() - t0
            ep_results['ETKF-medium'] = etkf_m
            print(f"  ETKF-med:  nRMSE={etkf_m['avg_nRMSE']:.4f} "
                  f"({etkf_m['avg_inference_ms']:.1f}ms/f, total {etkf_m_time:.1f}s)")
        except Exception as e:
            print(f"  ETKF-medium FAILED: {e}")
            ep_results['ETKF-medium'] = {'error': str(e)}

        # ETKF fine
        t0 = time.time()
        try:
            etkf_f = run_etkf(h5_path, 'fine', verbose=verbose)
            etkf_f_time = time.time() - t0
            ep_results['ETKF-fine'] = etkf_f
            print(f"  ETKF-fine: nRMSE={etkf_f['avg_nRMSE']:.4f} "
                  f"({etkf_f['avg_inference_ms']:.1f}ms/f, total {etkf_f_time:.1f}s)")
        except Exception as e:
            print(f"  ETKF-fine FAILED: {e}")
            ep_results['ETKF-fine'] = {'error': str(e)}

        all_results[short] = ep_results

    # Summary table
    print(f"\n{'='*80}")
    print(f"SUMMARY TABLE")
    print(f"{'='*80}")
    print(f"{'Episode':<30} {'WS nRMSE':>10} {'ETKF-m':>10} {'ETKF-f':>10} "
          f"{'WS ms':>8} {'ETKF-m ms':>10} {'ETKF-f ms':>10}")
    print(f"{'-'*88}")

    ws_nrmses, etkf_m_nrmses, etkf_f_nrmses = [], [], []
    ws_speeds, etkf_m_speeds, etkf_f_speeds = [], [], []

    for ep, res in all_results.items():
        ws = res.get('WaveSplat', {})
        em = res.get('ETKF-medium', {})
        ef = res.get('ETKF-fine', {})

        ws_n = ws.get('avg_nRMSE', float('nan'))
        em_n = em.get('avg_nRMSE', float('nan'))
        ef_n = ef.get('avg_nRMSE', float('nan'))

        ws_s = ws.get('avg_inference_ms', float('nan'))
        em_s = em.get('avg_inference_ms', float('nan'))
        ef_s = ef.get('avg_inference_ms', float('nan'))

        print(f"{ep:<30} {ws_n:>10.4f} {em_n:>10.4f} {ef_n:>10.4f} "
              f"{ws_s:>8.1f} {em_s:>10.1f} {ef_s:>10.1f}")

        if not np.isnan(ws_n): ws_nrmses.append(ws_n)
        if not np.isnan(em_n): etkf_m_nrmses.append(em_n)
        if not np.isnan(ef_n): etkf_f_nrmses.append(ef_n)
        if not np.isnan(ws_s): ws_speeds.append(ws_s)
        if not np.isnan(em_s): etkf_m_speeds.append(em_s)
        if not np.isnan(ef_s): etkf_f_speeds.append(ef_s)

    print(f"{'-'*88}")
    if ws_nrmses and etkf_m_nrmses:
        print(f"{'AVERAGE':<30} {np.mean(ws_nrmses):>10.4f} "
              f"{np.mean(etkf_m_nrmses):>10.4f} "
              f"{np.mean(etkf_f_nrmses):>10.4f} "
              f"{np.mean(ws_speeds):>8.1f} "
              f"{np.mean(etkf_m_speeds):>10.1f} "
              f"{np.mean(etkf_f_speeds):>10.1f}")

    # Save
    out_path = os.path.join(output_dir, 'etkf_comparison.json')
    with open(out_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")


if __name__ == '__main__':
    main()
