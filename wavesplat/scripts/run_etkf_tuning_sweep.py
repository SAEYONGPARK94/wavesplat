"""
run_etkf_tuning_sweep.py
Phase B2: ETKF hyperparameter sweep on 12 representative episodes.

Goal: find the best ETKF configuration for fair baseline comparison (Rev.4 defense).
Sweep dimensions:
  - grid resolution: (40,30), (80,60)
  - ensemble size: 50, 100
  - multiplicative inflation: 1.0, 1.02, 1.05
  - additive inflation: 0.005, 0.02
Total: 2 * 2 * 3 * 2 = 24 configurations

The best configuration by average nRMSE across 12 episodes is then used as the
ETKF baseline in Phase B1 (100 episode full run).
"""

import sys
import os
import json
import time
import numpy as np
from itertools import product

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from wavesplat.data_loader import EpisodeDataLoader
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

SWEEP = {
    'grid': [(40, 30), (80, 60)],
    'ensemble': [50, 100],
    'mult_inflation': [1.0, 1.02, 1.05],
    'add_inflation': [0.005, 0.02],
}


def run_etkf_one(h5_path, grid_nx, grid_ny, n_ens, mult_inf, add_inf):
    np.random.seed(42)
    loader = EpisodeDataLoader(h5_path)
    gx = loader.gt_grid_x.cpu().numpy()
    gy = loader.gt_grid_y.cpu().numpy()
    egx = np.linspace(gx[0], gx[-1], grid_nx)
    egy = np.linspace(gy[0], gy[-1], grid_ny)
    etkf = ETKFGrid(egx, egy, depth=loader.depth,
                    n_ensemble=n_ens,
                    mult_inflation=mult_inf,
                    add_inflation=add_inf)

    nrmses = []
    hs_errs = []
    ms_list = []
    for frame in loader:
        x = frame.x_obs.cpu().numpy()
        y = frame.y_obs.cpu().numpy()
        z = frame.z_obs.cpu().numpy()
        elapsed = etkf.update(x, y, z)
        pred_grid = etkf.predict_grid()
        interp = RegularGridInterpolator(
            (egy, egx), pred_grid,
            method='linear', bounds_error=False, fill_value=0.0)
        Xg, Yg = np.meshgrid(gx, gy)
        pred = interp(np.column_stack([Yg.ravel(), Xg.ravel()])).reshape(
            len(gy), len(gx))
        gt = frame.gt_field.cpu().numpy()
        m = compute_all_metrics(pred, gt, gx, gy, loader.Hs_true,
                                obs_x=x if len(x) > 0 else None,
                                obs_y=y if len(y) > 0 else None)
        nrmses.append(m['nRMSE'])
        hs_errs.append(m.get('Hs_error', float('nan')))
        ms_list.append(elapsed)

    return {
        'avg_nRMSE': float(np.mean(nrmses)),
        'avg_Hs_error': float(np.mean(hs_errs)),
        'avg_ms': float(np.mean(ms_list)),
        'n_frames': len(nrmses),
    }


def main():
    data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'results')
    os.makedirs(output_dir, exist_ok=True)

    configs = list(product(
        SWEEP['grid'],
        SWEEP['ensemble'],
        SWEEP['mult_inflation'],
        SWEEP['add_inflation'],
    ))
    print(f"ETKF Sweep: {len(configs)} configs × {len(EPISODES)} episodes = "
          f"{len(configs) * len(EPISODES)} runs")
    print('=' * 80)

    all_results = {}
    t_start = time.time()

    for ci, (grid, n_ens, mi, ai) in enumerate(configs):
        cfg_key = f"g{grid[0]}x{grid[1]}_ne{n_ens}_mi{mi}_ai{ai}"
        print(f"\n[{ci + 1}/{len(configs)}] {cfg_key}", flush=True)
        cfg_res = {}
        ep_nrmses = []
        for ep in EPISODES:
            h5 = os.path.join(data_dir, ep)
            if not os.path.exists(h5):
                continue
            short = ep.replace('episode_', '').replace('.h5', '')
            try:
                r = run_etkf_one(h5, grid[0], grid[1], n_ens, mi, ai)
                cfg_res[short] = r
                ep_nrmses.append(r['avg_nRMSE'])
                print(f"  {short:<32} nRMSE={r['avg_nRMSE']:.4f} "
                      f"Hs={r['avg_Hs_error']:.3f} {r['avg_ms']:.1f}ms",
                      flush=True)
            except Exception as e:
                cfg_res[short] = {'error': str(e)}
                print(f"  {short}: FAILED {e}", flush=True)

        mean_nrmse = float(np.mean(ep_nrmses)) if ep_nrmses else float('nan')
        cfg_res['_mean_nRMSE'] = mean_nrmse
        cfg_res['_params'] = {
            'grid': list(grid),
            'n_ensemble': n_ens,
            'mult_inflation': mi,
            'add_inflation': ai,
        }
        all_results[cfg_key] = cfg_res
        elapsed = time.time() - t_start
        print(f"  => mean nRMSE = {mean_nrmse:.4f}  |  elapsed {elapsed:.1f}s",
              flush=True)

        # Save incrementally
        out_path = os.path.join(output_dir, 'etkf_tuning_sweep.json')
        with open(out_path, 'w') as f:
            json.dump(all_results, f, indent=2, default=str)

    # Rank
    print(f"\n{'=' * 80}")
    print("RANKED RESULTS (top 10 by mean nRMSE)")
    print('=' * 80)
    ranked = sorted(all_results.items(),
                    key=lambda kv: kv[1].get('_mean_nRMSE', 1e9))
    for rank, (key, cfg) in enumerate(ranked[:10], 1):
        p = cfg['_params']
        print(f"  {rank:2d}. {key:<45} "
              f"nRMSE={cfg['_mean_nRMSE']:.4f}")

    print(f"\nTotal time: {time.time() - t_start:.1f}s")


if __name__ == '__main__':
    main()
