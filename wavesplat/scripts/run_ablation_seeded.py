"""
Seeded ablation experiment for P0-2.
Runs 4 configs × episodes × 3 algorithm seeds to assess:
  1. Dispersion vs no-dispersion (A1)
  2. Confidence vs no-confidence (A2)
  3. Run-to-run variance (multi-seed)
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import json
import numpy as np
from wavesplat.run import run_estimation

EPISODES = [
    'data/episode_BF3_1comp_seed0_uniform.h5',
    'data/episode_BF3_1comp_seed0_slope.h5',
    'data/episode_BF3_2comp_seed0_uniform.h5',
    'data/episode_BF3_2comp_seed0_slope.h5',
    'data/episode_BF3_5comp_seed0_uniform.h5',
    'data/episode_BF3_5comp_seed0_slope.h5',
]

CONFIGS = {
    'Full': {'use_dispersion': True, 'use_confidence': True, 'init_method': 'v1'},
    'A1_noDisp': {'use_dispersion': False, 'use_confidence': True, 'init_method': 'v1'},
    'A2_noConf': {'use_dispersion': True, 'use_confidence': False, 'init_method': 'v1'},
    'v2+disp': {'use_dispersion': True, 'use_confidence': True, 'init_method': 'v2'},
}

SEEDS = [42, 123, 456]

results = []

for ep in EPISODES:
    ep_name = os.path.basename(ep).replace('episode_', '').replace('.h5', '')
    for cfg_name, cfg_overrides in CONFIGS.items():
        seed_metrics = []
        for seed in SEEDS:
            config = {**cfg_overrides, 'seed': seed}
            try:
                r = run_estimation(ep, config, verbose=False)
                seed_metrics.append({
                    'seed': seed,
                    'avg_nRMSE': r['avg_nRMSE'],
                    'avg_RMSE_z': r['avg_RMSE_z'],
                    'avg_Hs_error': r['avg_Hs_error'],
                    'init_nrmse': r['init_nrmse'],
                    'init_rmse': r['init_rmse'],
                    'T_converge': r['T_converge'],
                    'avg_inference_ms': r['avg_inference_ms'],
                })
            except Exception as e:
                print(f"  ERROR: {ep_name} / {cfg_name} / seed={seed}: {e}")
                seed_metrics.append({'seed': seed, 'error': str(e)})

        # Compute mean/std across seeds
        valid = [m for m in seed_metrics if 'error' not in m]
        if valid:
            mean_nrmse = np.mean([m['avg_nRMSE'] for m in valid])
            std_nrmse = np.std([m['avg_nRMSE'] for m in valid])
            mean_hs = np.mean([m['avg_Hs_error'] for m in valid])
            mean_init = np.mean([m['init_nrmse'] for m in valid])
            print(f"{ep_name:30s} | {cfg_name:12s} | "
                  f"nRMSE={mean_nrmse:.4f}±{std_nrmse:.4f} | "
                  f"Hs_err={mean_hs:.4f} | init_nRMSE={mean_init:.4f}")
        else:
            print(f"{ep_name:30s} | {cfg_name:12s} | ALL FAILED")

        results.append({
            'episode': ep_name,
            'config': cfg_name,
            'seed_results': seed_metrics,
        })

# Save full results
os.makedirs('results', exist_ok=True)
with open('results/ablation_seeded.json', 'w') as f:
    json.dump(results, f, indent=2, default=str)

print("\nResults saved to results/ablation_seeded.json")
