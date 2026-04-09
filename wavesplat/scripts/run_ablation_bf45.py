"""
run_ablation_bf45.py
Phase B3: Dispersion constraint ablation on BF4 and BF5 episodes.

Motivation: The existing BF3 ablation (run_ablation_seeded.py) showed A1 (noDisp)
slightly better than Full, which Rev.2 flagged as a weakness in the dispersion
constraint argument. We expect BF4/BF5 (more spectral crowding) to show
unconstrained omega estimation failing.

Config matches paper main results: init_method='v2_konly', use_confidence=True,
adaptive_n=True, tracking_window=3.
"""
import sys
import os
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from wavesplat.run import run_estimation

# BF4/BF5 episodes (all 5 seeds, uniform + slope)
EPISODES = []
for bf in ['BF4', 'BF5']:
    for ncomp in ['5comp', '15comp', '30comp']:
        for seed in range(5):
            for ray in ['uniform', 'slope']:
                EPISODES.append(
                    f'data/episode_{bf}_{ncomp}_seed{seed}_{ray}.h5')

CONFIGS = {
    'Full':       {'use_dispersion': True,  'use_confidence': True,
                   'init_method': 'v2_konly', 'adaptive_n': True,
                   'tracking_window': 3},
    'A1_noDisp':  {'use_dispersion': False, 'use_confidence': True,
                   'init_method': 'v2_konly', 'adaptive_n': True,
                   'tracking_window': 3},
    'A2_noConf':  {'use_dispersion': True,  'use_confidence': False,
                   'init_method': 'v2_konly', 'adaptive_n': True,
                   'tracking_window': 3},
}

SEEDS = [42, 123, 456]

results = []
print(f"BF4-5 Ablation: {len(EPISODES)} episodes × {len(CONFIGS)} configs × "
      f"{len(SEEDS)} seeds = {len(EPISODES) * len(CONFIGS) * len(SEEDS)} runs")
print('=' * 80)

for ep in EPISODES:
    if not os.path.exists(ep):
        continue
    ep_name = os.path.basename(ep).replace('episode_', '').replace('.h5', '')
    for cfg_name, cfg_overrides in CONFIGS.items():
        seed_metrics = []
        for seed in SEEDS:
            config = {**cfg_overrides, 'seed': seed}
            try:
                r = run_estimation(ep, config, verbose=False)
                seed_metrics.append({
                    'seed': seed,
                    'avg_nRMSE': float(r['avg_nRMSE']),
                    'avg_RMSE_z': float(r['avg_RMSE_z']),
                    'avg_Hs_error': float(r['avg_Hs_error']),
                    'init_nrmse': float(r['init_nrmse']),
                    'T_converge': r['T_converge'],
                    'avg_inference_ms': float(r['avg_inference_ms']),
                })
            except Exception as e:
                seed_metrics.append({'seed': seed, 'error': str(e)})

        valid = [m for m in seed_metrics if 'error' not in m]
        if valid:
            mean_nrmse = np.mean([m['avg_nRMSE'] for m in valid])
            std_nrmse = np.std([m['avg_nRMSE'] for m in valid])
            mean_hs = np.mean([m['avg_Hs_error'] for m in valid])
            print(f"{ep_name:30s} | {cfg_name:12s} | "
                  f"nRMSE={mean_nrmse:.4f}±{std_nrmse:.4f} | "
                  f"Hs={mean_hs:.3f}", flush=True)
        else:
            print(f"{ep_name:30s} | {cfg_name:12s} | ALL FAILED", flush=True)

        results.append({
            'episode': ep_name,
            'config': cfg_name,
            'seed_results': seed_metrics,
        })

    # Incremental save
    os.makedirs('results', exist_ok=True)
    with open('results/ablation_bf45.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)

# Summary: mean nRMSE per config
print(f"\n{'=' * 80}\nSUMMARY (BF4-5 mean nRMSE per config)")
print('=' * 80)
for cfg_name in CONFIGS:
    nrmses = []
    for r in results:
        if r['config'] == cfg_name:
            valid = [m for m in r['seed_results'] if 'error' not in m]
            if valid:
                nrmses.append(np.mean([m['avg_nRMSE'] for m in valid]))
    if nrmses:
        print(f"  {cfg_name:12s}: {np.mean(nrmses):.4f} ± {np.std(nrmses):.4f} "
              f"(n={len(nrmses)})")

print("\nResults saved to results/ablation_bf45.json")
