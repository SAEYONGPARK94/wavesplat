#!/usr/bin/env python3
"""
Compare RELAX init vs Beamforming init: end-to-end WaveSplat pipeline.

Runs the full WaveSplat estimation (init + tracking) with both init methods
on all available episodes and compares nRMSE, correlation, and timing.

Usage:
    cd wavesplat && python3 -u -m scripts.compare_relax_init_e2e
"""

import sys
import os
import json
import time
import numpy as np

sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', buffering=1)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from wavesplat.run import run_estimation


EPISODES = [
    'episode_BF3_1comp_seed0_slope.h5',
    'episode_BF3_2comp_seed0_slope.h5',
    'episode_BF3_5comp_seed0_slope.h5',
    'episode_BF3_5comp_seed0_uniform.h5',
    'episode_BF4_1comp_seed0_slope.h5',
    'episode_BF4_5comp_seed0_uniform.h5',
    'episode_BF4_5comp_seed0_slope.h5',
    'episode_BF4_15comp_seed0_slope.h5',
    'episode_BF4_15comp_seed0_uniform.h5',
    'episode_BF4_30comp_seed0_uniform.h5',
    'episode_BF5_15comp_seed0_uniform.h5',
    'episode_BF5_30comp_seed0_uniform.h5',
]


def run_with_init(h5_path, init_method):
    """Run full WaveSplat with specified init method."""
    config = {
        'init_method': init_method,
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
        'init_nRMSE': float(result.get('init_nRMSE', 0)),
        'final_n': result.get('n_gaussians', 15),
    }


def main():
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            'data')

    print("=" * 80)
    print("RELAX Init vs Beamforming Init: End-to-End WaveSplat Comparison")
    print("=" * 80)

    all_results = []

    for ep_name in EPISODES:
        ep_path = os.path.join(data_dir, ep_name)
        if not os.path.exists(ep_path):
            print(f"\n  {ep_name}: NOT FOUND, skipping")
            continue

        print(f"\n[{ep_name}]")

        # BF init (v2_konly baseline)
        t0 = time.time()
        bf_result = run_with_init(ep_path, 'v2_konly')
        bf_total = time.time() - t0
        print(f"  BF:    nRMSE={bf_result['avg_nRMSE']:.4f}, "
              f"init={bf_result['init_nRMSE']:.4f}, "
              f"{bf_result['avg_inference_ms']:.1f}ms/f, "
              f"N={bf_result['final_n']}, {bf_total:.1f}s")

        # RELAX init
        t0 = time.time()
        relax_result = run_with_init(ep_path, 'relax')
        relax_total = time.time() - t0
        print(f"  RELAX: nRMSE={relax_result['avg_nRMSE']:.4f}, "
              f"init={relax_result['init_nRMSE']:.4f}, "
              f"{relax_result['avg_inference_ms']:.1f}ms/f, "
              f"N={relax_result['final_n']}, {relax_total:.1f}s")

        # Improvement
        if bf_result['avg_nRMSE'] > 1e-6:
            improv = (bf_result['avg_nRMSE'] - relax_result['avg_nRMSE']) / bf_result['avg_nRMSE'] * 100
            print(f"  → nRMSE change: {improv:+.1f}%")

        all_results.append({
            'episode': ep_name,
            'bf_nRMSE': bf_result['avg_nRMSE'],
            'bf_init_nRMSE': bf_result['init_nRMSE'],
            'bf_ms': bf_result['avg_inference_ms'],
            'bf_n': bf_result['final_n'],
            'relax_nRMSE': relax_result['avg_nRMSE'],
            'relax_init_nRMSE': relax_result['init_nRMSE'],
            'relax_ms': relax_result['avg_inference_ms'],
            'relax_n': relax_result['final_n'],
        })

    # Summary
    if all_results:
        print("\n" + "=" * 80)
        print("SUMMARY")
        print("=" * 80)

        bf_nrmses = [r['bf_nRMSE'] for r in all_results]
        relax_nrmses = [r['relax_nRMSE'] for r in all_results]
        bf_init = [r['bf_init_nRMSE'] for r in all_results]
        relax_init = [r['relax_init_nRMSE'] for r in all_results]

        print(f"\n  BF init:    avg nRMSE={np.mean(bf_nrmses):.4f}, "
              f"avg init={np.mean(bf_init):.4f}")
        print(f"  RELAX init: avg nRMSE={np.mean(relax_nrmses):.4f}, "
              f"avg init={np.mean(relax_init):.4f}")

        n_better = sum(1 for r in all_results if r['relax_nRMSE'] < r['bf_nRMSE'])
        print(f"\n  RELAX wins {n_better}/{len(all_results)} episodes on tracking nRMSE")

        n_init_better = sum(1 for r in all_results if r['relax_init_nRMSE'] < r['bf_init_nRMSE'])
        print(f"  RELAX wins {n_init_better}/{len(all_results)} episodes on init nRMSE")

        overall_improv = (np.mean(bf_nrmses) - np.mean(relax_nrmses)) / np.mean(bf_nrmses) * 100
        print(f"\n  Overall nRMSE improvement: {overall_improv:+.1f}%")

        # Save results
        results_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                    'results')
        os.makedirs(results_dir, exist_ok=True)
        out_path = os.path.join(results_dir, 'relax_init_e2e.json')
        with open(out_path, 'w') as f:
            json.dump({
                'summary': {
                    'bf_avg_nRMSE': float(np.mean(bf_nrmses)),
                    'relax_avg_nRMSE': float(np.mean(relax_nrmses)),
                    'improvement_pct': float(overall_improv),
                    'relax_wins': n_better,
                    'total_episodes': len(all_results),
                },
                'episodes': all_results,
            }, f, indent=2)
        print(f"\n  Results saved to {out_path}")


if __name__ == '__main__':
    main()
