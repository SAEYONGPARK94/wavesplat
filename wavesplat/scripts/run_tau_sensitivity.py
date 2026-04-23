"""
run_tau_sensitivity.py
τ_R sensitivity sweep: grow_residual_threshold ∈ {0.02, 0.05, 0.08, 0.10, 0.15} m

목적: τ_R 선택의 robustness 입증 (논문 §3.2.2 뒷받침)
에피소드: 12 representative episodes (p1_comparison과 동일)
결과: results/tau_sensitivity.json
"""

import sys
import os
import json
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from wavesplat.run import run_estimation, set_seed

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

TAU_VALUES = [0.02, 0.05, 0.08, 0.10, 0.15]


def run_wavesplat(h5_path, tau_r):
    config = {
        'init_method': 'v2_konly',
        'use_dispersion': True,
        'use_confidence': True,
        'adaptive_n': True,
        'tracking_window': 3,
        'grow_residual_threshold': tau_r,
        'seed': 42,
    }
    result = run_estimation(h5_path, config=config)
    return {
        'avg_nRMSE': float(result['avg_nRMSE']),
        'avg_inference_ms': float(result['avg_inference_ms']),
        'final_n': result.get('n_gaussians', 15),
    }


def main():
    data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'results')
    os.makedirs(output_dir, exist_ok=True)

    results = {}  # {episode: {tau: {avg_nRMSE, ...}}}

    print("τ_R Sensitivity Sweep")
    print(f"τ values: {TAU_VALUES}")
    print("=" * 80)

    header = f"{'Episode':<28}" + "".join(f"  τ={t:.2f}" for t in TAU_VALUES)
    print(header)
    print("-" * 80)

    for ep_name in EPISODES:
        h5_path = os.path.join(data_dir, ep_name)
        if not os.path.exists(h5_path):
            print(f"SKIP: {ep_name}")
            continue

        short = ep_name.replace('episode_', '').replace('.h5', '').replace('_seed0', '')
        results[short] = {}
        row = f"{short:<28}"

        for tau in TAU_VALUES:
            try:
                set_seed(42)
                r = run_wavesplat(h5_path, tau)
                results[short][str(tau)] = r
                row += f"  {r['avg_nRMSE']:.4f}"
            except Exception as e:
                results[short][str(tau)] = {'error': str(e)}
                row += f"  {'ERR':>6}"
                print(f"  [{short}] τ={tau} FAILED: {e}")

        print(row, flush=True)

    # Summary: per-tau average
    print("\n" + "=" * 80)
    print("Per-τ average nRMSE:")
    print(f"{'τ_R':>6}  {'avg nRMSE':>10}  {'vs τ=0.05':>10}  {'N_avg':>6}")
    print("-" * 40)

    tau_avgs = {}
    tau_n_avgs = {}
    for tau in TAU_VALUES:
        vals = []
        n_vals = []
        for ep_res in results.values():
            r = ep_res.get(str(tau), {})
            if 'avg_nRMSE' in r and not np.isnan(r['avg_nRMSE']):
                vals.append(r['avg_nRMSE'])
            if 'final_n' in r:
                n_vals.append(r['final_n'])
        tau_avgs[tau] = np.mean(vals) if vals else float('nan')
        tau_n_avgs[tau] = np.mean(n_vals) if n_vals else float('nan')

    ref = tau_avgs.get(0.05, float('nan'))
    for tau in TAU_VALUES:
        avg = tau_avgs[tau]
        delta = (avg - ref) / ref * 100 if not np.isnan(ref) and ref > 0 else float('nan')
        marker = " ← baseline" if tau == 0.05 else ""
        print(f"{tau:>6.2f}  {avg:>10.4f}  {delta:>+10.1f}%  {tau_n_avgs[tau]:>6.1f}{marker}")

    out_path = os.path.join(output_dir, 'tau_sensitivity.json')
    with open(out_path, 'w') as f:
        json.dump({'tau_values': TAU_VALUES, 'results': results,
                   'summary': {str(t): tau_avgs[t] for t in TAU_VALUES}}, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == '__main__':
    main()
