"""
run_super_resolution_ablation.py
4-way ablation: isolate contributions of (1) multi-frame fusion and (2) dispersion coupling
to super-resolution (resolving wavenumbers beyond per-frame spatial Nyquist).

Variants:
  A: v1 init (spatial FFT)  + no dispersion + single-frame  → Nyquist-limited baseline
  B: v2 init (beamform)     + no dispersion + multi-frame    → temporal helps, but w free
  C: v2 init (beamform)     + dispersion    + single-frame   → dispersion helps init, but 1-frame tracking
  D: v2 init (beamform)     + dispersion    + multi-frame    → full WaveSplat (current best)

Additionally reports:
  - Spatial Nyquist wavenumber k_Nyq for each episode
  - Recovered max |k| from each variant
  - Super-resolution ratio: k_max_recovered / k_Nyq
"""

import sys
import os
import json
import time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from wavesplat.data_loader import EpisodeDataLoader
from wavesplat.run import run_estimation


# Same 12 episodes as run_p1_comparison.py for direct comparability
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

# Ablation configs
VARIANTS = {
    'A: spatial-only': {
        'init_method': 'v1',
        'use_dispersion': False,
        'tracking_window': 1,
        'adaptive_n': False,
        'seed': 42,
    },
    'B: multi-frame, no disp': {
        'init_method': 'v2_konly',
        'use_dispersion': False,
        'tracking_window': 3,
        'adaptive_n': False,
        'seed': 42,
    },
    'C: disp-coupled, single-frame': {
        'init_method': 'v2_konly',
        'use_dispersion': True,
        'tracking_window': 1,
        'adaptive_n': False,
        'seed': 42,
    },
    'D: full WaveSplat': {
        'init_method': 'v2_konly',
        'use_dispersion': True,
        'tracking_window': 3,
        'adaptive_n': True,
        'seed': 42,
    },
}


def estimate_spatial_nyquist(loader):
    """
    Compute the spatial Nyquist wavenumber for v1 (FFT grid) initialization.

    v1 bins sparse LiDAR points onto a 32x24 grid over ~240m x 200m domain.
    The grid spacing determines the FFT Nyquist limit: k_Nyq = pi / dx.
    Beamforming (v2) bypasses this by working on raw irregular point locations.
    """
    # Match initializer.py defaults
    x_range = (10.0, 250.0)
    y_range = (-100.0, 100.0)
    nx, ny = 32, 24

    dx = (x_range[1] - x_range[0]) / nx
    dy = (y_range[1] - y_range[0]) / ny

    # Conservative: use the smaller Nyquist
    k_nyquist = np.pi / max(dx, dy)
    return float(k_nyquist)


def get_gt_max_wavenumber(loader):
    """Get maximum wavenumber from ground truth wave components."""
    wave_comps = loader.data.get('wave_components', None)
    if wave_comps is None:
        return float('nan')

    # wave_components columns: [A, omega, kx, ky, phi, Q]
    kx = wave_comps[:, 2]
    ky = wave_comps[:, 3]
    k_mag = np.sqrt(kx**2 + ky**2)
    return float(np.max(k_mag))


def get_recovered_max_k(result):
    """Extract maximum recovered wavenumber from estimation result."""
    params = result.get('spectrum_params', {})
    if not params:
        return float('nan')

    kx = np.array(params.get('kx', []))
    ky = np.array(params.get('ky', []))
    w = np.array(params.get('w', []))

    if len(kx) == 0:
        return float('nan')

    # Only consider components with non-negligible amplitude
    k_mag = np.sqrt(kx**2 + ky**2)
    if len(w) > 0:
        # Filter: amplitude > 10% of max
        significant = np.abs(w) > 0.1 * np.max(np.abs(w))
        if np.any(significant):
            k_mag = k_mag[significant]

    return float(np.max(k_mag))


def main():
    data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'results')
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 100)
    print("Super-Resolution 4-Way Ablation")
    print("=" * 100)

    all_results = {}

    for ep_name in EPISODES:
        h5_path = os.path.join(data_dir, ep_name)
        if not os.path.exists(h5_path):
            print(f"  SKIP: {ep_name}")
            continue

        short = ep_name.replace('episode_', '').replace('.h5', '')
        print(f"\n{'='*80}")
        print(f"[{short}]")

        # Load metadata
        loader = EpisodeDataLoader(h5_path)
        k_nyquist = estimate_spatial_nyquist(loader)
        k_gt_max = get_gt_max_wavenumber(loader)

        print(f"  k_Nyq = {k_nyquist:.3f} rad/m,  k_GT_max = {k_gt_max:.3f} rad/m,  "
              f"GT super-res ratio = {k_gt_max/k_nyquist:.1f}x")

        ep_results = {
            'k_nyquist': k_nyquist,
            'k_gt_max': k_gt_max,
            'gt_super_res_ratio': k_gt_max / k_nyquist if k_nyquist > 0 else float('nan'),
            'Hs_true': loader.Hs_true,
            'sea_state': loader.sea_state,
            'variants': {},
        }

        for var_name, config in VARIANTS.items():
            try:
                t0 = time.time()
                result = run_estimation(h5_path, config=config)
                elapsed = time.time() - t0

                k_recovered = get_recovered_max_k(result)
                super_res_ratio = k_recovered / k_nyquist if k_nyquist > 0 else float('nan')

                var_result = {
                    'avg_nRMSE': float(result['avg_nRMSE']),
                    'avg_RMSE_z': float(result['avg_RMSE_z']),
                    'avg_inference_ms': float(result['avg_inference_ms']),
                    'init_nrmse': float(result['init_nrmse']),
                    'n_gaussians': result.get('n_gaussians', 15),
                    'k_recovered_max': k_recovered,
                    'super_res_ratio': super_res_ratio,
                    'total_time_s': elapsed,
                }
                ep_results['variants'][var_name] = var_result

                marker = " ***" if var_name.startswith('D') else ""
                print(f"  {var_name:<30s}  nRMSE={var_result['avg_nRMSE']:.4f}  "
                      f"k_max={k_recovered:.3f}  SR={super_res_ratio:.1f}x  "
                      f"{var_result['avg_inference_ms']:.1f}ms{marker}")

            except Exception as e:
                print(f"  {var_name:<30s}  FAILED: {e}")
                ep_results['variants'][var_name] = {'error': str(e)}

        all_results[short] = ep_results

    # Summary table
    print(f"\n{'='*100}")
    print("SUMMARY")
    print(f"{'='*100}")

    variant_names = list(VARIANTS.keys())
    header = f"{'Episode':<30s} {'k_Nyq':>6s} {'k_GT':>6s}"
    for v in variant_names:
        tag = v.split(':')[0]
        header += f" {'nRMSE-'+tag:>10s} {'SR-'+tag:>6s}"
    print(header)
    print("-" * len(header))

    # Collect averages
    avg_nrmse = {v: [] for v in variant_names}
    avg_sr = {v: [] for v in variant_names}

    for ep, res in all_results.items():
        row = f"{ep:<30s} {res['k_nyquist']:>6.2f} {res['k_gt_max']:>6.2f}"
        for v in variant_names:
            vr = res['variants'].get(v, {})
            n = vr.get('avg_nRMSE', float('nan'))
            sr = vr.get('super_res_ratio', float('nan'))
            row += f" {n:>10.4f} {sr:>6.1f}"
            if not np.isnan(n):
                avg_nrmse[v].append(n)
            if not np.isnan(sr):
                avg_sr[v].append(sr)
        print(row)

    print("-" * len(header))
    row = f"{'AVERAGE':<30s} {'':>6s} {'':>6s}"
    for v in variant_names:
        n_avg = np.mean(avg_nrmse[v]) if avg_nrmse[v] else float('nan')
        sr_avg = np.mean(avg_sr[v]) if avg_sr[v] else float('nan')
        row += f" {n_avg:>10.4f} {sr_avg:>6.1f}"
    print(row)

    # Improvement analysis
    print(f"\n--- Improvement Analysis ---")
    for v in variant_names[1:]:
        base = avg_nrmse[variant_names[0]]
        curr = avg_nrmse[v]
        if base and curr:
            imp = (np.mean(base) - np.mean(curr)) / np.mean(base) * 100
            print(f"  {v} vs A: {imp:+.1f}% nRMSE improvement")

    if avg_nrmse[variant_names[-1]] and avg_nrmse[variant_names[0]]:
        d_avg = np.mean(avg_nrmse[variant_names[-1]])
        a_avg = np.mean(avg_nrmse[variant_names[0]])
        print(f"\n  Overall D vs A: {(a_avg - d_avg)/a_avg*100:+.1f}% nRMSE,  "
              f"SR: {np.mean(avg_sr[variant_names[0]]):.1f}x -> {np.mean(avg_sr[variant_names[-1]]):.1f}x")

    # Save results
    out_path = os.path.join(output_dir, 'super_resolution_ablation.json')
    with open(out_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")


if __name__ == '__main__':
    main()
