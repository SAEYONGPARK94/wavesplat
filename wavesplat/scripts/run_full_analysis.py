"""
run_full_analysis.py
Comprehensive analysis: WaveSplat 6-metric, SS bootstrap CI,
LETKF loc_radius sweep, and apples-to-apples comparison table.
"""

import sys
import os
import json
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from wavesplat.run import run_estimation
from wavesplat.data_loader import EpisodeDataLoader
from wavesplat.baselines.letkf_grid import LETKFGrid
from wavesplat.evaluation.metrics import compute_all_metrics
from scipy.interpolate import RegularGridInterpolator

EPISODES = [
    'episode_BF3_2comp_seed0_uniform.h5',
    'episode_BF3_5comp_seed0_uniform.h5',
    'episode_BF4_1comp_seed0_uniform.h5',
    'episode_BF4_5comp_seed0_uniform.h5',
    'episode_BF4_15comp_seed0_uniform.h5',
    'episode_BF5_15comp_seed0_uniform.h5',
]

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')


# ── 1. WaveSplat full 6-metric ─────────────────────────────────────────

def compute_wavesplat_6metric():
    """Extract all 6 metrics from WaveSplat frame_results."""
    print("=" * 80)
    print("TASK 1: WaveSplat full 6-metric panel")
    print("=" * 80)

    ws_results = {}
    for ep_name in EPISODES:
        h5_path = os.path.join(DATA_DIR, ep_name)
        if not os.path.exists(h5_path):
            print(f"  SKIP: {ep_name}")
            continue

        short = ep_name.replace('episode_', '').replace('.h5', '')
        print(f"\n  [{short}]", flush=True)

        config = {
            'init_method': 'v2_konly',
            'use_dispersion': True,
            'use_confidence': True,
            'adaptive_n': True,
            'tracking_window': 3,
            'seed': 42,
        }
        result = run_estimation(h5_path, config=config)
        frames = result['frame_results']

        # Average all 6 metrics across tracking frames
        keys = ['nRMSE', 'Variance_Ratio', 'Pearson_rho', 'Skill_Score',
                'Directional_R2', 'Hs_error']
        avg = {}
        for k in keys:
            vals = [f[k] for f in frames if k in f and not np.isnan(f.get(k, float('nan')))]
            avg[k] = float(np.mean(vals)) if vals else float('nan')
        avg['ms'] = float(np.mean([f['elapsed_ms'] for f in frames]))
        avg['N'] = result.get('n_gaussians', -1)

        ws_results[short] = avg
        print(f"    nRMSE={avg['nRMSE']:.4f}  VR={avg['Variance_Ratio']:.3f}  "
              f"rho={avg['Pearson_rho']:.3f}  SS={avg['Skill_Score']:+.3f}  "
              f"Dir_R2={avg['Directional_R2']:.3f}  Hs_err={avg['Hs_error']:.3f}  "
              f"({avg['ms']:.1f}ms)", flush=True)

    return ws_results


# ── 2. Bootstrap CI for Skill Score ─────────────────────────────────────

def bootstrap_ss_ci(ws_results, n_boot=10000, ci=0.95):
    """Bootstrap confidence interval for mean SS across episodes."""
    print("\n" + "=" * 80)
    print("TASK 2: Bootstrap CI for Skill Score")
    print("=" * 80)

    ss_values = [v['Skill_Score'] for v in ws_results.values()
                 if not np.isnan(v.get('Skill_Score', float('nan')))]

    if len(ss_values) < 2:
        print("  Not enough episodes for bootstrap")
        return {}

    ss_arr = np.array(ss_values)
    observed_mean = np.mean(ss_arr)

    rng = np.random.default_rng(42)
    boot_means = np.array([
        np.mean(rng.choice(ss_arr, size=len(ss_arr), replace=True))
        for _ in range(n_boot)
    ])

    alpha = 1 - ci
    lo = np.percentile(boot_means, 100 * alpha / 2)
    hi = np.percentile(boot_means, 100 * (1 - alpha / 2))

    result = {
        'observed_mean_SS': float(observed_mean),
        'ci_lower': float(lo),
        'ci_upper': float(hi),
        'ci_level': ci,
        'n_episodes': len(ss_values),
        'n_bootstrap': n_boot,
        'includes_zero': bool(lo <= 0 <= hi),
        'per_episode_SS': {k: v['Skill_Score'] for k, v in ws_results.items()},
    }

    print(f"  Mean SS = {observed_mean:+.4f}")
    print(f"  95% CI  = [{lo:+.4f}, {hi:+.4f}]")
    print(f"  CI includes zero: {result['includes_zero']}")
    print(f"  Per-episode SS: {[f'{v:.3f}' for v in ss_values]}")

    return result


# ── 3. LETKF loc_radius sensitivity sweep ──────────────────────────────

def run_letkf_sweep():
    """Sweep loc_radius over [20, 50, 100, 200] m on all 6 episodes."""
    print("\n" + "=" * 80)
    print("TASK 3: LETKF loc_radius sensitivity sweep")
    print("=" * 80)

    radii = [20, 50, 100, 200]
    sweep_results = {r: {} for r in radii}

    for ep_name in EPISODES:
        h5_path = os.path.join(DATA_DIR, ep_name)
        if not os.path.exists(h5_path):
            continue

        short = ep_name.replace('episode_', '').replace('.h5', '')
        print(f"\n  [{short}]", flush=True)

        for radius in radii:
            np.random.seed(42)
            loader = EpisodeDataLoader(h5_path)
            gt_grid_x = loader.gt_grid_x.cpu().numpy()
            gt_grid_y = loader.gt_grid_y.cpu().numpy()

            etkf_nx, etkf_ny = 40, 30
            etkf_gx = np.linspace(gt_grid_x[0], gt_grid_x[-1], etkf_nx)
            etkf_gy = np.linspace(gt_grid_y[0], gt_grid_y[-1], etkf_ny)

            letkf = LETKFGrid(etkf_gx, etkf_gy, depth=loader.depth,
                               n_ensemble=100, mult_inflation=1.0,
                               add_inflation=0.005, loc_radius=float(radius))

            frame_metrics = []
            for frame in loader:
                x = frame.x_obs.cpu().numpy()
                y = frame.y_obs.cpu().numpy()
                z = frame.z_obs.cpu().numpy()

                elapsed = letkf.update(x, y, z)

                pred = letkf.predict_grid()
                interp = RegularGridInterpolator(
                    (etkf_gy, etkf_gx), pred,
                    method='linear', bounds_error=False, fill_value=0.0)
                Xg, Yg = np.meshgrid(gt_grid_x, gt_grid_y)
                pred_full = interp(np.column_stack([Yg.ravel(), Xg.ravel()])).reshape(
                    len(gt_grid_y), len(gt_grid_x))

                gt = frame.gt_field.cpu().numpy()
                metrics = compute_all_metrics(
                    pred_full, gt, gt_grid_x, gt_grid_y, loader.Hs_true,
                    obs_x=x if len(x) > 0 else None,
                    obs_y=y if len(y) > 0 else None)
                metrics['elapsed_ms'] = elapsed
                frame_metrics.append(metrics)

            if frame_metrics:
                avg = {}
                for key in ['nRMSE', 'Variance_Ratio', 'Pearson_rho',
                            'Skill_Score', 'Directional_R2', 'Hs_error', 'elapsed_ms']:
                    vals = [m[key] for m in frame_metrics
                            if not np.isnan(m.get(key, float('nan')))]
                    avg[key] = float(np.mean(vals)) if vals else float('nan')
                sweep_results[radius][short] = avg

            print(f"    loc_radius={radius:>3d}m  nRMSE={avg['nRMSE']:.4f}  "
                  f"SS={avg['Skill_Score']:+.3f}  "
                  f"({avg['elapsed_ms']:.1f}ms)", flush=True)

    # Summary
    print(f"\n  {'LETKF sweep summary':^60}")
    print(f"  {'loc_radius':<12}", end="")
    for r in radii:
        print(f"  {r:>6d}m", end="")
    print()
    print(f"  {'-'*12}", end="")
    for _ in radii:
        print(f"  {'-'*7}", end="")
    print()

    for metric_name in ['nRMSE', 'Skill_Score']:
        print(f"  {metric_name:<12}", end="")
        for r in radii:
            vals = [v[metric_name] for v in sweep_results[r].values()
                    if not np.isnan(v.get(metric_name, float('nan')))]
            avg = np.mean(vals) if vals else float('nan')
            if metric_name == 'Skill_Score':
                print(f"  {avg:>+7.3f}", end="")
            else:
                print(f"  {avg:>7.4f}", end="")
        print()

    return sweep_results


# ── 4. Apples-to-apples comparison table ───────────────────────────────

def print_comparison_table(ws_results, existing_path):
    """Print full 6-metric comparison table."""
    print("\n" + "=" * 120)
    print(f"{'TASK 4: APPLES-TO-APPLES 6-METRIC COMPARISON':^120}")
    print("=" * 120)

    with open(existing_path) as f:
        baseline_results = json.load(f)

    methods = ['WaveSplat', 'LETKF(loc50)', 'ETKF(tuned)', 'GPR-1', 'GPR-disp(3f)']
    metrics = ['nRMSE', 'VR', 'rho', 'SS', 'Dir_R2', 'Hs_err']
    metric_keys_ws = ['nRMSE', 'Variance_Ratio', 'Pearson_rho', 'Skill_Score',
                      'Directional_R2', 'Hs_error']
    metric_keys_bl = ['nRMSE', 'VR', 'rho', 'SS', 'Dir_R2', 'Hs_err']

    # Collect all data
    all_data = {}  # {episode: {method: {metric: val}}}
    for ep_name in EPISODES:
        short = ep_name.replace('episode_', '').replace('.h5', '')
        all_data[short] = {}

        # WaveSplat
        if short in ws_results:
            ws = ws_results[short]
            all_data[short]['WaveSplat'] = {
                m: ws.get(mk, float('nan'))
                for m, mk in zip(metrics, metric_keys_ws)
            }
            all_data[short]['WaveSplat']['ms'] = ws.get('ms', float('nan'))

        # Baselines
        if short in baseline_results:
            for method in methods[1:]:
                bl = baseline_results[short].get(method, {})
                all_data[short][method] = {
                    m: bl.get(mk, float('nan'))
                    for m, mk in zip(metrics, metric_keys_bl)
                }
                all_data[short][method]['ms'] = bl.get('ms', float('nan'))

    # Print per-metric tables
    for metric in metrics:
        print(f"\n  {metric:^80}")
        header = f"  {'Episode':<28}"
        for m in methods:
            header += f"  {m:>14}"
        print(header)
        print(f"  {'-'*28}" + f"  {'-'*14}" * len(methods))

        method_vals = {m: [] for m in methods}
        for ep, data in all_data.items():
            row = f"  {ep:<28}"
            for m in methods:
                val = data.get(m, {}).get(metric, float('nan'))
                if not np.isnan(val):
                    method_vals[m].append(val)
                if metric == 'SS':
                    row += f"  {val:>+14.4f}"
                else:
                    row += f"  {val:>14.4f}"
            print(row)

        # Average
        avg_row = f"  {'AVERAGE':<28}"
        for m in methods:
            vals = method_vals[m]
            avg = np.mean(vals) if vals else float('nan')
            if metric == 'SS':
                avg_row += f"  {avg:>+14.4f}"
            else:
                avg_row += f"  {avg:>14.4f}"
        print(avg_row)

    # Speed comparison
    print(f"\n  {'Inference time (ms)':^80}")
    header = f"  {'Episode':<28}"
    for m in methods:
        header += f"  {m:>14}"
    print(header)
    print(f"  {'-'*28}" + f"  {'-'*14}" * len(methods))

    ms_vals = {m: [] for m in methods}
    for ep, data in all_data.items():
        row = f"  {ep:<28}"
        for m in methods:
            val = data.get(m, {}).get('ms', float('nan'))
            if not np.isnan(val):
                ms_vals[m].append(val)
            row += f"  {val:>14.1f}"
        print(row)

    avg_row = f"  {'AVERAGE':<28}"
    for m in methods:
        vals = ms_vals[m]
        avg = np.mean(vals) if vals else float('nan')
        avg_row += f"  {avg:>14.1f}"
    print(avg_row)

    return all_data


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    existing_path = os.path.join(OUTPUT_DIR, 'strong_baseline_comparison.json')

    # Task 1
    ws_results = compute_wavesplat_6metric()

    # Task 2
    ss_ci = bootstrap_ss_ci(ws_results)

    # Task 3
    letkf_sweep = run_letkf_sweep()

    # Task 4
    comparison = print_comparison_table(ws_results, existing_path)

    # Save all results
    output = {
        'wavesplat_6metric': ws_results,
        'ss_bootstrap_ci': ss_ci,
        'letkf_loc_sweep': {str(k): v for k, v in letkf_sweep.items()},
    }

    out_path = os.path.join(OUTPUT_DIR, 'full_analysis.json')
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nAll results saved to {out_path}")


if __name__ == '__main__':
    main()
