"""
run_12ep_full_analysis.py
Full 6-metric analysis on all 12 main-comparison episodes.
WaveSplat + LETKF(loc50) + ETKF(tuned) + GPR-1 + GPR-disp(3f)
"""

import sys
import os
import json
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from wavesplat.run import run_estimation
from wavesplat.data_loader import EpisodeDataLoader
from wavesplat.baselines.etkf_grid import ETKFGrid
from wavesplat.baselines.letkf_grid import LETKFGrid
from wavesplat.baselines.gpr_baseline import GPRBaseline
from wavesplat.baselines.gpr_dispersion import GPRDispersion
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

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')

METRIC_KEYS = ['nRMSE', 'Variance_Ratio', 'Pearson_rho', 'Skill_Score',
               'Directional_R2', 'Hs_error']
SHORT_KEYS = ['nRMSE', 'VR', 'rho', 'SS', 'Dir_R2', 'Hs_err']


def run_wavesplat(h5_path):
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

    avg = {}
    for k in METRIC_KEYS:
        vals = [f[k] for f in frames if k in f and not np.isnan(f.get(k, float('nan')))]
        avg[k] = float(np.mean(vals)) if vals else float('nan')
    avg['elapsed_ms'] = float(np.mean([f['elapsed_ms'] for f in frames]))
    avg['N'] = result.get('n_gaussians', -1)
    return avg


def run_grid_baseline(h5_path, baseline_class, baseline_kwargs):
    np.random.seed(42)
    loader = EpisodeDataLoader(h5_path)
    gt_grid_x = loader.gt_grid_x.cpu().numpy()
    gt_grid_y = loader.gt_grid_y.cpu().numpy()

    etkf_nx, etkf_ny = 40, 30
    etkf_gx = np.linspace(gt_grid_x[0], gt_grid_x[-1], etkf_nx)
    etkf_gy = np.linspace(gt_grid_y[0], gt_grid_y[-1], etkf_ny)

    baseline = baseline_class(etkf_gx, etkf_gy, depth=loader.depth,
                              **baseline_kwargs)

    frame_metrics = []
    for frame in loader:
        x = frame.x_obs.cpu().numpy()
        y = frame.y_obs.cpu().numpy()
        z = frame.z_obs.cpu().numpy()

        elapsed = baseline.update(x, y, z)

        pred = baseline.predict_grid()
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
        for key in METRIC_KEYS + ['elapsed_ms']:
            vals = [m[key] for m in frame_metrics
                    if not np.isnan(m.get(key, float('nan')))]
            avg[key] = float(np.mean(vals)) if vals else float('nan')
        return avg
    return {}


def run_gpr_baseline(h5_path, gpr_class, gpr_kwargs):
    np.random.seed(42)
    loader = EpisodeDataLoader(h5_path)
    gt_grid_x = loader.gt_grid_x.cpu().numpy()
    gt_grid_y = loader.gt_grid_y.cpu().numpy()

    gpr = gpr_class(**gpr_kwargs)

    frame_metrics = []
    for frame in loader:
        x = frame.x_obs.cpu().numpy()
        y = frame.y_obs.cpu().numpy()
        z = frame.z_obs.cpu().numpy()

        elapsed = gpr.update(x, y, z)
        pred = gpr.predict_grid(gt_grid_x, gt_grid_y)

        gt = frame.gt_field.cpu().numpy()
        metrics = compute_all_metrics(
            pred, gt, gt_grid_x, gt_grid_y, loader.Hs_true,
            obs_x=x if len(x) > 0 else None,
            obs_y=y if len(y) > 0 else None)
        metrics['elapsed_ms'] = elapsed
        frame_metrics.append(metrics)

    if frame_metrics:
        avg = {}
        for key in METRIC_KEYS + ['elapsed_ms']:
            vals = [m[key] for m in frame_metrics
                    if not np.isnan(m.get(key, float('nan')))]
            avg[key] = float(np.mean(vals)) if vals else float('nan')
        return avg
    return {}


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    methods = {
        'WaveSplat': None,
        'ETKF(tuned)': ('grid', ETKFGrid,
                         {'n_ensemble': 100, 'mult_inflation': 1.0,
                          'add_inflation': 0.005}),
        'LETKF(loc50)': ('grid', LETKFGrid,
                          {'n_ensemble': 100, 'mult_inflation': 1.0,
                           'add_inflation': 0.005, 'loc_radius': 50.0}),
        'GPR-1': ('gpr', GPRBaseline,
                  {'length_scale': 20.0, 'noise_sigma': 0.02,
                   'window_frames': 1}),
        'GPR-disp(3f)': ('gpr', GPRDispersion,
                          {'length_scale': 20.0, 'noise_sigma': 0.02,
                           'window_frames': 3, 'depth': 50.0}),
    }

    method_names = list(methods.keys())
    all_results = {}

    print(f"12-Episode Full 6-Metric Analysis")
    print(f"Episodes: {len(EPISODES)}")
    print(f"Methods: {method_names}")
    print(f"{'='*120}")

    for ep_name in EPISODES:
        h5_path = os.path.join(DATA_DIR, ep_name)
        if not os.path.exists(h5_path):
            print(f"  SKIP: {ep_name}")
            continue

        short = ep_name.replace('episode_', '').replace('.h5', '')
        print(f"\n[{short}]", flush=True)
        ep_results = {}

        # WaveSplat
        try:
            ws = run_wavesplat(h5_path)
            ep_results['WaveSplat'] = {
                sk: ws.get(mk, float('nan'))
                for sk, mk in zip(SHORT_KEYS, METRIC_KEYS)
            }
            ep_results['WaveSplat']['ms'] = ws.get('elapsed_ms', float('nan'))
            ep_results['WaveSplat']['N'] = ws.get('N', -1)
            print(f"  WaveSplat:     nRMSE={ws['nRMSE']:.4f}  "
                  f"SS={ws['Skill_Score']:+.3f}  ({ws['elapsed_ms']:.1f}ms)",
                  flush=True)
        except Exception as e:
            print(f"  WaveSplat FAILED: {e}")
            ep_results['WaveSplat'] = {'error': str(e)}

        # Baselines
        for name, spec in methods.items():
            if spec is None:
                continue
            mtype, cls, kwargs = spec

            try:
                if mtype == 'grid':
                    res = run_grid_baseline(h5_path, cls, kwargs)
                else:
                    res = run_gpr_baseline(h5_path, cls, kwargs)

                ep_results[name] = {
                    sk: res.get(mk, float('nan'))
                    for sk, mk in zip(SHORT_KEYS, METRIC_KEYS)
                }
                ep_results[name]['ms'] = res.get('elapsed_ms', float('nan'))
                print(f"  {name:<14s} nRMSE={res.get('nRMSE', float('nan')):.4f}  "
                      f"SS={res.get('Skill_Score', float('nan')):+.3f}  "
                      f"({res.get('elapsed_ms', float('nan')):.1f}ms)", flush=True)
            except Exception as e:
                print(f"  {name} FAILED: {e}")
                ep_results[name] = {'error': str(e)}

        all_results[short] = ep_results

    # ── Summary tables ──────────────────────────────────────────────
    print(f"\n{'='*120}")
    print(f"{'12-EPISODE SUMMARY':^120}")
    print(f"{'='*120}")

    for metric, short_key in zip(METRIC_KEYS, SHORT_KEYS):
        print(f"\n  {short_key:^100}")
        header = f"  {'Episode':<30}"
        for m in method_names:
            header += f"  {m:>14}"
        print(header)
        print(f"  {'-'*30}" + f"  {'-'*14}" * len(method_names))

        method_vals = {m: [] for m in method_names}
        for ep, res in all_results.items():
            row = f"  {ep:<30}"
            for m in method_names:
                val = res.get(m, {}).get(short_key, float('nan'))
                if not np.isnan(val):
                    method_vals[m].append(val)
                if short_key == 'SS':
                    row += f"  {val:>+14.4f}"
                else:
                    row += f"  {val:>14.4f}"
            print(row)

        avg_row = f"  {'AVERAGE':<30}"
        for m in method_names:
            vals = method_vals[m]
            avg = np.mean(vals) if vals else float('nan')
            if short_key == 'SS':
                avg_row += f"  {avg:>+14.4f}"
            else:
                avg_row += f"  {avg:>14.4f}"
        print(avg_row)

    # Speed
    print(f"\n  {'Speed (ms)':^100}")
    header = f"  {'Episode':<30}"
    for m in method_names:
        header += f"  {m:>14}"
    print(header)
    print(f"  {'-'*30}" + f"  {'-'*14}" * len(method_names))

    ms_vals = {m: [] for m in method_names}
    for ep, res in all_results.items():
        row = f"  {ep:<30}"
        for m in method_names:
            val = res.get(m, {}).get('ms', float('nan'))
            if not np.isnan(val):
                ms_vals[m].append(val)
            row += f"  {val:>14.1f}"
        print(row)

    avg_row = f"  {'AVERAGE':<30}"
    for m in method_names:
        vals = ms_vals[m]
        avg = np.mean(vals) if vals else float('nan')
        avg_row += f"  {avg:>14.1f}"
    print(avg_row)

    # ── Bootstrap CI for SS ─────────────────────────────────────────
    print(f"\n{'='*80}")
    print(f"Bootstrap CI for Skill Score (12 episodes)")
    print(f"{'='*80}")

    for m in method_names:
        ss_vals = []
        for ep, res in all_results.items():
            val = res.get(m, {}).get('SS', float('nan'))
            if not np.isnan(val):
                ss_vals.append(val)

        if len(ss_vals) < 2:
            print(f"  {m}: insufficient data")
            continue

        ss_arr = np.array(ss_vals)
        rng = np.random.default_rng(42)
        boot_means = np.array([
            np.mean(rng.choice(ss_arr, size=len(ss_arr), replace=True))
            for _ in range(10000)
        ])
        lo = np.percentile(boot_means, 2.5)
        hi = np.percentile(boot_means, 97.5)
        includes_zero = lo <= 0 <= hi

        print(f"  {m:<14s}  mean={np.mean(ss_arr):+.4f}  "
              f"95%CI=[{lo:+.4f}, {hi:+.4f}]  "
              f"includes_zero={includes_zero}")

    # ── Save ────────────────────────────────────────────────────────
    out_path = os.path.join(OUTPUT_DIR, '12ep_full_analysis.json')
    with open(out_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")


if __name__ == '__main__':
    main()
