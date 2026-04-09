"""
run_b1_full_comparison.py
Phase B1: Full 100-episode comparison across 4 methods.

Methods:
  1. WaveSplat (v2_konly init + adaptive N + tracking_window=3)
  2. ETKF (best config from Phase B2 sweep — default to fine 80x60, ne=100, mi=1.02, ai=0.005)
  3. GPR-single (window_frames=1)
  4. GPR-sliding (window_frames=3)

All pre-registered metrics (nRMSE/RMSE/Hs_err/VR/rho/SS/RMSE_far) computed per frame.
Outputs: results/b1_full_comparison.json

Usage:
  python3 scripts/run_b1_full_comparison.py  # all 100 episodes
  python3 scripts/run_b1_full_comparison.py --bf BF4  # BF4 only
  python3 scripts/run_b1_full_comparison.py --max 20  # first 20 episodes
"""

import sys
import os
import json
import time
import argparse
import warnings
import numpy as np

warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from wavesplat.data_loader import EpisodeDataLoader
from wavesplat.run import run_estimation
from wavesplat.baselines.etkf_grid import ETKFGrid
from wavesplat.baselines.gpr_baseline import GPRBaseline
from wavesplat.evaluation.metrics import compute_all_metrics
from scipy.interpolate import RegularGridInterpolator


# ETKF best config — confirmed from Phase B2 sweep (24 configs × 12 eps)
# Winner: g40x30_ne100_mi1.0_ai0.005, mean nRMSE = 0.3584 across 12 episodes
ETKF_BEST = {
    'grid': (40, 30),
    'n_ensemble': 100,
    'mult_inflation': 1.0,
    'add_inflation': 0.005,
}


def _metrics_keys():
    return ['nRMSE', 'RMSE_z', 'Hs_error', 'Variance_Ratio',
            'Pearson_rho', 'Skill_Score', 'RMSE_far',
            'RMSE_unobserved', 'Directional_R2']


def _agg_metrics(frame_metrics):
    out = {}
    for k in _metrics_keys():
        vals = [m.get(k, float('nan')) for m in frame_metrics]
        vals = [v for v in vals if v is not None and not np.isnan(v)]
        out[f'avg_{k}'] = float(np.mean(vals)) if vals else float('nan')
    ms = [m.get('elapsed_ms', float('nan')) for m in frame_metrics]
    ms = [v for v in ms if not np.isnan(v)]
    out['avg_inference_ms'] = float(np.mean(ms)) if ms else float('nan')
    out['n_frames'] = len(frame_metrics)
    return out


def run_wavesplat(h5_path):
    config = {
        'init_method': 'v2_konly',
        'use_dispersion': True,
        'use_confidence': True,
        'adaptive_n': True,
        'tracking_window': 3,
        'seed': 42,
    }
    r = run_estimation(h5_path, config=config, verbose=False)
    frame_results = r.get('frame_results', [])
    agg = _agg_metrics(frame_results) if frame_results else {}
    if not agg:
        agg = {
            'avg_nRMSE': float(r.get('avg_nRMSE', float('nan'))),
            'avg_RMSE_z': float(r.get('avg_RMSE_z', float('nan'))),
            'avg_Hs_error': float(r.get('avg_Hs_error', float('nan'))),
            'avg_inference_ms': float(r.get('avg_inference_ms', float('nan'))),
            'n_frames': r.get('tracking_frames', 0),
        }
    agg['T_converge'] = r.get('T_converge')
    return agg


def run_etkf(h5_path, cfg):
    np.random.seed(42)
    loader = EpisodeDataLoader(h5_path)
    gx = loader.gt_grid_x.cpu().numpy()
    gy = loader.gt_grid_y.cpu().numpy()
    egx = np.linspace(gx[0], gx[-1], cfg['grid'][0])
    egy = np.linspace(gy[0], gy[-1], cfg['grid'][1])
    etkf = ETKFGrid(egx, egy, depth=loader.depth,
                    n_ensemble=cfg['n_ensemble'],
                    mult_inflation=cfg['mult_inflation'],
                    add_inflation=cfg['add_inflation'])

    frame_results = []
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
        m['elapsed_ms'] = elapsed
        frame_results.append(m)
    return _agg_metrics(frame_results)


def run_gpr(h5_path, window_frames=1):
    np.random.seed(42)
    loader = EpisodeDataLoader(h5_path)
    gx = loader.gt_grid_x.cpu().numpy()
    gy = loader.gt_grid_y.cpu().numpy()
    gpr = GPRBaseline(kernel_type='matern', length_scale=20.0,
                      noise_sigma=0.05, window_frames=window_frames)

    frame_results = []
    for frame in loader:
        x = frame.x_obs.cpu().numpy()
        y = frame.y_obs.cpu().numpy()
        z = frame.z_obs.cpu().numpy()
        elapsed = gpr.update(x, y, z)
        pred = gpr.predict_grid(gx, gy)
        gt = frame.gt_field.cpu().numpy()
        m = compute_all_metrics(pred, gt, gx, gy, loader.Hs_true,
                                obs_x=x if len(x) > 0 else None,
                                obs_y=y if len(y) > 0 else None)
        m['elapsed_ms'] = elapsed
        frame_results.append(m)
    return _agg_metrics(frame_results)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--bf', type=str, default=None,
                    help='Filter by BF (e.g., BF4)')
    ap.add_argument('--max', type=int, default=None,
                    help='Max episodes to run')
    ap.add_argument('--skip-gpr-sliding', action='store_true')
    ap.add_argument('--etkf-grid', type=int, nargs=2, default=None)
    ap.add_argument('--etkf-ne', type=int, default=None)
    ap.add_argument('--etkf-mi', type=float, default=None)
    ap.add_argument('--etkf-ai', type=float, default=None)
    ap.add_argument('--out', type=str, default='results/b1_full_comparison.json')
    args = ap.parse_args()

    etkf_cfg = dict(ETKF_BEST)
    if args.etkf_grid:
        etkf_cfg['grid'] = tuple(args.etkf_grid)
    if args.etkf_ne is not None:
        etkf_cfg['n_ensemble'] = args.etkf_ne
    if args.etkf_mi is not None:
        etkf_cfg['mult_inflation'] = args.etkf_mi
    if args.etkf_ai is not None:
        etkf_cfg['add_inflation'] = args.etkf_ai

    data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
    out_path = os.path.join(os.path.dirname(__file__), '..', args.out)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    episodes = sorted(f for f in os.listdir(data_dir)
                      if f.startswith('episode_') and f.endswith('.h5'))
    if args.bf:
        episodes = [e for e in episodes if f'_{args.bf}_' in e]
    if args.max:
        episodes = episodes[:args.max]

    print(f"B1 Full Comparison: {len(episodes)} episodes")
    print(f"ETKF config: {etkf_cfg}")
    print(f"Sliding GPR: {'off' if args.skip_gpr_sliding else 'on'}")
    print('=' * 80, flush=True)

    all_results = {}
    t_start = time.time()

    for i, ep in enumerate(episodes):
        h5 = os.path.join(data_dir, ep)
        short = ep.replace('episode_', '').replace('.h5', '')
        print(f"\n[{i + 1}/{len(episodes)}] {short}", flush=True)
        ep_res = {}

        # WaveSplat
        try:
            t0 = time.time()
            ep_res['WaveSplat'] = run_wavesplat(h5)
            print(f"  WS:         nRMSE={ep_res['WaveSplat'].get('avg_nRMSE',float('nan')):.4f} "
                  f"Hs={ep_res['WaveSplat'].get('avg_Hs_error',float('nan')):.3f} "
                  f"VR={ep_res['WaveSplat'].get('avg_Variance_Ratio',float('nan')):.3f} "
                  f"({time.time()-t0:.1f}s)", flush=True)
        except Exception as e:
            ep_res['WaveSplat'] = {'error': str(e)}
            print(f"  WS FAILED: {e}", flush=True)

        # ETKF
        try:
            t0 = time.time()
            ep_res['ETKF'] = run_etkf(h5, etkf_cfg)
            print(f"  ETKF:       nRMSE={ep_res['ETKF'].get('avg_nRMSE',float('nan')):.4f} "
                  f"Hs={ep_res['ETKF'].get('avg_Hs_error',float('nan')):.3f} "
                  f"VR={ep_res['ETKF'].get('avg_Variance_Ratio',float('nan')):.3f} "
                  f"({time.time()-t0:.1f}s)", flush=True)
        except Exception as e:
            ep_res['ETKF'] = {'error': str(e)}
            print(f"  ETKF FAILED: {e}", flush=True)

        # GPR single
        try:
            t0 = time.time()
            ep_res['GPR-1'] = run_gpr(h5, window_frames=1)
            print(f"  GPR-1:      nRMSE={ep_res['GPR-1'].get('avg_nRMSE',float('nan')):.4f} "
                  f"Hs={ep_res['GPR-1'].get('avg_Hs_error',float('nan')):.3f} "
                  f"VR={ep_res['GPR-1'].get('avg_Variance_Ratio',float('nan')):.3f} "
                  f"({time.time()-t0:.1f}s)", flush=True)
        except Exception as e:
            ep_res['GPR-1'] = {'error': str(e)}
            print(f"  GPR-1 FAILED: {e}", flush=True)

        # GPR sliding
        if not args.skip_gpr_sliding:
            try:
                t0 = time.time()
                ep_res['GPR-3'] = run_gpr(h5, window_frames=3)
                print(f"  GPR-3:      nRMSE={ep_res['GPR-3'].get('avg_nRMSE',float('nan')):.4f} "
                      f"Hs={ep_res['GPR-3'].get('avg_Hs_error',float('nan')):.3f} "
                      f"VR={ep_res['GPR-3'].get('avg_Variance_Ratio',float('nan')):.3f} "
                      f"({time.time()-t0:.1f}s)", flush=True)
            except Exception as e:
                ep_res['GPR-3'] = {'error': str(e)}
                print(f"  GPR-3 FAILED: {e}", flush=True)

        all_results[short] = ep_res

        # Incremental save
        with open(out_path, 'w') as f:
            json.dump({'etkf_config': etkf_cfg, 'results': all_results},
                      f, indent=2, default=str)

        elapsed = time.time() - t_start
        avg_per_ep = elapsed / (i + 1)
        remaining = avg_per_ep * (len(episodes) - i - 1)
        print(f"  [{elapsed:.0f}s elapsed, ~{remaining:.0f}s remaining]",
              flush=True)

    # Summary
    print(f"\n{'=' * 80}\nSUMMARY (mean across episodes)\n{'=' * 80}")
    methods = ['WaveSplat', 'ETKF', 'GPR-1', 'GPR-3']
    metric_cols = ['avg_nRMSE', 'avg_Hs_error', 'avg_Variance_Ratio',
                   'avg_Pearson_rho', 'avg_Skill_Score']
    header = f"{'Method':<12}" + ''.join(f"{m[4:]:>14}" for m in metric_cols)
    print(header)
    for meth in methods:
        row = f"{meth:<12}"
        for mc in metric_cols:
            vals = [r.get(meth, {}).get(mc, float('nan'))
                    for r in all_results.values()]
            vals = [v for v in vals if isinstance(v, (int, float))
                    and not np.isnan(v)]
            mean = np.mean(vals) if vals else float('nan')
            row += f"{mean:>14.4f}"
        print(row)

    print(f"\nTotal: {time.time() - t_start:.1f}s")
    print(f"Saved: {out_path}")


if __name__ == '__main__':
    main()
