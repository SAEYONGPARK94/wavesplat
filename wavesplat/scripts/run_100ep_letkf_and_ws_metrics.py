"""
run_100ep_letkf_and_ws_metrics.py
1) WaveSplat full 6-metric on all 100 episodes (BF/complexity breakdown for Dir R² gap)
2) LETKF(loc50) on all 100 episodes
"""

import sys
import os
import json
import glob
import re
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from wavesplat.run import run_estimation
from wavesplat.data_loader import EpisodeDataLoader
from wavesplat.baselines.letkf_grid import LETKFGrid
from wavesplat.evaluation.metrics import compute_all_metrics
from scipy.interpolate import RegularGridInterpolator

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')

METRIC_KEYS = ['nRMSE', 'Variance_Ratio', 'Pearson_rho', 'Skill_Score',
               'Directional_R2', 'Hs_error']
SHORT_KEYS = ['nRMSE', 'VR', 'rho', 'SS', 'Dir_R2', 'Hs_err']


def parse_episode_name(fname):
    """Extract BF, ncomp, seed, raydrop from filename."""
    m = re.match(r'episode_BF(\d+)_(\d+)comp_seed(\d+)_(\w+)\.h5', fname)
    if m:
        return int(m.group(1)), int(m.group(2)), int(m.group(3)), m.group(4)
    return None, None, None, None


def run_wavesplat_6metric(h5_path):
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


def run_letkf(h5_path):
    np.random.seed(42)
    loader = EpisodeDataLoader(h5_path)
    gt_grid_x = loader.gt_grid_x.cpu().numpy()
    gt_grid_y = loader.gt_grid_y.cpu().numpy()

    etkf_nx, etkf_ny = 40, 30
    etkf_gx = np.linspace(gt_grid_x[0], gt_grid_x[-1], etkf_nx)
    etkf_gy = np.linspace(gt_grid_y[0], gt_grid_y[-1], etkf_ny)

    letkf = LETKFGrid(etkf_gx, etkf_gy, depth=loader.depth,
                       n_ensemble=100, mult_inflation=1.0,
                       add_inflation=0.005, loc_radius=50.0)

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
        for key in METRIC_KEYS + ['elapsed_ms']:
            vals = [m[key] for m in frame_metrics
                    if not np.isnan(m.get(key, float('nan')))]
            avg[key] = float(np.mean(vals)) if vals else float('nan')
        return avg
    return {}


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    episodes = sorted(glob.glob(os.path.join(DATA_DIR, 'episode_*.h5')))
    print(f"100-ep WaveSplat 6-metric + LETKF comparison")
    print(f"Episodes found: {len(episodes)}")
    print("=" * 100)

    all_results = {}

    for i, h5_path in enumerate(episodes):
        fname = os.path.basename(h5_path)
        bf, ncomp, seed, raydrop = parse_episode_name(fname)
        short = fname.replace('episode_', '').replace('.h5', '')

        print(f"\n[{i+1}/{len(episodes)}] {short}", flush=True)

        ep = {'bf': bf, 'ncomp': ncomp, 'seed': seed, 'raydrop': raydrop}

        # WaveSplat
        try:
            ws = run_wavesplat_6metric(h5_path)
            ep['WaveSplat'] = {sk: ws.get(mk, float('nan'))
                               for sk, mk in zip(SHORT_KEYS, METRIC_KEYS)}
            ep['WaveSplat']['ms'] = ws.get('elapsed_ms', float('nan'))
            ep['WaveSplat']['N'] = ws.get('N', -1)
            print(f"  WS:    nRMSE={ws['nRMSE']:.4f}  SS={ws['Skill_Score']:+.3f}  "
                  f"Dir_R2={ws['Directional_R2']:+.3f}  ({ws['elapsed_ms']:.1f}ms)",
                  flush=True)
        except Exception as e:
            print(f"  WS FAILED: {e}")
            ep['WaveSplat'] = {'error': str(e)}

        # LETKF
        try:
            lk = run_letkf(h5_path)
            ep['LETKF'] = {sk: lk.get(mk, float('nan'))
                           for sk, mk in zip(SHORT_KEYS, METRIC_KEYS)}
            ep['LETKF']['ms'] = lk.get('elapsed_ms', float('nan'))
            print(f"  LETKF: nRMSE={lk['nRMSE']:.4f}  SS={lk['Skill_Score']:+.3f}  "
                  f"Dir_R2={lk['Directional_R2']:+.3f}  ({lk['elapsed_ms']:.1f}ms)",
                  flush=True)
        except Exception as e:
            print(f"  LETKF FAILED: {e}")
            ep['LETKF'] = {'error': str(e)}

        all_results[short] = ep

        # Periodic save
        if (i + 1) % 10 == 0:
            tmp_path = os.path.join(OUTPUT_DIR, '100ep_ws_letkf_partial.json')
            with open(tmp_path, 'w') as f:
                json.dump(all_results, f, indent=2, default=str)
            print(f"  [checkpoint saved: {i+1}/{len(episodes)}]", flush=True)

    # ── Analysis ────────────────────────────────────────────────────

    print("\n" + "=" * 100)
    print("ANALYSIS")
    print("=" * 100)

    # 1. Overall averages
    for method in ['WaveSplat', 'LETKF']:
        print(f"\n  {method} — 100-ep averages:")
        for sk in SHORT_KEYS + ['ms']:
            vals = [r[method][sk] for r in all_results.values()
                    if method in r and isinstance(r[method], dict)
                    and sk in r[method]
                    and not np.isnan(r[method].get(sk, float('nan')))]
            if vals:
                print(f"    {sk:<8s}: {np.mean(vals):+.4f} (std {np.std(vals):.4f}, n={len(vals)})")

    # 2. Dir R² breakdown by BF
    print("\n  Dir R² breakdown by sea state (WaveSplat):")
    for bf in [2, 3, 4, 5]:
        vals = [r['WaveSplat']['Dir_R2'] for r in all_results.values()
                if r.get('bf') == bf and 'WaveSplat' in r
                and isinstance(r['WaveSplat'], dict)
                and not np.isnan(r['WaveSplat'].get('Dir_R2', float('nan')))]
        if vals:
            print(f"    BF{bf}: Dir_R2 = {np.mean(vals):+.4f} (std {np.std(vals):.4f}, n={len(vals)})")

    # 3. Dir R² breakdown by complexity
    print("\n  Dir R² breakdown by complexity (WaveSplat):")
    for ncomp in [1, 2, 5, 15, 30]:
        vals = [r['WaveSplat']['Dir_R2'] for r in all_results.values()
                if r.get('ncomp') == ncomp and 'WaveSplat' in r
                and isinstance(r['WaveSplat'], dict)
                and not np.isnan(r['WaveSplat'].get('Dir_R2', float('nan')))]
        if vals:
            print(f"    {ncomp:>2d} comp: Dir_R2 = {np.mean(vals):+.4f} (std {np.std(vals):.4f}, n={len(vals)})")

    # 4. Dir R² breakdown by BF × complexity
    print("\n  Dir R² by BF × complexity (WaveSplat):")
    for bf in [2, 3, 4, 5]:
        for ncomp in [1, 2, 5, 15, 30]:
            vals = [r['WaveSplat']['Dir_R2'] for r in all_results.values()
                    if r.get('bf') == bf and r.get('ncomp') == ncomp
                    and 'WaveSplat' in r and isinstance(r['WaveSplat'], dict)
                    and not np.isnan(r['WaveSplat'].get('Dir_R2', float('nan')))]
            if vals:
                print(f"    BF{bf}-{ncomp:>2d}comp: {np.mean(vals):+.4f} (n={len(vals)})")

    # 5. SS breakdown by complexity
    print("\n  SS breakdown by complexity:")
    for ncomp in [1, 2, 5, 15, 30]:
        ws_vals = [r['WaveSplat']['SS'] for r in all_results.values()
                   if r.get('ncomp') == ncomp and 'WaveSplat' in r
                   and isinstance(r['WaveSplat'], dict)
                   and not np.isnan(r['WaveSplat'].get('SS', float('nan')))]
        lk_vals = [r['LETKF']['SS'] for r in all_results.values()
                   if r.get('ncomp') == ncomp and 'LETKF' in r
                   and isinstance(r['LETKF'], dict)
                   and not np.isnan(r['LETKF'].get('SS', float('nan')))]
        if ws_vals and lk_vals:
            print(f"    {ncomp:>2d} comp: WS {np.mean(ws_vals):+.4f}  LETKF {np.mean(lk_vals):+.4f}  "
                  f"(n={len(ws_vals)})")

    # 6. Head-to-head: how many episodes does WS win?
    ws_wins_nrmse = 0
    ws_wins_ss = 0
    total = 0
    for r in all_results.values():
        if ('WaveSplat' in r and 'LETKF' in r
            and isinstance(r['WaveSplat'], dict) and isinstance(r['LETKF'], dict)):
            ws_n = r['WaveSplat'].get('nRMSE', float('nan'))
            lk_n = r['LETKF'].get('nRMSE', float('nan'))
            ws_s = r['WaveSplat'].get('SS', float('nan'))
            lk_s = r['LETKF'].get('SS', float('nan'))
            if not np.isnan(ws_n) and not np.isnan(lk_n):
                total += 1
                if ws_n < lk_n:
                    ws_wins_nrmse += 1
                if ws_s > lk_s:
                    ws_wins_ss += 1
    print(f"\n  Head-to-head (100 episodes):")
    print(f"    WS wins nRMSE: {ws_wins_nrmse}/{total}")
    print(f"    WS wins SS:    {ws_wins_ss}/{total}")

    # 7. Bootstrap CI
    print("\n  Bootstrap 95% CI for SS (100 episodes):")
    for method in ['WaveSplat', 'LETKF']:
        ss_vals = [r[method]['SS'] for r in all_results.values()
                   if method in r and isinstance(r[method], dict)
                   and not np.isnan(r[method].get('SS', float('nan')))]
        if len(ss_vals) >= 2:
            ss_arr = np.array(ss_vals)
            rng = np.random.default_rng(42)
            boot = np.array([np.mean(rng.choice(ss_arr, len(ss_arr), replace=True))
                             for _ in range(10000)])
            lo, hi = np.percentile(boot, [2.5, 97.5])
            print(f"    {method:<10s}: mean={np.mean(ss_arr):+.4f}  "
                  f"95%CI=[{lo:+.4f}, {hi:+.4f}]  "
                  f"includes_zero={lo <= 0 <= hi}")

    # Save
    out_path = os.path.join(OUTPUT_DIR, '100ep_ws_letkf.json')
    with open(out_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")


if __name__ == '__main__':
    main()
