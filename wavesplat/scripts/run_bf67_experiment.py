"""
run_bf67_experiment.py
BF6-7 scalability experiment: Generate higher sea state episodes and compare
WaveSplat vs LETKF to verify degradation under nonlinear wave conditions.

BF6: Hs=3.0m, Tp=10.0s (ε ≈ 0.05, mild nonlinearity)
BF7: Hs=5.0m, Tp=12.0s (ε ≈ 0.07, moderate nonlinearity)
"""

import sys
import os
import json
import numpy as np
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from wavesim.pipeline import run_episode, SEA_STATE
from wavesim.config import WaveSimConfig
from wavesplat.run import run_estimation
from wavesplat.data_loader import EpisodeDataLoader
from wavesplat.baselines.letkf_grid import LETKFGrid
from wavesplat.evaluation.metrics import compute_all_metrics
from scipy.interpolate import RegularGridInterpolator

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')

# Extend sea state definitions
SEA_STATE['BF6'] = dict(Hs=3.0, Tp=10.0)
SEA_STATE['BF7'] = dict(Hs=5.0, Tp=12.0)

METRIC_KEYS = ['nRMSE', 'Variance_Ratio', 'Pearson_rho', 'Skill_Score',
               'Directional_R2', 'Hs_error']
SHORT_KEYS = ['nRMSE', 'VR', 'rho', 'SS', 'Dir_R2', 'Hs_err']

# Episodes to generate: BF6 and BF7, varying complexity
SCENARIOS = [
    ('BF6', 1, 0, 'uniform'),
    ('BF6', 5, 0, 'uniform'),
    ('BF6', 15, 0, 'uniform'),
    ('BF6', 30, 0, 'uniform'),
    ('BF7', 1, 0, 'uniform'),
    ('BF7', 5, 0, 'uniform'),
    ('BF7', 15, 0, 'uniform'),
    ('BF7', 30, 0, 'uniform'),
]


def generate_episode(sea_state, n_comp, seed, raydrop):
    """Generate a single episode if it doesn't exist."""
    fname = f"episode_{sea_state}_{n_comp}comp_seed{seed}_{raydrop}.h5"
    fpath = os.path.join(DATA_DIR, fname)

    if os.path.exists(fpath):
        print(f"  [EXISTS] {fname}")
        return fpath

    cfg = WaveSimConfig()
    cfg.sea_states['BF6'] = {'Hs': 3.0, 'Tp': 10.0}
    cfg.sea_states['BF7'] = {'Hs': 5.0, 'Tp': 12.0}

    print(f"  [GENERATING] {fname} ...", flush=True)
    t0 = time.time()
    fpath = run_episode(
        sea_state=sea_state,
        n_components=n_comp,
        n_pts=cfg.episode.n_pts,
        seed=seed,
        raydrop_model=raydrop,
        duration=cfg.episode.duration,
        dt=cfg.episode.dt,
        depth=cfg.episode.depth,
        output_dir=DATA_DIR,
        verbose=False,
        cfg=cfg,
    )
    elapsed = time.time() - t0
    print(f"  [DONE] {elapsed:.0f}s", flush=True)
    return fpath


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

    print("=" * 80)
    print("BF6-7 Scalability Experiment")
    print("BF6: Hs=3.0m, Tp=10.0s  |  BF7: Hs=5.0m, Tp=12.0s")
    print("=" * 80)

    # Phase 1: Generate episodes
    print("\n--- Phase 1: Episode Generation ---")
    episode_paths = {}
    for sea_state, n_comp, seed, raydrop in SCENARIOS:
        key = f"{sea_state}_{n_comp}comp_seed{seed}_{raydrop}"
        try:
            path = generate_episode(sea_state, n_comp, seed, raydrop)
            episode_paths[key] = path
        except Exception as e:
            print(f"  FAILED: {key} — {e}")

    # Phase 2: Run WaveSplat + LETKF
    print("\n--- Phase 2: WaveSplat vs LETKF ---")
    all_results = {}

    for key, h5_path in episode_paths.items():
        print(f"\n[{key}]", flush=True)
        ep = {}

        # WaveSplat
        try:
            ws = run_wavesplat(h5_path)
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

        all_results[key] = ep

    # Phase 3: Analysis
    print("\n" + "=" * 80)
    print("ANALYSIS: BF6-7 vs BF2-5")
    print("=" * 80)

    # Load 100-ep results for BF2-5 comparison
    ref_path = os.path.join(OUTPUT_DIR, '100ep_ws_letkf.json')
    ref_data = {}
    if os.path.exists(ref_path):
        with open(ref_path) as f:
            ref_data = json.load(f)

    # Per-BF summary
    for bf in ['BF6', 'BF7']:
        print(f"\n  {bf} results:")
        for method in ['WaveSplat', 'LETKF']:
            keys = [k for k in all_results if k.startswith(bf)]
            for sk in SHORT_KEYS + ['ms']:
                vals = [all_results[k][method][sk] for k in keys
                        if method in all_results[k]
                        and isinstance(all_results[k][method], dict)
                        and sk in all_results[k][method]
                        and not np.isnan(all_results[k][method].get(sk, float('nan')))]
                if vals:
                    print(f"    {method:<8s} {sk:<8s}: {np.mean(vals):+.4f} "
                          f"(std {np.std(vals):.4f}, n={len(vals)})")

    # Cross-BF comparison (BF2-5 from ref, BF6-7 from this run)
    print("\n  Cross-BF nRMSE and SS comparison:")
    print(f"  {'BF':<6s}  {'WS nRMSE':>10s}  {'LETKF nRMSE':>12s}  "
          f"{'WS SS':>8s}  {'LETKF SS':>10s}")
    print(f"  {'-'*6}  {'-'*10}  {'-'*12}  {'-'*8}  {'-'*10}")

    # BF2-5 from reference data
    if ref_data:
        import re
        for bf_num in [2, 3, 4, 5]:
            bf_key = f"BF{bf_num}"
            ws_nrmse = []
            lk_nrmse = []
            ws_ss = []
            lk_ss = []
            for k, v in ref_data.items():
                m = re.match(rf'BF{bf_num}_', k)
                if m:
                    if 'WaveSplat' in v and isinstance(v['WaveSplat'], dict):
                        ws_nrmse.append(v['WaveSplat'].get('nRMSE', float('nan')))
                        ws_ss.append(v['WaveSplat'].get('SS', float('nan')))
                    if 'LETKF' in v and isinstance(v['LETKF'], dict):
                        lk_nrmse.append(v['LETKF'].get('nRMSE', float('nan')))
                        lk_ss.append(v['LETKF'].get('SS', float('nan')))
            ws_nrmse = [v for v in ws_nrmse if not np.isnan(v)]
            lk_nrmse = [v for v in lk_nrmse if not np.isnan(v)]
            ws_ss = [v for v in ws_ss if not np.isnan(v)]
            lk_ss = [v for v in lk_ss if not np.isnan(v)]
            if ws_nrmse and lk_nrmse:
                print(f"  {bf_key:<6s}  {np.mean(ws_nrmse):>10.4f}  "
                      f"{np.mean(lk_nrmse):>12.4f}  "
                      f"{np.mean(ws_ss):>+8.3f}  {np.mean(lk_ss):>+10.3f}")

    # BF6-7 from this run
    for bf_num in [6, 7]:
        bf_key = f"BF{bf_num}"
        keys = [k for k in all_results if k.startswith(bf_key)]
        ws_nrmse = [all_results[k]['WaveSplat']['nRMSE'] for k in keys
                    if 'WaveSplat' in all_results[k]
                    and isinstance(all_results[k]['WaveSplat'], dict)
                    and not np.isnan(all_results[k]['WaveSplat'].get('nRMSE', float('nan')))]
        lk_nrmse = [all_results[k]['LETKF']['nRMSE'] for k in keys
                    if 'LETKF' in all_results[k]
                    and isinstance(all_results[k]['LETKF'], dict)
                    and not np.isnan(all_results[k]['LETKF'].get('nRMSE', float('nan')))]
        ws_ss = [all_results[k]['WaveSplat']['SS'] for k in keys
                 if 'WaveSplat' in all_results[k]
                 and isinstance(all_results[k]['WaveSplat'], dict)
                 and not np.isnan(all_results[k]['WaveSplat'].get('SS', float('nan')))]
        lk_ss = [all_results[k]['LETKF']['SS'] for k in keys
                 if 'LETKF' in all_results[k]
                 and isinstance(all_results[k]['LETKF'], dict)
                 and not np.isnan(all_results[k]['LETKF'].get('SS', float('nan')))]
        if ws_nrmse and lk_nrmse:
            print(f"  {bf_key:<6s}  {np.mean(ws_nrmse):>10.4f}  "
                  f"{np.mean(lk_nrmse):>12.4f}  "
                  f"{np.mean(ws_ss):>+8.3f}  {np.mean(lk_ss):>+10.3f}")

    # Per-episode detail table
    print(f"\n  {'Episode':<30s}  {'WS nRMSE':>10s}  {'LK nRMSE':>10s}  "
          f"{'WS SS':>8s}  {'LK SS':>8s}  {'WS Dir_R2':>10s}  {'LK Dir_R2':>10s}")
    print(f"  {'-'*30}  {'-'*10}  {'-'*10}  {'-'*8}  {'-'*8}  {'-'*10}  {'-'*10}")
    for key in sorted(all_results.keys()):
        r = all_results[key]
        ws = r.get('WaveSplat', {})
        lk = r.get('LETKF', {})
        if isinstance(ws, dict) and isinstance(lk, dict):
            print(f"  {key:<30s}  "
                  f"{ws.get('nRMSE', float('nan')):>10.4f}  "
                  f"{lk.get('nRMSE', float('nan')):>10.4f}  "
                  f"{ws.get('SS', float('nan')):>+8.3f}  "
                  f"{lk.get('SS', float('nan')):>+8.3f}  "
                  f"{ws.get('Dir_R2', float('nan')):>+10.3f}  "
                  f"{lk.get('Dir_R2', float('nan')):>+10.3f}")

    # Save
    out_path = os.path.join(OUTPUT_DIR, 'bf67_experiment.json')
    with open(out_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")


if __name__ == '__main__':
    main()
