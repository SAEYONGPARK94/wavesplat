"""
run_multiseed_extended.py
Phase B4: Extended multi-seed statistical validation.

Extends run_multiseed_comparison.py to both uniform + slope raydrop models:
  - 10 (BF, comp) pairs × 5 data seeds × 2 raydrop models = 100 paired runs
  - Methods: WaveSplat-P1, ETKF, GPR-sliding (window=3)
  - Paired Wilcoxon signed-rank test (WS vs each baseline)

Outputs: results/multiseed_extended.json
"""

import sys
import os
import json
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
from scipy.stats import wilcoxon

BF_COMP = [
    ('BF2', '1comp'), ('BF3', '1comp'), ('BF3', '2comp'), ('BF3', '5comp'),
    ('BF4', '1comp'), ('BF4', '5comp'), ('BF4', '15comp'), ('BF4', '30comp'),
    ('BF5', '15comp'), ('BF5', '30comp'),
]
SEEDS = [0, 1, 2, 3, 4]
RAYDROPS = ['uniform', 'slope']

# Best ETKF config from Phase B2 sweep (24 configs × 12 eps)
# Winner: g40x30_ne100_mi1.0_ai0.005, mean nRMSE = 0.3584
ETKF_CFG = {'grid': (40, 30), 'ne': 100, 'mi': 1.0, 'ai': 0.005}


def _summarize(frame_results):
    keys = ['nRMSE', 'Hs_error', 'Variance_Ratio', 'Pearson_rho', 'Skill_Score']
    out = {}
    for k in keys:
        v = [r.get(k, float('nan')) for r in frame_results]
        v = [x for x in v if not np.isnan(x)]
        out[k] = float(np.mean(v)) if v else float('nan')
    out['ms'] = float(np.mean([r.get('elapsed_ms', 0) for r in frame_results]))
    return out


def run_ws(h5, seed):
    r = run_estimation(h5, config={
        'init_method': 'v2_konly', 'use_dispersion': True, 'use_confidence': True,
        'adaptive_n': True, 'tracking_window': 3, 'seed': 42 + seed,
    }, verbose=False)
    return _summarize(r.get('frame_results', []))


def run_etkf(h5, seed):
    np.random.seed(42 + seed)
    loader = EpisodeDataLoader(h5)
    gx = loader.gt_grid_x.cpu().numpy()
    gy = loader.gt_grid_y.cpu().numpy()
    egx = np.linspace(gx[0], gx[-1], ETKF_CFG['grid'][0])
    egy = np.linspace(gy[0], gy[-1], ETKF_CFG['grid'][1])
    etkf = ETKFGrid(egx, egy, depth=loader.depth,
                    n_ensemble=ETKF_CFG['ne'],
                    mult_inflation=ETKF_CFG['mi'],
                    add_inflation=ETKF_CFG['ai'])
    frames = []
    for frame in loader:
        x = frame.x_obs.cpu().numpy()
        y = frame.y_obs.cpu().numpy()
        z = frame.z_obs.cpu().numpy()
        ms = etkf.update(x, y, z)
        pg = etkf.predict_grid()
        interp = RegularGridInterpolator((egy, egx), pg,
            method='linear', bounds_error=False, fill_value=0.0)
        Xg, Yg = np.meshgrid(gx, gy)
        pred = interp(np.column_stack([Yg.ravel(), Xg.ravel()])).reshape(
            len(gy), len(gx))
        gt = frame.gt_field.cpu().numpy()
        m = compute_all_metrics(pred, gt, gx, gy, loader.Hs_true,
            obs_x=x if len(x) > 0 else None,
            obs_y=y if len(y) > 0 else None)
        m['elapsed_ms'] = ms
        frames.append(m)
    return _summarize(frames)


def run_gpr(h5, seed, window=3):
    np.random.seed(42 + seed)
    loader = EpisodeDataLoader(h5)
    gx = loader.gt_grid_x.cpu().numpy()
    gy = loader.gt_grid_y.cpu().numpy()
    gpr = GPRBaseline(kernel_type='matern', length_scale=20.0,
                      noise_sigma=0.05, window_frames=window)
    frames = []
    for frame in loader:
        x = frame.x_obs.cpu().numpy()
        y = frame.y_obs.cpu().numpy()
        z = frame.z_obs.cpu().numpy()
        ms = gpr.update(x, y, z)
        pred = gpr.predict_grid(gx, gy)
        gt = frame.gt_field.cpu().numpy()
        m = compute_all_metrics(pred, gt, gx, gy, loader.Hs_true,
            obs_x=x if len(x) > 0 else None,
            obs_y=y if len(y) > 0 else None)
        m['elapsed_ms'] = ms
        frames.append(m)
    return _summarize(frames)


def main():
    data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
    out_path = os.path.join(os.path.dirname(__file__), '..', 'results',
                            'multiseed_extended.json')

    all_res = {}
    ws_list, etkf_list, gpr_list = [], [], []

    n_total = len(BF_COMP) * len(SEEDS) * len(RAYDROPS)
    print(f"Extended multi-seed: {n_total} paired runs")
    print(f"ETKF config: {ETKF_CFG}")
    print('=' * 90)

    i = 0
    for bf, comp in BF_COMP:
        for seed in SEEDS:
            for ray in RAYDROPS:
                i += 1
                name = f"{bf}_{comp}_seed{seed}_{ray}"
                h5 = os.path.join(data_dir, f"episode_{name}.h5")
                if not os.path.exists(h5):
                    continue
                try:
                    ws = run_ws(h5, seed)
                    et = run_etkf(h5, seed)
                    gp = run_gpr(h5, seed, window=3)
                    all_res[name] = {'WS': ws, 'ETKF': et, 'GPR-3': gp}
                    ws_list.append(ws['nRMSE'])
                    etkf_list.append(et['nRMSE'])
                    gpr_list.append(gp['nRMSE'])
                    print(f"[{i:3d}/{n_total}] {name:35s} "
                          f"WS={ws['nRMSE']:.4f} ETKF={et['nRMSE']:.4f} "
                          f"GPR3={gp['nRMSE']:.4f}", flush=True)
                except Exception as e:
                    all_res[name] = {'error': str(e)}
                    print(f"[{i:3d}/{n_total}] {name}: FAILED {e}", flush=True)

                # Save incrementally
                os.makedirs(os.path.dirname(out_path), exist_ok=True)
                with open(out_path, 'w') as f:
                    json.dump({'etkf_config': ETKF_CFG, 'results': all_res},
                              f, indent=2, default=str)

    # Paired Wilcoxon tests
    print(f"\n{'=' * 80}\nWilcoxon signed-rank (WS vs baselines)")
    if len(ws_list) >= 5:
        ws_arr = np.array(ws_list)
        et_arr = np.array(etkf_list)
        gp_arr = np.array(gpr_list)

        try:
            w_e, p_e = wilcoxon(ws_arr, et_arr)
            print(f"  WS vs ETKF:   n={len(ws_list)} "
                  f"mean diff={np.mean(ws_arr - et_arr):+.4f} "
                  f"W={w_e:.1f} p={p_e:.4g}")
        except Exception as e:
            print(f"  WS vs ETKF: {e}")

        try:
            w_g, p_g = wilcoxon(ws_arr, gp_arr)
            print(f"  WS vs GPR-3:  n={len(ws_list)} "
                  f"mean diff={np.mean(ws_arr - gp_arr):+.4f} "
                  f"W={w_g:.1f} p={p_g:.4g}")
        except Exception as e:
            print(f"  WS vs GPR-3: {e}")

    print(f"\nSaved: {out_path}")


if __name__ == '__main__':
    main()
