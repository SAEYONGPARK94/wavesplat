"""
run_multiseed_comparison.py
Multi-seed statistical validation: run WaveSplat-P1 and ETKF on 5 seeds per episode.

Outputs mean±std for each episode and method, plus Wilcoxon signed-rank test.
"""

import sys
import os
import json
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from wavesplat.data_loader import EpisodeDataLoader
from wavesplat.run import run_estimation
from wavesplat.baselines.etkf_grid import ETKFGrid
from wavesplat.evaluation.metrics import compute_all_metrics
from scipy.interpolate import RegularGridInterpolator
from scipy.stats import wilcoxon

# BF/comp combinations × seed0-4
BF_COMP = [
    ('BF2', '1comp'),
    ('BF3', '1comp'),
    ('BF3', '2comp'),
    ('BF3', '5comp'),
    ('BF4', '1comp'),
    ('BF4', '5comp'),
    ('BF4', '15comp'),
    ('BF4', '30comp'),
    ('BF5', '15comp'),
    ('BF5', '30comp'),
]
SEEDS = [0, 1, 2, 3, 4]
RAYDROP = 'uniform'  # primary comparison on uniform


def run_wavesplat_p1(h5_path, seed):
    config = {
        'init_method': 'v2_konly',
        'use_dispersion': True,
        'use_confidence': True,
        'adaptive_n': True,
        'tracking_window': 3,
        'seed': 42 + seed,  # vary optimization seed
    }
    result = run_estimation(h5_path, config=config)
    return {
        'avg_nRMSE': float(result['avg_nRMSE']),
        'avg_inference_ms': float(result['avg_inference_ms']),
        'final_n': result.get('n_gaussians', 15),
        'T_converge': result.get('T_converge'),
    }


def run_etkf(h5_path, seed):
    np.random.seed(42 + seed)
    loader = EpisodeDataLoader(h5_path)
    gt_grid_x = loader.gt_grid_x.cpu().numpy()
    gt_grid_y = loader.gt_grid_y.cpu().numpy()

    etkf = ETKFGrid(
        np.linspace(gt_grid_x[0], gt_grid_x[-1], 40),
        np.linspace(gt_grid_y[0], gt_grid_y[-1], 30),
        depth=loader.depth, n_ensemble=50)

    frame_results = []
    for frame in loader:
        x = frame.x_obs.cpu().numpy()
        y = frame.y_obs.cpu().numpy()
        z = frame.z_obs.cpu().numpy()
        elapsed = etkf.update(x, y, z)

        etkf_pred = etkf.predict_grid()
        interp = RegularGridInterpolator(
            (np.linspace(gt_grid_y[0], gt_grid_y[-1], 30),
             np.linspace(gt_grid_x[0], gt_grid_x[-1], 40)),
            etkf_pred, method='linear', bounds_error=False, fill_value=0.0)
        Xg, Yg = np.meshgrid(gt_grid_x, gt_grid_y)
        pred = interp(np.column_stack([Yg.ravel(), Xg.ravel()])).reshape(
            len(gt_grid_y), len(gt_grid_x))

        gt = frame.gt_field.cpu().numpy()
        metrics = compute_all_metrics(
            pred, gt, gt_grid_x, gt_grid_y, loader.Hs_true,
            obs_x=x if len(x) > 0 else None,
            obs_y=y if len(y) > 0 else None)
        metrics['elapsed_ms'] = elapsed
        frame_results.append(metrics)

    if frame_results:
        return {
            'avg_nRMSE': float(np.mean([r['nRMSE'] for r in frame_results])),
            'avg_inference_ms': float(np.mean([r['elapsed_ms'] for r in frame_results])),
        }
    return {'avg_nRMSE': float('nan'), 'avg_inference_ms': float('nan')}


def main():
    data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'results')
    os.makedirs(output_dir, exist_ok=True)

    results = {}
    all_ws_nrmses = []
    all_etkf_nrmses = []

    print(f"Multi-seed comparison: {len(BF_COMP)} episodes × {len(SEEDS)} seeds × 2 methods")
    print("=" * 100)

    for bf, comp in BF_COMP:
        pair_key = f"{bf}_{comp}"
        results[pair_key] = {'WS-P1': [], 'ETKF': []}

        for seed in SEEDS:
            ep_name = f"episode_{bf}_{comp}_seed{seed}_{RAYDROP}.h5"
            h5_path = os.path.join(data_dir, ep_name)

            if not os.path.exists(h5_path):
                print(f"  SKIP: {ep_name}")
                continue

            print(f"[{pair_key} seed{seed}]", end=" ", flush=True)

            # WaveSplat
            try:
                ws = run_wavesplat_p1(h5_path, seed)
                results[pair_key]['WS-P1'].append(ws)
                print(f"WS={ws['avg_nRMSE']:.4f}", end=" ", flush=True)
            except Exception as e:
                print(f"WS-FAIL({e})", end=" ", flush=True)

            # ETKF
            try:
                etkf = run_etkf(h5_path, seed)
                results[pair_key]['ETKF'].append(etkf)
                print(f"ETKF={etkf['avg_nRMSE']:.4f}", flush=True)
            except Exception as e:
                print(f"ETKF-FAIL({e})", flush=True)

    # Summary
    print(f"\n{'=' * 100}")
    print(f"{'Episode':<15} {'WS mean±std':>16} {'ETKF mean±std':>16} "
          f"{'Δ%':>8} {'p-value':>10}")
    print(f"{'-' * 70}")

    for pair_key in results:
        ws_vals = [r['avg_nRMSE'] for r in results[pair_key]['WS-P1']]
        et_vals = [r['avg_nRMSE'] for r in results[pair_key]['ETKF']]

        if ws_vals and et_vals:
            ws_m, ws_s = np.mean(ws_vals), np.std(ws_vals)
            et_m, et_s = np.mean(et_vals), np.std(et_vals)
            delta = (et_m - ws_m) / et_m * 100

            all_ws_nrmses.extend(ws_vals)
            all_etkf_nrmses.extend(et_vals)

            # Wilcoxon test (paired)
            if len(ws_vals) >= 5:
                try:
                    stat, pval = wilcoxon(ws_vals, et_vals, alternative='less')
                    pval_str = f"{pval:.4f}"
                except Exception:
                    pval_str = "N/A"
            else:
                pval_str = "N/A"

            print(f"{pair_key:<15} {ws_m:>7.3f}±{ws_s:.3f} "
                  f"{et_m:>7.3f}±{et_s:.3f} {delta:>+7.1f}% {pval_str:>10}")

    print(f"{'-' * 70}")
    if all_ws_nrmses and all_etkf_nrmses:
        print(f"{'OVERALL':<15} {np.mean(all_ws_nrmses):>7.3f}±{np.std(all_ws_nrmses):.3f} "
              f"{np.mean(all_etkf_nrmses):>7.3f}±{np.std(all_etkf_nrmses):.3f}")

        # Global Wilcoxon
        try:
            stat, pval = wilcoxon(all_ws_nrmses, all_etkf_nrmses, alternative='less')
            print(f"\nGlobal Wilcoxon signed-rank test: stat={stat:.1f}, p={pval:.6f}")
            print(f"  {'*** Significant (p<0.001)' if pval < 0.001 else '** Significant (p<0.01)' if pval < 0.01 else '* Significant (p<0.05)' if pval < 0.05 else 'Not significant'}")
        except Exception as e:
            print(f"\nWilcoxon test failed: {e}")

    out_path = os.path.join(output_dir, 'multiseed_comparison.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")


if __name__ == '__main__':
    main()
