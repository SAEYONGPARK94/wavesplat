"""
run_raydrop_comparison.py
Compare WaveSplat-P1 and ETKF on paired uniform vs slope raydrop episodes.

For each BF/comp combination, runs both raydrop models and compares:
- Does WaveSplat handle slope raydrop well?
- Does ETKF benefit from specular observations?
- How does adaptive N behave under slope vs uniform?
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


# All BF/comp combinations with seed0
BF_COMP_PAIRS = [
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


def run_wavesplat_p1(h5_path):
    """P1: adaptive N + multi-frame tracking window."""
    config = {
        'init_method': 'v2_konly',
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
        'final_n': result.get('n_gaussians', 15),
    }


def run_etkf(h5_path):
    """Run ETKF-grid baseline (medium resolution)."""
    np.random.seed(42)
    loader = EpisodeDataLoader(h5_path)
    gt_grid_x = loader.gt_grid_x.cpu().numpy()
    gt_grid_y = loader.gt_grid_y.cpu().numpy()

    etkf_nx, etkf_ny = 40, 30
    etkf_gx = np.linspace(gt_grid_x[0], gt_grid_x[-1], etkf_nx)
    etkf_gy = np.linspace(gt_grid_y[0], gt_grid_y[-1], etkf_ny)

    etkf = ETKFGrid(etkf_gx, etkf_gy, depth=loader.depth, n_ensemble=50)

    frame_results = []
    for frame in loader:
        x = frame.x_obs.cpu().numpy()
        y = frame.y_obs.cpu().numpy()
        z = frame.z_obs.cpu().numpy()

        elapsed = etkf.update(x, y, z)

        etkf_pred = etkf.predict_grid()
        interp = RegularGridInterpolator(
            (etkf_gy, etkf_gx), etkf_pred,
            method='linear', bounds_error=False, fill_value=0.0)
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

    print("Raydrop Comparison: Uniform vs Slope")
    print(f"Pairs: {len(BF_COMP_PAIRS)}")
    print("=" * 100)

    for bf, comp in BF_COMP_PAIRS:
        pair_key = f"{bf}_{comp}"
        results[pair_key] = {}

        for raydrop in ['uniform', 'slope']:
            ep_name = f"episode_{bf}_{comp}_seed0_{raydrop}.h5"
            h5_path = os.path.join(data_dir, ep_name)

            if not os.path.exists(h5_path):
                print(f"  SKIP: {ep_name}")
                continue

            short = f"{bf}_{comp}_{raydrop}"
            print(f"\n[{short}]", flush=True)

            ep_result = {}

            # WaveSplat P1
            try:
                ws = run_wavesplat_p1(h5_path)
                ep_result['WS-P1'] = ws
                print(f"  WS-P1: nRMSE={ws['avg_nRMSE']:.4f} "
                      f"({ws['avg_inference_ms']:.1f}ms, N={ws['final_n']})",
                      flush=True)
            except Exception as e:
                print(f"  WS-P1 FAILED: {e}")
                ep_result['WS-P1'] = {'error': str(e)}

            # ETKF
            try:
                etkf = run_etkf(h5_path)
                ep_result['ETKF'] = etkf
                print(f"  ETKF:  nRMSE={etkf['avg_nRMSE']:.4f} "
                      f"({etkf['avg_inference_ms']:.1f}ms)", flush=True)
            except Exception as e:
                print(f"  ETKF FAILED: {e}")
                ep_result['ETKF'] = {'error': str(e)}

            results[pair_key][raydrop] = ep_result

    # ── Summary Table ──
    print(f"\n{'=' * 100}")
    print(f"{'Pair':<20} {'WS-U nRMSE':>12} {'WS-S nRMSE':>12} "
          f"{'ETKF-U':>10} {'ETKF-S':>10} "
          f"{'N_U':>5} {'N_S':>5} {'WS Δ%':>8} {'ETKF Δ%':>8}")
    print(f"{'-' * 100}")

    ws_u_all, ws_s_all, et_u_all, et_s_all = [], [], [], []

    for pair_key in results:
        u = results[pair_key].get('uniform', {})
        s = results[pair_key].get('slope', {})

        ws_u = u.get('WS-P1', {}).get('avg_nRMSE', float('nan'))
        ws_s = s.get('WS-P1', {}).get('avg_nRMSE', float('nan'))
        et_u = u.get('ETKF', {}).get('avg_nRMSE', float('nan'))
        et_s = s.get('ETKF', {}).get('avg_nRMSE', float('nan'))
        n_u = u.get('WS-P1', {}).get('final_n', '?')
        n_s = s.get('WS-P1', {}).get('final_n', '?')

        # Compute Δ% (slope vs uniform): positive = slope is worse
        ws_delta = (ws_s - ws_u) / ws_u * 100 if not (np.isnan(ws_u) or np.isnan(ws_s) or ws_u == 0) else float('nan')
        et_delta = (et_s - et_u) / et_u * 100 if not (np.isnan(et_u) or np.isnan(et_s) or et_u == 0) else float('nan')

        print(f"{pair_key:<20} {ws_u:>12.4f} {ws_s:>12.4f} "
              f"{et_u:>10.4f} {et_s:>10.4f} "
              f"{str(n_u):>5} {str(n_s):>5} "
              f"{ws_delta:>+7.1f}% {et_delta:>+7.1f}%")

        if not np.isnan(ws_u): ws_u_all.append(ws_u)
        if not np.isnan(ws_s): ws_s_all.append(ws_s)
        if not np.isnan(et_u): et_u_all.append(et_u)
        if not np.isnan(et_s): et_s_all.append(et_s)

    print(f"{'-' * 100}")
    if ws_u_all and ws_s_all:
        ws_u_avg = np.mean(ws_u_all)
        ws_s_avg = np.mean(ws_s_all)
        et_u_avg = np.mean(et_u_all)
        et_s_avg = np.mean(et_s_all)
        print(f"{'AVERAGE':<20} {ws_u_avg:>12.4f} {ws_s_avg:>12.4f} "
              f"{et_u_avg:>10.4f} {et_s_avg:>10.4f} "
              f"{'':>5} {'':>5} "
              f"{(ws_s_avg-ws_u_avg)/ws_u_avg*100:>+7.1f}% "
              f"{(et_s_avg-et_u_avg)/et_u_avg*100:>+7.1f}%")

        print(f"\n--- Key Findings ---")
        print(f"WaveSplat-P1 uniform avg: {ws_u_avg:.3f}")
        print(f"WaveSplat-P1 slope avg:   {ws_s_avg:.3f} "
              f"({(ws_s_avg-ws_u_avg)/ws_u_avg*100:+.1f}%)")
        print(f"ETKF uniform avg:         {et_u_avg:.3f}")
        print(f"ETKF slope avg:           {et_s_avg:.3f} "
              f"({(et_s_avg-et_u_avg)/et_u_avg*100:+.1f}%)")
        print(f"\nWS wins on uniform: "
              f"{sum(1 for w, e in zip(ws_u_all, et_u_all) if w < e)}/{len(ws_u_all)}")
        print(f"WS wins on slope:   "
              f"{sum(1 for w, e in zip(ws_s_all, et_s_all) if w < e)}/{len(ws_s_all)}")

    # Save
    out_path = os.path.join(output_dir, 'raydrop_comparison.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")


if __name__ == '__main__':
    main()
