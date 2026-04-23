"""
run_gpr_confidence_ablation.py
GPR + confidence weighting ablation

목적: C3(confidence weighting)이 WaveSplat 고유 기여인지,
      GPR에도 도움이 되는지 검증 (논문 C3 novelty 방어)

비교:
  A) WaveSplat (with confidence)        ← 기준
  B) WaveSplat (no confidence, A2)      ← 기존 ablation
  C) GPR-1 (no confidence, standard)    ← 기존 baseline
  D) GPR-1 + confidence weighting       ← 신규: GPR에 c_j 적용

C3 novelty 판정 기준:
  - D ≈ C (confidence가 GPR에 도움 안됨) → C3는 WaveSplat과의 상호작용
  - D < C (confidence가 GPR에도 도움됨) → C3는 general observation weighting

에피소드: 10 uniform episodes (raydrop 비교와 동일)
결과: results/gpr_confidence_ablation.json
"""

import sys
import os
import json
import numpy as np
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from wavesplat.data_loader import EpisodeDataLoader
from wavesplat.run import run_estimation, set_seed
from wavesplat.baselines.gpr_baseline import GPRBaseline
from wavesplat.evaluation.metrics import compute_all_metrics


EPISODES = [
    'episode_BF2_1comp_seed0_uniform.h5',
    'episode_BF3_1comp_seed0_uniform.h5',
    'episode_BF3_2comp_seed0_uniform.h5',
    'episode_BF3_5comp_seed0_uniform.h5',
    'episode_BF4_1comp_seed0_uniform.h5',
    'episode_BF4_5comp_seed0_uniform.h5',
    'episode_BF4_15comp_seed0_uniform.h5',
    'episode_BF4_30comp_seed0_uniform.h5',
    'episode_BF5_15comp_seed0_uniform.h5',
    'episode_BF5_30comp_seed0_uniform.h5',
]


def run_wavesplat_with_conf(h5_path):
    """WaveSplat with confidence (full system)."""
    config = {
        'init_method': 'v2_konly',
        'use_dispersion': True,
        'use_confidence': True,
        'adaptive_n': True,
        'tracking_window': 3,
        'seed': 42,
    }
    r = run_estimation(h5_path, config=config)
    return {'avg_nRMSE': float(r['avg_nRMSE']),
            'avg_inference_ms': float(r['avg_inference_ms'])}


def run_wavesplat_no_conf(h5_path):
    """WaveSplat without confidence (A2 ablation)."""
    config = {
        'init_method': 'v2_konly',
        'use_dispersion': True,
        'use_confidence': False,
        'adaptive_n': True,
        'tracking_window': 3,
        'seed': 42,
    }
    r = run_estimation(h5_path, config=config)
    return {'avg_nRMSE': float(r['avg_nRMSE']),
            'avg_inference_ms': float(r['avg_inference_ms'])}


def _compute_confidence_weights(x_obs, y_obs, z_obs, d_ref=125.0):
    """
    Flat-surface confidence weights: c_j = cos(theta_j) / (d_j / d_ref)
    where theta_j = arccos(|dz| / |ray|) assuming flat surface normal = z-hat,
    and d_j = sqrt(x_j^2 + y_j^2 + (mean_z)^2) approximation.

    For the ablation we use the same flat-surface approximation as real_data.py:
      - ray direction from origin to (x_j, y_j, z_obs_j)
      - surface normal ~ z-hat
      - theta_j = arccos(|z_j_component| / |ray|)
    """
    d_j = np.sqrt(x_obs**2 + y_obs**2 + z_obs**2)
    d_j = np.maximum(d_j, 1.0)  # avoid div by zero
    cos_theta = np.abs(z_obs) / d_j  # flat-surface approximation
    cos_theta = np.clip(cos_theta, 0.0, 1.0)
    w = cos_theta / (d_j / d_ref)
    w = np.maximum(w, 1e-6)
    return w


def run_gpr_no_conf(h5_path):
    """Standard GPR (no confidence weighting, uniform weights)."""
    loader = EpisodeDataLoader(h5_path)
    gt_x = loader.gt_grid_x.cpu().numpy()
    gt_y = loader.gt_grid_y.cpu().numpy()

    gpr = GPRBaseline(kernel_type='matern', length_scale=20.0,
                      noise_sigma=0.02, window_frames=1)
    frame_results = []
    for frame in loader:
        x = frame.x_obs.cpu().numpy()
        y = frame.y_obs.cpu().numpy()
        z = frame.z_obs.cpu().numpy()
        gpr.reset()
        elapsed = gpr.update(x, y, z)
        pred = gpr.predict_grid(gt_x, gt_y)
        gt = frame.gt_field.cpu().numpy()
        m = compute_all_metrics(pred, gt, gt_x, gt_y, loader.Hs_true)
        m['elapsed_ms'] = elapsed
        frame_results.append(m)

    return {'avg_nRMSE': float(np.mean([r['nRMSE'] for r in frame_results])),
            'avg_inference_ms': float(np.mean([r['elapsed_ms'] for r in frame_results]))}


def run_gpr_with_conf(h5_path):
    """GPR with confidence weighting applied as sample_weight in sklearn GP fit."""
    loader = EpisodeDataLoader(h5_path)
    gt_x = loader.gt_grid_x.cpu().numpy()
    gt_y = loader.gt_grid_y.cpu().numpy()

    frame_results = []
    for frame in loader:
        x = frame.x_obs.cpu().numpy()
        y = frame.y_obs.cpu().numpy()
        z = frame.z_obs.cpu().numpy()

        t0 = time.time()
        weights = _compute_confidence_weights(x, y, z)

        if len(x) < 3:
            elapsed = (time.time() - t0) * 1000
            frame_results.append({'nRMSE': float('nan'), 'elapsed_ms': elapsed})
            continue

        # sklearn GP does not directly support sample_weight in fit().
        # Equivalent: scale noise by 1/sqrt(w_j), i.e., use diagonal alpha.
        # We implement this by modifying alpha per-sample.
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import Matern, WhiteKernel

        X = np.column_stack([x, y])
        kernel = Matern(length_scale=20.0, nu=2.5,
                        length_scale_bounds=(1.0, 200.0))
        # alpha_j = sigma^2 / w_j  (higher weight → lower noise)
        base_noise = 0.02 ** 2
        alpha = base_noise / np.clip(weights, 1e-4, None)

        gp = GaussianProcessRegressor(kernel=kernel, alpha=alpha,
                                      normalize_y=True, n_restarts_optimizer=2)
        gp.fit(X, z)

        Xg, Yg = np.meshgrid(gt_x, gt_y)
        X_pred = np.column_stack([Xg.ravel(), Yg.ravel()])
        pred = gp.predict(X_pred).reshape(len(gt_y), len(gt_x))
        elapsed = (time.time() - t0) * 1000

        gt = frame.gt_field.cpu().numpy()
        m = compute_all_metrics(pred, gt, gt_x, gt_y, loader.Hs_true)
        m['elapsed_ms'] = elapsed
        frame_results.append(m)

    valid = [r for r in frame_results if not np.isnan(r.get('nRMSE', float('nan')))]
    return {
        'avg_nRMSE': float(np.mean([r['nRMSE'] for r in valid])) if valid else float('nan'),
        'avg_inference_ms': float(np.mean([r['elapsed_ms'] for r in frame_results])),
    }


def main():
    data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'results')
    os.makedirs(output_dir, exist_ok=True)

    methods = {
        'WS+conf': run_wavesplat_with_conf,
        'WS-conf': run_wavesplat_no_conf,
        'GPR':     run_gpr_no_conf,
        'GPR+conf': run_gpr_with_conf,
    }

    print("GPR + Confidence Ablation")
    print("Goal: does confidence weighting help GPR as much as WaveSplat?")
    print("=" * 72)
    print(f"{'Episode':<22}{'WS+conf':>9}{'WS-conf':>9}{'GPR':>9}{'GPR+conf':>10}")
    print("-" * 72)

    all_results = {}

    for ep_name in EPISODES:
        h5_path = os.path.join(data_dir, ep_name)
        if not os.path.exists(h5_path):
            print(f"SKIP: {ep_name}")
            continue

        short = ep_name.replace('episode_', '').replace('.h5', '').replace('_seed0_uniform', '')
        all_results[short] = {}
        row_vals = {}

        for label, fn in methods.items():
            try:
                set_seed(42)
                r = fn(h5_path)
                all_results[short][label] = r
                row_vals[label] = r['avg_nRMSE']
            except Exception as e:
                all_results[short][label] = {'error': str(e)}
                row_vals[label] = float('nan')
                print(f"  [{short}] {label} FAILED: {e}")

        print(f"{short:<22}"
              f"{row_vals.get('WS+conf', float('nan')):>9.4f}"
              f"{row_vals.get('WS-conf', float('nan')):>9.4f}"
              f"{row_vals.get('GPR', float('nan')):>9.4f}"
              f"{row_vals.get('GPR+conf', float('nan')):>10.4f}",
              flush=True)

    # Summary
    print("\n" + "=" * 72)
    summary = {}
    for label in methods:
        vals = [all_results[ep][label]['avg_nRMSE']
                for ep in all_results
                if label in all_results[ep] and 'avg_nRMSE' in all_results[ep][label]]
        summary[label] = float(np.mean(vals)) if vals else float('nan')

    print(f"{'Method':<12}{'Avg nRMSE':>12}")
    print("-" * 26)
    for label, avg in summary.items():
        print(f"{label:<12}{avg:>12.4f}")

    ws_gain = (summary['WS-conf'] - summary['WS+conf']) / summary['WS-conf'] * 100
    gpr_gain = (summary['GPR'] - summary['GPR+conf']) / summary['GPR'] * 100
    print(f"\nConf weighting gain — WaveSplat: {ws_gain:+.1f}%")
    print(f"Conf weighting gain — GPR:       {gpr_gain:+.1f}%")

    if abs(gpr_gain) < 2.0:
        print("\n→ Confidence weighting does NOT significantly help GPR")
        print("  → C3 is specific to WaveSplat's spectral parameterization")
    elif gpr_gain > 2.0:
        print("\n→ Confidence weighting helps GPR too")
        print("  → C3 is a general observation-weighting technique")
        print("  → Recommend reframing C3 as 'heteroscedastic observation model'")
    else:
        print("\n→ Mixed: GPR slightly improved but far less than WaveSplat")

    out_path = os.path.join(output_dir, 'gpr_confidence_ablation.json')
    with open(out_path, 'w') as f:
        json.dump({'summary': summary, 'results': all_results}, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == '__main__':
    main()
