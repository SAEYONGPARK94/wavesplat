"""
GP K_disp UQ Calibration Analysis (ws_ms5_reg_amp)

Formal coverage probability 분석:
  - 50/68/80/90/95% prediction interval의 empirical coverage
  - Calibration curve: nominal vs empirical coverage
  - Sharpness: average posterior std
  - CRPS (Continuous Ranked Probability Score)

6 pilot episodes, 결과: results/gp_uq_coverage_eval.json
"""
import os, sys, json
import numpy as np
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from wavesplat.data_loader import EpisodeDataLoader
from scripts.gp_disp_pilot import (
    estimate_components_wavesplat_opt, build_K_disp_oracle,
    compute_metrics_simple, TEST_EPISODES, N_FRAMES, SIGMA_OBS
)

OUT_PATH = 'results/gp_uq_coverage_eval.json'
# Use integer percent as keys to avoid float-string issues
PCTS = [50, 68, 80, 90, 95]
G = 9.81


def gp_predict_full(K_oo, K_po, z_obs, sigma_obs, beam_components):
    """GP posterior mean + variance. Returns (mu, std_post)."""
    n = K_oo.shape[0]
    K_oo_noise = K_oo + sigma_obs**2 * np.eye(n)
    # Add small jitter for numerical stability
    K_oo_noise += 1e-6 * np.trace(K_oo_noise) / n * np.eye(n)
    try:
        L = np.linalg.cholesky(K_oo_noise)
    except np.linalg.LinAlgError:
        return None, None

    alpha = np.linalg.solve(L.T, np.linalg.solve(L, z_obs))
    mu = K_po @ alpha

    # V = L^{-1} K_po.T  →  shape (n, n_pred)
    V = np.linalg.solve(L, K_po.T)
    reduction = np.sum(V**2, axis=0)   # (n_pred,)

    # Prior diagonal: K_disp(x*, x*) = sum_n A_n^2/2
    A = beam_components[:, 0]   # amplitude = column 0 (NOT 3, which is ky)
    prior_var = float(np.sum(A**2 / 2.0))
    prior_diag = prior_var * np.ones(K_po.shape[0])

    var_post = np.maximum(prior_diag - reduction, 1e-8)
    std_post = np.sqrt(var_post)
    return mu, std_post


def eval_episode_uq(h5_path, n_frames=N_FRAMES):
    loader = EpisodeDataLoader(h5_path)
    Hs = loader.Hs_true
    gt_x = loader.gt_grid_x.cpu().numpy()
    gt_y = loader.gt_grid_y.cpu().numpy()
    Xg, Yg = np.meshgrid(gt_x, gt_y)
    X_pred = np.column_stack([Xg.ravel(), Yg.ravel()])

    beam_components = estimate_components_wavesplat_opt(
        loader, apply_reg_amp=True, n_starts=5, lambda_reg=2.0,
        init_tp_factor=2.0, sigma_obs=SIGMA_OBS
    )

    eval_frames = loader.get_tracking_frames(tp_factor=2.0)

    # Initialize per-pct coverage lists
    cov_lists = {p: [] for p in PCTS}
    sharpness_list = []
    crps_list = []
    nrmse_list = []
    frame_count = 0

    for frame in eval_frames:
        if frame_count >= n_frames:
            break
        x_obs = frame.x_obs.cpu().numpy()
        y_obs = frame.y_obs.cpu().numpy()
        z_obs = frame.z_obs.cpu().numpy()
        gt = frame.gt_field.cpu().numpy()
        if len(x_obs) < 5:
            frame_count += 1
            continue

        X_obs = np.column_stack([x_obs, y_obs])
        K_oo = build_K_disp_oracle(X_obs, X_obs, beam_components)
        K_po = build_K_disp_oracle(X_pred, X_obs, beam_components)

        mu, std_post = gp_predict_full(K_oo, K_po, z_obs, SIGMA_OBS, beam_components)
        if mu is None:
            frame_count += 1
            continue

        pred = mu.reshape(len(gt_y), len(gt_x))
        gt_flat = gt.ravel()
        residuals = gt_flat - mu

        m = compute_metrics_simple(pred, gt, Hs)
        nrmse_list.append(m['nRMSE'])

        for pct in PCTS:
            p = pct / 100.0
            z = stats.norm.ppf((1 + p) / 2)
            within = np.abs(residuals) <= z * std_post
            cov_lists[pct].append(float(np.mean(within)))

        sharpness_list.append(float(np.mean(std_post)))

        # CRPS for Normal: sigma*(u*(2Phi(u)-1) + 2phi(u) - 1/sqrt(pi))
        u = residuals / (std_post + 1e-8)
        crps = std_post * (u * (2*stats.norm.cdf(u) - 1) +
                           2*stats.norm.pdf(u) - 1/np.sqrt(np.pi))
        crps_list.append(float(np.mean(crps)))

        frame_count += 1

    if not nrmse_list:
        return None

    return {
        'nRMSE': float(np.mean(nrmse_list)),
        'coverage': {p: float(np.mean(cov_lists[p])) for p in PCTS},
        'sharpness_mean_std': float(np.mean(sharpness_list)),
        'crps': float(np.mean(crps_list)),
        'n_frames': frame_count,
    }


def main():
    print("=" * 70)
    print("GP K_disp UQ Coverage Analysis (ws_ms5_reg_amp)")
    print(f"Episodes: {len(TEST_EPISODES)}, n_frames={N_FRAMES}")
    print("=" * 70)

    all_results = {}
    cov_all = {p: [] for p in PCTS}
    crps_all = []
    sharpness_all = []

    for h5_name in TEST_EPISODES:
        h5_path = os.path.join('data', h5_name)
        ep_key = h5_name.replace('episode_', '').replace('.h5', '')
        print(f"\n  [{ep_key}]")

        r = eval_episode_uq(h5_path)
        if r is None:
            print("    SKIP (no valid frames)")
            continue

        all_results[ep_key] = r
        print(f"    nRMSE={r['nRMSE']:.4f}  CRPS={r['crps']:.4f}  "
              f"sharpness(std)={r['sharpness_mean_std']:.4f}m  frames={r['n_frames']}")
        print("    Coverage  nominal→empirical (error):")
        for pct in PCTS:
            emp = r['coverage'][pct]
            err = emp - pct/100.0
            bar = ' ← over-confident' if err < -0.05 else (' ← under-confident' if err > 0.05 else '')
            print(f"      {pct:2d}%: {emp:.3f}  ({err:+.3f}){bar}")

        for pct in PCTS:
            cov_all[pct].append(r['coverage'][pct])
        crps_all.append(r['crps'])
        sharpness_all.append(r['sharpness_mean_std'])

    if not all_results:
        print("No valid results")
        return

    summary = {
        'coverage_mean':  {p: float(np.mean(cov_all[p])) for p in PCTS},
        'coverage_error': {p: float(np.mean(cov_all[p]) - p/100.0) for p in PCTS},
        'crps_mean':      float(np.mean(crps_all)),
        'sharpness_mean': float(np.mean(sharpness_all)),
    }

    print("\n" + "=" * 70)
    print("AGGREGATE SUMMARY")
    print("=" * 70)
    print(f"  {'Nominal':>8}  {'Empirical':>10}  {'Error':>8}  Status")
    print("  " + "-" * 50)
    for pct in PCTS:
        emp = summary['coverage_mean'][pct]
        err = summary['coverage_error'][pct]
        status = '✓ calibrated' if abs(err) < 0.05 else (
            'over-confident' if err < 0 else 'under-confident')
        print(f"  {pct:7d}%  {emp:10.3f}  {err:+8.3f}  {status}")
    print(f"\n  CRPS (lower=better): {summary['crps_mean']:.4f}")
    print(f"  Mean posterior std:  {summary['sharpness_mean']:.4f} m")

    output = {
        'per_episode': all_results,
        'summary': summary,
        'config': {'mode': 'ws_ms5_reg_amp', 'n_starts': 5,
                   'lambda_reg': 2.0, 'sigma_obs': SIGMA_OBS},
    }
    with open(OUT_PATH, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved: {OUT_PATH}")


if __name__ == '__main__':
    main()
