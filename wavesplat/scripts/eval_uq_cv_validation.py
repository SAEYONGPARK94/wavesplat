"""
GP K_disp UQ Cross-Validation: Apply learned scale factor to held-out episodes.

Protocol:
  - "Train" set: 6 pilot episodes (seed 0) → learned scale_factor = 5.40 (lambda=0)
  - "Test" set: 6 held-out episodes (seeds 1-4, diverse BF/comp)
  - Apply fixed scale_factor to held-out episodes and check coverage

If coverage holds on held-out episodes → practical UQ (no oracle needed).
Results: results/gp_uq_cv_validation.json
"""
import os, sys, json, time
import numpy as np
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from wavesplat.data_loader import EpisodeDataLoader
from scripts.gp_disp_pilot import (
    estimate_components_wavesplat_opt, build_K_disp_oracle,
    compute_metrics_simple, N_FRAMES, SIGMA_OBS
)

OUT_PATH = 'results/gp_uq_cv_validation.json'
PCTS = [50, 68, 80, 90, 95]

# Learned from 6 pilot episodes (lambda=0, n_starts=5)
LEARNED_SCALE = 5.40

# Held-out episodes: different seeds, same categories
HELD_OUT_EPISODES = [
    'episode_BF3_5comp_seed1_uniform.h5',   # simple, seed 1
    'episode_BF4_5comp_seed2_uniform.h5',   # simple, seed 2
    'episode_BF4_15comp_seed3_uniform.h5',  # medium, seed 3
    'episode_BF4_30comp_seed1_uniform.h5',  # complex, seed 1
    'episode_BF5_15comp_seed4_uniform.h5',  # medium-hard, seed 4
    'episode_BF5_30comp_seed2_uniform.h5',  # complex-hard, seed 2
]


def gp_predict_full(K_oo, K_po, z_obs, sigma_obs, beam_components):
    """GP posterior mean + variance."""
    n = K_oo.shape[0]
    K_oo_noise = K_oo + sigma_obs**2 * np.eye(n)
    K_oo_noise += 1e-6 * np.trace(K_oo_noise) / n * np.eye(n)
    try:
        L = np.linalg.cholesky(K_oo_noise)
    except np.linalg.LinAlgError:
        return None, None

    alpha = np.linalg.solve(L.T, np.linalg.solve(L, z_obs))
    mu = K_po @ alpha

    V = np.linalg.solve(L, K_po.T)
    reduction = np.sum(V**2, axis=0)

    A = beam_components[:, 0]
    prior_var = float(np.sum(A**2 / 2.0))
    prior_diag = prior_var * np.ones(K_po.shape[0])

    var_post = np.maximum(prior_diag - reduction, 1e-8)
    std_post = np.sqrt(var_post)
    return mu, std_post


def compute_coverage(residuals, std_post):
    """Compute coverage at multiple CI levels."""
    coverage = {}
    for pct in PCTS:
        p = pct / 100.0
        z = stats.norm.ppf((1 + p) / 2)
        within = np.abs(residuals) <= z * std_post
        coverage[pct] = float(np.mean(within))
    return coverage


def compute_crps(residuals, std_post):
    """CRPS for Gaussian predictive distribution."""
    u = residuals / (std_post + 1e-8)
    crps = std_post * (u * (2 * stats.norm.cdf(u) - 1) +
                       2 * stats.norm.pdf(u) - 1 / np.sqrt(np.pi))
    return float(np.mean(crps))


def eval_episode_uq(h5_path, fixed_scale, n_frames=N_FRAMES):
    """Evaluate UQ on one episode with fixed (learned) scale factor."""
    loader = EpisodeDataLoader(h5_path)
    Hs = loader.Hs_true
    gt_x = loader.gt_grid_x.cpu().numpy()
    gt_y = loader.gt_grid_y.cpu().numpy()
    Xg, Yg = np.meshgrid(gt_x, gt_y)
    X_pred = np.column_stack([Xg.ravel(), Yg.ravel()])

    # lambda=0 (no regularization)
    beam_components = estimate_components_wavesplat_opt(
        loader, apply_reg_amp=True, n_starts=5, lambda_reg=0.0,
        init_tp_factor=2.0, sigma_obs=SIGMA_OBS
    )

    eval_frames = loader.get_tracking_frames(tp_factor=2.0)

    all_residuals = []
    all_std_post = []
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

        all_residuals.append(residuals)
        all_std_post.append(std_post)
        frame_count += 1

    if not nrmse_list:
        return None

    all_res = np.concatenate(all_residuals)
    all_std = np.concatenate(all_std_post)

    # Raw (uncalibrated) metrics
    raw_cov = compute_coverage(all_res, all_std)
    raw_crps = compute_crps(all_res, all_std)
    raw_sharpness = float(np.mean(all_std))

    # Oracle scale (episode-specific optimal)
    oracle_scale = np.sqrt(np.mean(all_res**2)) / np.mean(all_std)

    # Fixed scale (learned from pilot)
    fixed_std = all_std * fixed_scale
    fixed_cov = compute_coverage(all_res, fixed_std)
    fixed_crps = compute_crps(all_res, fixed_std)
    fixed_sharpness = float(np.mean(fixed_std))

    # Oracle-scaled for reference
    oracle_std = all_std * oracle_scale
    oracle_cov = compute_coverage(all_res, oracle_std)
    oracle_crps = compute_crps(all_res, oracle_std)

    A = beam_components[:, 0]

    return {
        'nRMSE': float(np.mean(nrmse_list)),
        'oracle_scale': float(oracle_scale),
        'raw': {
            'coverage': raw_cov,
            'sharpness': raw_sharpness,
            'crps': raw_crps,
        },
        'fixed_scale': {
            'scale': float(fixed_scale),
            'coverage': fixed_cov,
            'sharpness': fixed_sharpness,
            'crps': fixed_crps,
        },
        'oracle_scaled': {
            'scale': float(oracle_scale),
            'coverage': oracle_cov,
            'crps': oracle_crps,
        },
        'n_frames': frame_count,
        'sum_A2_half': float(np.sum(A**2 / 2.0)),
    }


def main():
    print("=" * 70)
    print("GP K_disp UQ Cross-Validation")
    print(f"Learned scale factor: {LEARNED_SCALE:.2f} (from 6 pilot episodes)")
    print(f"Held-out episodes: {len(HELD_OUT_EPISODES)}")
    print("=" * 70)

    all_results = {}
    cov_fixed_all = {p: [] for p in PCTS}
    cov_oracle_all = {p: [] for p in PCTS}
    nrmse_all = []
    crps_fixed_all = []
    oracle_scales = []

    t_start = time.time()

    for h5_name in HELD_OUT_EPISODES:
        h5_path = os.path.join('data', h5_name)
        ep_key = h5_name.replace('episode_', '').replace('.h5', '')
        print(f"\n  [{ep_key}] ...", end=' ', flush=True)

        if not os.path.exists(h5_path):
            print("NOT FOUND, skip")
            continue

        r = eval_episode_uq(h5_path, fixed_scale=LEARNED_SCALE)
        if r is None:
            print("SKIP (no valid frames)")
            continue

        all_results[ep_key] = r
        nrmse_all.append(r['nRMSE'])
        oracle_scales.append(r['oracle_scale'])
        crps_fixed_all.append(r['fixed_scale']['crps'])

        for pct in PCTS:
            cov_fixed_all[pct].append(r['fixed_scale']['coverage'][pct])
            cov_oracle_all[pct].append(r['oracle_scaled']['coverage'][pct])

        print(f"nRMSE={r['nRMSE']:.4f}  oracle_scale={r['oracle_scale']:.2f}  "
              f"fixed_90%={r['fixed_scale']['coverage'][90]:.3f}")

    elapsed = time.time() - t_start

    if not all_results:
        print("No valid results")
        return

    # Summary
    print(f"\n{'='*70}")
    print("CROSS-VALIDATION SUMMARY")
    print(f"{'='*70}")
    print(f"  Learned scale: {LEARNED_SCALE:.2f}")
    print(f"  Oracle scales on held-out: {[f'{s:.2f}' for s in oracle_scales]}")
    print(f"  Mean oracle scale: {np.mean(oracle_scales):.2f} "
          f"(vs learned {LEARNED_SCALE:.2f}, diff={abs(np.mean(oracle_scales)-LEARNED_SCALE):.2f})")
    print(f"  nRMSE mean: {np.mean(nrmse_all):.4f}")

    print(f"\n  {'Nominal':>8}  {'Fixed-scale':>12}  {'Error':>8}  {'Oracle':>8}  Status")
    print("  " + "-" * 58)
    calibrated_count = 0
    for pct in PCTS:
        emp_fixed = np.mean(cov_fixed_all[pct])
        err_fixed = emp_fixed - pct/100.0
        emp_oracle = np.mean(cov_oracle_all[pct])
        status = 'OK' if abs(err_fixed) < 0.05 else (
            'over-conf' if err_fixed < 0 else 'under-conf')
        if abs(err_fixed) < 0.05:
            calibrated_count += 1
        print(f"  {pct:7d}%  {emp_fixed:12.3f}  {err_fixed:+8.3f}  {emp_oracle:8.3f}  {status}")

    print(f"\n  Fixed-scale CRPS: {np.mean(crps_fixed_all):.4f}")
    print(f"  Calibrated CIs: {calibrated_count}/{len(PCTS)}")
    print(f"  Elapsed: {elapsed:.0f}s")

    verdict = "PASS" if calibrated_count >= 4 else "FAIL"
    print(f"\n  VERDICT: {verdict}")
    if verdict == "PASS":
        print("  → Learned scale factor generalizes to unseen episodes!")
        print("  → GP K_disp provides practical calibrated UQ without oracle.")
    else:
        print("  → Scale factor does NOT generalize. UQ requires per-episode calibration.")

    summary = {
        'verdict': verdict,
        'learned_scale': LEARNED_SCALE,
        'mean_oracle_scale_heldout': float(np.mean(oracle_scales)),
        'nRMSE_mean': float(np.mean(nrmse_all)),
        'fixed_coverage': {p: float(np.mean(cov_fixed_all[p])) for p in PCTS},
        'fixed_coverage_error': {p: float(np.mean(cov_fixed_all[p]) - p/100.0) for p in PCTS},
        'oracle_coverage': {p: float(np.mean(cov_oracle_all[p])) for p in PCTS},
        'fixed_crps': float(np.mean(crps_fixed_all)),
        'calibrated_cis': f"{calibrated_count}/{len(PCTS)}",
        'elapsed_sec': elapsed,
    }

    output = {
        'description': 'CV validation: learned scale factor applied to held-out episodes',
        'protocol': {
            'train_set': '6 pilot episodes (seed 0), lambda=0',
            'test_set': f'{len(HELD_OUT_EPISODES)} held-out episodes (seeds 1-4)',
            'learned_scale': LEARNED_SCALE,
            'lambda_reg': 0.0,
            'n_starts': 5,
            'sigma_obs': SIGMA_OBS,
        },
        'per_episode': all_results,
        'summary': summary,
    }
    with open(OUT_PATH, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\n  Saved: {OUT_PATH}")


if __name__ == '__main__':
    main()
