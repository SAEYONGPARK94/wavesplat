"""
GP K_disp UQ Calibration Fix: lambda sweep + post-hoc variance scaling

Problem: lambda_reg=2.0 causes amplitude degeneration → posterior variance too small
         → 90% CI coverage only 24.8% (should be ~90%)

Fix approaches:
  1. Reduce lambda (0, 0.5, 2.0) to restore amplitude → improve prior variance
  2. Post-hoc scaling: scale_factor = empirical_std / mean_posterior_std

6 pilot episodes, results: results/gp_uq_calibration_fix.json
"""
import os, sys, json, time
import numpy as np
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from wavesplat.data_loader import EpisodeDataLoader
from scripts.gp_disp_pilot import (
    estimate_components_wavesplat_opt, build_K_disp_oracle,
    compute_metrics_simple, TEST_EPISODES, N_FRAMES, SIGMA_OBS,
    _apply_reg_ml_amp, _neg_log_ml_per_comp
)

OUT_PATH = 'results/gp_uq_calibration_fix.json'
PCTS = [50, 68, 80, 90, 95]
LAMBDA_VALUES = [0.0, 0.5, 2.0]


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


def compute_coverage_metrics(residuals, std_post):
    """Compute coverage, CRPS, sharpness from residuals and posterior std."""
    coverage = {}
    for pct in PCTS:
        p = pct / 100.0
        z = stats.norm.ppf((1 + p) / 2)
        within = np.abs(residuals) <= z * std_post
        coverage[pct] = float(np.mean(within))

    sharpness = float(np.mean(std_post))

    u = residuals / (std_post + 1e-8)
    crps = std_post * (u * (2 * stats.norm.cdf(u) - 1) +
                       2 * stats.norm.pdf(u) - 1 / np.sqrt(np.pi))
    crps_mean = float(np.mean(crps))

    return coverage, sharpness, crps_mean


def eval_episode_uq(h5_path, lambda_reg, n_frames=N_FRAMES):
    loader = EpisodeDataLoader(h5_path)
    Hs = loader.Hs_true
    gt_x = loader.gt_grid_x.cpu().numpy()
    gt_y = loader.gt_grid_y.cpu().numpy()
    Xg, Yg = np.meshgrid(gt_x, gt_y)
    X_pred = np.column_stack([Xg.ravel(), Yg.ravel()])

    beam_components = estimate_components_wavesplat_opt(
        loader, apply_reg_amp=True, n_starts=5, lambda_reg=lambda_reg,
        init_tp_factor=2.0, sigma_obs=SIGMA_OBS
    )

    eval_frames = loader.get_tracking_frames(tp_factor=2.0)

    cov_lists = {p: [] for p in PCTS}
    sharpness_list = []
    crps_list = []
    nrmse_list = []
    all_residuals = []
    all_std_post = []
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

        coverage, sharpness, crps_mean = compute_coverage_metrics(residuals, std_post)
        for pct in PCTS:
            cov_lists[pct].append(coverage[pct])
        sharpness_list.append(sharpness)
        crps_list.append(crps_mean)

        all_residuals.append(residuals)
        all_std_post.append(std_post)
        frame_count += 1

    if not nrmse_list:
        return None

    # Concatenate all residuals and std for post-hoc calibration
    all_res = np.concatenate(all_residuals)
    all_std = np.concatenate(all_std_post)

    # Post-hoc scaling factor: match empirical std to posterior std
    empirical_rmse = np.sqrt(np.mean(all_res**2))
    mean_post_std = np.mean(all_std)
    scale_factor = empirical_rmse / mean_post_std

    # Recalculate coverage with scaled std
    scaled_std = all_std * scale_factor
    cov_scaled, sharp_scaled, crps_scaled = compute_coverage_metrics(all_res, scaled_std)

    # Amplitude stats
    A = beam_components[:, 0]
    amp_stats = {
        'mean_A': float(np.mean(A)),
        'std_A': float(np.std(A)),
        'min_A': float(np.min(A)),
        'max_A': float(np.max(A)),
        'sum_A2_half': float(np.sum(A**2 / 2.0)),
    }

    return {
        'nRMSE': float(np.mean(nrmse_list)),
        'raw': {
            'coverage': {p: float(np.mean(cov_lists[p])) for p in PCTS},
            'sharpness_mean_std': float(np.mean(sharpness_list)),
            'crps': float(np.mean(crps_list)),
        },
        'posthoc_scaled': {
            'scale_factor': float(scale_factor),
            'coverage': cov_scaled,
            'sharpness_mean_std': float(sharp_scaled),
            'crps': float(crps_scaled),
        },
        'amp_stats': amp_stats,
        'n_frames': frame_count,
    }


def main():
    print("=" * 70)
    print("GP K_disp UQ Calibration Fix")
    print(f"Lambda values: {LAMBDA_VALUES}")
    print(f"Episodes: {len(TEST_EPISODES)}, n_frames={N_FRAMES}")
    print("=" * 70)

    results = {}

    for lam in LAMBDA_VALUES:
        lam_key = f"lambda_{lam}"
        print(f"\n{'='*60}")
        print(f"  Lambda = {lam}")
        print(f"{'='*60}")

        ep_results = {}
        cov_raw_all = {p: [] for p in PCTS}
        cov_scaled_all = {p: [] for p in PCTS}
        nrmse_all = []
        crps_raw_all = []
        crps_scaled_all = []
        sharp_raw_all = []
        sharp_scaled_all = []
        scale_factors = []

        t0 = time.time()
        for h5_name in TEST_EPISODES:
            h5_path = os.path.join('data', h5_name)
            ep_key = h5_name.replace('episode_', '').replace('.h5', '')
            print(f"\n  [{ep_key}] ...", end=' ', flush=True)

            r = eval_episode_uq(h5_path, lambda_reg=lam)
            if r is None:
                print("SKIP")
                continue

            ep_results[ep_key] = r
            nrmse_all.append(r['nRMSE'])
            scale_factors.append(r['posthoc_scaled']['scale_factor'])
            crps_raw_all.append(r['raw']['crps'])
            crps_scaled_all.append(r['posthoc_scaled']['crps'])
            sharp_raw_all.append(r['raw']['sharpness_mean_std'])
            sharp_scaled_all.append(r['posthoc_scaled']['sharpness_mean_std'])

            for pct in PCTS:
                cov_raw_all[pct].append(r['raw']['coverage'][pct])
                cov_scaled_all[pct].append(r['posthoc_scaled']['coverage'][pct])

            print(f"nRMSE={r['nRMSE']:.4f}  scale={r['posthoc_scaled']['scale_factor']:.2f}  "
                  f"sum(A²/2)={r['amp_stats']['sum_A2_half']:.4f}")

        elapsed = time.time() - t0

        if not ep_results:
            continue

        summary = {
            'nRMSE_mean': float(np.mean(nrmse_all)),
            'raw_coverage': {p: float(np.mean(cov_raw_all[p])) for p in PCTS},
            'raw_crps': float(np.mean(crps_raw_all)),
            'raw_sharpness': float(np.mean(sharp_raw_all)),
            'scaled_coverage': {p: float(np.mean(cov_scaled_all[p])) for p in PCTS},
            'scaled_crps': float(np.mean(crps_scaled_all)),
            'scaled_sharpness': float(np.mean(sharp_scaled_all)),
            'mean_scale_factor': float(np.mean(scale_factors)),
            'elapsed_sec': elapsed,
        }

        results[lam_key] = {
            'per_episode': ep_results,
            'summary': summary,
        }

        # Print summary
        print(f"\n  --- Lambda={lam} Summary ---")
        print(f"  nRMSE: {summary['nRMSE_mean']:.4f}")
        print(f"  Scale factor: {summary['mean_scale_factor']:.2f}")
        print(f"\n  RAW coverage:")
        for pct in PCTS:
            emp = summary['raw_coverage'][pct]
            err = emp - pct/100.0
            print(f"    {pct}%: {emp:.3f} (err={err:+.3f})")
        print(f"  RAW CRPS: {summary['raw_crps']:.4f}")

        print(f"\n  POST-HOC SCALED coverage:")
        for pct in PCTS:
            emp = summary['scaled_coverage'][pct]
            err = emp - pct/100.0
            status = 'OK' if abs(err) < 0.05 else ('over-conf' if err < 0 else 'under-conf')
            print(f"    {pct}%: {emp:.3f} (err={err:+.3f}) {status}")
        print(f"  Scaled CRPS: {summary['scaled_crps']:.4f}")

    # Save
    output = {
        'description': 'UQ calibration fix: lambda sweep + post-hoc variance scaling',
        'lambda_values': LAMBDA_VALUES,
        'episodes': TEST_EPISODES,
        'n_frames': N_FRAMES,
        'sigma_obs': SIGMA_OBS,
        'results': results,
    }
    with open(OUT_PATH, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved: {OUT_PATH}")


if __name__ == '__main__':
    main()
