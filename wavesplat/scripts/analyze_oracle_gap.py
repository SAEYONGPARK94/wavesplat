#!/usr/bin/env python3
"""
Oracle-Practical Gap 분석: 정확히 어디서 성능 차이가 발생하는지 정량화.

분석 항목:
1. 방향(kx,ky) 추정 오차 vs oracle
2. 성분 수 매칭 품질 (matched/unmatched components)
3. 커널 유사도: K_disp(estimated) vs K_disp(oracle)
4. 해상상태별 gap 분해

6 pilot episodes에서 빠르게 분석.
"""
import sys, os
import numpy as np
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from wavesplat.data_loader import EpisodeDataLoader

# Reuse functions from gp_disp_pilot
from scripts.gp_disp_pilot import (
    estimate_components_wavesplat_opt,
    build_K_disp_oracle,
    gp_predict,
    compute_metrics_simple,
    TEST_EPISODES,
    N_FRAMES,
    DATA_DIR,
    SIGMA_OBS,
)


def match_components(est_kx, est_ky, gt_kx, gt_ky):
    """Hungarian matching of estimated to ground-truth wave directions."""
    n_est = len(est_kx)
    n_gt = len(gt_kx)
    n = max(n_est, n_gt)

    # Cost matrix: angular distance in k-space
    cost = np.full((n, n), 1e6)
    for i in range(n_est):
        for j in range(n_gt):
            dk = np.sqrt((est_kx[i] - gt_kx[j])**2 + (est_ky[i] - gt_ky[j])**2)
            cost[i, j] = dk

    row_ind, col_ind = linear_sum_assignment(cost)

    matches = []
    for r, c in zip(row_ind, col_ind):
        if r < n_est and c < n_gt and cost[r, c] < 1e5:
            matches.append((r, c, cost[r, c]))

    return matches


def analyze_episode(h5_path, n_starts=5, lambda_reg=2.0):
    """Analyze one episode: direction quality, kernel similarity, prediction gap."""
    loader = EpisodeDataLoader(h5_path)
    gt_x = loader.gt_grid_x.cpu().numpy()
    gt_y = loader.gt_grid_y.cpu().numpy()
    Hs = loader.Hs_true
    depth = loader.depth

    Xg, Yg = np.meshgrid(gt_x, gt_y)
    X_pred = np.column_stack([Xg.ravel(), Yg.ravel()])

    # Ground truth components
    wave_comp = loader.wave_components  # (n_comp, 6): [A, omega, kx, ky, phi, Q]
    gt_kx = wave_comp[:, 2]
    gt_ky = wave_comp[:, 3]
    gt_A = wave_comp[:, 0]
    n_gt = len(gt_kx)

    # Estimated components
    est_comp = estimate_components_wavesplat_opt(
        loader, apply_reg_amp=True, n_starts=n_starts,
        lambda_reg=lambda_reg, init_tp_factor=2.0
    )
    est_kx = est_comp[:, 2]
    est_ky = est_comp[:, 3]
    est_A = est_comp[:, 0]
    n_est = len(est_kx)

    # 1. Direction matching
    matches = match_components(est_kx, est_ky, gt_kx, gt_ky)

    matched_dk = [m[2] for m in matches]
    matched_gt_energy = [gt_A[m[1]]**2 for m in matches]

    # Energy-weighted direction error
    total_gt_energy = np.sum(gt_A**2)
    matched_gt_indices = set(m[1] for m in matches)
    unmatched_gt_energy = sum(gt_A[j]**2 for j in range(n_gt) if j not in matched_gt_indices)

    if matched_dk:
        mean_dk = np.mean(matched_dk)
        energy_weighted_dk = np.sum([dk * e for dk, e in zip(matched_dk, matched_gt_energy)]) / max(sum(matched_gt_energy), 1e-12)
    else:
        mean_dk = float('inf')
        energy_weighted_dk = float('inf')

    # 2. Kernel similarity (Frobenius norm of difference)
    K_oracle = build_K_disp_oracle(X_pred[:100], X_pred[:100], wave_comp)
    K_est = build_K_disp_oracle(X_pred[:100], X_pred[:100], est_comp)

    K_oracle_norm = np.linalg.norm(K_oracle, 'fro')
    K_diff_norm = np.linalg.norm(K_oracle - K_est, 'fro')
    kernel_rel_error = K_diff_norm / max(K_oracle_norm, 1e-12)

    # 3. Prediction comparison on first tracking frame
    eval_frames = loader.get_tracking_frames(tp_factor=2.0)
    if eval_frames:
        fr = eval_frames[0]
        X_obs = np.column_stack([fr.x_obs.cpu().numpy(), fr.y_obs.cpu().numpy()])
        z_obs = fr.z_obs.cpu().numpy()
        gt_field = fr.gt_field.cpu().numpy()

        # Oracle GP
        K_oo_or = build_K_disp_oracle(X_obs, X_obs, wave_comp)
        K_po_or = build_K_disp_oracle(X_pred, X_obs, wave_comp)
        mu_or = gp_predict(K_oo_or, K_po_or, z_obs, SIGMA_OBS)
        mu_or = mu_or.reshape(len(gt_x), len(gt_y)).T
        m_or = compute_metrics_simple(mu_or, gt_field, Hs)

        # Estimated GP
        K_oo_est = build_K_disp_oracle(X_obs, X_obs, est_comp)
        K_po_est = build_K_disp_oracle(X_pred, X_obs, est_comp)
        mu_est = gp_predict(K_oo_est, K_po_est, z_obs, SIGMA_OBS)
        mu_est = mu_est.reshape(len(gt_x), len(gt_y)).T
        m_est = compute_metrics_simple(mu_est, gt_field, Hs)

        pred_gap = m_est['nRMSE'] - m_or['nRMSE']
    else:
        m_or = m_est = {'nRMSE': float('nan')}
        pred_gap = float('nan')

    return {
        'n_gt_comp': n_gt,
        'n_est_comp': n_est,
        'n_matched': len(matches),
        'mean_dk': float(mean_dk),
        'energy_weighted_dk': float(energy_weighted_dk),
        'unmatched_gt_energy_frac': float(unmatched_gt_energy / max(total_gt_energy, 1e-12)),
        'kernel_rel_error': float(kernel_rel_error),
        'oracle_nRMSE': float(m_or['nRMSE']),
        'est_nRMSE': float(m_est['nRMSE']),
        'pred_gap': float(pred_gap),
        'est_total_energy': float(np.sum(est_A**2)),
        'gt_total_energy': float(total_gt_energy),
        'energy_ratio': float(np.sum(est_A**2) / max(total_gt_energy, 1e-12)),
    }


if __name__ == '__main__':
    import json

    results = {}

    print("=" * 80)
    print("Oracle-Practical Gap Analysis (6 pilot episodes)")
    print("=" * 80)

    for h5_name in TEST_EPISODES:
        h5_path = os.path.join(DATA_DIR, h5_name)
        ep_key = h5_name.replace('episode_', '').replace('.h5', '')
        print(f"\n--- {ep_key} ---")

        try:
            r = analyze_episode(h5_path, n_starts=5, lambda_reg=2.0)
            results[ep_key] = r

            print(f"  Components: GT={r['n_gt_comp']}, Est={r['n_est_comp']}, Matched={r['n_matched']}")
            print(f"  Direction error: mean_dk={r['mean_dk']:.4f} rad/m, energy-weighted={r['energy_weighted_dk']:.4f}")
            print(f"  Unmatched GT energy: {r['unmatched_gt_energy_frac']:.1%}")
            print(f"  Kernel relative error: {r['kernel_rel_error']:.3f}")
            print(f"  Energy ratio (est/gt): {r['energy_ratio']:.3f}")
            print(f"  nRMSE: oracle={r['oracle_nRMSE']:.4f}, est={r['est_nRMSE']:.4f}, gap={r['pred_gap']:+.4f}")
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback; traceback.print_exc()

    # Summary
    if results:
        print("\n" + "=" * 80)
        print("SUMMARY")
        print("=" * 80)

        all_r = list(results.values())
        print(f"\n{'Metric':<35} {'Mean':>8} {'Std':>8} {'Min':>8} {'Max':>8}")
        print("-" * 75)
        for key in ['mean_dk', 'energy_weighted_dk', 'unmatched_gt_energy_frac',
                     'kernel_rel_error', 'energy_ratio', 'oracle_nRMSE', 'est_nRMSE', 'pred_gap']:
            vals = [r[key] for r in all_r if not np.isnan(r[key]) and not np.isinf(r[key])]
            if vals:
                print(f"  {key:<33} {np.mean(vals):>8.4f} {np.std(vals):>8.4f} "
                      f"{np.min(vals):>8.4f} {np.max(vals):>8.4f}")

        # Correlation analysis
        gaps = [r['pred_gap'] for r in all_r if not np.isnan(r['pred_gap'])]
        dk_vals = [r['energy_weighted_dk'] for r in all_r if not np.isnan(r['pred_gap'])]
        ker_vals = [r['kernel_rel_error'] for r in all_r if not np.isnan(r['pred_gap'])]
        unmatch_vals = [r['unmatched_gt_energy_frac'] for r in all_r if not np.isnan(r['pred_gap'])]

        if len(gaps) >= 3:
            from scipy.stats import pearsonr
            print(f"\nCorrelation with prediction gap:")
            for name, vals in [('energy_weighted_dk', dk_vals),
                               ('kernel_rel_error', ker_vals),
                               ('unmatched_gt_energy_frac', unmatch_vals)]:
                r, p = pearsonr(vals, gaps)
                print(f"  {name:<33} r={r:.3f}, p={p:.3f}")

    out_path = 'wavesplat/results/oracle_gap_analysis.json'
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {out_path}")
