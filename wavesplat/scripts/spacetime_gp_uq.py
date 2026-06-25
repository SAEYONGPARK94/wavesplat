"""
spacetime_gp_uq.py — UQ calibration for space-time dispersion-fused GP.

Computes full posterior (mean + variance) for the space-time GP, then finds
and validates a post-hoc scale factor for calibrated uncertainty quantification.

Protocol:
  - Train set: first 3 seeds (seed 0-2) — learn optimal scale factor
  - Test set:  last 2 seeds (seed 3-4) — validate coverage

Output: results/spacetime_gp_uq.json
"""
import os, sys, json, time
import numpy as np
from scipy import stats, optimize

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
os.chdir(os.path.join(os.path.dirname(__file__), '..'))

from wavesplat.data_loader import EpisodeDataLoader
from scripts.gp_disp_pilot import estimate_components_wavesplat_opt

G = 9.81
SIGMA_OBS = 0.02
W = 10
N_TARGETS = 10   # fast-look (was 30); coverage averages over episodes×frames×grid,
                 # so 10 frames/ep already gives statistically tight estimates
COMP_MODE = 'ws_reg_amp'
PCTS = [50, 68, 80, 90, 95]

SEA_STATES = [
    ('BF3',  5), ('BF4',  5), ('BF4', 15),
    ('BF4', 30), ('BF5', 15), ('BF5', 30),
]

TRAIN_SEEDS = [0, 1, 2]
TEST_SEEDS  = [3, 4]


# ── Kernel builders ───────────────────────────────────────────────────────────

def comps_to_arrays(comps, depth):
    A = comps[:, 0]; kx = comps[:, 2]; ky = comps[:, 3]
    k = np.sqrt(kx**2 + ky**2 + 1e-12)
    omega = np.sqrt(G * k * np.tanh(k * depth))
    return A, omega, kx, ky


def build_K_spacetime(P1, P2, A, omega, kx, ky):
    dx = P1[:, 0:1] - P2[None, :, 0]
    dy = P1[:, 1:2] - P2[None, :, 1]
    dt = P1[:, 2:3] - P2[None, :, 2]
    K = np.zeros((P1.shape[0], P2.shape[0]))
    for n in range(len(A)):
        K += 0.5 * A[n]**2 * np.cos(kx[n]*dx + ky[n]*dy - omega[n]*dt)
    return K


def gp_posterior(K_oo, K_po, z, A_est, sigma=SIGMA_OBS):
    """Returns (mean, posterior_std) at prediction points."""
    N = len(z)
    prior_var = float(np.sum(A_est**2 / 2.0))
    K_noise = K_oo + (sigma**2 + 1e-6) * np.eye(N)
    try:
        L = np.linalg.cholesky(K_noise)
    except np.linalg.LinAlgError:
        alpha = np.linalg.lstsq(K_noise, z, rcond=None)[0]
        mu = K_po @ alpha
        return mu, np.full(len(mu), np.sqrt(prior_var))
    alpha = np.linalg.solve(L.T, np.linalg.solve(L, z))
    mu = K_po @ alpha
    V = np.linalg.solve(L, K_po.T)
    reduction = np.sum(V**2, axis=0)
    var_post = np.maximum(prior_var - reduction, 1e-8)
    return mu, np.sqrt(var_post)


def frame_obs(f):
    x = f.x_obs.cpu().numpy(); y = f.y_obs.cpu().numpy(); z = f.z_obs.cpu().numpy()
    return np.column_stack([x, y, np.full_like(x, float(f.t))]), z


# ── Per-episode: collect (residuals, raw_std) pairs ─────────────────────────

def collect_residuals(tag, h5_name):
    loader = EpisodeDataLoader(os.path.join('data', h5_name))
    Hs = loader.Hs_true; depth = loader.depth
    gx = loader.gt_grid_x.cpu().numpy(); gy = loader.gt_grid_y.cpu().numpy()
    Xg, Yg = np.meshgrid(gx, gy)
    grid_xy = np.column_stack([Xg.ravel(), Yg.ravel()])

    comps = estimate_components_wavesplat_opt(
        loader, apply_reg_amp=True, n_starts=1, lambda_reg=0.5)
    Ae, oe, kxe, kye = comps_to_arrays(comps, depth)

    frames = loader.get_tracking_frames(tp_factor=2.0)
    n = len(frames)
    targets = list(range(W, min(n, W + N_TARGETS)))

    all_residuals = []
    all_raw_std = []
    nRMSE_list = []

    for ti in targets:
        target = frames[ti]
        t_star = float(target.t)
        gt = target.gt_field.cpu().numpy().ravel()
        grid_pts = np.column_stack([grid_xy, np.full(len(grid_xy), t_star)])

        Ps, zs = [], []
        for j in range(ti - W + 1, ti + 1):
            Pj, zj = frame_obs(frames[j]); Ps.append(Pj); zs.append(zj)
        Pw = np.vstack(Ps); zw = np.concatenate(zs)

        K_oo = build_K_spacetime(Pw, Pw, Ae, oe, kxe, kye)
        K_po = build_K_spacetime(grid_pts, Pw, Ae, oe, kxe, kye)
        mu, raw_std = gp_posterior(K_oo, K_po, zw, Ae)

        residuals = mu - gt
        all_residuals.append(residuals)
        all_raw_std.append(raw_std)
        nRMSE_list.append(float(np.sqrt(np.mean(residuals**2)) / max(Hs, 1e-6)))

    return (np.concatenate(all_residuals), np.concatenate(all_raw_std),
            float(np.mean(nRMSE_list)), Hs)


# ── Coverage metrics ──────────────────────────────────────────────────────────

def compute_coverage(residuals, std, scale):
    cov = {}
    for pct in PCTS:
        z = stats.norm.ppf((1 + pct/100) / 2)
        within = np.abs(residuals) <= z * scale * std
        cov[pct] = float(np.mean(within))
    return cov


def find_optimal_scale(residuals_list, std_list):
    """Find scale that minimises sum of squared coverage errors."""
    all_r = np.concatenate(residuals_list)
    all_s = np.concatenate(std_list)

    def loss(log_s):
        s = np.exp(log_s[0])
        total = 0.0
        for pct in PCTS:
            z = stats.norm.ppf((1 + pct/100) / 2)
            cov = np.mean(np.abs(all_r) <= z * s * all_s)
            total += (cov - pct/100)**2
        return total

    res = optimize.minimize(loss, [np.log(3.0)], method='Nelder-Mead',
                            options={'xatol': 1e-4, 'fatol': 1e-6, 'maxiter': 500})
    return float(np.exp(res.x[0]))


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("Space-Time GP UQ Calibration")
    print(f"comp_mode={COMP_MODE}, W={W}")
    print("=" * 60)

    train_residuals, train_stds, train_results = [], [], {}
    test_residuals, test_stds, test_results = [], [], {}

    for ss, nc in SEA_STATES:
        for seed in range(5):
            tag = f'{ss}_{nc}c_s{seed}'
            h5 = f'episode_{ss}_{nc}comp_seed{seed}_uniform.h5'
            fpath = os.path.join('data', h5)
            if not os.path.exists(fpath) or os.path.getsize(fpath) < 1000:
                print(f"  SKIP {tag}")
                continue

            print(f"  {tag} ...", end='', flush=True)
            t0 = time.time()
            try:
                r, s, nRMSE, Hs = collect_residuals(tag, h5)
                elapsed = time.time() - t0
                print(f" {elapsed:.0f}s  nRMSE={nRMSE:.4f}  "
                      f"prior_std={np.mean(s):.4f}")
                ep_res = {'nRMSE': nRMSE, 'Hs': Hs,
                          'mean_raw_std': float(np.mean(s)),
                          'std_residuals': float(np.std(r))}
                if seed in TRAIN_SEEDS:
                    train_residuals.append(r); train_stds.append(s)
                    train_results[tag] = ep_res
                else:
                    test_residuals.append(r); test_stds.append(s)
                    test_results[tag] = ep_res
            except Exception as e:
                print(f" ERROR: {e}")

    if not train_residuals:
        print("No training episodes available.")
        return

    # ── Learn scale factor ────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"Learning scale factor from {len(train_results)} train episodes...")
    scale = find_optimal_scale(train_residuals, train_stds)
    print(f"Optimal scale factor: {scale:.3f}")

    all_train_r = np.concatenate(train_residuals)
    all_train_s = np.concatenate(train_stds)
    train_cov = compute_coverage(all_train_r, all_train_s, scale)
    train_cov_raw = compute_coverage(all_train_r, all_train_s, 1.0)

    print(f"\nTrain coverage (raw, scale=1.0):")
    for pct in PCTS:
        print(f"  {pct}%: {train_cov_raw[pct]:.3f} (err={train_cov_raw[pct]-pct/100:+.3f})")
    print(f"\nTrain coverage (calibrated, scale={scale:.3f}):")
    for pct in PCTS:
        print(f"  {pct}%: {train_cov[pct]:.3f} (err={train_cov[pct]-pct/100:+.3f})")

    # ── Validate on test set ──────────────────────────────────────────────────
    test_cov_result = {}
    if test_residuals:
        all_test_r = np.concatenate(test_residuals)
        all_test_s = np.concatenate(test_stds)
        test_cov = compute_coverage(all_test_r, all_test_s, scale)
        test_cov_raw = compute_coverage(all_test_r, all_test_s, 1.0)
        test_cov_result = {'calibrated': test_cov, 'raw': test_cov_raw}

        print(f"\nTest coverage (calibrated, scale={scale:.3f}, {len(test_results)} episodes):")
        n_calibrated = 0
        for pct in PCTS:
            err = test_cov[pct] - pct/100
            ok = abs(err) < 0.05
            if ok: n_calibrated += 1
            flag = "OK" if ok else "FAIL"
            print(f"  {pct}%: {test_cov[pct]:.3f} (err={err:+.3f}) [{flag}]")
        verdict = "PASS" if n_calibrated >= 4 else "PARTIAL" if n_calibrated >= 3 else "FAIL"
        print(f"\nVerdict: {verdict} ({n_calibrated}/5 CIs calibrated within ±5%)")

    # ── Save ──────────────────────────────────────────────────────────────────
    output = {
        'description': 'Space-time GP UQ calibration (post-hoc scale factor)',
        'config': {'comp_mode': COMP_MODE, 'W': W, 'n_targets': N_TARGETS,
                   'sigma_obs': SIGMA_OBS, 'train_seeds': TRAIN_SEEDS,
                   'test_seeds': TEST_SEEDS},
        'learned_scale': round(scale, 4),
        'train_n_episodes': len(train_results),
        'test_n_episodes': len(test_results),
        'train_coverage_raw': train_cov_raw,
        'train_coverage_calibrated': train_cov,
        'test_coverage': test_cov_result,
        'train_per_episode': train_results,
        'test_per_episode': test_results,
    }
    out_path = 'results/spacetime_gp_uq.json'
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == '__main__':
    main()
