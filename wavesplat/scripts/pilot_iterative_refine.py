#!/usr/bin/env python3
"""
Pilot: Iterative GP-Residual Refinement

현재 GP K_disp의 한계:
- WaveSplat Phase 1이 15개 성분만 추출 → 30-comp GT에서 절반을 놓침
- 방향 추정 local min → Pearson gap 큼

접근법:
1. WaveSplat으로 초기 N=15 성분 추출 (기존)
2. GP fit → observation에서의 LOO(leave-one-out) residual 계산
3. Residual에서 beamforming으로 missing 성분 추출
4. 성분 합치고 amplitude re-optimize (reg_amp)
5. NML이 개선되면 반복, 아니면 중단

빠른 파일럿: 2 episodes (BF4_30comp, BF5_30comp) × seed0만

또한 비교: n_gaussians=[3,8,15] vs [3,8,15,25] 직접 비교
"""
import sys, os, time
import numpy as np
from scipy.optimize import minimize

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from wavesplat.data_loader import EpisodeDataLoader
from wavesplat.optimizer.initializer import initialize_from_frames_v2

from scripts.gp_disp_pilot import (
    build_K_disp_oracle,
    gp_predict,
    compute_metrics_simple,
    _apply_reg_ml_amp,
    _apply_ml_scale,
    _collect_init_obs,
    _neg_log_ml,
    DATA_DIR,
    SIGMA_OBS,
    N_FRAMES,
)

# ── Test episodes (complex seas only, where gap is large) ─────────────────
PILOT_EPISODES = [
    'episode_BF4_30comp_seed0_uniform.h5',
    'episode_BF4_15comp_seed0_uniform.h5',
    'episode_BF5_30comp_seed0_uniform.h5',
    'episode_BF5_15comp_seed0_uniform.h5',
]


def beamform_from_residual(x_obs, y_obs, z_residual, depth, n_new=10, t_obs=None):
    """Beamforming on residual signal to find missing components."""
    import torch

    class FakeFrame:
        def __init__(self, x, y, z, t=0.0):
            self.x_obs = torch.tensor(x, dtype=torch.float32)
            self.y_obs = torch.tensor(y, dtype=torch.float32)
            self.z_obs = torch.tensor(z, dtype=torch.float32)
            self.t = t

    # Create fake frames from residual
    frames = [FakeFrame(x_obs, y_obs, z_residual)]

    try:
        kx, ky, w, phi = initialize_from_frames_v2(
            frames, n_gaussians=n_new, depth=depth
        )
        return kx.numpy(), ky.numpy(), np.abs(w.numpy())
    except Exception as e:
        print(f"    Beamform failed: {e}")
        return np.array([]), np.array([]), np.array([])


def compute_nml(kx, ky, X_obs, z_obs, sigma_obs=SIGMA_OBS):
    """Compute NML for given components with global ML scale."""
    n_comp = len(kx)
    N = len(z_obs)
    dx = X_obs[:, 0:1] - X_obs[:, 0:1].T
    dy = X_obs[:, 1:2] - X_obs[:, 1:2].T

    K_unit = np.zeros((N, N))
    for n in range(n_comp):
        K_unit += 0.5 * np.cos(kx[n] * dx + ky[n] * dy)

    # Optimize global scale
    from scipy.optimize import minimize_scalar
    res = minimize_scalar(
        lambda log_s: _neg_log_ml(K_unit, z_obs, log_s, sigma_obs),
        bounds=(-20, 5), method='bounded'
    )
    return res.fun


def eval_components(comps, loader, n_frames=5):
    """Evaluate K_disp with given components on first n_frames tracking frames."""
    gt_x = loader.gt_grid_x.cpu().numpy()
    gt_y = loader.gt_grid_y.cpu().numpy()
    Hs = loader.Hs_true
    Xg, Yg = np.meshgrid(gt_x, gt_y)
    X_pred = np.column_stack([Xg.ravel(), Yg.ravel()])

    eval_frames = loader.get_tracking_frames(tp_factor=2.0)
    metrics = []

    for fi, fr in enumerate(eval_frames[:n_frames]):
        X_obs = np.column_stack([fr.x_obs.cpu().numpy(), fr.y_obs.cpu().numpy()])
        z_obs = fr.z_obs.cpu().numpy()
        gt_field = fr.gt_field.cpu().numpy()

        K_oo = build_K_disp_oracle(X_obs, X_obs, comps)
        K_po = build_K_disp_oracle(X_pred, X_obs, comps)
        mu = gp_predict(K_oo, K_po, z_obs, SIGMA_OBS)
        mu = mu.reshape(len(gt_x), len(gt_y)).T
        m = compute_metrics_simple(mu, gt_field, Hs)
        metrics.append(m)

    avg = {k: float(np.mean([m[k] for m in metrics])) for k in metrics[0]}
    return avg


def run_iterative_refinement(h5_path, max_iter=3, n_new_per_iter=8):
    """Run iterative GP-residual refinement."""
    from wavesplat.optimizer.sequential_update import SequentialUpdater

    loader = EpisodeDataLoader(h5_path)
    init_frames = loader.get_init_window(tp_factor=2.0)
    if len(init_frames) < 3:
        init_frames = [loader[i] for i in range(min(10, loader.n_frames))]

    x_range = (float(loader.gt_grid_x.min()), float(loader.gt_grid_x.max()))
    y_range = (float(loader.gt_grid_y.min()), float(loader.gt_grid_y.max()))
    depth = loader.depth
    Hs = loader.Hs_true

    X_obs_init, z_obs_init = _collect_init_obs(init_frames)

    # Step 0: Baseline — standard WaveSplat n=15
    print("  [Iter 0] WaveSplat Phase1 (n=15, seed=42)...")
    t0 = time.time()
    cfg = {
        'n_gaussians': [3, 8, 15],
        'depth': depth, 'lr': 0.01,
        'lambda_l1': 0.001, 'lambda_smooth': 0.0001,
        'tracking_steps': 10, 'tracking_mode': 'w_phi',
        'tracking_lr': 0.005, 'init_method': 'v2_konly',
        'use_dispersion': True, 'use_confidence': True,
        'adaptive_n': False, 'device': 'cpu', 'seed': 42,
    }
    updater = SequentialUpdater(cfg)
    updater.initialize(init_frames, x_range=x_range, y_range=y_range)

    kx = updater.model.kx.detach().cpu().numpy()
    ky = updater.model.ky.detach().cpu().numpy()
    t_init = time.time() - t0

    # Apply reg_amp
    A, nml = _apply_reg_ml_amp(kx, ky, X_obs_init, z_obs_init,
                                lambda_reg=2.0, sigma_obs=SIGMA_OBS)

    comps = np.zeros((len(kx), 6))
    comps[:, 0] = A
    comps[:, 2] = kx
    comps[:, 3] = ky

    m0 = eval_components(comps, loader, n_frames=5)
    print(f"    N={len(kx)}, NML={nml:.2f}, nRMSE={m0['nRMSE']:.4f}, "
          f"Pearson={m0['Pearson']:.3f}, VR={m0['VR']:.3f} [{t_init:.0f}s]")

    results = [{'iter': 0, 'n_comp': len(kx), 'nml': float(nml), **m0, 'time_s': t_init}]

    # Iterative refinement
    for it in range(1, max_iter + 1):
        print(f"  [Iter {it}] Beamforming on GP residual...")
        t0 = time.time()

        # Compute GP residual at init obs points
        K_oo = build_K_disp_oracle(X_obs_init, X_obs_init, comps)
        mu_obs = gp_predict(K_oo, K_oo, z_obs_init, SIGMA_OBS)
        residual = z_obs_init - mu_obs

        residual_energy = np.sum(residual**2)
        obs_energy = np.sum(z_obs_init**2)
        print(f"    Residual energy: {residual_energy:.6f} ({residual_energy/max(obs_energy,1e-12):.1%} of obs)")

        if residual_energy / max(obs_energy, 1e-12) < 0.01:
            print(f"    Residual < 1% — stopping")
            break

        # Beamform on residual
        new_kx, new_ky, new_A = beamform_from_residual(
            X_obs_init[:, 0], X_obs_init[:, 1], residual,
            depth=depth, n_new=n_new_per_iter
        )

        if len(new_kx) == 0:
            print(f"    No new components found — stopping")
            break

        # Merge: old + new (deduplicate if very close)
        old_kx, old_ky = comps[:, 2], comps[:, 3]
        merged_kx = list(old_kx)
        merged_ky = list(old_ky)
        n_added = 0

        for i in range(len(new_kx)):
            # Check if duplicate (within 0.05 rad/m)
            dists = [np.sqrt((new_kx[i] - ok)**2 + (new_ky[i] - oky)**2)
                     for ok, oky in zip(merged_kx, merged_ky)]
            if min(dists) > 0.05:
                merged_kx.append(new_kx[i])
                merged_ky.append(new_ky[i])
                n_added += 1

        if n_added == 0:
            print(f"    All new components are duplicates — stopping")
            break

        merged_kx = np.array(merged_kx)
        merged_ky = np.array(merged_ky)

        # Re-optimize amplitudes with regularization
        A_new, nml_new = _apply_reg_ml_amp(
            merged_kx, merged_ky, X_obs_init, z_obs_init,
            lambda_reg=2.0, sigma_obs=SIGMA_OBS
        )

        # Build new comps
        n_total = len(merged_kx)
        comps_new = np.zeros((n_total, 6))
        comps_new[:, 0] = A_new
        comps_new[:, 2] = merged_kx
        comps_new[:, 3] = merged_ky

        m_new = eval_components(comps_new, loader, n_frames=5)
        t_iter = time.time() - t0

        improved = nml_new < nml
        print(f"    N={n_total} (+{n_added}), NML={nml_new:.2f} ({'better' if improved else 'worse'}), "
              f"nRMSE={m_new['nRMSE']:.4f}, Pearson={m_new['Pearson']:.3f}, VR={m_new['VR']:.3f} [{t_iter:.0f}s]")

        results.append({'iter': it, 'n_comp': n_total, 'n_added': n_added,
                        'nml': float(nml_new), **m_new, 'time_s': t_iter})

        if improved:
            comps = comps_new
            nml = nml_new
        else:
            print(f"    NML worsened — reverting and stopping")
            break

    return results


def run_larger_n_gaussians(h5_path, n_list=[15, 20, 25, 30]):
    """Compare different n_gaussians directly."""
    from wavesplat.optimizer.sequential_update import SequentialUpdater

    loader = EpisodeDataLoader(h5_path)
    init_frames = loader.get_init_window(tp_factor=2.0)
    if len(init_frames) < 3:
        init_frames = [loader[i] for i in range(min(10, loader.n_frames))]

    x_range = (float(loader.gt_grid_x.min()), float(loader.gt_grid_x.max()))
    y_range = (float(loader.gt_grid_y.min()), float(loader.gt_grid_y.max()))
    depth = loader.depth
    X_obs_init, z_obs_init = _collect_init_obs(init_frames)

    results = []
    for n_max in n_list:
        print(f"  n_gaussians={n_max}...")
        t0 = time.time()

        # Build schedule
        schedule = [3, 8]
        if n_max <= 15:
            schedule.append(n_max)
        else:
            schedule.append(15)
            schedule.append(n_max)

        cfg = {
            'n_gaussians': schedule,
            'depth': depth, 'lr': 0.01,
            'lambda_l1': 0.001, 'lambda_smooth': 0.0001,
            'tracking_steps': 10, 'tracking_mode': 'w_phi',
            'tracking_lr': 0.005, 'init_method': 'v2_konly',
            'use_dispersion': True, 'use_confidence': True,
            'adaptive_n': False, 'device': 'cpu', 'seed': 42,
        }
        updater = SequentialUpdater(cfg)
        updater.initialize(init_frames, x_range=x_range, y_range=y_range)

        kx = updater.model.kx.detach().cpu().numpy()
        ky = updater.model.ky.detach().cpu().numpy()
        t_init = time.time() - t0

        A, nml = _apply_reg_ml_amp(kx, ky, X_obs_init, z_obs_init,
                                    lambda_reg=2.0, sigma_obs=SIGMA_OBS)

        comps = np.zeros((len(kx), 6))
        comps[:, 0] = A
        comps[:, 2] = kx
        comps[:, 3] = ky

        m = eval_components(comps, loader, n_frames=5)
        print(f"    N={len(kx)}, NML={nml:.2f}, nRMSE={m['nRMSE']:.4f}, "
              f"Pearson={m['Pearson']:.3f}, VR={m['VR']:.3f} [{t_init:.0f}s]")

        results.append({'n_max': n_max, 'n_comp': len(kx), 'nml': float(nml),
                        **m, 'time_s': t_init})

    return results


if __name__ == '__main__':
    import json

    all_results = {}

    print("=" * 80)
    print("EXPERIMENT 1: Iterative GP-Residual Refinement")
    print("=" * 80)

    for h5_name in PILOT_EPISODES:
        h5_path = os.path.join(DATA_DIR, h5_name)
        ep_key = h5_name.replace('episode_', '').replace('.h5', '')
        print(f"\n{'='*60}")
        print(f"{ep_key}")
        print(f"{'='*60}")

        try:
            r = run_iterative_refinement(h5_path, max_iter=3, n_new_per_iter=8)
            all_results[f'iterative_{ep_key}'] = r
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback; traceback.print_exc()

    print("\n" + "=" * 80)
    print("EXPERIMENT 2: Larger n_gaussians")
    print("=" * 80)

    for h5_name in PILOT_EPISODES:
        h5_path = os.path.join(DATA_DIR, h5_name)
        ep_key = h5_name.replace('episode_', '').replace('.h5', '')
        print(f"\n{'='*60}")
        print(f"{ep_key}")
        print(f"{'='*60}")

        try:
            r = run_larger_n_gaussians(h5_path, n_list=[15, 20, 25])
            all_results[f'larger_n_{ep_key}'] = r
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback; traceback.print_exc()

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    for key, r in all_results.items():
        if isinstance(r, list) and len(r) > 0:
            print(f"\n{key}:")
            for entry in r:
                n = entry.get('n_comp', entry.get('n_max', '?'))
                it = entry.get('iter', entry.get('n_max', '?'))
                print(f"  N={n}: nRMSE={entry['nRMSE']:.4f}, Pearson={entry['Pearson']:.3f}, "
                      f"NML={entry.get('nml', 0):.1f}")

    out_path = 'results/pilot_iterative_refine.json'
    with open(out_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved: {out_path}")
