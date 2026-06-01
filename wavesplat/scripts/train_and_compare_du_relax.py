#!/usr/bin/env python3
"""
Train Deep Unfolded RELAX and compare with classical RELAX and Beamforming.

Strategy:
  1. Load episodes, split into train/test
  2. Train DU-RELAX on train episodes (self-supervised: reconstruction loss)
  3. Evaluate all three methods on test episodes

Usage:
    cd wavesplat && python3 -u -m scripts.train_and_compare_du_relax
"""

import sys
import os
import time
import numpy as np
import torch

sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', buffering=1)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from wavesplat.data_loader import EpisodeDataLoader
from wavesplat.optimizer.relax_estimator import relax_estimate
from wavesplat.optimizer.initializer import (
    _beamform_coarse, _beamform_refine, _linear_projection
)
from wavesplat.optimizer.deep_relax import DeepUnfoldedRELAX, train_du_relax

G = 9.81


def collect_obs(loader, tp_factor=2.0):
    """Collect all observations from init window."""
    frames = loader.get_init_window(tp_factor=tp_factor)
    all_x, all_y, all_z, all_t = [], [], [], []
    for f in frames:
        n = len(f.x_obs)
        if n == 0:
            continue
        all_x.append(f.x_obs.numpy())
        all_y.append(f.y_obs.numpy())
        all_z.append(f.z_obs.numpy())
        all_t.append(np.full(n, f.t))

    if not all_x:
        return None
    return (np.concatenate(all_x), np.concatenate(all_y),
            np.concatenate(all_z), np.concatenate(all_t),
            loader.depth)


def compute_nrmse_corr(x, y, z, t, depth, components):
    """Compute nRMSE and correlation from component list."""
    z_centered = z - z.mean()
    z_pred = np.zeros(len(x))
    for comp in components:
        if isinstance(comp, (list, tuple)):
            kx, ky, amp, phi = comp[0], comp[1], comp[2], comp[3]
        else:
            kx, ky, amp, phi = comp
        k_mag = np.sqrt(kx**2 + ky**2 + 1e-12)
        omega = np.sqrt(G * k_mag * np.tanh(k_mag * depth))
        phase = kx * x + ky * y - omega * t + phi
        z_pred += amp * np.cos(phase)

    rmse = np.sqrt(np.mean((z_pred - z_centered)**2))
    std_z = np.std(z_centered)
    if std_z < 1e-10:
        return 0.0, 0.0
    nrmse = rmse / std_z
    corr = np.corrcoef(z_pred, z_centered)[0, 1]
    return nrmse, corr


def run_beamforming(x, y, z, t, depth, K):
    """Beamforming baseline."""
    z_centered = z - z.mean()
    power, kx_axis, ky_axis = _beamform_coarse(
        x, y, z_centered, t, depth, n_kx=81, n_ky=81)
    dkx = kx_axis[1] - kx_axis[0]
    dky = ky_axis[1] - ky_axis[0]
    cx = np.argmin(np.abs(kx_axis))
    cy = np.argmin(np.abs(ky_axis))
    power[max(0, cy-1):cy+2, max(0, cx-1):cx+2] = 0.0

    z_residual = z_centered.copy()
    components = []
    for _ in range(K):
        flat_idx = np.argmax(power.ravel())
        iy, ix = np.unravel_index(flat_idx, power.shape)
        if power[iy, ix] < 1e-12:
            break
        kx_c, ky_c = kx_axis[ix], ky_axis[iy]
        kx_f, ky_f, _ = _beamform_refine(x, y, z_residual, t, depth, kx_c, ky_c)
        amp, phi = _linear_projection(x, y, z_residual, t, kx_f, ky_f, depth)
        components.append([kx_f, ky_f, amp, phi])
        k_mag = np.sqrt(kx_f**2 + ky_f**2 + 1e-12)
        omega = np.sqrt(G * k_mag * np.tanh(k_mag * depth))
        z_residual -= amp * np.cos(kx_f*x + ky_f*y - omega*t + phi)
        sy, sx = int(0.15/dky)+1, int(0.15/dkx)+1
        power[max(0,iy-sy):iy+sy+1, max(0,ix-sx):ix+sx+1] = 0.0
    return components


def run_du_relax(model, x, y, z, t, depth, device='cpu'):
    """Run trained DU-RELAX model (needs grad for internal gradient steps)."""
    model.eval()
    xt = torch.tensor(x, dtype=torch.float32, device=device)
    yt = torch.tensor(y, dtype=torch.float32, device=device)
    zt = torch.tensor(z, dtype=torch.float32, device=device)
    tt = torch.tensor(t, dtype=torch.float32, device=device)

    kx, ky, amp, phi, _ = model(xt, yt, zt, tt, float(depth))

    components = []
    for i in range(len(kx)):
        components.append([
            kx[i].detach().item(), ky[i].detach().item(),
                amp[i].detach().item(), phi[i].detach().item()
            ])
    return components


def main():
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            'data')

    # All available episodes
    all_episodes = [
        'episode_BF3_2comp_seed0_slope.h5',
        'episode_BF3_5comp_seed0_slope.h5',
        'episode_BF4_5comp_seed0_slope.h5',
        'episode_BF4_15comp_seed0_slope.h5',
        'episode_BF4_30comp_seed0_slope.h5',
        'episode_BF5_15comp_seed0_slope.h5',
        'episode_BF3_1comp_seed0_slope.h5',
        'episode_BF4_1comp_seed0_slope.h5',
    ]

    K = 15
    device = 'cpu'

    print("=" * 80)
    print("Deep Unfolded RELAX: Train & Compare")
    print("=" * 80)

    # Load all episodes
    episode_data = []
    episode_names = []
    for ep_name in all_episodes:
        ep_path = os.path.join(data_dir, ep_name)
        if not os.path.exists(ep_path):
            continue
        obs = collect_obs(EpisodeDataLoader(ep_path))
        if obs is None:
            continue
        episode_data.append(obs)
        episode_names.append(ep_name)

    print(f"\nLoaded {len(episode_data)} episodes")

    # Split: use first episodes for training, last 2 for testing
    n_train = max(1, len(episode_data) - 2)
    train_data = episode_data[:n_train]
    test_data = episode_data[n_train:]
    test_names = episode_names[n_train:]

    # If few episodes, use all for both train and test
    if len(episode_data) <= 3:
        train_data = episode_data
        test_data = episode_data
        test_names = episode_names

    print(f"Train: {n_train} episodes, Test: {len(test_data)} episodes")

    # === Train DU-RELAX ===
    print("\n" + "=" * 80)
    print("Training DU-RELAX")
    print("=" * 80)

    model = DeepUnfoldedRELAX(
        n_components=K,
        n_grad_steps=5,
        n_cycles=2,
        n_coarse=41,
    )
    print(f"  Model parameters: {model.count_parameters()}")

    t0 = time.time()
    losses = train_du_relax(model, train_data, n_epochs=100, lr=1e-3, device=device)
    train_time = time.time() - t0
    print(f"  Training done in {train_time:.1f}s, final loss={losses[-1]:.6f}")

    # === Evaluate on test episodes ===
    print("\n" + "=" * 80)
    print("Evaluation on test episodes")
    print("=" * 80)

    all_results = []

    for ep_idx, (obs, ep_name) in enumerate(zip(test_data, test_names)):
        x, y, z, t, depth = obs

        loader = EpisodeDataLoader(os.path.join(data_dir, ep_name))
        wc = loader.wave_components
        n_true = len(wc) if wc is not None else 0

        print(f"\n[{ep_name}] N_true={n_true}, N_obs={len(x)}")

        # Beamforming
        t0 = time.time()
        bf_comps = run_beamforming(x, y, z, t, depth, K)
        bf_time = time.time() - t0
        bf_nrmse, bf_corr = compute_nrmse_corr(x, y, z, t, depth, bf_comps)
        print(f"  BF:       nRMSE={bf_nrmse:.4f}, ρ={bf_corr:.3f}, {bf_time:.1f}s")

        # RELAX
        t0 = time.time()
        relax_comps = relax_estimate(x, y, z, t, depth, K,
                                      max_relax_cycles=5, n_coarse=81)
        relax_time = time.time() - t0
        relax_nrmse, relax_corr = compute_nrmse_corr(x, y, z, t, depth, relax_comps)
        print(f"  RELAX:    nRMSE={relax_nrmse:.4f}, ρ={relax_corr:.3f}, {relax_time:.1f}s")

        # DU-RELAX
        t0 = time.time()
        du_comps = run_du_relax(model, x, y, z, t, depth, device)
        du_time = time.time() - t0
        du_nrmse, du_corr = compute_nrmse_corr(x, y, z, t, depth, du_comps)
        print(f"  DU-RELAX: nRMSE={du_nrmse:.4f}, ρ={du_corr:.3f}, {du_time:.1f}s")

        # Improvements
        if bf_nrmse > 1e-6:
            print(f"  → DU vs BF:    {(bf_nrmse - du_nrmse)/bf_nrmse*100:+.1f}%")
            print(f"  → DU vs RELAX: {(relax_nrmse - du_nrmse)/relax_nrmse*100:+.1f}%")

        all_results.append({
            'episode': ep_name,
            'bf_nrmse': bf_nrmse, 'bf_corr': bf_corr, 'bf_time': bf_time,
            'relax_nrmse': relax_nrmse, 'relax_corr': relax_corr, 'relax_time': relax_time,
            'du_nrmse': du_nrmse, 'du_corr': du_corr, 'du_time': du_time,
        })

    # Summary
    if all_results:
        print("\n" + "=" * 80)
        print("SUMMARY")
        print("=" * 80)
        for method in ['bf', 'relax', 'du']:
            nrmses = [r[f'{method}_nrmse'] for r in all_results]
            corrs = [r[f'{method}_corr'] for r in all_results]
            times = [r[f'{method}_time'] for r in all_results]
            label = {'bf': 'Beamforming', 'relax': 'RELAX', 'du': 'DU-RELAX'}[method]
            print(f"  {label:12s}: nRMSE={np.mean(nrmses):.4f}, "
                  f"ρ={np.mean(corrs):.3f}, time={np.mean(times):.2f}s")


if __name__ == '__main__':
    main()
