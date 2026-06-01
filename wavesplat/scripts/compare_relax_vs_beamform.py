#!/usr/bin/env python3
"""
Compare RELAX vs Beamforming spectral initialization quality.

Evaluates on Gerstner GT episodes (discrete components -> sparse problem).
Measures init-only quality: how well does each method recover the true
spectral components from multi-frame observations?

Usage:
    cd wavesplat && python3 -m scripts.compare_relax_vs_beamform
"""

import sys
import os
import time
import numpy as np

# Unbuffered output
sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', buffering=1)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from wavesplat.data_loader import EpisodeDataLoader
from wavesplat.optimizer.initializer import (
    _collect_observations, _beamform_coarse, _beamform_refine, _linear_projection
)
from wavesplat.optimizer.relax_estimator import relax_estimate

G = 9.81


def compute_init_nrmse(x, y, z, t, depth, components):
    """Compute nRMSE of initialized model vs observations."""
    z_centered = z - z.mean()
    z_pred = np.zeros(len(x))
    for kx, ky, amp, phi in components:
        k_mag = np.sqrt(kx**2 + ky**2 + 1e-12)
        omega = np.sqrt(G * k_mag * np.tanh(k_mag * depth))
        phase = kx * x + ky * y - omega * t + phi
        z_pred += amp * np.cos(phase)

    rmse = np.sqrt(np.mean((z_pred - z_centered)**2))
    std_z = np.std(z_centered)
    if std_z < 1e-10:
        return 0.0, 0.0

    nrmse = rmse / std_z
    corr = np.corrcoef(z_pred, z_centered)[0, 1] if std_z > 1e-10 else 0.0
    return nrmse, corr


def run_beamforming(x, y, z, t, depth, n_comp, n_coarse=81):
    """Run beamforming extraction (same as v2 initializer)."""
    z_centered = z - z.mean()

    power, kx_axis, ky_axis = _beamform_coarse(
        x, y, z_centered, t, depth, n_kx=n_coarse, n_ky=n_coarse
    )

    # Zero DC
    dkx = kx_axis[1] - kx_axis[0]
    dky = ky_axis[1] - ky_axis[0]
    cx = np.argmin(np.abs(kx_axis))
    cy = np.argmin(np.abs(ky_axis))
    power[max(0, cy-1):cy+2, max(0, cx-1):cx+2] = 0.0

    z_residual = z_centered.copy()
    components = []
    min_dk = 0.15

    for _ in range(n_comp):
        flat_idx = np.argmax(power.ravel())
        iy_peak, ix_peak = np.unravel_index(flat_idx, power.shape)

        if power[iy_peak, ix_peak] < 1e-12:
            break

        kx_coarse = kx_axis[ix_peak]
        ky_coarse = ky_axis[iy_peak]

        kx_fine, ky_fine, _ = _beamform_refine(
            x, y, z_residual, t, depth, kx_coarse, ky_coarse
        )

        amp, phi = _linear_projection(x, y, z_residual, t, kx_fine, ky_fine, depth)

        components.append([kx_fine, ky_fine, amp, phi])

        # Subtract
        k_mag = np.sqrt(kx_fine**2 + ky_fine**2 + 1e-12)
        omega = np.sqrt(G * k_mag * np.tanh(k_mag * depth))
        ph = kx_fine * x + ky_fine * y - omega * t + phi
        z_residual = z_residual - amp * np.cos(ph)

        # Suppress neighborhood
        suppress_kx = int(min_dk / dkx) + 1
        suppress_ky = int(min_dk / dky) + 1
        y_lo = max(0, iy_peak - suppress_ky)
        y_hi = min(power.shape[0], iy_peak + suppress_ky + 1)
        x_lo = max(0, ix_peak - suppress_kx)
        x_hi = min(power.shape[1], ix_peak + suppress_kx + 1)
        power[y_lo:y_hi, x_lo:x_hi] = 0.0

    return components


def wavenumber_matching_error(components, true_kx, true_ky):
    """Average nearest-neighbor wavenumber error from true to estimated."""
    if len(components) == 0:
        return float('inf')

    est_kx = np.array([c[0] for c in components])
    est_ky = np.array([c[1] for c in components])

    errors = []
    for tkx, tky in zip(true_kx, true_ky):
        dists = np.sqrt((est_kx - tkx)**2 + (est_ky - tky)**2)
        errors.append(np.min(dists))

    return np.mean(errors)


def main():
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            'data')

    episodes = [
        'episode_BF3_2comp_seed0_slope.h5',
        'episode_BF3_5comp_seed0_slope.h5',
        'episode_BF4_5comp_seed0_slope.h5',
        'episode_BF4_15comp_seed0_slope.h5',
        'episode_BF4_30comp_seed0_slope.h5',
        'episode_BF5_15comp_seed0_slope.h5',
    ]

    K = 15  # Number of estimated components

    print("=" * 80)
    print("RELAX vs Beamforming: Init-only Quality Comparison")
    print("=" * 80)

    all_results = []

    for ep_name in episodes:
        ep_path = os.path.join(data_dir, ep_name)
        if not os.path.exists(ep_path):
            print(f"\n  {ep_name}: NOT FOUND, skipping")
            continue

        loader = EpisodeDataLoader(ep_path)
        frames = loader.get_init_window(tp_factor=2.0)

        # True components: [A, omega, kx, ky, phi, Q]
        wc = loader.wave_components
        if wc is not None:
            true_kx = wc[:, 2]
            true_ky = wc[:, 3]
            n_true = len(true_kx)
        else:
            true_kx, true_ky = np.array([]), np.array([])
            n_true = 0

        # Collect observations
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
            continue

        x = np.concatenate(all_x)
        y = np.concatenate(all_y)
        z = np.concatenate(all_z)
        t = np.concatenate(all_t)

        depth = loader.depth
        Hs = loader.Hs_true

        print(f"\n[{ep_name}]")
        print(f"  N_true={n_true}, Hs={Hs:.2f}m, N_obs={len(x)}, "
              f"N_frames={len(frames)}, depth={depth:.0f}m")

        # --- Beamforming ---
        t0 = time.time()
        bf_comps = run_beamforming(x, y, z, t, depth, K)
        bf_time = time.time() - t0

        bf_nrmse, bf_corr = compute_init_nrmse(x, y, z, t, depth, bf_comps)
        bf_kerr = wavenumber_matching_error(bf_comps, true_kx, true_ky)

        print(f"    BF_K{K:2d}    : nRMSE={bf_nrmse:.4f}, ρ={bf_corr:.3f}, "
              f"k_err={bf_kerr:.4f} rad/m, {bf_time:.1f}s")

        # --- RELAX ---
        t0 = time.time()
        relax_comps = relax_estimate(
            x, y, z, t, depth, K,
            max_relax_cycles=5,
            n_coarse=81
        )
        relax_time = time.time() - t0

        relax_nrmse, relax_corr = compute_init_nrmse(x, y, z, t, depth, relax_comps)
        relax_kerr = wavenumber_matching_error(relax_comps, true_kx, true_ky)

        print(f"    RELAX_K{K:2d} : nRMSE={relax_nrmse:.4f}, ρ={relax_corr:.3f}, "
              f"k_err={relax_kerr:.4f} rad/m, {relax_time:.1f}s")

        # Improvement
        if bf_nrmse > 1e-6:
            improv = (bf_nrmse - relax_nrmse) / bf_nrmse * 100
            print(f"    → nRMSE improvement: {improv:+.1f}%")
        if bf_kerr > 1e-6:
            k_improv = (bf_kerr - relax_kerr) / bf_kerr * 100
            print(f"    → k_error improvement: {k_improv:+.1f}%")

        all_results.append({
            'episode': ep_name,
            'n_true': n_true,
            'bf_nrmse': bf_nrmse, 'bf_corr': bf_corr,
            'bf_kerr': bf_kerr, 'bf_time': bf_time,
            'relax_nrmse': relax_nrmse, 'relax_corr': relax_corr,
            'relax_kerr': relax_kerr, 'relax_time': relax_time,
        })

    # Summary
    if all_results:
        print("\n" + "=" * 80)
        print("SUMMARY")
        print("=" * 80)

        bf_nrmses = [r['bf_nrmse'] for r in all_results]
        relax_nrmses = [r['relax_nrmse'] for r in all_results]
        bf_corrs = [r['bf_corr'] for r in all_results]
        relax_corrs = [r['relax_corr'] for r in all_results]
        bf_kerrs = [r['bf_kerr'] for r in all_results]
        relax_kerrs = [r['relax_kerr'] for r in all_results]

        print(f"\n  Beamforming avg: nRMSE={np.mean(bf_nrmses):.4f}, "
              f"ρ={np.mean(bf_corrs):.3f}, k_err={np.mean(bf_kerrs):.4f}")
        print(f"  RELAX avg:      nRMSE={np.mean(relax_nrmses):.4f}, "
              f"ρ={np.mean(relax_corrs):.3f}, k_err={np.mean(relax_kerrs):.4f}")

        n_better = sum(1 for r in all_results if r['relax_nrmse'] < r['bf_nrmse'])
        print(f"\n  RELAX wins {n_better}/{len(all_results)} episodes on nRMSE")

        n_k_better = sum(1 for r in all_results if r['relax_kerr'] < r['bf_kerr'])
        print(f"  RELAX wins {n_k_better}/{len(all_results)} episodes on k_error")


if __name__ == '__main__':
    main()
