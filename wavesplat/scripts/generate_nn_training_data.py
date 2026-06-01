#!/usr/bin/env python3
"""
Generate training data for Neural Residual Corrector.

For each sample:
  1. Generate JONSWAP spectrum → initialize HOS M=1, M=2, M=3
  2. Run all three to multiple prediction horizons
  3. Extract top-N parametric components from initial state (NN input)
  4. Compute three residual targets:
     - residual_full = HOS_M3 - HOS_M1 (full nonlinear correction)
     - residual_3rd  = HOS_M3 - HOS_M2 (pure 3rd+ order effect)
     - residual_nn   = HOS_M3 - HOS_M1 - Stokes2(N_comp) (NN target when Stokes2 is used)
  5. Also compute Stokes2 correction from N components for reference

Usage:
    python3 -m scripts.generate_nn_training_data [--n_samples 1000] [--output data/nn_train_1k.npz]
"""

import numpy as np
import time
import sys
import os
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from wavesim.hos_simulator import HOSSimulator, HOSConfig

G = 9.81


def _compute_stokes2_correction_np(kx, ky, w, phi, omega, depth, x, y, t):
    """Compute Stokes 2nd order correction using N-component parametric model.

    This is the NumPy version of Stokes2Transfer.correction_np, inlined to
    avoid import dependency issues during data generation.
    """
    N = len(kx)

    # Transfer functions (Sharma-Dean)
    k_mag = np.sqrt(kx**2 + ky**2)
    kh = k_mag * depth
    tanh_kh = np.tanh(np.clip(kh, 0, 20))
    R = k_mag * tanh_kh
    sqrt_R = np.sqrt(np.maximum(R, 1e-12))

    k_safe = np.maximum(k_mag, 1e-10)
    ux = kx / k_safe
    uy = ky / k_safe
    cos_dtheta = ux[:, None] * ux[None, :] + uy[:, None] * uy[None, :]
    cos_dtheta = np.clip(cos_dtheta, -1.0, 1.0)

    kn_km_cos = k_mag[:, None] * k_mag[None, :] * cos_dtheta
    Rn_Rm = R[:, None] * R[None, :]
    sqrt_Rn_Rm = sqrt_R[:, None] * sqrt_R[None, :]
    k2_minus_R2 = k_mag**2 - R**2

    L_plus = np.zeros((N, N))
    L_minus = np.zeros((N, N))

    for sign in [+1, -1]:
        sqR_pm = sqrt_R[:, None] + sign * sqrt_R[None, :]
        sqR_pm_sq = sqR_pm**2

        kx_pm = kx[:, None] + sign * kx[None, :]
        ky_pm = ky[:, None] + sign * ky[None, :]
        k_pm = np.sqrt(kx_pm**2 + ky_pm**2)
        k_pm_h = k_pm * depth
        k_pm_tanh = k_pm * np.tanh(np.clip(k_pm_h, 0, 20))

        denom = sqR_pm_sq - k_pm_tanh
        denom = np.where(np.abs(denom) < 1e-8, np.sign(denom + 1e-20) * 1e-8, denom)

        bracket1 = (sqrt_R[None, :] * k2_minus_R2[:, None]
                     + sign * sqrt_R[:, None] * k2_minus_R2[None, :])
        num1 = sqR_pm * bracket1
        inner2 = kn_km_cos - sign * Rn_Rm
        num2 = 2.0 * sqR_pm_sq * inner2
        D_pm = (num1 + num2) / denom

        middle_term = (kn_km_cos - sign * Rn_Rm) / np.maximum(sqrt_Rn_Rm, 1e-12)
        R_sum = R[:, None] + R[None, :]
        L = 0.25 * (D_pm - middle_term + R_sum)

        if sign == +1:
            L_plus = L
        else:
            L_minus = L

    np.fill_diagonal(L_minus, 0.0)
    zero_k = k_mag < 1e-8
    L_plus[zero_k, :] = 0.0
    L_plus[:, zero_k] = 0.0
    L_minus[zero_k, :] = 0.0
    L_minus[:, zero_k] = 0.0

    # Evaluate correction at query points
    phase = (kx[None, :] * x[:, None]
             + ky[None, :] * y[:, None]
             - omega[None, :] * t
             + phi[None, :])
    C = np.cos(phase)
    S = np.sin(phase)

    ww = w[:, None] * w[None, :]
    W_plus = ww * L_plus
    W_minus = ww * L_minus

    CW_plus = C @ W_plus
    SW_plus = S @ W_plus
    z_sum = np.sum(CW_plus * C, axis=1) - np.sum(SW_plus * S, axis=1)

    CW_minus = C @ W_minus
    SW_minus = S @ W_minus
    z_diff = np.sum(CW_minus * C, axis=1) + np.sum(SW_minus * S, axis=1)

    return z_sum + z_diff


def extract_top_n_components(sim, N_comp=30):
    """Extract top-N strongest spectral components from HOS state."""
    eta_hat = sim.eta_hat
    Nx, Ny = sim.Nx, sim.Ny

    kx_all, ky_all, amp_all, phi_all, omega_all = [], [], [], [], []

    for ix in range(Nx):
        for iy in range(Ny):
            a = eta_hat[ix, iy]
            if abs(a) < 1e-15:
                continue
            k_mag = sim.K[ix, iy]
            if k_mag < 1e-10:
                continue

            kx_all.append(sim.KX[ix, iy])
            ky_all.append(sim.KY[ix, iy])
            amp_all.append(abs(a))
            phi_all.append(np.angle(a))
            omega_all.append(sim.omega[ix, iy])

    kx_all = np.array(kx_all)
    ky_all = np.array(ky_all)
    amp_all = np.array(amp_all)
    phi_all = np.array(phi_all)
    omega_all = np.array(omega_all)

    idx = np.argsort(amp_all)[::-1][:N_comp]
    return kx_all[idx], ky_all[idx], amp_all[idx], phi_all[idx], omega_all[idx]


def generate_sample(Hs, Tp, direction_deg, spread_deg, seed, N_comp,
                    horizons, Nx=64, Ny=64, Lx=500.0, Ly=400.0, depth=50.0,
                    n_grid=32):
    """Generate one training sample with M=1, M=2, M=3 HOS + Stokes2(N_comp)."""
    dt_hos = 0.1

    # Initialize three HOS simulators: M=1, M=2, M=3
    sims = {}
    for M in [1, 2, 3]:
        cfg = HOSConfig(Lx=Lx, Ly=Ly, Nx=Nx, Ny=Ny, depth=depth, M=M)
        sim = HOSSimulator(cfg)
        sim.initialize_jonswap(Hs=Hs, Tp=Tp, gamma=3.3,
                               direction_deg=direction_deg,
                               direction_spread_deg=spread_deg,
                               seed=seed)
        sims[M] = sim

    # Extract top-N components from initial state
    kx_c, ky_c, w_c, phi_c, omega_c = extract_top_n_components(sims[1], N_comp)

    # Evaluation grid (subsampled)
    step_x = max(1, Nx // n_grid)
    step_y = max(1, Ny // n_grid)
    dx = Lx / Nx
    dy = Ly / Ny
    x_1d = np.arange(0, Nx, step_x) * dx
    y_1d = np.arange(0, Ny, step_y) * dy
    X, Y = np.meshgrid(x_1d[:n_grid], y_1d[:n_grid], indexing='ij')
    x_flat = X.ravel()
    y_flat = Y.ravel()

    Hs_actual = 4.0 * np.std(sims[3].get_eta())

    sample = {
        'kx': kx_c, 'ky': ky_c, 'w': w_c, 'phi': phi_c, 'omega': omega_c,
        'Hs': Hs, 'Tp': Tp, 'Hs_actual': Hs_actual, 'depth': depth,
        'x_grid': x_flat, 'y_grid': y_flat,
        'horizons': {},
    }

    for horizon in horizons:
        # Run all three simulators to this horizon
        for M in [1, 2, 3]:
            sims[M].run_to(float(horizon), dt=dt_hos)

        # Get fields on subsampled grid
        eta_m1_full = sims[1].get_eta()
        eta_m2_full = sims[2].get_eta()
        eta_m3_full = sims[3].get_eta()

        eta_m1 = eta_m1_full[::step_x, ::step_y][:n_grid, :n_grid].ravel()
        eta_m2 = eta_m2_full[::step_x, ::step_y][:n_grid, :n_grid].ravel()
        eta_m3 = eta_m3_full[::step_x, ::step_y][:n_grid, :n_grid].ravel()

        # Compute Stokes2 correction from N parametric components
        stokes2_corr = _compute_stokes2_correction_np(
            kx_c, ky_c, w_c, phi_c, omega_c, depth,
            x_flat, y_flat, float(horizon))

        # Three residual targets:
        # 1. Full nonlinear: M3 - M1 (for use without Stokes2)
        residual_full = eta_m3 - eta_m1
        # 2. Pure 3rd+: M3 - M2 (HOS-level, theoretical minimum for NN)
        residual_3rd = eta_m3 - eta_m2
        # 3. NN target when Stokes2(N_comp) is used: M3 - M1 - Stokes2(N)
        residual_nn = eta_m3 - eta_m1 - stokes2_corr

        sample['horizons'][horizon] = {
            'eta_m1': eta_m1,
            'eta_m2': eta_m2,
            'eta_m3': eta_m3,
            'stokes2_corr': stokes2_corr,
            'residual_full': residual_full,
            'residual_3rd': residual_3rd,
            'residual_nn': residual_nn,
        }

    return sample


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_samples', type=int, default=1000)
    parser.add_argument('--N_comp', type=int, default=30)
    parser.add_argument('--output', type=str, default='data/nn_train_1k.npz')
    parser.add_argument('--n_grid', type=int, default=32)
    args = parser.parse_args()

    sea_configs = [
        {'Hs': 0.6, 'Tp': 5.5},   # BF3
        {'Hs': 0.9, 'Tp': 6.0},   # BF3-4
        {'Hs': 1.2, 'Tp': 7.0},   # BF4
        {'Hs': 1.5, 'Tp': 8.0},   # BF4-5
        {'Hs': 2.0, 'Tp': 9.0},   # BF5
    ]
    horizons = [5.0, 10.0]

    rng = np.random.default_rng(42)

    all_samples = []
    t0 = time.time()
    n_errors = 0

    print(f"Generating {args.n_samples} training samples with N={args.N_comp} components...")
    print(f"Method: HOS M=1,2,3 + Stokes2(N_comp) — three residual targets")

    for i in range(args.n_samples):
        cfg = sea_configs[rng.integers(len(sea_configs))]
        Hs = cfg['Hs'] * rng.uniform(0.8, 1.2)
        Tp = cfg['Tp'] * rng.uniform(0.9, 1.1)
        direction = rng.uniform(-30, 30)
        spread = rng.uniform(15, 45)
        seed = int(rng.integers(10000))

        try:
            sample = generate_sample(
                Hs=Hs, Tp=Tp, direction_deg=direction,
                spread_deg=spread, seed=seed,
                N_comp=args.N_comp, horizons=horizons,
                n_grid=args.n_grid,
            )
            # Check for NaN
            has_nan = False
            for h in horizons:
                for key in ['residual_full', 'residual_3rd', 'residual_nn']:
                    if np.any(np.isnan(sample['horizons'][h][key])):
                        has_nan = True
                        break
            if has_nan:
                print(f"  Sample {i}: NaN detected, skipping")
                n_errors += 1
                continue

            all_samples.append(sample)
        except Exception as e:
            print(f"  Sample {i}: ERROR {e}")
            n_errors += 1
            continue

        if (i + 1) % 20 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta_min = (args.n_samples - i - 1) / rate / 60
            n_ok = len(all_samples)
            print(f"  [{i+1}/{args.n_samples}] {n_ok} ok, {n_errors} err, "
                  f"{elapsed:.0f}s, ETA {eta_min:.1f}min")

    # Pack into arrays
    N = len(all_samples)
    N_comp = args.N_comp
    n_grid2 = args.n_grid * args.n_grid
    n_horizons = len(horizons)

    kx_arr = np.zeros((N, N_comp))
    ky_arr = np.zeros((N, N_comp))
    w_arr = np.zeros((N, N_comp))
    phi_arr = np.zeros((N, N_comp))
    omega_arr = np.zeros((N, N_comp))
    Hs_arr = np.zeros(N)
    Tp_arr = np.zeros(N)

    residuals_full = np.zeros((N, n_horizons, n_grid2))
    residuals_3rd = np.zeros((N, n_horizons, n_grid2))
    residuals_nn = np.zeros((N, n_horizons, n_grid2))
    eta_m1_all = np.zeros((N, n_horizons, n_grid2))
    eta_m2_all = np.zeros((N, n_horizons, n_grid2))
    eta_m3_all = np.zeros((N, n_horizons, n_grid2))
    stokes2_all = np.zeros((N, n_horizons, n_grid2))

    for i, s in enumerate(all_samples):
        nc = len(s['kx'])
        kx_arr[i, :nc] = s['kx']
        ky_arr[i, :nc] = s['ky']
        w_arr[i, :nc] = s['w']
        phi_arr[i, :nc] = s['phi']
        omega_arr[i, :nc] = s['omega']
        Hs_arr[i] = s['Hs']
        Tp_arr[i] = s['Tp']

        for j, h in enumerate(horizons):
            hdata = s['horizons'][h]
            residuals_full[i, j] = hdata['residual_full']
            residuals_3rd[i, j] = hdata['residual_3rd']
            residuals_nn[i, j] = hdata['residual_nn']
            eta_m1_all[i, j] = hdata['eta_m1']
            eta_m2_all[i, j] = hdata['eta_m2']
            eta_m3_all[i, j] = hdata['eta_m3']
            stokes2_all[i, j] = hdata['stokes2_corr']

    # Save
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    np.savez_compressed(
        args.output,
        kx=kx_arr, ky=ky_arr, w=w_arr, phi=phi_arr, omega=omega_arr,
        Hs=Hs_arr, Tp=Tp_arr,
        # Legacy key for backward compat with train script
        residuals=residuals_full,
        # New detailed targets
        residuals_full=residuals_full,
        residuals_3rd=residuals_3rd,
        residuals_nn=residuals_nn,
        eta_m1=eta_m1_all, eta_m2=eta_m2_all, eta_m3=eta_m3_all,
        # Keep old keys for compat
        eta_hos=eta_m3_all, eta_pred=eta_m1_all,
        stokes2_corr=stokes2_all,
        horizons=np.array(horizons),
        x_grid=all_samples[0]['x_grid'],
        y_grid=all_samples[0]['y_grid'],
        depth=50.0, N_comp=N_comp, n_grid=args.n_grid,
    )

    # Summary stats
    print(f"\nGenerated {N} samples ({n_errors} errors) in {time.time()-t0:.0f}s")
    print(f"Saved to {args.output}")

    for j, h in enumerate(horizons):
        m3_std = np.mean(np.std(eta_m3_all[:, j], axis=1))

        res_full_std = np.mean(np.std(residuals_full[:, j], axis=1))
        res_3rd_std = np.mean(np.std(residuals_3rd[:, j], axis=1))
        res_nn_std = np.mean(np.std(residuals_nn[:, j], axis=1))
        s2_std = np.mean(np.std(stokes2_all[:, j], axis=1))

        print(f"\n  t+{h:.0f}s (HOS_M3 std={m3_std:.4f}m):")
        print(f"    residual_full (M3-M1): std={res_full_std:.4f}m, ratio={res_full_std/m3_std:.3f}")
        print(f"    residual_3rd  (M3-M2): std={res_3rd_std:.4f}m, ratio={res_3rd_std/m3_std:.3f}")
        print(f"    residual_nn (M3-M1-S2): std={res_nn_std:.4f}m, ratio={res_nn_std/m3_std:.3f}")
        print(f"    stokes2(N_comp):        std={s2_std:.4f}m, ratio={s2_std/m3_std:.3f}")


if __name__ == '__main__':
    main()
