#!/usr/bin/env python3
"""
HOS vs Linear Prediction Gap Experiment.

Compares nonlinear HOS evolution against linear prediction at multiple
prediction horizons (10s, 30s, 60s) for different sea states.

Two comparison methods:
  A) No spinup: progressive wave init → compare HOS vs phi+=omega*dt
     (directly measures nonlinear divergence from linear model)
  B) With spinup: coupled linear prediction using (eta_hat, phi_s_hat)
     (measures divergence from general linearized evolution)

Purpose: Determine if Neural ODE for prediction is worthwhile.
- Gap < 5%  → Neural ODE has low value → KalmanNet only
- Gap > 10% → Neural ODE is valuable → pursue both directions

Usage:
    python -m scripts.hos_vs_linear_gap
"""

import numpy as np
import json
import time
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from wavesim.hos_simulator import HOSSimulator, HOSConfig

G = 9.81


def compute_nrmse(pred, gt):
    """Normalized RMSE: RMSE / std(gt)."""
    rmse = np.sqrt(np.mean((pred - gt)**2))
    std_gt = np.std(gt)
    if std_gt < 1e-10:
        return 0.0
    return rmse / std_gt


def run_experiment_no_spinup(sea_state_name, Hs, Tp, M_order, seed,
                              horizons=(10, 30, 60)):
    """
    Method A: No spinup. Initialize as progressive waves, compare directly.
    This is the most relevant for WaveSplat (which uses progressive wave model).
    """
    print(f"\n--- {sea_state_name}: Hs={Hs}m, Tp={Tp}s, M={M_order}, seed={seed} (no spinup) ---")

    Nx, Ny = 128, 128
    Lx, Ly = 500.0, 400.0
    depth = 50.0
    dt_hos = 0.05

    cfg = HOSConfig(Lx=Lx, Ly=Ly, Nx=Nx, Ny=Ny, depth=depth, M=M_order)
    sim = HOSSimulator(cfg)
    sim.initialize_jonswap(Hs=Hs, Tp=Tp, gamma=3.3,
                           direction_deg=0.0, direction_spread_deg=30.0,
                           seed=seed)

    # Save initial state (progressive waves)
    eta_hat_0 = sim.eta_hat.copy()
    Hs_actual = 4.0 * np.std(sim.get_eta())
    print(f"  Hs actual: {Hs_actual:.3f}m")

    results = {'sea_state': sea_state_name, 'Hs': Hs, 'Tp': Tp,
               'M': M_order, 'seed': seed, 'method': 'no_spinup',
               'Hs_actual': float(Hs_actual), 'horizons': {}}

    t0_wall = time.time()
    max_h = max(horizons)
    print(f"  Running HOS to {max_h}s...", flush=True)

    sorted_horizons = sorted(horizons)
    for horizon in sorted_horizons:
        sim.run_to(float(horizon), dt=dt_hos)

        # HOS ground truth on spectral grid
        eta_hos = sim.get_eta()

        # Linear prediction: each mode = progressive wave rotating at omega
        eta_hat_lin = eta_hat_0 * np.exp(-1j * sim.omega * horizon)
        eta_lin = np.real(np.fft.ifft2(eta_hat_lin)) * (Nx * Ny)

        nrmse = compute_nrmse(eta_lin, eta_hos)
        rmse = np.sqrt(np.mean((eta_lin - eta_hos)**2))
        corr = np.corrcoef(eta_lin.ravel(), eta_hos.ravel())[0, 1]
        Hs_hos = 4.0 * np.std(eta_hos)

        results['horizons'][str(horizon)] = {
            'nrmse': float(nrmse),
            'rmse': float(rmse),
            'correlation': float(corr),
            'Hs_hos': float(Hs_hos),
        }
        elapsed = time.time() - t0_wall
        print(f"  t+{horizon:3d}s: nRMSE={nrmse:.4f}, corr={corr:.4f}, Hs={Hs_hos:.3f}m [{elapsed:.0f}s]")

    print(f"  Total wall time: {time.time()-t0_wall:.0f}s")
    return results


def run_experiment_with_spinup(sea_state_name, Hs, Tp, M_order, seed,
                                horizons=(10, 30, 60)):
    """
    Method B: With spinup, using coupled linear prediction (eta + phi_s).
    This measures the gap when nonlinear features are already present.
    """
    print(f"\n--- {sea_state_name}: Hs={Hs}m, Tp={Tp}s, M={M_order}, seed={seed} (with spinup) ---")

    Nx, Ny = 128, 128
    Lx, Ly = 500.0, 400.0
    depth = 50.0
    dt_hos = 0.05

    cfg = HOSConfig(Lx=Lx, Ly=Ly, Nx=Nx, Ny=Ny, depth=depth, M=M_order)
    sim = HOSSimulator(cfg)
    sim.initialize_jonswap(Hs=Hs, Tp=Tp, gamma=3.3,
                           direction_deg=0.0, direction_spread_deg=30.0,
                           seed=seed)

    L1 = sim.Ln[1]
    omega = sim.omega
    omega_safe = np.where(omega > 1e-10, omega, 1.0)

    spinup_time = 2.0 * Tp
    print(f"  Spinup: {spinup_time:.0f}s...", flush=True)
    t0_wall = time.time()

    sim.run_to(spinup_time, dt=dt_hos)
    eta_hat_t0 = sim.eta_hat.copy()
    phi_hat_t0 = sim.phi_s_hat.copy()

    Hs_actual = 4.0 * np.std(sim.get_eta())
    print(f"  Hs after spinup: {Hs_actual:.3f}m")

    results = {'sea_state': sea_state_name, 'Hs': Hs, 'Tp': Tp,
               'M': M_order, 'seed': seed, 'method': 'spinup_coupled',
               'Hs_actual': float(Hs_actual), 'horizons': {}}

    sorted_horizons = sorted(horizons)
    for horizon in sorted_horizons:
        sim.run_to(spinup_time + horizon, dt=dt_hos)
        eta_hos = sim.get_eta()

        # Coupled linear prediction: eta(t) = eta0*cos(wt) + phi0*L1/w*sin(wt)
        cos_wt = np.cos(omega * horizon)
        sin_wt = np.sin(omega * horizon)
        eta_hat_pred = eta_hat_t0 * cos_wt + phi_hat_t0 * L1 / omega_safe * sin_wt
        eta_hat_pred[0, 0] = 0
        eta_pred = np.real(np.fft.ifft2(eta_hat_pred)) * (Nx * Ny)

        nrmse = compute_nrmse(eta_pred, eta_hos)
        rmse = np.sqrt(np.mean((eta_pred - eta_hos)**2))
        corr = np.corrcoef(eta_pred.ravel(), eta_hos.ravel())[0, 1]
        Hs_hos = 4.0 * np.std(eta_hos)

        results['horizons'][str(horizon)] = {
            'nrmse': float(nrmse),
            'rmse': float(rmse),
            'correlation': float(corr),
            'Hs_hos': float(Hs_hos),
        }
        elapsed = time.time() - t0_wall
        print(f"  t+{horizon:3d}s: nRMSE={nrmse:.4f}, corr={corr:.4f}, Hs={Hs_hos:.3f}m [{elapsed:.0f}s]")

    print(f"  Total wall time: {time.time()-t0_wall:.0f}s")
    return results


def main():
    sea_states = {
        'BF3': {'Hs': 0.6, 'Tp': 5.5},
        'BF4': {'Hs': 1.2, 'Tp': 7.0},
        'BF5': {'Hs': 2.0, 'Tp': 9.0},
    }

    horizons = [10, 30, 60]
    seeds = [0, 1, 2]
    M_orders = [2, 3]

    all_results = []

    # Method A: No spinup (most relevant for WaveSplat)
    print("=" * 80)
    print("METHOD A: No spinup — progressive wave init")
    print("=" * 80)
    for sea_name, sea_params in sea_states.items():
        for M_order in M_orders:
            for seed in seeds:
                try:
                    result = run_experiment_no_spinup(
                        sea_name, sea_params['Hs'], sea_params['Tp'],
                        M_order, seed, horizons=horizons
                    )
                    all_results.append(result)
                except Exception as e:
                    print(f"  ERROR: {e}")
                    import traceback
                    traceback.print_exc()

    # Method B: With spinup + coupled prediction
    print("\n" + "=" * 80)
    print("METHOD B: With spinup — coupled linear prediction")
    print("=" * 80)
    for sea_name, sea_params in sea_states.items():
        for M_order in [3]:  # Only M=3 for spinup (most realistic)
            for seed in [0]:  # Single seed to save time
                try:
                    result = run_experiment_with_spinup(
                        sea_name, sea_params['Hs'], sea_params['Tp'],
                        M_order, seed, horizons=horizons
                    )
                    all_results.append(result)
                except Exception as e:
                    print(f"  ERROR: {e}")
                    import traceback
                    traceback.print_exc()

    # Summary
    from collections import defaultdict

    print("\n" + "=" * 80)
    print("SUMMARY: Method A (no spinup)")
    print("=" * 80)
    print(f"{'Sea':6s} {'M':4s} {'10s':>12s} {'30s':>12s} {'60s':>12s}")
    print("-" * 50)

    agg_a = defaultdict(list)
    for r in all_results:
        if r['method'] == 'no_spinup':
            for h_str, m in r['horizons'].items():
                agg_a[(r['sea_state'], r['M'], int(h_str))].append(m['nrmse'])

    for sea in ['BF3', 'BF4', 'BF5']:
        for M in [2, 3]:
            vals = {}
            for h in [10, 30, 60]:
                key = (sea, M, h)
                if key in agg_a:
                    arr = agg_a[key]
                    vals[h] = f"{np.mean(arr):.4f}±{np.std(arr):.4f}"
                else:
                    vals[h] = "N/A"
            print(f"{sea:6s} M={M}  {vals[10]:>12s} {vals[30]:>12s} {vals[60]:>12s}")

    print("\n" + "=" * 80)
    print("SUMMARY: Method B (spinup + coupled)")
    print("=" * 80)
    for r in all_results:
        if r['method'] == 'spinup_coupled':
            sea = r['sea_state']
            M = r['M']
            for h_str, m in sorted(r['horizons'].items(), key=lambda x: int(x[0])):
                print(f"  {sea} M={M} {h_str:>3s}s: nRMSE={m['nrmse']:.4f}, corr={m['correlation']:.4f}")

    # Decision
    print("\n" + "=" * 80)
    print("DECISION (Method A, 30s horizon, M=3):")
    gap_30 = []
    for (sea, M, h), vals in agg_a.items():
        if h == 30 and M == 3:
            gap_30.extend(vals)
    if gap_30:
        mean_gap = np.mean(gap_30)
        print(f"  Mean nRMSE gap at 30s: {mean_gap:.4f} ({mean_gap*100:.1f}%)")
        if mean_gap < 0.05:
            print("  -> Gap < 5%: Neural ODE LOW value. KalmanNet only.")
        elif mean_gap < 0.10:
            print("  -> Gap 5-10%: Neural ODE MODERATE value. Consider both.")
        else:
            print("  -> Gap > 10%: Neural ODE HIGH value. Pursue both.")

    # Save
    output_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                               'results', 'hos_vs_linear_gap.json')
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == '__main__':
    main()
