#!/usr/bin/env python3
"""
3-way gap experiment: Linear vs HOS-M=2 vs HOS-M=3.

Purpose: Determine how much of the nonlinear gap is 2nd-order vs 3rd-order.

Method (no spinup, progressive wave init):
  1. Initialize three HOS simulators (M=1, M=2, M=3) with identical JONSWAP
  2. Run all three to prediction horizons
  3. Linear prediction: simple phase rotation (exp(-i*omega*t))
  4. Compare nRMSE of each vs HOS M=3 ground truth

Key insight: HOS M=2 naturally captures 2nd-order nonlinear physics
in the time evolution, giving a clean comparison against the total gap.

For WaveSplat integration, the Stokes 2nd order module (stokes2.py) computes
the STATIC bound harmonics from parametric components (~30-50 N), which is
efficient and differentiable. This experiment validates the physics value.

Usage:
    python3 -m scripts.stokes2_gap
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
    rmse = np.sqrt(np.mean((pred - gt)**2))
    std_gt = np.std(gt)
    if std_gt < 1e-10:
        return 0.0
    return rmse / std_gt


def run_experiment(sea_state_name, Hs, Tp, seed, horizons=(10, 30, 60)):
    """Run 3-way comparison: Linear vs HOS-M2 vs HOS-M3."""
    print(f"\n--- {sea_state_name}: Hs={Hs}m, Tp={Tp}s, seed={seed} ---")

    Nx, Ny = 128, 128
    Lx, Ly = 500.0, 400.0
    depth = 50.0
    dt_hos = 0.05

    # Create three HOS simulators with same init
    sims = {}
    eta_hat_0 = None
    omega = None

    for M in [1, 2, 3]:
        cfg = HOSConfig(Lx=Lx, Ly=Ly, Nx=Nx, Ny=Ny, depth=depth, M=M)
        sim = HOSSimulator(cfg)
        sim.initialize_jonswap(Hs=Hs, Tp=Tp, gamma=3.3,
                               direction_deg=0.0, direction_spread_deg=30.0,
                               seed=seed)
        sims[M] = sim
        if eta_hat_0 is None:
            eta_hat_0 = sim.eta_hat.copy()
            omega = sim.omega.copy()

    Hs_actual = 4.0 * np.std(sims[3].get_eta())
    print(f"  Hs actual: {Hs_actual:.3f}m")

    results = {
        'sea_state': sea_state_name, 'Hs': Hs, 'Tp': Tp,
        'seed': seed, 'Hs_actual': float(Hs_actual), 'horizons': {}
    }

    t0_wall = time.time()
    sorted_horizons = sorted(horizons)

    for horizon in sorted_horizons:
        # Run all HOS simulations
        for M in [1, 2, 3]:
            sims[M].run_to(float(horizon), dt=dt_hos)

        # Ground truth: HOS M=3
        eta_m3 = sims[3].get_eta()

        # HOS M=2 (captures 2nd-order nonlinear evolution)
        eta_m2 = sims[2].get_eta()

        # HOS M=1 (linear W, but nonlinear BCs)
        eta_m1 = sims[1].get_eta()

        # Pure linear: simple phase rotation
        eta_hat_lin = eta_hat_0 * np.exp(-1j * omega * horizon)
        eta_lin = np.real(np.fft.ifft2(eta_hat_lin)) * (Nx * Ny)

        # Compute metrics vs M=3 ground truth
        nrmse_lin = compute_nrmse(eta_lin.ravel(), eta_m3.ravel())
        nrmse_m1 = compute_nrmse(eta_m1.ravel(), eta_m3.ravel())
        nrmse_m2 = compute_nrmse(eta_m2.ravel(), eta_m3.ravel())

        corr_lin = np.corrcoef(eta_lin.ravel(), eta_m3.ravel())[0, 1]
        corr_m2 = np.corrcoef(eta_m2.ravel(), eta_m3.ravel())[0, 1]

        # Decomposition of the gap
        # Total gap: linear vs M3
        # 2nd order gap: what M=2 explains (linear - M2) / (linear - M3)
        gap_m2_explains = 0.0
        if nrmse_lin > 1e-6:
            gap_m2_explains = (nrmse_lin - nrmse_m2) / nrmse_lin * 100

        # 3rd order residual: M=2 vs M=3
        gap_3rd_order = nrmse_m2

        results['horizons'][str(horizon)] = {
            'nrmse_linear': float(nrmse_lin),
            'nrmse_hos_m1': float(nrmse_m1),
            'nrmse_hos_m2': float(nrmse_m2),
            'corr_linear': float(corr_lin),
            'corr_hos_m2': float(corr_m2),
            'gap_m2_explains_pct': float(gap_m2_explains),
            'Hs_m3': float(4.0 * np.std(eta_m3)),
        }

        elapsed = time.time() - t0_wall
        print(f"  t+{horizon:3d}s: Lin={nrmse_lin:.4f}, M1={nrmse_m1:.4f}, "
              f"M2={nrmse_m2:.4f}, M2 explains {gap_m2_explains:.1f}% [{elapsed:.0f}s]")

    return results


def main():
    sea_states = {
        'BF3': {'Hs': 0.6, 'Tp': 5.5},
        'BF4': {'Hs': 1.2, 'Tp': 7.0},
        'BF5': {'Hs': 2.0, 'Tp': 9.0},
    }

    horizons = [10, 30, 60]
    seeds = [0, 1, 2]

    all_results = []

    print("=" * 80)
    print("3-WAY GAP: Linear vs HOS-M1 vs HOS-M2 vs HOS-M3")
    print("=" * 80)

    for sea_name, sea_params in sea_states.items():
        for seed in seeds:
            try:
                result = run_experiment(
                    sea_name, sea_params['Hs'], sea_params['Tp'],
                    seed, horizons=horizons,
                )
                all_results.append(result)
            except Exception as e:
                print(f"  ERROR: {e}")
                import traceback
                traceback.print_exc()

    # Summary table
    from collections import defaultdict

    print("\n" + "=" * 80)
    print("SUMMARY (mean over 3 seeds, vs HOS M=3 ground truth)")
    print("=" * 80)
    print(f"{'Sea':6s} {'t':>5s} {'Linear':>10s} {'HOS-M1':>10s} {'HOS-M2':>10s} {'M2 expl%':>10s}")
    print("-" * 55)

    agg = defaultdict(lambda: defaultdict(list))
    for r in all_results:
        for h_str, m in r['horizons'].items():
            key = (r['sea_state'], int(h_str))
            agg[key]['lin'].append(m['nrmse_linear'])
            agg[key]['m1'].append(m['nrmse_hos_m1'])
            agg[key]['m2'].append(m['nrmse_hos_m2'])
            agg[key]['gap'].append(m['gap_m2_explains_pct'])

    for sea in ['BF3', 'BF4', 'BF5']:
        for h in horizons:
            key = (sea, h)
            if key in agg:
                d = agg[key]
                print(f"{sea:6s} {h:>3d}s  {np.mean(d['lin']):>9.4f}  "
                      f"{np.mean(d['m1']):>9.4f}  {np.mean(d['m2']):>9.4f}  "
                      f"{np.mean(d['gap']):>9.1f}")

    # Analysis
    print("\n" + "=" * 80)
    print("ANALYSIS:")

    all_m2_explain = []
    for key, d in agg.items():
        all_m2_explain.extend(d['gap'])

    mean_explain = np.mean(all_m2_explain) if all_m2_explain else 0
    print(f"\n  Mean gap explained by M=2 (2nd order): {mean_explain:.1f}%")
    print(f"  → Remaining gap (3rd+ order): {100-mean_explain:.1f}%")

    # Separate by horizon
    for h in horizons:
        vals = []
        for sea in ['BF3', 'BF4', 'BF5']:
            key = (sea, h)
            if key in agg:
                vals.extend(agg[key]['gap'])
        if vals:
            print(f"  At {h}s horizon: M=2 explains {np.mean(vals):.1f}% of gap")

    print("\n  Implications for WaveSplat nonlinear extension:")
    if mean_explain > 70:
        print("  → 2nd order dominates: Stokes2 correction very valuable")
        print("  → NN residual (3rd+) is small: lightweight NN sufficient")
    elif mean_explain > 40:
        print("  → 2nd order significant: Stokes2 correction valuable")
        print("  → NN residual (3rd+) moderate: NN needed for full correction")
    else:
        print("  → 3rd+ order dominates: Stokes2 alone insufficient")
        print("  → NN must capture substantial nonlinear effects")

    # Save
    output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                               'results')
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'stokes2_gap.json')
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == '__main__':
    main()
