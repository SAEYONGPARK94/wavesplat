#!/usr/bin/env python3
"""
HOS Prediction Gap: Measuring Stokes2 value via HOS order comparison.

Uses HOS at different nonlinear orders as natural baselines:
  - M=1: linear prediction (equivalent to frozen-spectrum linear propagation)
  - M=2: 2nd-order (includes Stokes2 bound harmonics)
  - M=3: full nonlinear GT

This properly measures how much of the nonlinear gap Stokes2 closes,
using consistent physics (irrotational potential flow throughout).

Two evaluation methods:
  A) Full-field comparison on spectral grid (best-case)
  B) WaveSplat-scale comparison: extract K dominant components at t=0,
     predict forward with linear and Stokes2, compare to M=3 GT

Usage:
    python3 -m scripts.hos_prediction_gap
"""

import numpy as np
import json
import time
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from wavesim.hos_simulator import HOSSimulator, HOSConfig
from wavesplat.model.stokes2 import Stokes2Transfer

G = 9.81


def compute_nrmse(pred, gt):
    """Normalized RMSE: RMSE / std(gt)."""
    rmse = np.sqrt(np.mean((pred - gt)**2))
    std_gt = np.std(gt)
    if std_gt < 1e-10:
        return 0.0
    return rmse / std_gt


def extract_top_k_components(sim, K=20):
    """Extract top-K dominant spectral components from HOS state."""
    eta_hat = sim.eta_hat.copy()
    energy = np.abs(eta_hat)**2
    # Flatten, sort by energy
    flat_idx = np.argsort(energy.ravel())[::-1]
    top_idx = flat_idx[:K]
    ix = top_idx // sim.Ny
    iy = top_idx % sim.Ny

    return {
        'kx': sim.KX[ix, iy],
        'ky': sim.KY[ix, iy],
        'amp': np.abs(eta_hat[ix, iy]),
        'phase': np.angle(eta_hat[ix, iy]),
        'omega': sim.omega[ix, iy],
        'n_comp': K,
    }


def predict_linear_components(comp, x, y, t):
    """Linear prediction from K components."""
    phase = (comp['kx'][None, :] * x[:, None]
             + comp['ky'][None, :] * y[:, None]
             - comp['omega'][None, :] * t
             + comp['phase'][None, :])
    return np.sum(comp['amp'][None, :] * np.cos(phase), axis=1)


def predict_stokes2_components(comp, x, y, t, depth):
    """Linear + Stokes2 correction from K components."""
    eta_lin = predict_linear_components(comp, x, y, t)
    try:
        st = Stokes2Transfer(comp['kx'], comp['ky'], depth)
        eta2 = st.correction_np(x, y, t, comp['amp'],
                                comp['kx'], comp['ky'],
                                comp['phase'], comp['omega'])
        return eta_lin + eta2
    except Exception as e:
        print(f"    Stokes2 failed: {e}")
        return eta_lin


def run_method_a(sea_name, Hs, Tp, seed, horizons):
    """
    Method A: Full-field comparison via HOS at different orders.

    M=1 vs M=2 vs M=3, all from same initial JONSWAP spectrum.
    """
    print(f"\n--- Method A: {sea_name} Hs={Hs}m Tp={Tp}s seed={seed} ---")

    # Adaptive resolution: lower Nx and smaller dt for higher sea states
    if Hs <= 0.8:
        Nx, Ny = 128, 128
        dt = 0.05
    elif Hs <= 1.5:
        Nx, Ny = 64, 64
        dt = 0.02
    else:
        Nx, Ny = 64, 64
        dt = 0.01
    Lx, Ly = 500.0, 400.0
    depth = 50.0

    results = {}

    # Run each M order from identical initial conditions
    for M_order in [1, 2, 3]:
        cfg = HOSConfig(Lx=Lx, Ly=Ly, Nx=Nx, Ny=Ny, depth=depth, M=M_order)
        sim = HOSSimulator(cfg)
        sim.initialize_jonswap(Hs=Hs, Tp=Tp, gamma=3.3,
                               direction_deg=0.0, direction_spread_deg=30.0,
                               seed=seed)

        Hs_actual = sim.get_Hs()
        if M_order == 3:
            print(f"  Hs actual: {Hs_actual:.3f}m")

        t0 = time.time()
        for horizon in sorted(horizons):
            sim.run_to(float(horizon), dt=dt)
            eta = sim.get_eta().copy()

            key = str(horizon)
            if key not in results:
                results[key] = {}
            results[key][f'M{M_order}'] = eta

        elapsed = time.time() - t0
        print(f"  M={M_order} done in {elapsed:.0f}s")

    # Compare
    horizon_results = {}
    for h in horizons:
        key = str(h)
        gt = results[key]['M3']
        eta_m1 = results[key]['M1']
        eta_m2 = results[key]['M2']

        nrmse_m1 = compute_nrmse(eta_m1, gt)
        nrmse_m2 = compute_nrmse(eta_m2, gt)
        corr_m1 = np.corrcoef(eta_m1.ravel(), gt.ravel())[0, 1]
        corr_m2 = np.corrcoef(eta_m2.ravel(), gt.ravel())[0, 1]

        # Gap closed by M=2 (Stokes2)
        gap_closed = 0.0
        if nrmse_m1 > 1e-6:
            gap_closed = (nrmse_m1 - nrmse_m2) / nrmse_m1 * 100

        horizon_results[key] = {
            'nrmse_M1': float(nrmse_m1),
            'nrmse_M2': float(nrmse_m2),
            'corr_M1': float(corr_m1),
            'corr_M2': float(corr_m2),
            'gap_closed_pct': float(gap_closed),
            'Hs_M3': float(4.0 * np.std(gt)),
        }
        print(f"  t+{h:3d}s: M1={nrmse_m1:.4f}, M2={nrmse_m2:.4f}, "
              f"gap_closed={gap_closed:+.1f}%")

    return {
        'sea_state': sea_name, 'Hs': Hs, 'Tp': Tp, 'seed': seed,
        'method': 'full_field',
        'horizons': horizon_results,
    }


def run_method_b(sea_name, Hs, Tp, seed, horizons, K_values=(10, 20, 30)):
    """
    Method B: WaveSplat-scale component prediction.

    Extract top-K components from initial spectrum, predict with
    linear and Stokes2, compare to HOS M=3 GT.
    """
    print(f"\n--- Method B: {sea_name} Hs={Hs}m Tp={Tp}s seed={seed} ---")

    Nx, Ny = 128, 128
    Lx, Ly = 500.0, 400.0
    depth = 50.0
    dt = 0.02 if Hs >= 1.0 else 0.05

    # Run M=3 GT
    cfg3 = HOSConfig(Lx=Lx, Ly=Ly, Nx=Nx, Ny=Ny, depth=depth, M=3)
    sim_gt = HOSSimulator(cfg3)
    sim_gt.initialize_jonswap(Hs=Hs, Tp=Tp, gamma=3.3,
                              direction_deg=0.0, direction_spread_deg=30.0,
                              seed=seed)

    # Extract components before any evolution
    comp_by_k = {}
    for K in K_values:
        comp_by_k[K] = extract_top_k_components(sim_gt, K=K)
        print(f"  K={K}: max_amp={comp_by_k[K]['amp'][0]:.4f}m")

    # Evaluation grid (subsampled, like real scenario)
    eval_nx, eval_ny = 50, 50
    eval_x = np.linspace(50, 450, eval_nx)
    eval_y = np.linspace(50, 350, eval_ny)
    Xg, Yg = np.meshgrid(eval_x, eval_y, indexing='xy')
    x_flat = Xg.ravel()
    y_flat = Yg.ravel()

    results_by_k = {K: {} for K in K_values}

    t0 = time.time()
    for horizon in sorted(horizons):
        sim_gt.run_to(float(horizon), dt=dt)
        gt_field = sim_gt.get_eta_at(Xg, Yg)
        gt_flat = gt_field.ravel()

        for K in K_values:
            comp = comp_by_k[K]
            eta_lin = predict_linear_components(comp, x_flat, y_flat, float(horizon))
            eta_s2 = predict_stokes2_components(comp, x_flat, y_flat,
                                                float(horizon), depth)

            nrmse_lin = compute_nrmse(eta_lin, gt_flat)
            nrmse_s2 = compute_nrmse(eta_s2, gt_flat)
            corr_lin = np.corrcoef(eta_lin, gt_flat)[0, 1]
            corr_s2 = np.corrcoef(eta_s2, gt_flat)[0, 1]

            gap_closed = 0.0
            if nrmse_lin > 1e-6:
                gap_closed = (nrmse_lin - nrmse_s2) / nrmse_lin * 100

            results_by_k[K][str(horizon)] = {
                'nrmse_linear': float(nrmse_lin),
                'nrmse_stokes2': float(nrmse_s2),
                'corr_linear': float(corr_lin),
                'corr_stokes2': float(corr_s2),
                'gap_closed_pct': float(gap_closed),
            }

        elapsed = time.time() - t0
        # Print for K=20 (default WaveSplat scale)
        m = results_by_k[20][str(horizon)]
        print(f"  t+{horizon:3d}s (K=20): lin={m['nrmse_linear']:.4f}, "
              f"s2={m['nrmse_stokes2']:.4f}, "
              f"gap_closed={m['gap_closed_pct']:+.1f}% [{elapsed:.0f}s]")

    return {
        'sea_state': sea_name, 'Hs': Hs, 'Tp': Tp, 'seed': seed,
        'method': 'component_prediction',
        'K_values': list(K_values),
        'results_by_K': {str(K): results_by_k[K] for K in K_values},
    }


def main():
    sea_states = {
        'BF3': {'Hs': 0.6, 'Tp': 5.5},
        'BF4': {'Hs': 1.2, 'Tp': 7.0},
        'BF5': {'Hs': 2.0, 'Tp': 9.0},
    }

    horizons = [5, 10, 20, 30, 60]
    seeds = [0, 1]

    all_results_a = []
    all_results_b = []

    # ======================== METHOD A ========================
    print("=" * 80)
    print("METHOD A: Full-field HOS M=1 vs M=2 vs M=3")
    print("=" * 80)

    for sea_name, sp in sea_states.items():
        for seed in seeds:
            try:
                result = run_method_a(sea_name, sp['Hs'], sp['Tp'],
                                      seed, horizons)
                all_results_a.append(result)
            except Exception as e:
                print(f"  ERROR: {e}")
                import traceback
                traceback.print_exc()

    # Summary A
    print("\n" + "=" * 80)
    print("SUMMARY A: Full-field (M=1 → M=2 → M=3 GT)")
    print("=" * 80)

    from collections import defaultdict
    agg_a = defaultdict(lambda: defaultdict(list))
    for r in all_results_a:
        for h_str, m in r['horizons'].items():
            agg_a[r['sea_state']][int(h_str)].append(m)

    header = f"{'Sea':6s}"
    for h in horizons:
        header += f" {'t+' + str(h) + 's':>28s}"
    print(header)
    print("-" * (6 + 29 * len(horizons)))

    for sea in ['BF3', 'BF4', 'BF5']:
        if sea not in agg_a:
            continue
        row = f"{sea:6s}"
        for h in horizons:
            if h not in agg_a[sea]:
                row += f" {'N/A':>28s}"
                continue
            ms = agg_a[sea][h]
            m1 = np.mean([m['nrmse_M1'] for m in ms])
            m2 = np.mean([m['nrmse_M2'] for m in ms])
            gc = np.mean([m['gap_closed_pct'] for m in ms])
            row += f" {m1:.3f}→{m2:.3f} (gc={gc:+.0f}%)"
        print(row)

    # ======================== METHOD B ========================
    print("\n" + "=" * 80)
    print("METHOD B: Component-level prediction (K=10,20,30)")
    print("=" * 80)

    for sea_name, sp in sea_states.items():
        for seed in seeds[:1]:  # single seed (method B is slower)
            try:
                result = run_method_b(sea_name, sp['Hs'], sp['Tp'],
                                      seed, horizons, K_values=(10, 20, 30))
                all_results_b.append(result)
            except Exception as e:
                print(f"  ERROR: {e}")
                import traceback
                traceback.print_exc()

    # Summary B (K=20)
    print("\n" + "=" * 80)
    print("SUMMARY B: Component prediction (K=20)")
    print("=" * 80)
    for r in all_results_b:
        sea = r['sea_state']
        k20 = r['results_by_K']['20']
        print(f"\n  {sea}:")
        for h in horizons:
            h_str = str(h)
            if h_str in k20:
                m = k20[h_str]
                print(f"    t+{h:3d}s: lin={m['nrmse_linear']:.4f}, "
                      f"s2={m['nrmse_stokes2']:.4f}, "
                      f"gap_closed={m['gap_closed_pct']:+.1f}%")

    # ======================== DECISION ========================
    print("\n" + "=" * 80)
    print("DECISION ANALYSIS")
    print("=" * 80)

    # Method A averages
    print("\n  Method A (best-case, full spectrum):")
    for h in [10, 30, 60]:
        all_gc = [m['gap_closed_pct']
                  for r in all_results_a
                  for m in [r['horizons'].get(str(h))]
                  if m is not None]
        if all_gc:
            print(f"    t+{h}s: M=2 closes {np.mean(all_gc):.1f}% ± {np.std(all_gc):.1f}% of gap")

    # Method B averages (K=20)
    print("\n  Method B (WaveSplat-scale, K=20):")
    for h in [10, 30, 60]:
        all_gc = [r['results_by_K']['20'][str(h)]['gap_closed_pct']
                  for r in all_results_b
                  if str(h) in r['results_by_K']['20']]
        if all_gc:
            print(f"    t+{h}s: Stokes2 closes {np.mean(all_gc):.1f}% ± {np.std(all_gc):.1f}% of gap")

    # Save
    output_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                               'results', 'hos_prediction_gap.json')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    output = {
        'method_a': all_results_a,
        'method_b': all_results_b,
    }
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {output_path}")


if __name__ == '__main__':
    main()
