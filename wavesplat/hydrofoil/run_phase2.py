"""Phase 2: PD-only vs PD+Feedforward comparison.

Compares four control strategies across multiple sea states:
  1. PD-only:           feedback only (baseline)
  2. PD + GT FF:        perfect wave preview (upper bound)
  3. PD + WaveSplat FF: simulated ~20% nRMSE estimation error
  4. PD + Noisy FF:     simulated ~40% nRMSE (poor estimation)

Metrics: heave RMS, pitch RMS, accel RMS, comfort rating, saturation %
"""

import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + '/..')

from wavesim.wave_generator import jonswap_components
from hydrofoil.config import candela_c8
from hydrofoil.controller import PDControllerConfig, FeedforwardConfig
from hydrofoil.sim import run_simulation


def comfort_rating(rms_accel):
    if rms_accel < 0.315:
        return "Comfortable"
    elif rms_accel < 0.63:
        return "Slight"
    elif rms_accel < 1.0:
        return "Fairly"
    elif rms_accel < 1.6:
        return "Uncomfortable"
    else:
        return "Very Uncomf."


def saturation_pct(flap_arr, delta_max=0.26):
    """Percentage of time flap is saturated (within 1% of limit)."""
    return np.mean(np.abs(flap_arr) > delta_max * 0.99) * 100


def run_comparison(scenario_name, Hs, Tp, direction_deg,
                   duration=30.0, seed=42):
    """Run all 4 control strategies for one scenario."""
    cfg = candela_c8()
    cfg['sim'].duration = duration

    components = jonswap_components(
        Hs=Hs, Tp=Tp, n_components=15,
        depth=cfg['sim'].depth,
        direction_deg=direction_deg,
        direction_spread_deg=20.0,
        seed=seed,
    )

    ctrl = PDControllerConfig()

    # --- Strategy definitions ---
    strategies = {
        'PD-only': None,
        'PD+GT_FF': FeedforwardConfig(
            preview_dt=0.5, K_ff=0.8,
            noise_std=0.0, slope_noise_std=0.0,
        ),
        'PD+WS_FF': FeedforwardConfig(
            preview_dt=0.5, K_ff=0.8,
            noise_std=Hs * 0.20,           # ~20% nRMSE on elevation
            slope_noise_std=0.02,           # slope noise ~1.1°
        ),
        'PD+Noisy_FF': FeedforwardConfig(
            preview_dt=0.5, K_ff=0.8,
            noise_std=Hs * 0.40,           # ~40% nRMSE
            slope_noise_std=0.05,           # slope noise ~2.9°
        ),
    }

    results = {}
    for name, ff_cfg in strategies.items():
        # Fix random seed for reproducibility of noise
        np.random.seed(seed)
        r = run_simulation(cfg, components, vessel_heading=0.0,
                           controller=ctrl, feedforward=ff_cfg)
        results[name] = r

    return results


def analyze_results(results, scenario_name, Hs):
    """Print comparison table for one scenario."""
    print(f"\n{'=' * 90}")
    print(f"  {scenario_name}")
    print(f"{'=' * 90}")
    print(f"  {'Strategy':<16} {'z_RMS':>8} {'θ_RMS':>8} {'a_RMS':>8} "
          f"{'z_max':>8} {'θ_max':>8} {'Sat%':>6} {'Comfort':>13}")
    print(f"  {'':16} {'[cm]':>8} {'[deg]':>8} {'[m/s²]':>8} "
          f"{'[cm]':>8} {'[deg]':>8} {'':>6} {'':>13}")
    print("  " + "-" * 84)

    baseline = None
    for name, r in results.items():
        mask = r.time > 5.0
        z = r.z[mask]
        theta = r.theta[mask]
        z_ddot = r.heave_accel[mask]
        flap_f = r.flap_front[mask]
        flap_r = r.flap_rear[mask]

        z_rms = np.std(z) * 100  # cm
        th_rms = np.degrees(np.std(theta))
        a_rms = np.std(z_ddot)
        z_max = np.max(np.abs(z)) * 100
        th_max = np.degrees(np.max(np.abs(theta)))
        sat = max(saturation_pct(flap_f), saturation_pct(flap_r))
        comfort = comfort_rating(a_rms)

        if baseline is None:
            baseline = (z_rms, th_rms, a_rms)

        # Improvement vs baseline
        z_imp = (1 - z_rms / baseline[0]) * 100 if baseline[0] > 0 else 0
        a_imp = (1 - a_rms / baseline[2]) * 100 if baseline[2] > 0 else 0

        imp_str = "" if name == 'PD-only' else f" ({z_imp:+.0f}%z, {a_imp:+.0f}%a)"

        print(f"  {name:<16} {z_rms:>8.2f} {th_rms:>8.3f} {a_rms:>8.3f} "
              f"{z_max:>8.2f} {th_max:>8.3f} {sat:>5.1f}% {comfort:>13}{imp_str}")


def main():
    print("=" * 90)
    print("  Phase 2: Feedforward Control Comparison")
    print("  Vessel: Candela C-8 class, 22 knots")
    print("  Preview: 0.5s spatial (5.65m ahead)")
    print("  FF gain: 0.8 (80% compensation)")
    print("=" * 90)

    scenarios = [
        ("BF3 Head Sea  (Hs=0.88m Tp=5.0s)", 0.88, 5.0, 180.0),
        ("BF4 Head Sea  (Hs=1.88m Tp=7.5s)", 1.88, 7.5, 180.0),
        ("BF3 Beam Sea  (Hs=0.88m Tp=5.0s)", 0.88, 5.0, 90.0),
        ("BF3 Quarter   (Hs=0.88m Tp=5.0s)", 0.88, 5.0, 135.0),
    ]

    all_results = {}
    for name, Hs, Tp, d in scenarios:
        print(f"\n  Running: {name}...", end=" ", flush=True)
        results = run_comparison(name, Hs, Tp, d)
        all_results[name] = (results, Hs)
        print("done")

    for name, (results, Hs) in all_results.items():
        analyze_results(results, name, Hs)

    # --- Summary: feedforward improvement ---
    print(f"\n{'=' * 90}")
    print("  SUMMARY: Feedforward Improvement over PD-only")
    print(f"{'=' * 90}")
    print(f"  {'Scenario':<35} {'GT FF':>12} {'WaveSplat FF':>12} {'Noisy FF':>12}")
    print(f"  {'':35} {'Δ accel':>12} {'Δ accel':>12} {'Δ accel':>12}")
    print("  " + "-" * 71)

    for name, (results, Hs) in all_results.items():
        mask = results['PD-only'].time > 5.0
        a_base = np.std(results['PD-only'].heave_accel[mask])

        improvements = []
        for strat in ['PD+GT_FF', 'PD+WS_FF', 'PD+Noisy_FF']:
            a = np.std(results[strat].heave_accel[mask])
            imp = (1 - a / a_base) * 100
            improvements.append(imp)

        short_name = name[:35]
        print(f"  {short_name:<35} {improvements[0]:>+11.1f}% {improvements[1]:>+11.1f}% "
              f"{improvements[2]:>+11.1f}%")

    print(f"\n  Note: WaveSplat FF simulated with ~20% nRMSE noise on wave estimation")
    print(f"  Note: Noisy FF simulated with ~40% nRMSE (degraded estimation)")
    print(f"  → GT FF shows theoretical upper bound of feedforward benefit")
    print(f"  → WaveSplat FF shows realistic benefit given current estimation accuracy")


if __name__ == '__main__':
    main()
