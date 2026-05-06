"""Single-point bow sensor vs WaveSplat feedforward comparison.

Quantifies what a Candela-style single downward-facing bow sensor can
achieve compared to WaveSplat's full 2D spectral feedforward.

Control strategies compared
---------------------------
PD-only        Feedback only — lower bound.
GT-FF          Perfect full-spectral model, per-foil evaluation — upper bound.
WS-FF          WaveSplat (~20% nRMSE noise), per-foil evaluation.
SP-1pt         Best-case single-point: GT components at ONE preview point,
               SAME signal for both foils.  Isolates the per-foil effect.
SP-temporal    Realistic bow sensor: temporal differentiation + head-sea
               slope assumption.  Fails systematically in beam/oblique seas.

Scenarios
---------
BF3 Head sea  — expected SP ≈ WS (1D head sea, single-point works)
BF4 Head sea  — foil separation / λ larger → per-foil effect matters more
BF3 Beam sea  — SP-temporal head-sea assumption wrong → may degrade
BF3 Quarter sea (135°) — oblique sea, intermediate case
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + '/..')

from wavesim.wave_generator import jonswap_components
from hydrofoil.config import candela_c8
from hydrofoil.controller import PDControllerConfig, FeedforwardConfig, FeedforwardFilter
from hydrofoil.sim import run_simulation
from hydrofoil.single_point_ff import compute_single_point_ff_info, TemporalBowSensor


# ─────────────────────────────────────────────────────────────────────────────
# Phase-velocity helper
# ─────────────────────────────────────────────────────────────────────────────

def peak_phase_velocity(Tp: float, depth: float = 50.0, g: float = 9.81) -> float:
    """Dispersion-accurate phase velocity at peak period [m/s]."""
    omega_p = 2 * np.pi / Tp
    # Newton iteration for k: omega^2 = g*k*tanh(k*d)
    k = omega_p ** 2 / g  # deep-water initial guess
    for _ in range(30):
        k = omega_p ** 2 / (g * np.tanh(k * depth))
    return omega_p / k


# ─────────────────────────────────────────────────────────────────────────────
# ff_info_provider factory for SP-1pt
# ─────────────────────────────────────────────────────────────────────────────

def make_sp1pt_provider(components: list):
    """Return a ff_info_provider callable using SP-1pt (analytic GT, 1 point)."""
    def provider(state, front_foil, rear_foil, sim, comps, t, heading, ff_cfg):
        return compute_single_point_ff_info(
            x_cg=state.x,
            components=components,
            t=t,
            vessel_heading=heading,
            speed=sim.speed,
            preview_dt=ff_cfg.preview_dt,
        )
    return provider


# ─────────────────────────────────────────────────────────────────────────────
# ff_info_provider factory for SP-temporal
# ─────────────────────────────────────────────────────────────────────────────

def make_sp_temporal_provider(components: list, Tp: float, dt: float,
                              depth: float = 50.0):
    """Return a ff_info_provider callable using SP-temporal (realistic sensor)."""
    c_phase = peak_phase_velocity(Tp, depth)
    sensor = TemporalBowSensor(dt=dt, c_phase=c_phase, noise_std=0.02, bow_offset=4.0)

    def provider(state, front_foil, rear_foil, sim, comps, t, heading, ff_cfg):
        return sensor.compute_ff_info(
            x_cg=state.x,
            components=components,
            t=t,
            vessel_heading=heading,
            speed=sim.speed,
            preview_dt=ff_cfg.preview_dt,
        )
    return provider


# ─────────────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────────────

def comfort_rating(rms_accel: float) -> str:
    if rms_accel < 0.315:
        return "Comfortable"
    elif rms_accel < 0.63:
        return "Slight"
    elif rms_accel < 1.0:
        return "Fairly"
    elif rms_accel < 1.6:
        return "Uncomf."
    else:
        return "Very Uncomf."


def saturation_pct(flap_arr: np.ndarray, delta_max: float = 0.26) -> float:
    return np.mean(np.abs(flap_arr) > delta_max * 0.99) * 100


# ─────────────────────────────────────────────────────────────────────────────
# Main comparison
# ─────────────────────────────────────────────────────────────────────────────

PREVIEW_DT = 0.5    # [s] — same for all strategies (5.65 m ahead at 11.3 m/s)
DURATION   = 30.0   # [s]
SEED       = 42     # default seed; main() uses multi-seed averaging
SEEDS      = [42, 0, 1, 2, 3]  # five seeds for paper Table~\ref{tab:single_point}


def run_scenario(scenario_name: str, Hs: float, Tp: float,
                 direction_deg: float, seed: int = SEED) -> dict:
    """Run all five strategies for one scenario with a given random seed."""
    cfg = candela_c8()
    cfg['sim'].duration = DURATION
    dt = cfg['sim'].dt  # 0.01 s

    components = jonswap_components(
        Hs=Hs, Tp=Tp, n_components=15,
        depth=cfg['sim'].depth,
        direction_deg=direction_deg,
        direction_spread_deg=20.0,
        seed=seed,
    )

    ctrl = PDControllerConfig()
    ff_base = FeedforwardConfig(preview_dt=PREVIEW_DT, K_ff=0.8,
                                noise_std=0.0, slope_noise_std=0.0)

    # Providers for single-point strategies
    sp1pt_provider    = make_sp1pt_provider(components)
    sp_temp_provider  = make_sp_temporal_provider(components, Tp, dt,
                                                  cfg['sim'].depth)

    strategies = {
        'PD-only':     (None,    None),
        'GT-FF':       (ff_base, None),
        'WS-FF':       (FeedforwardConfig(preview_dt=PREVIEW_DT, K_ff=0.8,
                                          noise_std=Hs * 0.20,
                                          slope_noise_std=0.02), None),
        'SP-1pt':      (ff_base, sp1pt_provider),
        'SP-temporal': (ff_base, sp_temp_provider),
    }

    results = {}
    for name, (ff_cfg, provider) in strategies.items():
        np.random.seed(seed)
        r = run_simulation(
            cfg, components, vessel_heading=0.0,
            controller=ctrl,
            feedforward=ff_cfg,
            ff_info_provider=provider,
        )
        results[name] = r

    return results


def print_scenario(results: dict, scenario_name: str) -> dict:
    """Print comparison table and return improvement dict."""
    print(f"\n{'=' * 100}")
    print(f"  {scenario_name}")
    print(f"{'=' * 100}")
    print(f"  {'Strategy':<14} {'a_RMS':>8} {'z_RMS':>8} {'θ_RMS':>8} "
          f"{'Sat%':>6} {'Comfort':>13} {'vs PD':>10} {'vs WS':>10}")
    print(f"  {'':14} {'[m/s²]':>8} {'[cm]':>8} {'[deg]':>8} "
          f"{'':>6} {'':>13} {'Δa%':>10} {'Δa%':>10}")
    print("  " + "-" * 90)

    mask_fn = lambda r: r.time > 5.0
    metrics = {}

    for name, r in results.items():
        mask = mask_fn(r)
        a_rms = np.std(r.heave_accel[mask])
        z_rms = np.std(r.z[mask]) * 100
        th_rms = np.degrees(np.std(r.theta[mask]))
        sat = max(saturation_pct(r.flap_front[mask]),
                  saturation_pct(r.flap_rear[mask]))
        metrics[name] = a_rms

    a_pd = metrics['PD-only']
    a_ws = metrics['WS-FF']

    for name, r in results.items():
        mask = mask_fn(r)
        a_rms = metrics[name]
        z_rms = np.std(r.z[mask]) * 100
        th_rms = np.degrees(np.std(r.theta[mask]))
        sat = max(saturation_pct(r.flap_front[mask] if r.flap_front is not None
                                 else np.zeros(mask.sum())),
                  saturation_pct(r.flap_rear[mask] if r.flap_rear is not None
                                 else np.zeros(mask.sum())))
        comfort = comfort_rating(a_rms)
        vs_pd = (1 - a_rms / a_pd) * 100 if name != 'PD-only' else 0.0
        vs_ws = (1 - a_rms / a_ws) * 100 if name not in ('PD-only', 'WS-FF') else 0.0

        vs_pd_str = f"{vs_pd:+.1f}%" if name != 'PD-only' else "—"
        vs_ws_str = f"{vs_ws:+.1f}%" if name not in ('PD-only', 'WS-FF') else "—"

        print(f"  {name:<14} {a_rms:>8.3f} {z_rms:>8.2f} {th_rms:>8.3f} "
              f"{sat:>5.1f}% {comfort:>13} {vs_pd_str:>10} {vs_ws_str:>10}")

    return metrics


def main():
    """Run multi-seed comparison reproducing Table tab:single_point in the paper.

    Each scenario is run over SEEDS = [42, 0, 1, 2, 3].  Results are reported
    as mean ± std over the five wave realisations, matching the paper table.
    """
    scenarios = [
        ("BF3 Head sea   (Hs=0.88m Tp=5.0s μ=180°)", 0.88, 5.0, 180.0),
        ("BF4 Head sea   (Hs=1.88m Tp=7.5s μ=180°)", 1.88, 7.5, 180.0),
        ("BF3 Beam sea   (Hs=0.88m Tp=5.0s μ=90°)",  0.88, 5.0,  90.0),
        ("BF3 Quarter    (Hs=0.88m Tp=5.0s μ=135°)", 0.88, 5.0, 135.0),
    ]

    print("=" * 110)
    print("  Single-Point Bow Sensor vs WaveSplat Feedforward  (N={} seeds)".format(
        len(SEEDS)))
    print("  Reference vessel: Candela C-8  |  V=11.3 m/s  |  preview_dt=0.5 s")
    print("  SP-1pt:    GT components, single preview point, same signal both foils")
    print("  SP-temporal: temporal dη/dt from noisy bow sensor, head-sea slope assumption")
    print("=" * 110)

    # ── Collect per-seed improvements ─────────────────────────────────────
    # imps[scenario_name][strategy_name] = list of improvement values
    imps = {name: {'GT-FF': [], 'WS-FF': [], 'SP-1pt': [], 'SP-temporal': []}
            for name, *_ in scenarios}

    for name, Hs, Tp, d in scenarios:
        for seed in SEEDS:
            print(f"  {name}  seed={seed}...", end=" ", flush=True)
            results = run_scenario(name, Hs, Tp, d, seed=seed)
            mask = results['PD-only'].time > 5.0
            a_pd = np.std(results['PD-only'].heave_accel[mask])
            for strat in ('GT-FF', 'WS-FF', 'SP-1pt', 'SP-temporal'):
                imp = (1 - np.std(results[strat].heave_accel[mask]) / a_pd) * 100
                imps[name][strat].append(imp)
            print("done")
        print()

    # ── Per-scenario detail table ──────────────────────────────────────────
    for name, Hs, Tp, d in scenarios:
        results_seed42 = run_scenario(name, Hs, Tp, d, seed=42)
        print_scenario(results_seed42, name + "  [seed=42 detail]")

    # ── Multi-seed summary table ───────────────────────────────────────────
    print(f"\n{'=' * 110}")
    print(f"  SUMMARY (mean ± std over N={len(SEEDS)} seeds)  — reproduces paper Table tab:single_point")
    print(f"{'=' * 110}")
    hdr = "  {:<44} {:>14} {:>14} {:>14} {:>14} {:>9}".format(
        "Scenario", "GT-FF", "WS-FF", "SP-1pt", "SP-temporal", "SP1/GT%")
    print(hdr)
    print("  " + "-" * 108)

    for name, *_ in scenarios:
        v = imps[name]
        def fmt(lst):
            return "{:+.1f}(±{:.1f})".format(np.mean(lst), np.std(lst))
        gt_mean = np.mean(v['GT-FF'])
        sp1_mean = np.mean(v['SP-1pt'])
        ratio = sp1_mean / gt_mean * 100 if gt_mean != 0 else float('nan')
        print("  {:<44} {:>14} {:>14} {:>14} {:>14} {:>8.0f}%".format(
            name[:44],
            fmt(v['GT-FF']), fmt(v['WS-FF']),
            fmt(v['SP-1pt']), fmt(v['SP-temporal']),
            ratio))

    print(f"\n  SP1/GT% = mean(SP-1pt improvement) / mean(GT-FF improvement) × 100")
    print(f"\n  Key findings:")
    print(f"    BF3 Head: SP-1pt consistently degrades (structural per-foil phase mismatch)")
    print(f"    BF3 Beam: SP-1pt highly variable (directional spreading → seed-dependent)")
    print(f"    SP-temporal: reliably harmful in beam/oblique sea (head-sea assumption)")


if __name__ == '__main__':
    main()
