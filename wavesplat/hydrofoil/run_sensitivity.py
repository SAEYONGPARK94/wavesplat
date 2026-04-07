"""Sensitivity analyses for hydrofoil feedforward controller.

Analysis 1: nRMSE Sensitivity Sweep
  - How much wave estimation error can we tolerate before feedforward hurts?
  - Sweeps nRMSE from 0% (GT) to 50% across two sea state scenarios.

Analysis 2: Preview Distance Sweep
  - What is the optimal spatial preview distance?
  - Compares GT vs WaveSplat (20% nRMSE) across preview distances.
"""

import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + '/..')

from wavesim.wave_generator import jonswap_components
from hydrofoil.config import candela_c8
from hydrofoil.controller import PDControllerConfig, FeedforwardConfig
from hydrofoil.sim import run_simulation


# ── Shared constants ─────────────────────────────────────────────────────────

DURATION = 30.0
TRANSIENT = 5.0          # skip first 5 s
SEED = 42
N_COMPONENTS = 15
DIR_SPREAD = 20.0

SCENARIOS = [
    ("BF3 Head Sea", 0.88, 5.0, 180.0),
    ("BF3 Beam Sea", 0.88, 5.0, 90.0),
]


def _make_cfg_and_components(Hs, Tp, direction_deg):
    cfg = candela_c8()
    cfg['sim'].duration = DURATION
    components = jonswap_components(
        Hs=Hs, Tp=Tp, n_components=N_COMPONENTS,
        depth=cfg['sim'].depth,
        direction_deg=direction_deg,
        direction_spread_deg=DIR_SPREAD,
        seed=SEED,
    )
    return cfg, components


def _run_one(cfg, components, ff_cfg=None):
    """Run a single simulation with fixed seed."""
    np.random.seed(SEED)
    ctrl = PDControllerConfig()
    r = run_simulation(cfg, components, vessel_heading=0.0,
                       controller=ctrl, feedforward=ff_cfg)
    return r


def _metrics(r):
    """Return (heave_rms_cm, accel_rms) after transient."""
    mask = r.time > TRANSIENT
    z_rms = np.std(r.z[mask]) * 100       # cm
    a_rms = np.std(r.heave_accel[mask])   # m/s^2
    return z_rms, a_rms


# ── Analysis 1: nRMSE Sensitivity Sweep ─────────────────────────────────────

def run_nrmse_sweep():
    print("=" * 90)
    print("  Analysis 1: nRMSE Sensitivity Sweep")
    print("  Feedforward: K_ff=0.8, lp_alpha=0.05, preview_dt=0.5s")
    print("=" * 90)

    nrmse_levels = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]

    for sc_name, Hs, Tp, direction_deg in SCENARIOS:
        cfg, components = _make_cfg_and_components(Hs, Tp, direction_deg)

        # PD-only baseline
        print(f"\n  Running {sc_name}: PD-only baseline...", end=" ", flush=True)
        r_base = _run_one(cfg, components, ff_cfg=None)
        z_base, a_base = _metrics(r_base)
        print("done")

        # Sweep nRMSE
        rows = []
        for nrmse in nrmse_levels:
            print(f"  Running {sc_name}: nRMSE={nrmse*100:.0f}%...", end=" ", flush=True)
            ff_cfg = FeedforwardConfig(
                preview_dt=0.5,
                K_ff=0.8,
                lp_alpha=0.05,
                noise_std=Hs * nrmse,
                slope_noise_std=nrmse * 0.1,
            )
            r = _run_one(cfg, components, ff_cfg=ff_cfg)
            z_rms, a_rms = _metrics(r)

            z_imp = (1 - z_rms / z_base) * 100 if z_base > 0 else 0
            a_imp = (1 - a_rms / a_base) * 100 if a_base > 0 else 0
            rows.append((nrmse, z_rms, a_rms, z_imp, a_imp))
            print("done")

        # Print table
        print(f"\n  {'─' * 72}")
        print(f"  {sc_name}  (Hs={Hs}m, Tp={Tp}s, dir={direction_deg}°)")
        print(f"  PD-only baseline: heave RMS = {z_base:.2f} cm,  accel RMS = {a_base:.3f} m/s²")
        print(f"  {'─' * 72}")
        print(f"  {'nRMSE':>8} {'z_RMS [cm]':>12} {'a_RMS [m/s²]':>14} "
              f"{'Δ heave':>10} {'Δ accel':>10}")
        print(f"  {'─' * 72}")

        for nrmse, z_rms, a_rms, z_imp, a_imp in rows:
            label = "GT" if nrmse == 0 else f"{nrmse*100:.0f}%"
            print(f"  {label:>8} {z_rms:>12.2f} {a_rms:>14.3f} "
                  f"{z_imp:>+9.1f}% {a_imp:>+9.1f}%")

        # Find threshold where accel improvement drops below 0%
        threshold = None
        for nrmse, _, _, _, a_imp in rows:
            if a_imp < 0:
                threshold = nrmse
                break

        print(f"  {'─' * 72}")
        if threshold is not None:
            print(f"  >>> Accel degradation threshold: nRMSE = {threshold*100:.0f}%")
        else:
            print(f"  >>> Feedforward beneficial across all tested nRMSE levels")
        print()


# ── Analysis 2: Preview Distance Sweep ──────────────────────────────────────

def run_preview_sweep():
    print("=" * 90)
    print("  Analysis 2: Preview Distance Sweep")
    print("  Feedforward: K_ff=0.8, lp_alpha=0.05")
    print("  Comparing GT (noise=0) vs WaveSplat (20% nRMSE)")
    print("=" * 90)

    preview_dts = [0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0]

    for sc_name, Hs, Tp, direction_deg in SCENARIOS:
        cfg, components = _make_cfg_and_components(Hs, Tp, direction_deg)

        # PD-only baseline
        print(f"\n  Running {sc_name}: PD-only baseline...", end=" ", flush=True)
        r_base = _run_one(cfg, components, ff_cfg=None)
        z_base, a_base = _metrics(r_base)
        print("done")

        rows = []
        for pdt in preview_dts:
            # GT run
            print(f"  Running {sc_name}: preview_dt={pdt:.1f}s GT...", end=" ", flush=True)
            ff_gt = FeedforwardConfig(
                preview_dt=pdt, K_ff=0.8, lp_alpha=0.05,
                noise_std=0.0, slope_noise_std=0.0,
            )
            r_gt = _run_one(cfg, components, ff_cfg=ff_gt)
            z_gt, a_gt = _metrics(r_gt)
            print("done")

            # WaveSplat (20% nRMSE) run
            print(f"  Running {sc_name}: preview_dt={pdt:.1f}s WS...", end=" ", flush=True)
            ff_ws = FeedforwardConfig(
                preview_dt=pdt, K_ff=0.8, lp_alpha=0.05,
                noise_std=Hs * 0.20,
                slope_noise_std=0.20 * 0.1,
            )
            r_ws = _run_one(cfg, components, ff_cfg=ff_ws)
            z_ws, a_ws = _metrics(r_ws)
            print("done")

            z_imp_gt = (1 - z_gt / z_base) * 100 if z_base > 0 else 0
            a_imp_gt = (1 - a_gt / a_base) * 100 if a_base > 0 else 0
            z_imp_ws = (1 - z_ws / z_base) * 100 if z_base > 0 else 0
            a_imp_ws = (1 - a_ws / a_base) * 100 if a_base > 0 else 0

            rows.append((pdt, z_imp_gt, a_imp_gt, z_imp_ws, a_imp_ws))

        # Print table
        print(f"\n  {'─' * 80}")
        print(f"  {sc_name}  (Hs={Hs}m, Tp={Tp}s, dir={direction_deg}°)")
        print(f"  PD-only baseline: heave RMS = {z_base:.2f} cm,  accel RMS = {a_base:.3f} m/s²")
        print(f"  {'─' * 80}")
        print(f"  {'preview_dt':>12} {'GT Δheave':>11} {'GT Δaccel':>11} "
              f"{'WS Δheave':>11} {'WS Δaccel':>11}")
        print(f"  {'[s]':>12} {'':>11} {'':>11} {'':>11} {'':>11}")
        print(f"  {'─' * 80}")

        for pdt, z_gt, a_gt, z_ws, a_ws in rows:
            print(f"  {pdt:>12.1f} {z_gt:>+10.1f}% {a_gt:>+10.1f}% "
                  f"{z_ws:>+10.1f}% {a_ws:>+10.1f}%")

        # Find optimal preview distance (max accel improvement)
        best_gt = max(rows, key=lambda r: r[2])
        best_ws = max(rows, key=lambda r: r[4])

        print(f"  {'─' * 80}")
        print(f"  >>> Optimal preview (GT):       dt={best_gt[0]:.1f}s  "
              f"(accel {best_gt[2]:+.1f}%, heave {best_gt[1]:+.1f}%)")
        print(f"  >>> Optimal preview (WaveSplat): dt={best_ws[0]:.1f}s  "
              f"(accel {best_ws[4]:+.1f}%, heave {best_ws[3]:+.1f}%)")
        print()


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    print()
    run_nrmse_sweep()
    run_preview_sweep()

    print("=" * 90)
    print("  Sensitivity analysis complete.")
    print("=" * 90)


if __name__ == '__main__':
    main()
