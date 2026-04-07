"""Generate publication-quality figures from hydrofoil sensitivity analysis.

Produces three figures:
  1. nRMSE Sensitivity: heave & accel improvement vs estimation error
  2. Preview Distance Sweep: GT vs WaveSplat across preview horizons
  3. Phase 2 Comparison: bar chart of accel improvement across scenarios
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + '/..')

from wavesim.wave_generator import jonswap_components
from hydrofoil.config import candela_c8
from hydrofoil.controller import PDControllerConfig, FeedforwardConfig
from hydrofoil.sim import run_simulation

# ── Style setup ──────────────────────────────────────────────────────────────

for _style in ['seaborn-v0_8-whitegrid', 'seaborn-whitegrid']:
    try:
        plt.style.use(_style)
        break
    except OSError:
        continue

FIGURES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figures')

# ── Shared constants (same as run_sensitivity.py) ────────────────────────────

DURATION = 30.0
TRANSIENT = 5.0
SEED = 42
N_COMPONENTS = 15
DIR_SPREAD = 20.0

SCENARIOS = [
    ("BF3 Head Sea", 0.88, 5.0, 180.0),
    ("BF3 Beam Sea", 0.88, 5.0, 90.0),
]

PHASE2_SCENARIOS = [
    ("BF3 Head (180\u00b0)",  0.88, 5.0, 180.0),
    ("BF4 Head (180\u00b0)",  1.88, 7.5, 180.0),
    ("BF3 Beam (90\u00b0)",   0.88, 5.0,  90.0),
    ("BF3 Quarter (135\u00b0)", 0.88, 5.0, 135.0),
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
    np.random.seed(SEED)
    ctrl = PDControllerConfig()
    r = run_simulation(cfg, components, vessel_heading=0.0,
                       controller=ctrl, feedforward=ff_cfg)
    return r


def _metrics(r):
    mask = r.time > TRANSIENT
    z_rms = np.std(r.z[mask]) * 100       # cm
    a_rms = np.std(r.heave_accel[mask])   # m/s^2
    return z_rms, a_rms


def _save(fig, name):
    os.makedirs(FIGURES_DIR, exist_ok=True)
    for ext in ('png', 'pdf'):
        path = os.path.join(FIGURES_DIR, f'{name}.{ext}')
        fig.savefig(path, dpi=150, bbox_inches='tight')
    print(f"  Saved {name}.png / .pdf")


# ── Figure 1: nRMSE Sensitivity ─────────────────────────────────────────────

def collect_nrmse_data():
    nrmse_levels = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]
    all_data = {}

    for sc_name, Hs, Tp, direction_deg in SCENARIOS:
        cfg, components = _make_cfg_and_components(Hs, Tp, direction_deg)

        print(f"  [{sc_name}] PD-only baseline...", end=" ", flush=True)
        r_base = _run_one(cfg, components, ff_cfg=None)
        z_base, a_base = _metrics(r_base)
        print("done")

        z_imps, a_imps = [], []
        for nrmse in nrmse_levels:
            print(f"  [{sc_name}] nRMSE={nrmse*100:.0f}%...", end=" ", flush=True)
            ff_cfg = FeedforwardConfig(
                preview_dt=0.5, K_ff=0.8, lp_alpha=0.05,
                noise_std=Hs * nrmse,
                slope_noise_std=nrmse * 0.1,
            )
            r = _run_one(cfg, components, ff_cfg=ff_cfg)
            z_rms, a_rms = _metrics(r)
            z_imps.append((1 - z_rms / z_base) * 100 if z_base > 0 else 0)
            a_imps.append((1 - a_rms / a_base) * 100 if a_base > 0 else 0)
            print("done")

        all_data[sc_name] = {
            'nrmse_pct': [n * 100 for n in nrmse_levels],
            'z_imp': z_imps,
            'a_imp': a_imps,
        }

    return all_data


def plot_nrmse_sensitivity(all_data):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, (sc_name, _Hs, _Tp, _d) in zip(axes, SCENARIOS):
        d = all_data[sc_name]
        x = d['nrmse_pct']

        # Left y-axis: heave improvement
        color_z = '#1f77b4'
        ax.plot(x, d['z_imp'], 'o-', color=color_z, linewidth=2,
                markersize=6, label='Heave RMS impr.')
        ax.set_xlabel('nRMSE [%]', fontsize=11)
        ax.set_ylabel('Heave RMS improvement [%]', fontsize=11, color=color_z)
        ax.tick_params(axis='y', labelcolor=color_z)

        # Right y-axis: accel improvement
        ax2 = ax.twinx()
        color_a = '#d62728'
        ax2.plot(x, d['a_imp'], 's--', color=color_a, linewidth=2,
                 markersize=6, label='Accel RMS impr.')
        ax2.set_ylabel('Accel RMS improvement [%]', fontsize=11, color=color_a)
        ax2.tick_params(axis='y', labelcolor=color_a)

        # Reference line at y=0
        ax.axhline(0, color='gray', linestyle='--', linewidth=0.8, zorder=0)

        # WaveSplat range shading
        ax.axvspan(15, 25, alpha=0.15, color='#1f77b4', zorder=0,
                   label='WaveSplat range')

        ax.set_title(sc_name, fontsize=13)
        ax.set_xticks(x)

        # Combined legend
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2,
                  loc='lower left', fontsize=9)

    fig.tight_layout()
    _save(fig, 'fig_nrmse_sensitivity')
    plt.close(fig)


# ── Figure 2: Preview Distance Sweep ────────────────────────────────────────

def collect_preview_data():
    preview_dts = [0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0]
    all_data = {}

    for sc_name, Hs, Tp, direction_deg in SCENARIOS:
        cfg, components = _make_cfg_and_components(Hs, Tp, direction_deg)

        print(f"  [{sc_name}] PD-only baseline...", end=" ", flush=True)
        r_base = _run_one(cfg, components, ff_cfg=None)
        z_base, a_base = _metrics(r_base)
        print("done")

        z_gt, a_gt, z_ws, a_ws = [], [], [], []
        for pdt in preview_dts:
            # GT
            print(f"  [{sc_name}] preview={pdt:.1f}s GT...", end=" ", flush=True)
            ff_gt = FeedforwardConfig(
                preview_dt=pdt, K_ff=0.8, lp_alpha=0.05,
                noise_std=0.0, slope_noise_std=0.0,
            )
            r = _run_one(cfg, components, ff_cfg=ff_gt)
            zr, ar = _metrics(r)
            z_gt.append((1 - zr / z_base) * 100 if z_base > 0 else 0)
            a_gt.append((1 - ar / a_base) * 100 if a_base > 0 else 0)
            print("done")

            # WaveSplat (20% nRMSE)
            print(f"  [{sc_name}] preview={pdt:.1f}s WS...", end=" ", flush=True)
            ff_ws = FeedforwardConfig(
                preview_dt=pdt, K_ff=0.8, lp_alpha=0.05,
                noise_std=Hs * 0.20,
                slope_noise_std=0.20 * 0.1,
            )
            r = _run_one(cfg, components, ff_cfg=ff_ws)
            zr, ar = _metrics(r)
            z_ws.append((1 - zr / z_base) * 100 if z_base > 0 else 0)
            a_ws.append((1 - ar / a_base) * 100 if a_base > 0 else 0)
            print("done")

        all_data[sc_name] = {
            'preview_dts': preview_dts,
            'z_gt': z_gt, 'a_gt': a_gt,
            'z_ws': z_ws, 'a_ws': a_ws,
        }

    return all_data


def plot_preview_sweep(all_data):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, (sc_name, _Hs, _Tp, _d) in zip(axes, SCENARIOS):
        d = all_data[sc_name]
        x = d['preview_dts']

        ax.plot(x, d['z_gt'], 'o-', color='#1f77b4', linewidth=2,
                markersize=5, label='GT heave')
        ax.plot(x, d['a_gt'], 's-', color='#d62728', linewidth=2,
                markersize=5, label='GT accel')
        ax.plot(x, d['z_ws'], 'o--', color='#6baed6', linewidth=2,
                markersize=5, label='WS heave')
        ax.plot(x, d['a_ws'], 's--', color='#fc9272', linewidth=2,
                markersize=5, label='WS accel')

        # Mark optimal points (max accel improvement)
        idx_gt = int(np.argmax(d['a_gt']))
        idx_ws = int(np.argmax(d['a_ws']))
        ax.plot(x[idx_gt], d['a_gt'][idx_gt], '*', color='#d62728',
                markersize=16, zorder=5, markeredgecolor='black',
                markeredgewidth=0.5)
        ax.plot(x[idx_ws], d['a_ws'][idx_ws], '*', color='#fc9272',
                markersize=16, zorder=5, markeredgecolor='black',
                markeredgewidth=0.5)

        ax.axhline(0, color='gray', linestyle='--', linewidth=0.8, zorder=0)
        ax.set_xlabel('Preview distance [s]', fontsize=11)
        ax.set_ylabel('RMS improvement [%]', fontsize=11)
        ax.set_title(sc_name, fontsize=13)
        ax.legend(fontsize=9, loc='best')

    fig.tight_layout()
    _save(fig, 'fig_preview_sweep')
    plt.close(fig)


# ── Figure 3: Phase 2 Comparison Bar Chart ──────────────────────────────────

def collect_phase2_data():
    all_data = {}

    for sc_name, Hs, Tp, direction_deg in PHASE2_SCENARIOS:
        cfg = candela_c8()
        cfg['sim'].duration = DURATION
        components = jonswap_components(
            Hs=Hs, Tp=Tp, n_components=N_COMPONENTS,
            depth=cfg['sim'].depth,
            direction_deg=direction_deg,
            direction_spread_deg=DIR_SPREAD,
            seed=SEED,
        )
        ctrl = PDControllerConfig()

        strategies = {
            'PD-only': None,
            'GT FF': FeedforwardConfig(
                preview_dt=0.5, K_ff=0.8,
                noise_std=0.0, slope_noise_std=0.0,
            ),
            'WaveSplat FF': FeedforwardConfig(
                preview_dt=0.5, K_ff=0.8,
                noise_std=Hs * 0.20,
                slope_noise_std=0.02,
            ),
            'Noisy FF': FeedforwardConfig(
                preview_dt=0.5, K_ff=0.8,
                noise_std=Hs * 0.40,
                slope_noise_std=0.05,
            ),
        }

        accel_imps = {}
        a_base = None
        for strat_name, ff_cfg in strategies.items():
            print(f"  [{sc_name}] {strat_name}...", end=" ", flush=True)
            np.random.seed(SEED)
            r = run_simulation(cfg, components, vessel_heading=0.0,
                               controller=ctrl, feedforward=ff_cfg)
            mask = r.time > TRANSIENT
            a_rms = np.std(r.heave_accel[mask])

            if strat_name == 'PD-only':
                a_base = a_rms
                accel_imps[strat_name] = 0.0
            else:
                accel_imps[strat_name] = (1 - a_rms / a_base) * 100
            print("done")

        all_data[sc_name] = accel_imps

    return all_data


def plot_phase2_comparison(all_data):
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))

    scenario_names = [s[0] for s in PHASE2_SCENARIOS]
    strat_names = ['GT FF', 'WaveSplat FF', 'Noisy FF']
    colors = {'GT FF': '#2ca02c', 'WaveSplat FF': '#1f77b4', 'Noisy FF': '#ff7f0e'}

    n_scenarios = len(scenario_names)
    n_strats = len(strat_names)
    bar_width = 0.22
    x = np.arange(n_scenarios)

    for i, strat in enumerate(strat_names):
        vals = [all_data[sc][strat] for sc in scenario_names]
        offset = (i - (n_strats - 1) / 2) * bar_width
        bars = ax.bar(x + offset, vals, bar_width, label=strat,
                      color=colors[strat], edgecolor='white', linewidth=0.5)

        # Value labels on bars
        for bar, v in zip(bars, vals):
            y_pos = bar.get_height()
            va = 'bottom' if y_pos >= 0 else 'top'
            ax.text(bar.get_x() + bar.get_width() / 2, y_pos,
                    f'{v:+.1f}%', ha='center', va=va, fontsize=9,
                    fontweight='bold')

    ax.axhline(0, color='gray', linestyle='--', linewidth=0.8, zorder=0)
    ax.set_xticks(x)
    ax.set_xticklabels(scenario_names, fontsize=11)
    ax.set_ylabel('Accel RMS improvement [%]', fontsize=11)
    ax.set_title('Feedforward Control: Accel Improvement over PD-only', fontsize=13)
    ax.legend(fontsize=10, loc='best')

    fig.tight_layout()
    _save(fig, 'fig_phase2_comparison')
    plt.close(fig)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    print()
    print("=" * 70)
    print("  Generating hydrofoil sensitivity analysis figures")
    print("=" * 70)

    # Figure 1
    print("\n[Figure 1] nRMSE Sensitivity Sweep")
    nrmse_data = collect_nrmse_data()
    plot_nrmse_sensitivity(nrmse_data)

    # Figure 2
    print("\n[Figure 2] Preview Distance Sweep")
    preview_data = collect_preview_data()
    plot_preview_sweep(preview_data)

    # Figure 3
    print("\n[Figure 3] Phase 2 Comparison")
    phase2_data = collect_phase2_data()
    plot_phase2_comparison(phase2_data)

    print("\n" + "=" * 70)
    print(f"  All figures saved to {FIGURES_DIR}")
    print("=" * 70)


if __name__ == '__main__':
    main()
