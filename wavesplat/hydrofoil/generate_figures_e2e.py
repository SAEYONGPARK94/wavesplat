"""Generate publication-quality figures from Phase 3 end-to-end results.

Produces two figures:
  Figure 4: Time-domain comparison (heave + accel) for best/worst WaveSplat episodes
  Figure 5: WaveSplat component accuracy (GT vs estimated in omega-amplitude space)
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import h5py

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + '/..')

from hydrofoil.run_phase3_e2e import run_e2e_comparison, _flip_to_head_sea
from hydrofoil.wavesplat_bridge import h5_gt_to_components, wavesplat_to_components
from wavesplat.data_loader import EpisodeDataLoader
from wavesplat.optimizer.sequential_update import SequentialUpdater
from wavesplat.run import DEFAULT_CONFIG

# ── Style setup ──────────────────────────────────────────────────────────────

for _style in ['seaborn-v0_8-whitegrid', 'seaborn-whitegrid']:
    try:
        plt.style.use(_style)
        break
    except OSError:
        continue

FIGURES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figures')
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data')

# ── Episodes ─────────────────────────────────────────────────────────────────

EPISODES = [
    ("BF3 2-comp (simple)", "episode_BF3_2comp_seed0_uniform.h5"),
    ("BF4 5-comp (rough)",  "episode_BF4_5comp_seed0_uniform.h5"),
]

TRANSIENT = 5.0  # skip first 5s


def _save(fig, name):
    os.makedirs(FIGURES_DIR, exist_ok=True)
    for ext in ('png', 'pdf'):
        path = os.path.join(FIGURES_DIR, f'{name}.{ext}')
        fig.savefig(path, dpi=150, bbox_inches='tight')
    print(f"  Saved {name}.png / .pdf")


def _rms(x):
    """RMS of array (standard deviation around zero — matches print_comparison)."""
    return np.std(x)


# ── Run E2E + extract components ─────────────────────────────────────────────

def run_all_episodes():
    """Run E2E comparison for both episodes and extract component info."""
    all_data = {}

    for ep_name, filename in EPISODES:
        h5_path = os.path.join(DATA_DIR, filename)
        if not os.path.exists(h5_path):
            print(f"  SKIP: {filename} not found")
            continue

        print(f"\n{'─' * 70}")
        print(f"  Running E2E: {ep_name}")
        print(f"{'─' * 70}")

        # Run E2E comparison (PD-only, GT FF, WS FF)
        e2e_data = run_e2e_comparison(h5_path, duration=20.0, verbose=True)

        # Also run WaveSplat separately to extract component parameters
        print(f"\n  [Components] Extracting GT and WS component parameters...")

        # GT components
        with h5py.File(h5_path, 'r') as f:
            gt_wc_array = f['config/wave_components'][()]
        gt_components = h5_gt_to_components(gt_wc_array)

        # WaveSplat estimation
        loader = EpisodeDataLoader(h5_path, device='cpu')
        ws_config = DEFAULT_CONFIG.copy()
        ws_config['device'] = 'cpu'

        updater = SequentialUpdater(ws_config)
        init_frames = loader.get_init_window(tp_factor=2.0)
        updater.initialize(
            init_frames,
            x_range=(float(loader.gt_grid_x[0]), float(loader.gt_grid_x[-1])),
            y_range=(float(loader.gt_grid_y[0]), float(loader.gt_grid_y[-1])),
            verbose=False,
        )
        tracking_frames = loader.get_tracking_frames(tp_factor=2.0)
        for frame in tracking_frames:
            updater.update(frame)

        ws_components = wavesplat_to_components(updater.model)

        # Flip to head sea for consistency (same as E2E pipeline)
        gt_components = _flip_to_head_sea(gt_components)
        ws_components = _flip_to_head_sea(ws_components)

        print(f"  GT: {len(gt_components)} components, WS: {len(ws_components)} components")

        all_data[ep_name] = {
            'e2e': e2e_data,
            'gt_components': gt_components,
            'ws_components': ws_components,
            'filename': filename,
        }

    return all_data


# ── Figure 4: Time-domain comparison ─────────────────────────────────────────

def plot_timeseries(all_data):
    """Figure 4: Heave and acceleration time series for 2 episodes."""
    print("\n[Figure 4] Time-domain comparison")

    fig, axes = plt.subplots(2, 2, figsize=(14, 7))

    colors = {
        'PD-only':   '#999999',
        'PD+GT_FF':  '#2ca02c',
        'PD+WS_FF':  '#1f77b4',
    }
    labels = {
        'PD-only':   'PD-only',
        'PD+GT_FF':  'GT FF',
        'PD+WS_FF':  'WaveSplat FF',
    }

    for col_idx, (ep_name, _filename) in enumerate(EPISODES):
        if ep_name not in all_data:
            continue
        data = all_data[ep_name]
        results = data['e2e']['results']

        for row_idx, (quantity, unit, scale, ylabel) in enumerate([
            ('z', 'cm', 100.0, 'Heave [cm]'),
            ('heave_accel', 'm/s²', 1.0, 'Heave acceleration [m/s²]'),
        ]):
            ax = axes[row_idx, col_idx]

            for strat_name in ['PD-only', 'PD+GT_FF', 'PD+WS_FF']:
                r = results[strat_name]
                t = r.time
                mask = t >= TRANSIENT
                t_plot = t[mask]

                if quantity == 'z':
                    y = r.z[mask] * scale
                else:
                    y = r.heave_accel[mask] * scale

                lw = 1.8 if strat_name != 'PD-only' else 1.2
                alpha = 1.0 if strat_name != 'PD-only' else 0.6
                ax.plot(t_plot, y, color=colors[strat_name],
                        linewidth=lw, alpha=alpha,
                        label=labels[strat_name])

                # Compute RMS for annotation
                rms_val = _rms(y)
                # Store for annotation box
                if not hasattr(ax, '_rms_texts'):
                    ax._rms_texts = []
                ax._rms_texts.append(f"{labels[strat_name]}: {rms_val:.2f}")

            # RMS annotation box
            rms_text = '\n'.join(ax._rms_texts)
            props = dict(boxstyle='round,pad=0.3', facecolor='white',
                         alpha=0.85, edgecolor='gray', linewidth=0.5)
            ax.text(0.97, 0.97, rms_text, transform=ax.transAxes,
                    fontsize=8, verticalalignment='top',
                    horizontalalignment='right', bbox=props,
                    family='monospace')

            ax.set_xlabel('Time [s]', fontsize=11)
            ax.set_ylabel(ylabel, fontsize=11)
            ax.set_xlim(TRANSIENT, 20.0)

            if row_idx == 0:
                ax.set_title(ep_name, fontsize=13)

            # Legend only in top-left subplot
            if row_idx == 0 and col_idx == 0:
                ax.legend(fontsize=9, loc='upper left')

    fig.tight_layout()
    _save(fig, 'fig_e2e_timeseries')
    plt.close(fig)


# ── Figure 5: Component accuracy ─────────────────────────────────────────────

def _match_components(gt_comps, ws_comps):
    """Match WS components to closest GT component by omega.

    Returns list of (gt_comp, ws_comp) pairs.
    Unmatched GT components get ws_comp=None.
    """
    gt_omegas = np.array([c.omega for c in gt_comps])
    ws_omegas = np.array([c.omega for c in ws_comps])

    pairs = []
    used_ws = set()

    for gt_c in gt_comps:
        if len(ws_omegas) == 0:
            pairs.append((gt_c, None))
            continue
        diffs = np.abs(ws_omegas - gt_c.omega)
        # Mask already-used
        for idx in used_ws:
            diffs[idx] = np.inf
        best_idx = int(np.argmin(diffs))
        if diffs[best_idx] < 1.0:  # reasonable match threshold
            pairs.append((gt_c, ws_comps[best_idx]))
            used_ws.add(best_idx)
        else:
            pairs.append((gt_c, None))

    # Unmatched WS components
    for i, ws_c in enumerate(ws_comps):
        if i not in used_ws:
            pairs.append((None, ws_c))

    return pairs


def plot_component_accuracy(all_data):
    """Figure 5: GT vs WaveSplat component comparison in omega-amplitude space."""
    print("\n[Figure 5] Component accuracy")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for col_idx, (ep_name, _filename) in enumerate(EPISODES):
        if ep_name not in all_data:
            continue
        data = all_data[ep_name]
        gt_comps = data['gt_components']
        ws_comps = data['ws_components']
        ax = axes[col_idx]

        pairs = _match_components(gt_comps, ws_comps)

        # Colormap based on direction
        all_dirs = []
        for gt_c, ws_c in pairs:
            if gt_c is not None:
                all_dirs.append(np.degrees(np.arctan2(gt_c.ky, gt_c.kx)))
            if ws_c is not None:
                all_dirs.append(np.degrees(np.arctan2(ws_c.ky, ws_c.kx)))

        if len(all_dirs) > 0:
            vmin = min(all_dirs) - 10
            vmax = max(all_dirs) + 10
        else:
            vmin, vmax = -180, 180

        cmap = plt.cm.coolwarm
        norm = plt.Normalize(vmin=vmin, vmax=vmax)

        # Plot GT components (filled circles)
        for gt_c, ws_c in pairs:
            if gt_c is not None:
                dir_deg = np.degrees(np.arctan2(gt_c.ky, gt_c.kx))
                color = cmap(norm(dir_deg))
                ax.scatter(gt_c.omega, abs(gt_c.A), c=[color],
                           s=80, marker='o', edgecolors='black',
                           linewidth=0.8, zorder=5)
                # kx sign label
                kx_sign = '+' if gt_c.kx >= 0 else '-'
                ax.annotate(kx_sign, (gt_c.omega, abs(gt_c.A)),
                            textcoords="offset points", xytext=(6, 6),
                            fontsize=8, fontweight='bold', color='black')

        # Plot WS components (open squares)
        for gt_c, ws_c in pairs:
            if ws_c is not None:
                dir_deg = np.degrees(np.arctan2(ws_c.ky, ws_c.kx))
                color = cmap(norm(dir_deg))
                ax.scatter(ws_c.omega, abs(ws_c.A), c=[color],
                           s=80, marker='s', facecolors='none',
                           edgecolors=color, linewidth=2.0, zorder=5)
                # kx sign label
                kx_sign = '+' if ws_c.kx >= 0 else '-'
                ax.annotate(kx_sign, (ws_c.omega, abs(ws_c.A)),
                            textcoords="offset points", xytext=(6, -10),
                            fontsize=8, fontweight='bold', color='gray')

        # Connect matched pairs with dashed lines
        for gt_c, ws_c in pairs:
            if gt_c is not None and ws_c is not None:
                ax.plot([gt_c.omega, ws_c.omega],
                        [abs(gt_c.A), abs(ws_c.A)],
                        '--', color='gray', linewidth=0.8, alpha=0.7,
                        zorder=3)

        ax.set_xlabel(r'$\omega$ [rad/s]', fontsize=11)
        ax.set_ylabel('|A| [m]', fontsize=11)
        ax.set_title(ep_name, fontsize=13)
        ax.set_ylim(bottom=0)

        # Legend with proxy artists
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor='gray',
                   markersize=9, markeredgecolor='black', label='GT'),
            Line2D([0], [0], marker='s', color='w', markerfacecolor='none',
                   markersize=9, markeredgecolor='gray',
                   markeredgewidth=2, label='WaveSplat'),
            Line2D([0], [0], linestyle='--', color='gray',
                   linewidth=0.8, label='Matched pair'),
        ]
        ax.legend(handles=legend_elements, fontsize=9, loc='upper right')

    # Add colorbar for direction
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, shrink=0.8, pad=0.02)
    cbar.set_label('Direction atan2(ky, kx) [deg]', fontsize=10)

    fig.tight_layout()
    _save(fig, 'fig_component_accuracy')
    plt.close(fig)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    print()
    print("=" * 70)
    print("  Generating E2E figures (Phase 3)")
    print("=" * 70)

    # Run all episodes
    all_data = run_all_episodes()

    if len(all_data) == 0:
        print("  No episodes found. Exiting.")
        return

    # Figure 4: Time-domain comparison
    plot_timeseries(all_data)

    # Figure 5: Component accuracy
    plot_component_accuracy(all_data)

    print()
    print("=" * 70)
    print(f"  All E2E figures saved to {FIGURES_DIR}")
    print("=" * 70)


if __name__ == '__main__':
    main()
