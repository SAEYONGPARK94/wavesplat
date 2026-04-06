"""
plot_results.py
Visualization for WaveSplat evaluation results.

Usage:
    python -m wavesplat.visualization.plot_results --results results/eval_batch.json --output figures/
    python -m wavesplat.visualization.plot_results --results results/eval_single.json --output figures/
"""

import argparse
import json
import os
import numpy as np

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


def plot_comparison_table(results: dict, output_dir: str):
    """
    Generate bar chart comparing methods across metrics.

    Args:
        results: dict of {method_name: {avg_RMSE_z, avg_nRMSE, avg_inference_ms, ...}}
        output_dir: directory for saved figures
    """
    methods = list(results.keys())
    rmse_vals = [results[m].get('avg_RMSE_z', 0) for m in methods]
    nrmse_vals = [results[m].get('avg_nRMSE', 0) for m in methods]
    ms_vals = [results[m].get('avg_inference_ms', 0) for m in methods]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # RMSE
    colors = ['#2196F3' if 'WaveSplat' in m and 'oracle' not in m.lower()
              else '#4CAF50' if 'oracle' in m.lower()
              else '#FF9800' for m in methods]

    axes[0].barh(methods, rmse_vals, color=colors)
    axes[0].set_xlabel('RMSE_z [m]')
    axes[0].set_title('Wave Elevation RMSE')
    axes[0].invert_yaxis()

    # nRMSE
    axes[1].barh(methods, nrmse_vals, color=colors)
    axes[1].set_xlabel('nRMSE (RMSE/Hs)')
    axes[1].set_title('Normalized RMSE')
    axes[1].invert_yaxis()

    # Inference time
    axes[2].barh(methods, ms_vals, color=colors)
    axes[2].set_xlabel('ms/frame')
    axes[2].set_title('Inference Time')
    axes[2].invert_yaxis()

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'comparison_bar.png'), dpi=150)
    plt.close()
    print(f"  Saved comparison_bar.png")


def plot_convergence_curve(frame_results: list, method_name: str,
                           output_dir: str):
    """
    Plot nRMSE over time for a single method run.

    Args:
        frame_results: list of per-frame metric dicts with 't' and 'nRMSE'
        method_name: e.g. 'WaveSplat'
        output_dir: directory for saved figures
    """
    times = [r['t'] for r in frame_results]
    nrmse_vals = [r['nRMSE'] for r in frame_results]

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(times, nrmse_vals, 'b-', linewidth=1.5, label=method_name)
    ax.axhline(y=0.05, color='r', linestyle='--', alpha=0.7, label='Target (5%)')
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('nRMSE')
    ax.set_title(f'{method_name} — Convergence')
    ax.legend()
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    safe_name = method_name.replace(' ', '_').replace('(', '').replace(')', '')
    plt.savefig(os.path.join(output_dir, f'convergence_{safe_name}.png'), dpi=150)
    plt.close()
    print(f"  Saved convergence_{safe_name}.png")


def plot_spatial_rmse(frame_results: list, output_dir: str):
    """
    Plot RMSE_near vs RMSE_far over time.
    """
    times = [r['t'] for r in frame_results if 'RMSE_near' in r]
    rmse_near = [r['RMSE_near'] for r in frame_results if 'RMSE_near' in r]
    rmse_far = [r['RMSE_far'] for r in frame_results if 'RMSE_near' in r]

    if not times:
        return

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(times, rmse_near, 'g-', linewidth=1.5, label='Near (<80m)')
    ax.plot(times, rmse_far, 'r-', linewidth=1.5, label='Far (≥80m)')
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('RMSE [m]')
    ax.set_title('Spatial RMSE: Near vs Far')
    ax.legend()
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'spatial_rmse.png'), dpi=150)
    plt.close()
    print(f"  Saved spatial_rmse.png")


def plot_batch_summary(all_results: dict, output_dir: str):
    """
    Heatmap-style summary for batch evaluation across episodes.

    Args:
        all_results: {episode_name: {method_name: {metrics...}}}
    """
    episodes = sorted(all_results.keys())
    episodes_clean = [e.replace('episode_', '').replace('.h5', '') for e in episodes]

    # Collect all methods
    all_methods = set()
    for ep_results in all_results.values():
        if isinstance(ep_results, dict) and 'error' not in ep_results:
            all_methods.update(ep_results.keys())
    methods = sorted(all_methods)

    if not methods:
        return

    # Build RMSE matrix
    matrix = np.full((len(episodes), len(methods)), np.nan)
    for i, ep in enumerate(episodes):
        if 'error' in all_results[ep]:
            continue
        for j, m in enumerate(methods):
            if m in all_results[ep]:
                matrix[i, j] = all_results[ep][m].get('avg_RMSE_z', np.nan)

    fig, ax = plt.subplots(figsize=(max(8, len(methods) * 2), max(6, len(episodes) * 0.4)))
    im = ax.imshow(matrix, aspect='auto', cmap='RdYlGn_r')
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels(methods, rotation=45, ha='right')
    ax.set_yticks(range(len(episodes)))
    ax.set_yticklabels(episodes_clean, fontsize=7)
    ax.set_title('RMSE_z [m] — Lower is Better')

    # Annotate cells
    for i in range(len(episodes)):
        for j in range(len(methods)):
            val = matrix[i, j]
            if not np.isnan(val):
                ax.text(j, i, f'{val:.3f}', ha='center', va='center', fontsize=6)

    plt.colorbar(im, ax=ax, shrink=0.8)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'batch_heatmap.png'), dpi=150)
    plt.close()
    print(f"  Saved batch_heatmap.png")


def generate_text_table(results: dict) -> str:
    """Generate ASCII comparison table."""
    lines = []
    lines.append(f"{'Method':<25} {'RMSE_z [m]':>12} {'nRMSE':>10} {'ms/frame':>10}")
    lines.append('-' * 57)
    for name, r in results.items():
        rmse_z = r.get('avg_RMSE_z', float('nan'))
        nrmse_v = r.get('avg_nRMSE', float('nan'))
        ms = r.get('avg_inference_ms', float('nan'))
        lines.append(f"{name:<25} {rmse_z:>12.4f} {nrmse_v:>10.4f} {ms:>10.1f}")
    return '\n'.join(lines)


def main():
    if not HAS_MPL:
        print("matplotlib not installed. Install with: pip install matplotlib")
        print("Generating text tables only.")

    parser = argparse.ArgumentParser(description='WaveSplat result visualization')
    parser.add_argument('--results', type=str, required=True,
                        help='Path to eval JSON (single or batch)')
    parser.add_argument('--output', type=str, default='figures/',
                        help='Output directory for figures')
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    with open(args.results) as f:
        data = json.load(f)

    # Detect single vs batch
    # Single: {method: {metrics}}, Batch: {episode: {method: {metrics}}}
    first_val = next(iter(data.values()))
    is_batch = isinstance(first_val, dict) and any(
        isinstance(v, dict) and 'avg_RMSE_z' in v
        for v in first_val.values()
    ) if isinstance(first_val, dict) else False

    if is_batch:
        print("Batch evaluation results:")
        # Print per-episode tables
        for ep_name, ep_results in data.items():
            if 'error' in ep_results:
                print(f"\n{ep_name}: FAILED — {ep_results['error']}")
                continue
            print(f"\n{ep_name}:")
            print(generate_text_table(ep_results))

        if HAS_MPL:
            plot_batch_summary(data, args.output)

    else:
        # Single episode
        print("Single episode results:")
        print(generate_text_table(data))

        if HAS_MPL:
            plot_comparison_table(data, args.output)

    # Save text table
    table_path = os.path.join(args.output, 'results_table.txt')
    with open(table_path, 'w') as f:
        if is_batch:
            for ep_name, ep_results in data.items():
                if 'error' in ep_results:
                    f.write(f"\n{ep_name}: FAILED\n")
                    continue
                f.write(f"\n{ep_name}:\n")
                f.write(generate_text_table(ep_results))
                f.write('\n')
        else:
            f.write(generate_text_table(data))
    print(f"\nText table saved to {table_path}")


if __name__ == '__main__':
    main()
