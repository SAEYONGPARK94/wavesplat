"""
generate_paper_figures.py
Generate matplotlib figures for the WaveSplat paper.

Figures:
1. Bar chart: WaveSplat vs ETKF nRMSE per episode
2. Adaptive N: N_true vs N_adapt scatter
3. Ablation: grouped bar chart
"""

import json
import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print("matplotlib not found, generating text-only summaries")

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
PAPER_DIR = os.path.join(os.path.dirname(__file__), '..', 'paper', 'figures')


def load_json(name):
    path = os.path.join(RESULTS_DIR, name)
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def fig1_main_comparison():
    """Bar chart: WaveSplat-P1 vs ETKF nRMSE per episode."""
    data = load_json('p1_comparison.json')
    if data is None or not HAS_MPL:
        return

    episodes = list(data.keys())
    short_names = [ep.replace('_seed0_', '\n').replace('uniform', 'U').replace('slope', 'S')
                   for ep in episodes]

    ws_vals = [data[ep].get('WS-P1', {}).get('avg_nRMSE', 0) for ep in episodes]
    etkf_vals = [data[ep].get('ETKF', {}).get('avg_nRMSE', 0) for ep in episodes]

    x = np.arange(len(episodes))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 5))
    bars1 = ax.bar(x - width/2, ws_vals, width, label='WaveSplat (ours)',
                   color='#2196F3', edgecolor='white')
    bars2 = ax.bar(x + width/2, etkf_vals, width, label='ETKF-grid',
                   color='#FF9800', edgecolor='white', alpha=0.8)

    ax.set_ylabel('nRMSE (lower is better)', fontsize=12)
    ax.set_title('WaveSplat vs ETKF-grid: Per-Episode nRMSE', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(short_names, fontsize=8, ha='center')
    ax.legend(fontsize=11)
    ax.set_ylim(0, 0.6)
    ax.axhline(y=np.mean(ws_vals), color='#2196F3', linestyle='--', alpha=0.5,
               label=f'WS avg={np.mean(ws_vals):.3f}')
    ax.axhline(y=np.mean(etkf_vals), color='#FF9800', linestyle='--', alpha=0.5,
               label=f'ETKF avg={np.mean(etkf_vals):.3f}')
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    out_path = os.path.join(PAPER_DIR, 'fig1_comparison.pdf')
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {out_path}")


def fig2_adaptive_n():
    """Scatter: N_true vs N_adapt with diagonal reference line."""
    data = load_json('p1_comparison.json')
    if data is None or not HAS_MPL:
        return

    n_true_list = []
    n_adapt_list = []
    nrmse_list = []

    for ep, res in data.items():
        p1 = res.get('WS-P1', {})
        n_adapt = p1.get('final_n', None)
        if n_adapt is None:
            continue

        # Extract N_true from episode name
        parts = ep.split('_')
        n_true = None
        for p in parts:
            if p.endswith('comp'):
                n_true = int(p.replace('comp', ''))
                break
        if n_true is None:
            continue

        n_true_list.append(n_true)
        n_adapt_list.append(n_adapt)
        nrmse_list.append(p1.get('avg_nRMSE', 0))

    fig, ax = plt.subplots(figsize=(6, 6))
    scatter = ax.scatter(n_true_list, n_adapt_list, c=nrmse_list,
                         cmap='RdYlGn_r', s=120, edgecolors='black',
                         linewidths=0.5, vmin=0, vmax=0.4)
    ax.plot([0, 35], [0, 35], 'k--', alpha=0.3, label='$N_{adapt} = N_{true}$')
    ax.set_xlabel('$N_{true}$ (ground-truth components)', fontsize=12)
    ax.set_ylabel('$N_{adapt}$ (adaptive model size)', fontsize=12)
    ax.set_title('Adaptive N vs Ground-Truth Complexity', fontsize=14)
    ax.set_xlim(0, 35)
    ax.set_ylim(0, 25)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)

    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('nRMSE', fontsize=11)

    plt.tight_layout()
    out_path = os.path.join(PAPER_DIR, 'fig2_adaptive_n.pdf')
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {out_path}")


def fig3_speed_comparison():
    """Speed comparison: WaveSplat vs ETKF ms/frame."""
    data = load_json('p1_comparison.json')
    if data is None or not HAS_MPL:
        return

    episodes = list(data.keys())
    short_names = [ep.replace('_seed0_', '\n').replace('uniform', 'U').replace('slope', 'S')
                   for ep in episodes]

    ws_speed = [data[ep].get('WS-P1', {}).get('avg_inference_ms', 0) for ep in episodes]
    etkf_speed = [data[ep].get('ETKF', {}).get('avg_inference_ms', 0) for ep in episodes]

    x = np.arange(len(episodes))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.bar(x - width/2, ws_speed, width, label='WaveSplat',
           color='#2196F3', edgecolor='white')
    ax.bar(x + width/2, etkf_speed, width, label='ETKF-grid',
           color='#FF9800', edgecolor='white', alpha=0.8)

    ax.set_ylabel('Inference time (ms/frame)', fontsize=12)
    ax.set_title('Per-Frame Inference Speed', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(short_names, fontsize=8, ha='center')
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    out_path = os.path.join(PAPER_DIR, 'fig3_speed.pdf')
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {out_path}")


def fig4_ablation():
    """Grouped bar chart for ablation study."""
    data = load_json('ablation_seeded.json')
    if data is None or not HAS_MPL:
        return

    configs = ['Full', 'A1_noDisp', 'A2_noConf']
    config_labels = ['Full model', 'w/o Dispersion', 'w/o Confidence']
    colors = ['#4CAF50', '#FF5722', '#9C27B0']

    episodes = {}
    for entry in data:
        ep = entry['episode']
        cfg = entry['config']
        if cfg not in configs:
            continue
        seeds = entry['seed_results']
        avg = np.mean([s['avg_nRMSE'] for s in seeds])
        std = np.std([s['avg_nRMSE'] for s in seeds])
        if ep not in episodes:
            episodes[ep] = {}
        episodes[ep][cfg] = (avg, std)

    ep_names = sorted(episodes.keys())
    short_names = [ep.replace('episode_', '').replace('_seed0_', '\n')
                      .replace('.h5', '').replace('uniform', 'U').replace('slope', 'S')
                   for ep in ep_names]

    x = np.arange(len(ep_names))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, (cfg, label, color) in enumerate(zip(configs, config_labels, colors)):
        vals = [episodes[ep].get(cfg, (0, 0))[0] for ep in ep_names]
        errs = [episodes[ep].get(cfg, (0, 0))[1] for ep in ep_names]
        ax.bar(x + (i - 1) * width, vals, width, yerr=errs,
               label=label, color=color, edgecolor='white',
               capsize=3, alpha=0.85)

    ax.set_ylabel('nRMSE (3-seed avg)', fontsize=12)
    ax.set_title('Ablation Study: Contribution of Each Component', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(short_names, fontsize=8, ha='center')
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    out_path = os.path.join(PAPER_DIR, 'fig4_ablation.pdf')
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {out_path}")


def fig5_raydrop():
    """Grouped bar chart: WaveSplat and ETKF under uniform vs slope raydrop."""
    data = load_json('raydrop_comparison.json')
    if data is None or not HAS_MPL:
        return

    pairs = list(data.keys())
    short_names = [p.replace('_', '\n') for p in pairs]

    ws_u = [data[p].get('uniform', {}).get('WS-P1', {}).get('avg_nRMSE', 0) for p in pairs]
    ws_s = [data[p].get('slope', {}).get('WS-P1', {}).get('avg_nRMSE', 0) for p in pairs]
    et_u = [data[p].get('uniform', {}).get('ETKF', {}).get('avg_nRMSE', 0) for p in pairs]
    et_s = [data[p].get('slope', {}).get('ETKF', {}).get('avg_nRMSE', 0) for p in pairs]

    x = np.arange(len(pairs))
    width = 0.2

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.bar(x - 1.5*width, ws_u, width, label='WS Uniform', color='#2196F3', edgecolor='white')
    ax.bar(x - 0.5*width, ws_s, width, label='WS Slope', color='#1565C0', edgecolor='white')
    ax.bar(x + 0.5*width, et_u, width, label='ETKF Uniform', color='#FF9800', edgecolor='white', alpha=0.8)
    ax.bar(x + 1.5*width, et_s, width, label='ETKF Slope', color='#E65100', edgecolor='white', alpha=0.8)

    ax.set_ylabel('nRMSE (lower is better)', fontsize=12)
    ax.set_title('Raydrop Model Comparison: Uniform vs Slope', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(short_names, fontsize=8, ha='center')
    ax.legend(fontsize=9, ncol=2)
    ax.set_ylim(0, 0.6)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    out_path = os.path.join(PAPER_DIR, 'fig5_raydrop.pdf')
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {out_path}")


def main():
    os.makedirs(PAPER_DIR, exist_ok=True)
    print("Generating paper figures...")
    fig1_main_comparison()
    fig2_adaptive_n()
    fig3_speed_comparison()
    fig4_ablation()
    fig5_raydrop()
    print("Done.")


if __name__ == '__main__':
    main()
