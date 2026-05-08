"""
Generate figures for lab meeting 2026-05-13 PPT.
Output: docs/ directory (PNG files for embedding in PPT).

2-way comparison only: WaveSplat vs Wang2025 (ETKF-orig removed).
"""
import json
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(SCRIPT_DIR, '..')
DOC_DIR = os.path.join(ROOT, 'docs')
RESULT_DIR = os.path.join(ROOT, 'results')

# Colors matching PPT theme
C_WS = '#007ACC'      # WaveSplat (accent blue)
C_WANG = '#E8553A'     # Wang2025 (accent orange-red)
C_BG = '#F0F4F8'

plt.rcParams.update({
    'font.size': 13,
    'font.family': 'sans-serif',
    'axes.facecolor': C_BG,
    'figure.facecolor': 'white',
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.color': '#AAAAAA',
})


def load_wang2025_data():
    path = os.path.join(RESULT_DIR, 'wang2025_comparison.json')
    with open(path) as f:
        data = json.load(f)
    return data


def parse_complexity(ep_name):
    """Extract number of components from episode name."""
    parts = ep_name.split('_')
    for p in parts:
        if p.endswith('comp'):
            return int(p.replace('comp', ''))
    return 0


def fig1_bar_chart_2way(data):
    """Bar chart: average nRMSE for 2 methods + speed annotation."""
    methods = ['WaveSplat', 'ETKF-Wang2025 (Ne=300, fuzzy)']
    labels = ['WaveSplat\n(Ours)', 'Wang2025\n(Ne=300, ETKF)']
    colors = [C_WS, C_WANG]

    avg_nrmse = []
    avg_ms = []
    for m in methods:
        vals = [ep[m]['avg_nRMSE'] for ep in data.values()]
        ms_vals = [ep[m]['avg_inference_ms'] for ep in data.values()]
        avg_nrmse.append(np.mean(vals))
        avg_ms.append(np.mean(ms_vals))

    fig, ax = plt.subplots(figsize=(7, 5))
    x = np.arange(len(methods))
    bars = ax.bar(x, avg_nrmse, width=0.5, color=colors, edgecolor='white',
                  linewidth=1.5, zorder=3)

    for i, (bar, nrmse, ms) in enumerate(zip(bars, avg_nrmse, avg_ms)):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{nrmse:.3f}', ha='center', va='bottom', fontsize=16,
                fontweight='bold', color=colors[i])
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height()/2,
                f'{ms:.0f} ms/frame', ha='center', va='center', fontsize=13,
                color='white', fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=14)
    ax.set_ylabel('Average nRMSE (lower is better)', fontsize=13)
    ax.set_title('12-Episode Comparison: WaveSplat vs Wang2025',
                 fontsize=16, fontweight='bold', pad=15)
    ax.set_ylim(0, 0.40)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    out = os.path.join(DOC_DIR, 'fig_3way_bar.png')
    fig.savefig(out, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'saved: {out}')


def fig2_scatter_complexity(data):
    """Improved scatter: complexity vs nRMSE, WaveSplat vs Wang2025 only.
    Connected lines, clear labels, bigger markers, zone annotations.
    """
    episodes = sorted(data.keys(), key=lambda e: (parse_complexity(e), e))

    ws_nrmse = []
    wang_nrmse = []
    n_comps = []
    ep_labels = []

    for ep in episodes:
        nc = parse_complexity(ep)
        n_comps.append(nc)
        ws_nrmse.append(data[ep]['WaveSplat']['avg_nRMSE'])
        wang_nrmse.append(data[ep]['ETKF-Wang2025 (Ne=300, fuzzy)']['avg_nRMSE'])
        parts = ep.split('_')
        label = f"{parts[0]}_{parts[1]}"
        if 'slope' in ep:
            label += '_s'
        ep_labels.append(label)

    n_comps = np.array(n_comps)
    ws_nrmse = np.array(ws_nrmse)
    wang_nrmse = np.array(wang_nrmse)

    fig, ax = plt.subplots(figsize=(11, 5.5))

    # Background zones
    ax.axvspan(0.5, 12, alpha=0.08, color='blue', zorder=1)
    ax.axvspan(12, 33, alpha=0.08, color='red', zorder=1)

    # Trend lines (mean per complexity)
    unique_comps = sorted(set(n_comps))
    ws_means = [np.mean(ws_nrmse[n_comps == nc]) for nc in unique_comps]
    wang_means = [np.mean(wang_nrmse[n_comps == nc]) for nc in unique_comps]
    ax.plot(unique_comps, ws_means, '-', color=C_WS, alpha=0.4, linewidth=2.5, zorder=3)
    ax.plot(unique_comps, wang_means, '-', color=C_WANG, alpha=0.4, linewidth=2.5, zorder=3)

    # Scatter points
    ax.scatter(n_comps, ws_nrmse, c=C_WS, s=120, zorder=5,
               label='WaveSplat (Ours)', edgecolors='white', linewidth=1.0)
    ax.scatter(n_comps, wang_nrmse, c=C_WANG, s=120, zorder=5,
               label='Wang2025 (ETKF)', marker='s', edgecolors='white', linewidth=1.0)

    # Zone labels
    ax.text(5, 0.47, 'WaveSplat\ndominates', fontsize=13,
            color=C_WS, ha='center', style='italic', fontweight='bold', alpha=0.8)
    ax.text(22, 0.47, 'Wang2025\nwins', fontsize=13,
            color='#CC3333', ha='center', style='italic', fontweight='bold', alpha=0.8)

    ax.set_xlabel('Number of Wave Components (complexity)', fontsize=14)
    ax.set_ylabel('nRMSE (lower is better)', fontsize=14)
    ax.set_title('Episode-Level: Complexity vs Accuracy', fontsize=16,
                 fontweight='bold', pad=12)
    ax.legend(loc='upper left', fontsize=12, framealpha=0.9)
    ax.set_xlim(0, 33)
    ax.set_ylim(0, 0.52)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    out = os.path.join(DOC_DIR, 'fig_complexity_scatter.png')
    fig.savefig(out, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'saved: {out}')


def fig3_ablation_bar():
    """Bar chart for 4-way ablation results with clear labels."""
    variants = [
        'A\n(FFT init,\nfree omega,\n1-frame)',
        'B\n(Beamform init,\nfree omega,\n3-frame)',
        'C\n(Beamform init,\nhard dispersion,\n1-frame)',
        'D (WaveSplat)\n(Beamform init,\nhard dispersion,\n3-frame)',
    ]
    nrmse = [0.213, 0.208, 0.226, 0.211]
    colors_ab = ['#999999', '#999999', '#999999', C_WS]

    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(variants))
    bars = ax.bar(x, nrmse, width=0.55, color=colors_ab, edgecolor='white',
                  linewidth=1.5, zorder=3)

    for bar, val, col in zip(bars, nrmse, colors_ab):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
                f'{val:.3f}', ha='center', va='bottom', fontsize=14,
                fontweight='bold', color=col if col != '#999999' else '#333333')

    ax.set_xticks(x)
    ax.set_xticklabels(variants, fontsize=10)
    ax.set_ylabel('Average nRMSE', fontsize=13)
    ax.set_title('4-Way Ablation Study (12 Episodes)', fontsize=15,
                 fontweight='bold', pad=12)
    ax.set_ylim(0.15, 0.26)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    out = os.path.join(DOC_DIR, 'fig_ablation_bar.png')
    fig.savefig(out, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'saved: {out}')


def fig4_speed_comparison():
    """Horizontal bar chart showing speed comparison (log scale). No ETKF-orig."""
    methods = ['WaveSplat\n(Ours)', 'Wang2025\n(Ne=300, ETKF)',
               'FNO\n(Ehlers 2023)', 'PFT-PINN\n(Ehlers 2025)']
    times_ms = [13.4, 632, 1.9, 30*60*1000]
    colors = [C_WS, C_WANG, '#2CA02C', '#D4A017']
    realtime_thresh = 100

    fig, ax = plt.subplots(figsize=(9, 3.5))
    y = np.arange(len(methods))
    bars = ax.barh(y, times_ms, height=0.55, color=colors, edgecolor='white',
                   linewidth=1.5, zorder=3)

    ax.set_xscale('log')
    ax.set_xlim(0.5, 5e6)

    ax.axvline(x=realtime_thresh, color='#CC3333', linestyle='--', linewidth=1.5,
               zorder=4, alpha=0.7)
    ax.text(realtime_thresh * 1.3, len(methods) - 0.3, 'Real-time\nthreshold\n(100 ms)',
            fontsize=9, color='#CC3333', va='top')

    for bar, t in zip(bars, times_ms):
        if t < 1000:
            label = f'{t:.1f} ms'
        elif t < 60000:
            label = f'{t/1000:.1f} s'
        else:
            label = f'{t/60000:.0f} min'
        ax.text(max(t * 1.3, 2), bar.get_y() + bar.get_height()/2,
                label, va='center', fontsize=12, fontweight='bold')

    ax.set_yticks(y)
    ax.set_yticklabels(methods, fontsize=11)
    ax.set_xlabel('Inference Time per Frame (log scale)', fontsize=12)
    ax.set_title('Speed Comparison Across Methods', fontsize=14, fontweight='bold',
                 pad=12)
    ax.invert_yaxis()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    out = os.path.join(DOC_DIR, 'fig_speed_comparison.png')
    fig.savefig(out, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'saved: {out}')


if __name__ == '__main__':
    os.makedirs(DOC_DIR, exist_ok=True)
    data = load_wang2025_data()
    fig1_bar_chart_2way(data)
    fig2_scatter_complexity(data)
    fig3_ablation_bar()
    fig4_speed_comparison()
    print('\nAll figures generated.')
