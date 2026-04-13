"""
generate_all_episode_figures.py
Generate GT vs WaveSplat heatmap figures for all 100 episodes.
Each episode produces a 1×2 figure: Ground Truth | WaveSplat prediction.
Used for thesis appendix.
"""

import sys
import os
import glob
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from wavesplat.data_loader import EpisodeDataLoader
from wavesplat.run import run_estimation, set_seed
from wavesplat.optimizer.sequential_update import SequentialUpdater
from wavesplat.evaluation.metrics import nrmse


TARGET_FRAME_IDX = 50  # Use frame 50 for all episodes


def get_ws_prediction(h5_path, target_frame_idx):
    """Run WaveSplat and return prediction at target frame."""
    set_seed(42)
    config = {
        'init_method': 'v2_konly',
        'use_dispersion': True,
        'use_confidence': True,
        'adaptive_n': True,
        'tracking_window': 3,
        'seed': 42,
    }
    result = run_estimation(h5_path, config=config)

    loader = EpisodeDataLoader(h5_path)
    cfg = config.copy()
    cfg['depth'] = loader.depth

    updater = SequentialUpdater(cfg)
    init_frames = loader.get_init_window(tp_factor=2.0)
    updater.initialize(
        init_frames,
        x_range=(float(loader.gt_grid_x[0]), float(loader.gt_grid_x[-1])),
        y_range=(float(loader.gt_grid_y[0]), float(loader.gt_grid_y[-1])),
    )

    tracking_frames = loader.get_tracking_frames(tp_factor=2.0)
    frame_idx = min(target_frame_idx, len(tracking_frames) - 1)

    for i in range(frame_idx + 1):
        updater.update(tracking_frames[i])

    pred = updater.predict_field(
        loader.gt_grid_x, loader.gt_grid_y, tracking_frames[frame_idx].t)

    avg_nrmse = result['avg_nRMSE']
    return (pred.cpu().numpy(),
            tracking_frames[frame_idx].gt_field.cpu().numpy(),
            loader.Hs_true,
            avg_nrmse)


def make_single_figure(gt, ws_pred, Hs, avg_nrmse, label, output_path):
    """Create 1×2 heatmap: GT | WaveSplat."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    vmax = max(abs(gt.min()), abs(gt.max()))
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)

    # GT
    im = axes[0].imshow(gt, cmap='RdBu_r', norm=norm,
                        aspect='auto', origin='lower')
    axes[0].set_title('Ground Truth', fontsize=12, fontweight='bold')
    axes[0].set_xticks([])
    axes[0].set_yticks([])

    # WaveSplat
    axes[1].imshow(ws_pred, cmap='RdBu_r', norm=norm,
                   aspect='auto', origin='lower')
    axes[1].set_title('WaveSplat', fontsize=12, fontweight='bold')
    axes[1].set_xticks([])
    axes[1].set_yticks([])

    # nRMSE annotation
    frame_nrmse = nrmse(ws_pred, gt, Hs)
    axes[1].text(0.02, 0.98, f'nRMSE={frame_nrmse:.3f}\navg={avg_nrmse:.3f}',
                 transform=axes[1].transAxes, fontsize=9,
                 verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    fig.colorbar(im, ax=axes.tolist(), shrink=0.8, label='η [m]')
    fig.suptitle(label, fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def main():
    data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
    fig_dir = os.path.join(os.path.dirname(__file__), '..', 'thesis',
                           'figures_episodes')
    os.makedirs(fig_dir, exist_ok=True)

    episodes = sorted(glob.glob(os.path.join(data_dir, 'episode_*.h5')))
    total = len(episodes)
    print(f"Found {total} episodes. Output → {fig_dir}/")

    for idx, h5_path in enumerate(episodes):
        ep_name = os.path.basename(h5_path).replace('.h5', '')
        out_path = os.path.join(fig_dir, f'{ep_name}.png')

        if os.path.exists(out_path):
            print(f"[{idx+1}/{total}] SKIP (exists): {ep_name}")
            continue

        print(f"[{idx+1}/{total}] {ep_name} ...", end=" ", flush=True)
        try:
            ws_pred, gt, Hs, avg_nrmse = get_ws_prediction(
                h5_path, TARGET_FRAME_IDX)
            label = ep_name.replace('episode_', '').replace('_', ' ')
            make_single_figure(gt, ws_pred, Hs, avg_nrmse, label, out_path)
            print(f"done (avg nRMSE={avg_nrmse:.3f})")
        except Exception as e:
            print(f"FAILED: {e}")

    print(f"\nAll done. Figures in {fig_dir}/")


if __name__ == '__main__':
    main()
