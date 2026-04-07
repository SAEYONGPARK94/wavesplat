"""
generate_tracking_animation.py
Generate MP4 animation showing WaveSplat tracking over time.
Left: GT wave field, Right: WaveSplat prediction, with nRMSE overlay.
"""

import sys
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import matplotlib.animation as animation

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from wavesplat.data_loader import EpisodeDataLoader
from wavesplat.optimizer.sequential_update import SequentialUpdater
from wavesplat.run import set_seed
from wavesplat.evaluation.metrics import nrmse


DEFAULT_EPISODE = 'episode_BF4_15comp_seed0_uniform.h5'


def generate_animation(h5_path, output_path, max_frames=150):
    """Generate side-by-side GT vs WaveSplat animation."""
    set_seed(42)
    loader = EpisodeDataLoader(h5_path)

    cfg = {
        'n_gaussians': [3, 8, 15],
        'depth': loader.depth,
        'lr': 0.01,
        'lambda_l1': 0.001,
        'lambda_smooth': 0.0001,
        'tracking_steps': 10,
        'tracking_mode': 'w_phi',
        'tracking_lr': 0.005,
        'init_method': 'v2_konly',
        'use_dispersion': True,
        'use_confidence': True,
        'adaptive_n': True,
        'prune_threshold': 0.10,
        'grow_residual_threshold': 0.05,
        'max_n': 30,
        'tracking_window': 3,
        'device': 'cpu',
        'seed': 42,
    }

    updater = SequentialUpdater(cfg)
    init_frames = loader.get_init_window(tp_factor=2.0)
    updater.initialize(
        init_frames,
        x_range=(float(loader.gt_grid_x[0]), float(loader.gt_grid_x[-1])),
        y_range=(float(loader.gt_grid_y[0]), float(loader.gt_grid_y[-1])),
    )

    grid_x = loader.gt_grid_x
    grid_y = loader.gt_grid_y
    tracking_frames = loader.get_tracking_frames(tp_factor=2.0)
    n_frames = min(len(tracking_frames), max_frames)

    # Collect all frames first
    print(f"Running tracking for {n_frames} frames...")
    gt_fields = []
    ws_fields = []
    nrmses = []
    times = []
    n_gaussians_list = []

    for i in range(n_frames):
        frame = tracking_frames[i]
        updater.update(frame)
        pred = updater.predict_field(grid_x, grid_y, frame.t)

        gt_np = frame.gt_field.cpu().numpy()
        pred_np = pred.cpu().numpy()

        gt_fields.append(gt_np)
        ws_fields.append(pred_np)
        nrmses.append(nrmse(pred_np, gt_np, loader.Hs_true))
        times.append(frame.t)
        n_gaussians_list.append(updater.model.n_gaussians)

        if i % 20 == 0:
            print(f"  Frame {i}/{n_frames}: t={frame.t:.1f}s, "
                  f"nRMSE={nrmses[-1]:.4f}, N={n_gaussians_list[-1]}")

    # Determine global color scale
    vmax = max(max(abs(g.min()), abs(g.max())) for g in gt_fields)
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)

    # Create animation
    print("Creating animation...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    im1 = ax1.imshow(gt_fields[0], cmap='RdBu_r', norm=norm,
                     aspect='auto', origin='lower')
    ax1.set_title('Ground Truth', fontsize=13, fontweight='bold')
    ax1.set_xticks([])
    ax1.set_yticks([])

    im2 = ax2.imshow(ws_fields[0], cmap='RdBu_r', norm=norm,
                     aspect='auto', origin='lower')
    ax2.set_title('WaveSplat', fontsize=13, fontweight='bold')
    ax2.set_xticks([])
    ax2.set_yticks([])

    fig.colorbar(im1, ax=[ax1, ax2], shrink=0.8, label='η [m]')

    info_text = ax2.text(0.02, 0.98, '', transform=ax2.transAxes,
                         fontsize=10, verticalalignment='top',
                         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    time_text = fig.suptitle('', fontsize=12)

    sea_state = os.path.basename(h5_path).replace('episode_', '').replace('.h5', '')
    fig.text(0.5, 0.02, f'Episode: {sea_state}  |  Hs={loader.Hs_true:.1f}m, Tp={loader.Tp_true:.1f}s',
             ha='center', fontsize=10, style='italic')

    def update(frame_idx):
        im1.set_data(gt_fields[frame_idx])
        im2.set_data(ws_fields[frame_idx])
        info_text.set_text(f'nRMSE={nrmses[frame_idx]:.3f}\nN={n_gaussians_list[frame_idx]}')
        time_text.set_text(f't = {times[frame_idx]:.1f}s')
        return im1, im2, info_text, time_text

    ani = animation.FuncAnimation(fig, update, frames=n_frames,
                                  interval=100, blit=False)

    # Save as MP4
    try:
        writer = animation.FFMpegWriter(fps=10, bitrate=2000)
        ani.save(output_path, writer=writer)
        print(f"Saved animation: {output_path}")
    except Exception:
        # Fallback: save as GIF
        gif_path = output_path.replace('.mp4', '.gif')
        ani.save(gif_path, writer='pillow', fps=10)
        print(f"FFmpeg not available. Saved as GIF: {gif_path}")

    plt.close()

    # Also save nRMSE convergence plot
    conv_path = output_path.replace('.mp4', '_convergence.png')
    fig2, ax = plt.subplots(figsize=(8, 4))
    ax.plot(times, nrmses, 'b-', linewidth=1.5)
    ax.axhline(y=np.mean(nrmses), color='r', linestyle='--', alpha=0.5,
               label=f'Mean={np.mean(nrmses):.3f}')
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('nRMSE')
    ax.set_title(f'Convergence: {sea_state}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(conv_path, dpi=150)
    print(f"Saved convergence: {conv_path}")
    plt.close()


def main():
    data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
    fig_dir = os.path.join(os.path.dirname(__file__), '..', 'paper', 'figures')
    os.makedirs(fig_dir, exist_ok=True)

    # Generate for default episode
    h5_path = os.path.join(data_dir, DEFAULT_EPISODE)
    if not os.path.exists(h5_path):
        print(f"Episode not found: {h5_path}")
        # Try any available episode
        import glob
        episodes = sorted(glob.glob(os.path.join(data_dir, '*.h5')))
        if episodes:
            h5_path = episodes[0]
            print(f"Using: {h5_path}")
        else:
            print("No episodes available.")
            return

    output_path = os.path.join(fig_dir, 'tracking_animation.mp4')
    generate_animation(h5_path, output_path)


if __name__ == '__main__':
    main()
