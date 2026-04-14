"""
generate_all_episode_figures.py
Generate GT vs WaveSplat heatmap figures for all 100 episodes.
Each episode produces a 1×2 figure: Ground Truth | WaveSplat prediction.
For 10 representative episodes (seed0_uniform), generates 1×4 comparison
with ETKF and GPR baselines.
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
import matplotlib.gridspec as gridspec

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from wavesplat.data_loader import EpisodeDataLoader
from wavesplat.run import run_estimation, set_seed
from wavesplat.optimizer.sequential_update import SequentialUpdater
from wavesplat.baselines.etkf_grid import ETKFGrid
from wavesplat.baselines.gpr_baseline import GPRBaseline
from wavesplat.evaluation.metrics import nrmse
from scipy.interpolate import RegularGridInterpolator


TARGET_FRAME_IDX = 50

# Representative episodes for 4-column comparison (1 per setting)
REPRESENTATIVE = {
    'episode_BF2_1comp_seed0_uniform',
    'episode_BF3_1comp_seed0_uniform',
    'episode_BF3_2comp_seed0_uniform',
    'episode_BF3_5comp_seed0_uniform',
    'episode_BF4_1comp_seed0_uniform',
    'episode_BF4_5comp_seed0_uniform',
    'episode_BF4_15comp_seed0_uniform',
    'episode_BF4_30comp_seed0_uniform',
    'episode_BF5_15comp_seed0_uniform',
    'episode_BF5_30comp_seed0_uniform',
}


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


def get_etkf_prediction(h5_path, target_frame_idx):
    """Run ETKF and return prediction at target frame."""
    np.random.seed(42)
    loader = EpisodeDataLoader(h5_path)
    gt_x = loader.gt_grid_x.cpu().numpy()
    gt_y = loader.gt_grid_y.cpu().numpy()

    etkf = ETKFGrid(
        np.linspace(gt_x[0], gt_x[-1], 40),
        np.linspace(gt_y[0], gt_y[-1], 30),
        depth=loader.depth, n_ensemble=50)

    for i, frame in enumerate(loader):
        x = frame.x_obs.cpu().numpy()
        y = frame.y_obs.cpu().numpy()
        z = frame.z_obs.cpu().numpy()
        etkf.update(x, y, z)

        if i == target_frame_idx:
            etkf_pred = etkf.predict_grid()
            interp = RegularGridInterpolator(
                (np.linspace(gt_y[0], gt_y[-1], 30),
                 np.linspace(gt_x[0], gt_x[-1], 40)),
                etkf_pred, method='linear', bounds_error=False, fill_value=0.0)
            Xg, Yg = np.meshgrid(gt_x, gt_y)
            pred = interp(np.column_stack([Yg.ravel(), Xg.ravel()])).reshape(
                len(gt_y), len(gt_x))
            return pred
    return None


def get_gpr_prediction(h5_path, target_frame_idx):
    """Run GPR and return prediction at target frame."""
    loader = EpisodeDataLoader(h5_path)
    gt_x = loader.gt_grid_x.cpu().numpy()
    gt_y = loader.gt_grid_y.cpu().numpy()

    gpr = GPRBaseline(kernel_type='matern', length_scale=20.0,
                      noise_sigma=0.02, window_frames=1)

    for i, frame in enumerate(loader):
        x = frame.x_obs.cpu().numpy()
        y = frame.y_obs.cpu().numpy()
        z = frame.z_obs.cpu().numpy()

        gpr.reset()
        gpr.update(x, y, z)

        if i == target_frame_idx:
            pred = gpr.predict_grid(gt_x, gt_y)
            return pred
    return None


def make_2col_figure(gt, ws_pred, Hs, avg_nrmse, label, output_path):
    """Create 1×2 heatmap with colorbar on the far right outside."""
    fig = plt.figure(figsize=(10, 4))
    gs = gridspec.GridSpec(1, 3, width_ratios=[1, 1, 0.05], wspace=0.08)

    vmax = max(abs(gt.min()), abs(gt.max()))
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)

    # GT
    ax0 = fig.add_subplot(gs[0, 0])
    im = ax0.imshow(gt, cmap='RdBu_r', norm=norm, aspect='auto', origin='lower')
    ax0.set_title('Ground Truth', fontsize=12, fontweight='bold')
    ax0.set_xticks([])
    ax0.set_yticks([])

    # WaveSplat
    ax1 = fig.add_subplot(gs[0, 1])
    ax1.imshow(ws_pred, cmap='RdBu_r', norm=norm, aspect='auto', origin='lower')
    frame_nrmse = nrmse(ws_pred, gt, Hs)
    ax1.set_title(f'WaveSplat  (nRMSE={frame_nrmse:.3f}, avg={avg_nrmse:.3f})',
                  fontsize=11, fontweight='bold')
    ax1.set_xticks([])
    ax1.set_yticks([])

    # Colorbar on far right
    cax = fig.add_subplot(gs[0, 2])
    fig.colorbar(im, cax=cax, label='η [m]')

    fig.suptitle(label, fontsize=13, fontweight='bold', y=1.02)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def make_4col_figure(gt, ws_pred, etkf_pred, gpr_pred, Hs, avg_nrmse,
                     label, output_path):
    """Create 1×4 heatmap: GT | WS | ETKF | GPR with colorbar outside."""
    fig = plt.figure(figsize=(18, 4))
    gs = gridspec.GridSpec(1, 5, width_ratios=[1, 1, 1, 1, 0.05], wspace=0.08)

    vmax = max(abs(gt.min()), abs(gt.max()))
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)

    panels = [
        ('Ground Truth', gt, None),
        ('WaveSplat', ws_pred, avg_nrmse),
        ('ETKF', etkf_pred, None),
        ('GPR', gpr_pred, None),
    ]

    for col, (method, data, extra) in enumerate(panels):
        ax = fig.add_subplot(gs[0, col])
        im = ax.imshow(data, cmap='RdBu_r', norm=norm,
                       aspect='auto', origin='lower')
        ax.set_xticks([])
        ax.set_yticks([])

        if col > 0 and data is not None:
            n = nrmse(data, gt, Hs)
            title = f'{method}  (nRMSE={n:.3f})'
            if extra is not None:
                title = f'{method}  (nRMSE={n:.3f}, avg={extra:.3f})'
            ax.set_title(title, fontsize=10, fontweight='bold')
        else:
            ax.set_title(method, fontsize=11, fontweight='bold')

    # Colorbar on far right
    cax = fig.add_subplot(gs[0, 4])
    fig.colorbar(im, cax=cax, label='η [m]')

    fig.suptitle(label, fontsize=13, fontweight='bold', y=1.02)
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
        is_rep = ep_name in REPRESENTATIVE

        if os.path.exists(out_path):
            print(f"[{idx+1}/{total}] SKIP (exists): {ep_name}")
            continue

        tag = " [4-col]" if is_rep else ""
        print(f"[{idx+1}/{total}]{tag} {ep_name} ...", end=" ", flush=True)

        try:
            ws_pred, gt, Hs, avg_nrmse = get_ws_prediction(
                h5_path, TARGET_FRAME_IDX)
            label = ep_name.replace('episode_', '').replace('_', ' ')

            if is_rep:
                # 4-column comparison
                print("ETKF...", end=" ", flush=True)
                etkf_pred = get_etkf_prediction(h5_path, TARGET_FRAME_IDX)
                print("GPR...", end=" ", flush=True)
                gpr_pred = get_gpr_prediction(h5_path, TARGET_FRAME_IDX)
                make_4col_figure(gt, ws_pred, etkf_pred, gpr_pred,
                                 Hs, avg_nrmse, label, out_path)
            else:
                # 2-column GT vs WS
                make_2col_figure(gt, ws_pred, Hs, avg_nrmse, label, out_path)

            print(f"done (avg nRMSE={avg_nrmse:.3f})")
        except Exception as e:
            print(f"FAILED: {e}")

    print(f"\nAll done. Figures in {fig_dir}/")


if __name__ == '__main__':
    main()
