"""
generate_simple_qualitative_figure.py

GT vs WaveSplat 의 단순 case (에러 작은 case) figure 생성.
- 2D heatmap (위 row): GT, WaveSplat, Error
- 3D surface (아래 row): GT, WaveSplat
출력:
- docs/fig_demo_simple_2d3d.png (단일 case, S13d 용)
- docs/fig_appendix_c_cases_2d3d.png (여러 단순 case, Appendix C 용)
"""
import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from wavesplat.data_loader import EpisodeDataLoader
from wavesplat.run import run_estimation, set_seed
from wavesplat.optimizer.sequential_update import SequentialUpdater

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(SCRIPT_DIR, '..')
DATA_DIR = os.path.join(ROOT, 'data')
DOC_DIR = os.path.join(ROOT, 'docs')

# Simple cases — 에러 작은 (단순) wave 들
# S13d 용 single case: 가장 단순 (BF2 N=1)
# Appendix C 용 multi case: 단순 case 4 개
SINGLE_CASE = {'episode': 'episode_BF2_1comp_seed0_uniform.h5',
               'label': 'BF2 / N=1 / uniform', 'frame_idx': 50}

SIMPLE_CASES = [
    {'episode': 'episode_BF2_1comp_seed0_uniform.h5', 'label': 'BF2 / N=1', 'frame_idx': 50},
    {'episode': 'episode_BF3_1comp_seed0_uniform.h5', 'label': 'BF3 / N=1', 'frame_idx': 50},
    {'episode': 'episode_BF3_2comp_seed0_uniform.h5', 'label': 'BF3 / N=2', 'frame_idx': 50},
    {'episode': 'episode_BF3_5comp_seed0_uniform.h5', 'label': 'BF3 / N=5', 'frame_idx': 50},
]


def get_ws_prediction(h5_path, target_frame_idx):
    """Run WaveSplat and return (pred, gt) at target frame."""
    set_seed(42)
    config = {
        'init_method': 'v2_konly',
        'use_dispersion': True,
        'use_confidence': True,
        'adaptive_n': True,
        'tracking_window': 3,
        'seed': 42,
    }
    run_estimation(h5_path, config=config)  # warmup
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
    gt = tracking_frames[frame_idx].gt_field.cpu().numpy()
    return pred.cpu().numpy(), gt, loader.gt_grid_x.cpu().numpy(), loader.gt_grid_y.cpu().numpy()


def plot_2d3d_single(case, out_path, figsize=(15, 8)):
    """Single case: 2D row (GT, WS, Err) + 3D row (GT, WS) — 5 panels total."""
    h5_path = os.path.join(DATA_DIR, case['episode'])
    if not os.path.exists(h5_path):
        print(f"SKIP: {case['episode']}")
        return None

    print(f"Processing {case['label']}...")
    pred, gt, gx, gy = get_ws_prediction(h5_path, case['frame_idx'])
    err = gt - pred
    nrmse_val = np.sqrt(np.mean(err**2)) / np.std(gt)

    Xg, Yg = np.meshgrid(gx, gy)
    vmax = max(np.abs(gt).max(), np.abs(pred).max())
    vmin = -vmax
    err_max = np.abs(err).max()
    err_norm = TwoSlopeNorm(vmin=-err_max, vcenter=0, vmax=err_max)

    fig = plt.figure(figsize=figsize)
    fig.suptitle(f"{case['label']}  ·  nRMSE = {nrmse_val:.3f}",
                 fontsize=14, fontweight='bold', y=0.98)

    # Row 1: 2D heatmaps
    ax1 = fig.add_subplot(2, 3, 1)
    im1 = ax1.pcolormesh(Xg, Yg, gt, cmap='RdBu_r', vmin=vmin, vmax=vmax,
                         shading='auto')
    ax1.set_title('Ground Truth (2D)', fontsize=11, fontweight='bold')
    ax1.set_aspect('equal')
    plt.colorbar(im1, ax=ax1, fraction=0.046, label='η [m]')

    ax2 = fig.add_subplot(2, 3, 2)
    im2 = ax2.pcolormesh(Xg, Yg, pred, cmap='RdBu_r', vmin=vmin, vmax=vmax,
                         shading='auto')
    ax2.set_title('WaveSplat (2D)', fontsize=11, fontweight='bold')
    ax2.set_aspect('equal')
    plt.colorbar(im2, ax=ax2, fraction=0.046, label='η [m]')

    ax3 = fig.add_subplot(2, 3, 3)
    im3 = ax3.pcolormesh(Xg, Yg, err, cmap='RdBu_r', norm=err_norm,
                         shading='auto')
    ax3.set_title(f'Error (GT − WaveSplat)', fontsize=11, fontweight='bold')
    ax3.set_aspect('equal')
    plt.colorbar(im3, ax=ax3, fraction=0.046, label='Δη [m]')

    # Row 2: 3D surfaces
    ax4 = fig.add_subplot(2, 3, 4, projection='3d')
    surf4 = ax4.plot_surface(Xg, Yg, gt, cmap='RdBu_r', vmin=vmin, vmax=vmax,
                              linewidth=0, antialiased=True, alpha=0.95)
    ax4.set_title('Ground Truth (3D)', fontsize=11, fontweight='bold')
    ax4.set_xlabel('x [m]')
    ax4.set_ylabel('y [m]')
    ax4.set_zlabel('η [m]')
    ax4.set_zlim(vmin, vmax)
    ax4.view_init(elev=25, azim=-60)

    ax5 = fig.add_subplot(2, 3, 5, projection='3d')
    surf5 = ax5.plot_surface(Xg, Yg, pred, cmap='RdBu_r', vmin=vmin, vmax=vmax,
                              linewidth=0, antialiased=True, alpha=0.95)
    ax5.set_title('WaveSplat (3D)', fontsize=11, fontweight='bold')
    ax5.set_xlabel('x [m]')
    ax5.set_ylabel('y [m]')
    ax5.set_zlabel('η [m]')
    ax5.set_zlim(vmin, vmax)
    ax5.view_init(elev=25, azim=-60)

    # 6th cell: empty
    ax6 = fig.add_subplot(2, 3, 6)
    ax6.axis('off')

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"saved: {out_path}")
    return nrmse_val


def plot_2d_multicase(cases, out_path, figsize=(13, 10)):
    """Multiple cases stacked vertically: each case = 2D row only (3 panels: GT, WS, Err)."""
    n_cases = len(cases)
    fig = plt.figure(figsize=figsize)

    nrmse_list = []
    for ci, case in enumerate(cases):
        h5_path = os.path.join(DATA_DIR, case['episode'])
        if not os.path.exists(h5_path):
            print(f"SKIP: {case['episode']}")
            continue
        print(f"Processing {case['label']}...")
        pred, gt, gx, gy = get_ws_prediction(h5_path, case['frame_idx'])
        err = gt - pred
        nrmse_val = np.sqrt(np.mean(err**2)) / np.std(gt)
        nrmse_list.append(nrmse_val)

        Xg, Yg = np.meshgrid(gx, gy)
        vmax = max(np.abs(gt).max(), np.abs(pred).max())
        vmin = -vmax
        err_max = np.abs(err).max()
        err_norm = TwoSlopeNorm(vmin=-err_max, vcenter=0, vmax=err_max)

        # Each case occupies 1 row of 3 panels (GT, WS, Err — 2D only)
        base_idx = ci * 3

        ax1 = fig.add_subplot(n_cases, 3, base_idx + 1)
        ax1.pcolormesh(Xg, Yg, gt, cmap='RdBu_r', vmin=vmin, vmax=vmax, shading='auto')
        if ci == 0:
            ax1.set_title('Ground Truth', fontsize=12, fontweight='bold')
        ax1.set_ylabel(f"{case['label']}\nnRMSE={nrmse_val:.3f}",
                       fontsize=11, fontweight='bold')
        ax1.set_aspect('equal')
        ax1.set_xticks([])
        ax1.set_yticks([])

        ax2 = fig.add_subplot(n_cases, 3, base_idx + 2)
        ax2.pcolormesh(Xg, Yg, pred, cmap='RdBu_r', vmin=vmin, vmax=vmax, shading='auto')
        if ci == 0:
            ax2.set_title('WaveSplat', fontsize=12, fontweight='bold')
        ax2.set_aspect('equal')
        ax2.set_xticks([])
        ax2.set_yticks([])

        ax3 = fig.add_subplot(n_cases, 3, base_idx + 3)
        ax3.pcolormesh(Xg, Yg, err, cmap='RdBu_r', norm=err_norm, shading='auto')
        if ci == 0:
            ax3.set_title('Error (GT - WS)', fontsize=12, fontweight='bold')
        ax3.set_aspect('equal')
        ax3.set_xticks([])
        ax3.set_yticks([])

    fig.suptitle("GT vs WaveSplat - simple cases (low nRMSE)",
                 fontsize=13, fontweight='bold', y=0.995)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"saved: {out_path}")
    return nrmse_list


def main():
    # Single simple case for S13d (BF2 N=1, smallest nRMSE)
    out_single = os.path.join(DOC_DIR, 'fig_demo_simple_2d3d.png')
    plot_2d3d_single(SINGLE_CASE, out_single)

    # Multi-case 2D-only for Appendix C
    out_multi = os.path.join(DOC_DIR, 'fig_appendix_c_cases_2d3d.png')
    plot_2d_multicase(SIMPLE_CASES, out_multi)

    print("Done.")


if __name__ == '__main__':
    main()
