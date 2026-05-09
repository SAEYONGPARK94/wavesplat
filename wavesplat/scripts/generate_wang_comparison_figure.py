"""
generate_wang_comparison_figure.py

GT vs WaveSplat vs Wang2025 비교 figure 생성.
2D heatmap 4-panel: GT | WaveSplat | Wang2025 | (Wave_WS - Wang_WS) error 비교
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
from wavesplat.baselines.etkf_wang2025 import ETKFWang2025
from scipy.interpolate import RegularGridInterpolator

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(SCRIPT_DIR, '..')
DATA_DIR = os.path.join(ROOT, 'data')
DOC_DIR = os.path.join(ROOT, 'docs')


def get_ws_prediction(h5_path, target_frame_idx):
    set_seed(42)
    config = {
        'init_method': 'v2_konly',
        'use_dispersion': True,
        'use_confidence': True,
        'adaptive_n': True,
        'tracking_window': 3,
        'seed': 42,
    }
    loader = EpisodeDataLoader(h5_path)
    cfg = config.copy()
    cfg['depth'] = loader.depth
    updater = SequentialUpdater(cfg)
    init_frames = loader.get_init_window(tp_factor=2.0)
    updater.initialize(init_frames,
                       x_range=(float(loader.gt_grid_x[0]), float(loader.gt_grid_x[-1])),
                       y_range=(float(loader.gt_grid_y[0]), float(loader.gt_grid_y[-1])))
    tracking_frames = loader.get_tracking_frames(tp_factor=2.0)
    frame_idx = min(target_frame_idx, len(tracking_frames) - 1)
    for i in range(frame_idx + 1):
        updater.update(tracking_frames[i])
    pred = updater.predict_field(loader.gt_grid_x, loader.gt_grid_y, tracking_frames[frame_idx].t)
    gt = tracking_frames[frame_idx].gt_field.cpu().numpy()
    return pred.cpu().numpy(), gt, loader.gt_grid_x.cpu().numpy(), loader.gt_grid_y.cpu().numpy()


def get_wang_prediction(h5_path, target_frame_idx):
    np.random.seed(42)
    loader = EpisodeDataLoader(h5_path)
    gt_x = loader.gt_grid_x.cpu().numpy()
    gt_y = loader.gt_grid_y.cpu().numpy()
    nx, ny = 80, 60
    grid_x = np.linspace(gt_x[0], gt_x[-1], nx)
    grid_y = np.linspace(gt_y[0], gt_y[-1], ny)
    wang = ETKFWang2025(grid_x=grid_x, grid_y=grid_y,
                        depth=loader.depth, n_ensemble=300)
    for i, frame in enumerate(loader):
        x = frame.x_obs.cpu().numpy()
        y = frame.y_obs.cpu().numpy()
        z = frame.z_obs.cpu().numpy()
        wang.update(x, y, z)
        if i == target_frame_idx:
            wang_pred = wang.predict_grid()
            interp = RegularGridInterpolator(
                (grid_y, grid_x), wang_pred, method='linear',
                bounds_error=False, fill_value=0.0)
            Xg, Yg = np.meshgrid(gt_x, gt_y)
            pred = interp(np.column_stack([Yg.ravel(), Xg.ravel()])).reshape(
                len(gt_y), len(gt_x))
            return pred, frame.gt_field.cpu().numpy()
    return None, None


def plot_wang_comparison_single(case, out_path, figsize=(15, 4)):
    h5_path = os.path.join(DATA_DIR, case['episode'])
    if not os.path.exists(h5_path):
        print(f"SKIP: {case['episode']}")
        return None

    print(f"Processing {case['label']}...")
    ws_pred, gt, gx, gy = get_ws_prediction(h5_path, case['frame_idx'])
    wang_pred, _ = get_wang_prediction(h5_path, case['frame_idx'])
    if wang_pred is None:
        print(f"  Wang prediction failed for {case['label']}")
        return None

    err_ws = gt - ws_pred
    err_wang = gt - wang_pred
    nrmse_ws = np.sqrt(np.mean(err_ws**2)) / np.std(gt)
    nrmse_wang = np.sqrt(np.mean(err_wang**2)) / np.std(gt)

    Xg, Yg = np.meshgrid(gx, gy)
    vmax = max(np.abs(gt).max(), np.abs(ws_pred).max(), np.abs(wang_pred).max())
    vmin = -vmax

    fig, axes = plt.subplots(1, 3, figsize=figsize)
    fig.suptitle(f"{case['label']}  ·  WaveSplat nRMSE={nrmse_ws:.3f}  vs  Wang2025 nRMSE={nrmse_wang:.3f}",
                 fontsize=13, fontweight='bold', y=1.02)

    for ax, data, title in zip(axes,
                                [gt, ws_pred, wang_pred],
                                ['Ground Truth', 'WaveSplat', 'Wang 2025 (Ne=300)']):
        im = ax.pcolormesh(Xg, Yg, data, cmap='RdBu_r', vmin=vmin, vmax=vmax,
                           shading='auto')
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_aspect('equal')
        plt.colorbar(im, ax=ax, fraction=0.046, label='η [m]')

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"saved: {out_path} (WS={nrmse_ws:.3f}, Wang={nrmse_wang:.3f})")
    return (nrmse_ws, nrmse_wang)


def main():
    # 사용자가 비교 보고싶은 case 들
    cases = [
        {'episode': 'episode_BF3_2comp_seed0_uniform.h5', 'label': 'BF3 / N=2', 'frame_idx': 200, 'tag': 'bf3_n2'},
        {'episode': 'episode_BF3_5comp_seed0_uniform.h5', 'label': 'BF3 / N=5', 'frame_idx': 200, 'tag': 'bf3_n5'},
        {'episode': 'episode_BF4_5comp_seed0_uniform.h5', 'label': 'BF4 / N=5', 'frame_idx': 200, 'tag': 'bf4_n5'},
    ]
    results = []
    for c in cases:
        out_path = os.path.join(DOC_DIR, f"fig_wang_compare_{c['tag']}.png")
        r = plot_wang_comparison_single(c, out_path)
        if r:
            results.append((c['label'], *r))

    print("\n=== 요약 ===")
    for label, ws, wang in results:
        print(f"  {label}: WaveSplat {ws:.3f} vs Wang2025 {wang:.3f}")


if __name__ == '__main__':
    main()
