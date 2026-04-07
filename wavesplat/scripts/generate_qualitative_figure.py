"""
generate_qualitative_figure.py
Generate side-by-side GT vs WaveSplat vs ETKF (vs GPR) 2D heatmaps
for selected frames. Used as Figure 8 in paper.
"""

import sys
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from wavesplat.data_loader import EpisodeDataLoader
from wavesplat.run import run_estimation, set_seed
from wavesplat.baselines.etkf_grid import ETKFGrid
from wavesplat.baselines.gpr_baseline import GPRBaseline
from wavesplat.optimizer.sequential_update import SequentialUpdater
from wavesplat.evaluation.metrics import rmse, nrmse
from scipy.interpolate import RegularGridInterpolator


# Selected episodes and frames for qualitative comparison
CASES = [
    {'episode': 'episode_BF3_2comp_seed0_uniform.h5', 'label': 'BF3 2-comp', 'frame_idx': 50},
    {'episode': 'episode_BF4_15comp_seed0_uniform.h5', 'label': 'BF4 15-comp', 'frame_idx': 50},
    {'episode': 'episode_BF5_30comp_seed0_uniform.h5', 'label': 'BF5 30-comp', 'frame_idx': 50},
]


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
    if target_frame_idx < len(result['frame_results']):
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
        return pred.cpu().numpy(), tracking_frames[frame_idx].gt_field.cpu().numpy()
    return None, None


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
            return pred, frame.gt_field.cpu().numpy()
    return None, None


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
            return pred, frame.gt_field.cpu().numpy()
    return None, None


def make_figure(cases_data, output_path):
    """Create n_cases × 4 heatmap grid: GT, WaveSplat, ETKF, GPR."""
    n_cases = len(cases_data)
    fig, axes = plt.subplots(n_cases, 4, figsize=(16, 3.5 * n_cases))
    if n_cases == 1:
        axes = axes.reshape(1, -1)

    methods = ['Ground Truth', 'WaveSplat', 'ETKF', 'GPR']

    for row, case in enumerate(cases_data):
        # Find global vmin/vmax from GT
        gt = case['gt']
        vmax = max(abs(gt.min()), abs(gt.max()))
        norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)

        for col, (method, data) in enumerate(zip(methods, [gt, case['ws'], case['etkf'], case['gpr']])):
            ax = axes[row, col]
            if data is not None:
                im = ax.imshow(data, cmap='RdBu_r', norm=norm,
                               aspect='auto', origin='lower')

                # Add nRMSE annotation (except GT)
                if col > 0 and gt is not None:
                    n = nrmse(data, gt, case['Hs'])
                    ax.text(0.02, 0.98, f'nRMSE={n:.3f}',
                            transform=ax.transAxes, fontsize=9,
                            verticalalignment='top',
                            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            else:
                ax.text(0.5, 0.5, 'N/A', transform=ax.transAxes,
                        ha='center', va='center', fontsize=14)

            if row == 0:
                ax.set_title(method, fontsize=12, fontweight='bold')
            if col == 0:
                ax.set_ylabel(case['label'], fontsize=11, fontweight='bold')

            ax.set_xticks([])
            ax.set_yticks([])

        # Colorbar
        fig.colorbar(im, ax=axes[row, :].tolist(), shrink=0.8, label='η [m]')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.savefig(output_path.replace('.png', '.pdf'), bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def main():
    data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
    fig_dir = os.path.join(os.path.dirname(__file__), '..', 'paper', 'figures')
    os.makedirs(fig_dir, exist_ok=True)

    cases_data = []

    for case in CASES:
        h5_path = os.path.join(data_dir, case['episode'])
        if not os.path.exists(h5_path):
            print(f"SKIP: {case['episode']}")
            continue

        print(f"\n[{case['label']}] Processing...", flush=True)

        loader = EpisodeDataLoader(h5_path)

        # WaveSplat
        print("  WaveSplat...", end=" ", flush=True)
        ws_pred, gt = get_ws_prediction(h5_path, case['frame_idx'])
        print("done")

        # ETKF
        print("  ETKF...", end=" ", flush=True)
        etkf_pred, _ = get_etkf_prediction(h5_path, case['frame_idx'])
        print("done")

        # GPR
        print("  GPR...", end=" ", flush=True)
        gpr_pred, _ = get_gpr_prediction(h5_path, case['frame_idx'])
        print("done")

        cases_data.append({
            'label': case['label'],
            'gt': gt,
            'ws': ws_pred,
            'etkf': etkf_pred,
            'gpr': gpr_pred,
            'Hs': loader.Hs_true,
        })

    if cases_data:
        make_figure(cases_data,
                    os.path.join(fig_dir, 'fig8_qualitative.png'))
    else:
        print("No episodes found. Skipping figure generation.")


if __name__ == '__main__':
    main()
