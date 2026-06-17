"""
Visualize GP K_disp vs WaveSplat vs GT comparison.

Outputs:
  - docs/fig_gp_comparison_{tag}.png : per-episode 4-column comparison (GT, WaveSplat, GP K_disp, GP UQ)
  - docs/fig_gp_multi_comparison.png : multi-episode comparison grid
  - docs/gif_gp_comparison_{tag}.gif : animated frame-by-frame GIF for selected episodes
"""
import os, sys, time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import matplotlib.cm as cm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from wavesplat.data_loader import EpisodeDataLoader
from wavesplat.run import set_seed
from wavesplat.optimizer.sequential_update import SequentialUpdater
from scripts.gp_disp_pilot import (
    estimate_components_wavesplat_opt, build_K_disp_oracle,
    gp_predict, compute_metrics_simple, SIGMA_OBS
)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(SCRIPT_DIR, '..')
DATA_DIR = os.path.join(ROOT, 'data')
DOC_DIR = os.path.join(ROOT, 'docs')
os.makedirs(DOC_DIR, exist_ok=True)

# Episodes to visualize (diverse complexity)
VIS_EPISODES = [
    {'episode': 'episode_BF3_5comp_seed0_slope.h5',   'label': 'BF3 / N=5',  'tag': 'bf3_5c'},
    {'episode': 'episode_BF4_5comp_seed0_uniform.h5',  'label': 'BF4 / N=5',  'tag': 'bf4_5c'},
    {'episode': 'episode_BF4_15comp_seed0_uniform.h5', 'label': 'BF4 / N=15', 'tag': 'bf4_15c'},
    {'episode': 'episode_BF5_15comp_seed0_uniform.h5', 'label': 'BF5 / N=15', 'tag': 'bf5_15c'},
    {'episode': 'episode_BF4_30comp_seed0_uniform.h5', 'label': 'BF4 / N=30', 'tag': 'bf4_30c'},
    {'episode': 'episode_BF5_30comp_seed0_uniform.h5', 'label': 'BF5 / N=30', 'tag': 'bf5_30c'},
]


def get_ws_prediction(loader, frame_idx):
    """Run WaveSplat and return prediction at target frame."""
    set_seed(42)
    cfg = {
        'init_method': 'v2_konly',
        'use_dispersion': True,
        'use_confidence': True,
        'adaptive_n': True,
        'tracking_window': 3,
        'seed': 42,
        'depth': loader.depth,
    }
    updater = SequentialUpdater(cfg)
    init_frames = loader.get_init_window(tp_factor=2.0)
    updater.initialize(
        init_frames,
        x_range=(float(loader.gt_grid_x[0]), float(loader.gt_grid_x[-1])),
        y_range=(float(loader.gt_grid_y[0]), float(loader.gt_grid_y[-1])),
    )

    tracking_frames = loader.get_tracking_frames(tp_factor=2.0)
    fidx = min(frame_idx, len(tracking_frames) - 1)
    for i in range(fidx + 1):
        updater.update(tracking_frames[i])

    pred = updater.predict_field(
        loader.gt_grid_x, loader.gt_grid_y, tracking_frames[fidx].t)
    return pred.cpu().numpy()


def get_gp_prediction(loader, beam_components, frame, gt_x, gt_y):
    """GP K_disp prediction + posterior std for one frame."""
    x_obs = frame.x_obs.cpu().numpy()
    y_obs = frame.y_obs.cpu().numpy()
    z_obs = frame.z_obs.cpu().numpy()
    if len(x_obs) < 5:
        return None, None

    Xg, Yg = np.meshgrid(gt_x, gt_y)
    X_pred = np.column_stack([Xg.ravel(), Yg.ravel()])
    X_obs = np.column_stack([x_obs, y_obs])

    K_oo = build_K_disp_oracle(X_obs, X_obs, beam_components)
    K_po = build_K_disp_oracle(X_pred, X_obs, beam_components)

    # Mean
    mu = gp_predict(K_oo, K_po, z_obs, SIGMA_OBS)

    # Posterior std
    n = K_oo.shape[0]
    K_oo_noise = K_oo + SIGMA_OBS**2 * np.eye(n)
    K_oo_noise += 1e-6 * np.trace(K_oo_noise) / n * np.eye(n)
    try:
        L = np.linalg.cholesky(K_oo_noise)
        V = np.linalg.solve(L, K_po.T)
        reduction = np.sum(V**2, axis=0)
        A = beam_components[:, 0]
        prior_var = float(np.sum(A**2 / 2.0))
        var_post = np.maximum(prior_var - reduction, 1e-8)
        std_post = np.sqrt(var_post).reshape(len(gt_y), len(gt_x))
    except np.linalg.LinAlgError:
        std_post = np.zeros((len(gt_y), len(gt_x)))

    pred = mu.reshape(len(gt_y), len(gt_x))
    return pred, std_post


def plot_single_frame_comparison(gt, ws_pred, gp_pred, gp_std, obs_xy,
                                  gt_x, gt_y, Hs, label, out_path):
    """5-column comparison: GT, WaveSplat, GP K_disp, GP Error, GP UQ (std)."""
    fig, axes = plt.subplots(1, 5, figsize=(24, 4.2))
    Xg, Yg = np.meshgrid(gt_x, gt_y)

    vmax = max(np.abs(gt).max(), np.abs(ws_pred).max(), np.abs(gp_pred).max())
    vmin = -vmax

    ws_err = gt - ws_pred
    gp_err = gt - gp_pred
    ws_nrmse = np.sqrt(np.mean(ws_err**2)) / max(Hs, 1e-6)
    gp_nrmse = np.sqrt(np.mean(gp_err**2)) / max(Hs, 1e-6)
    err_max = max(np.abs(ws_err).max(), np.abs(gp_err).max())

    # 1. GT
    im0 = axes[0].pcolormesh(Xg, Yg, gt, cmap='RdBu_r', vmin=vmin, vmax=vmax, shading='auto')
    if obs_xy is not None:
        axes[0].scatter(obs_xy[:, 0], obs_xy[:, 1], c='k', s=1, alpha=0.3, zorder=5)
    axes[0].set_title('Ground Truth', fontsize=11, fontweight='bold')
    axes[0].set_aspect('equal')
    axes[0].set_xticks([]); axes[0].set_yticks([])
    plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    # 2. WaveSplat
    im1 = axes[1].pcolormesh(Xg, Yg, ws_pred, cmap='RdBu_r', vmin=vmin, vmax=vmax, shading='auto')
    axes[1].set_title(f'WaveSplat  nRMSE={ws_nrmse:.3f}', fontsize=10, fontweight='bold')
    axes[1].set_aspect('equal')
    axes[1].set_xticks([]); axes[1].set_yticks([])
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    # 3. GP K_disp
    im2 = axes[2].pcolormesh(Xg, Yg, gp_pred, cmap='RdBu_r', vmin=vmin, vmax=vmax, shading='auto')
    axes[2].set_title(f'GP K_disp  nRMSE={gp_nrmse:.3f}', fontsize=10, fontweight='bold',
                      color='darkgreen' if gp_nrmse < ws_nrmse else 'darkred')
    axes[2].set_aspect('equal')
    axes[2].set_xticks([]); axes[2].set_yticks([])
    plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

    # 4. Error comparison (WS error top, GP error bottom)
    err_norm = TwoSlopeNorm(vmin=-err_max, vcenter=0, vmax=err_max)
    im3 = axes[3].pcolormesh(Xg, Yg, gp_err, cmap='RdBu_r', norm=err_norm, shading='auto')
    axes[3].set_title('GP Error (GT - GP)', fontsize=10, fontweight='bold')
    axes[3].set_aspect('equal')
    axes[3].set_xticks([]); axes[3].set_yticks([])
    plt.colorbar(im3, ax=axes[3], fraction=0.046, pad=0.04)

    # 5. GP UQ (posterior std)
    im4 = axes[4].pcolormesh(Xg, Yg, gp_std, cmap='YlOrRd', shading='auto')
    axes[4].set_title(f'GP Uncertainty (std)', fontsize=10, fontweight='bold')
    axes[4].set_aspect('equal')
    axes[4].set_xticks([]); axes[4].set_yticks([])
    plt.colorbar(im4, ax=axes[4], fraction=0.046, pad=0.04, label='std [m]')

    fig.suptitle(f'{label}', fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  saved: {out_path}")


def plot_multi_episode_grid(all_data, out_path):
    """Multi-episode grid: rows=episodes, cols=GT/WS/GP/UQ."""
    n_ep = len(all_data)
    fig, axes = plt.subplots(n_ep, 4, figsize=(18, 3.8 * n_ep))
    if n_ep == 1:
        axes = axes[np.newaxis, :]

    for ri, d in enumerate(all_data):
        gt_x, gt_y = d['gt_x'], d['gt_y']
        Xg, Yg = np.meshgrid(gt_x, gt_y)
        gt, ws, gp, std = d['gt'], d['ws'], d['gp'], d['gp_std']
        Hs = d['Hs']

        vmax = max(np.abs(gt).max(), np.abs(ws).max(), np.abs(gp).max())
        vmin = -vmax
        ws_nrmse = np.sqrt(np.mean((gt - ws)**2)) / max(Hs, 1e-6)
        gp_nrmse = np.sqrt(np.mean((gt - gp)**2)) / max(Hs, 1e-6)

        axes[ri, 0].pcolormesh(Xg, Yg, gt, cmap='RdBu_r', vmin=vmin, vmax=vmax, shading='auto')
        axes[ri, 0].set_ylabel(d['label'], fontsize=11, fontweight='bold')

        axes[ri, 1].pcolormesh(Xg, Yg, ws, cmap='RdBu_r', vmin=vmin, vmax=vmax, shading='auto')
        axes[ri, 2].pcolormesh(Xg, Yg, gp, cmap='RdBu_r', vmin=vmin, vmax=vmax, shading='auto')

        im_std = axes[ri, 3].pcolormesh(Xg, Yg, std, cmap='YlOrRd', shading='auto')

        if ri == 0:
            axes[ri, 0].set_title('Ground Truth', fontsize=11, fontweight='bold')
            axes[ri, 1].set_title('WaveSplat', fontsize=11, fontweight='bold')
            axes[ri, 2].set_title('GP K_disp', fontsize=11, fontweight='bold')
            axes[ri, 3].set_title('GP Uncertainty', fontsize=11, fontweight='bold')

        # nRMSE labels
        axes[ri, 1].text(0.02, 0.95, f'{ws_nrmse:.3f}', transform=axes[ri, 1].transAxes,
                         fontsize=9, fontweight='bold', va='top',
                         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        color = 'green' if gp_nrmse < ws_nrmse else 'red'
        axes[ri, 2].text(0.02, 0.95, f'{gp_nrmse:.3f}', transform=axes[ri, 2].transAxes,
                         fontsize=9, fontweight='bold', va='top', color=color,
                         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        for ci in range(4):
            axes[ri, ci].set_aspect('equal')
            axes[ri, ci].set_xticks([]); axes[ri, ci].set_yticks([])

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  saved: {out_path}")


def generate_gif(loader, beam_components, gt_x, gt_y, label, tag,
                 n_frames=30, fps=4):
    """Generate animated GIF comparing GT vs GP K_disp frame by frame."""
    import imageio.v2 as imageio

    tracking_frames = loader.get_tracking_frames(tp_factor=2.0)
    Hs = loader.Hs_true
    Xg, Yg = np.meshgrid(gt_x, gt_y)

    # Also run WaveSplat for comparison
    set_seed(42)
    cfg = {
        'init_method': 'v2_konly', 'use_dispersion': True,
        'use_confidence': True, 'adaptive_n': True,
        'tracking_window': 3, 'seed': 42, 'depth': loader.depth,
    }
    updater = SequentialUpdater(cfg)
    init_frames = loader.get_init_window(tp_factor=2.0)
    updater.initialize(init_frames,
                       x_range=(float(gt_x[0]), float(gt_x[-1])),
                       y_range=(float(gt_y[0]), float(gt_y[-1])))

    frames_png = []
    tmp_dir = os.path.join(DOC_DIR, f'_tmp_gif_{tag}')
    os.makedirs(tmp_dir, exist_ok=True)

    for fi in range(min(n_frames, len(tracking_frames))):
        frame = tracking_frames[fi]
        gt = frame.gt_field.cpu().numpy()

        # WaveSplat
        updater.update(frame)
        ws_pred = updater.predict_field(
            loader.gt_grid_x, loader.gt_grid_y, frame.t).cpu().numpy()

        # GP K_disp
        gp_pred, gp_std = get_gp_prediction(loader, beam_components, frame, gt_x, gt_y)
        if gp_pred is None:
            continue

        vmax = max(np.abs(gt).max(), 0.01)
        vmin = -vmax
        ws_nrmse = np.sqrt(np.mean((gt - ws_pred)**2)) / max(Hs, 1e-6)
        gp_nrmse = np.sqrt(np.mean((gt - gp_pred)**2)) / max(Hs, 1e-6)

        fig, axes = plt.subplots(1, 3, figsize=(14, 3.8))

        axes[0].pcolormesh(Xg, Yg, gt, cmap='RdBu_r', vmin=vmin, vmax=vmax, shading='auto')
        axes[0].set_title('Ground Truth', fontsize=11, fontweight='bold')

        axes[1].pcolormesh(Xg, Yg, ws_pred, cmap='RdBu_r', vmin=vmin, vmax=vmax, shading='auto')
        axes[1].set_title(f'WaveSplat  nRMSE={ws_nrmse:.3f}', fontsize=10, fontweight='bold')

        axes[2].pcolormesh(Xg, Yg, gp_pred, cmap='RdBu_r', vmin=vmin, vmax=vmax, shading='auto')
        color = 'darkgreen' if gp_nrmse < ws_nrmse else 'darkred'
        axes[2].set_title(f'GP K_disp  nRMSE={gp_nrmse:.3f}', fontsize=10,
                          fontweight='bold', color=color)

        for ax in axes:
            ax.set_aspect('equal')
            ax.set_xticks([]); ax.set_yticks([])

        fig.suptitle(f'{label}  Frame {fi+1}/{n_frames}', fontsize=12, fontweight='bold')
        plt.tight_layout()

        frame_path = os.path.join(tmp_dir, f'frame_{fi:03d}.png')
        fig.savefig(frame_path, dpi=100, bbox_inches='tight')
        plt.close(fig)
        frames_png.append(frame_path)

    # Assemble GIF
    if frames_png:
        gif_path = os.path.join(DOC_DIR, f'gif_gp_comparison_{tag}.gif')
        images = [imageio.imread(fp) for fp in frames_png]
        imageio.mimsave(gif_path, images, duration=1000//fps, loop=0)
        print(f"  GIF saved: {gif_path} ({len(frames_png)} frames, {fps} fps)")

    # Cleanup
    import shutil
    shutil.rmtree(tmp_dir, ignore_errors=True)


def main():
    print("=" * 70)
    print("GP K_disp Visualization: Comparison with WaveSplat")
    print("=" * 70)

    all_data = []

    for ep_info in VIS_EPISODES:
        h5_path = os.path.join(DATA_DIR, ep_info['episode'])
        if not os.path.exists(h5_path):
            print(f"SKIP: {ep_info['episode']}")
            continue

        label = ep_info['label']
        tag = ep_info['tag']
        print(f"\n[{label}] Loading...")

        loader = EpisodeDataLoader(h5_path)
        Hs = loader.Hs_true
        gt_x = loader.gt_grid_x.cpu().numpy()
        gt_y = loader.gt_grid_y.cpu().numpy()

        # GP K_disp components (lambda=0, n_starts=5)
        print(f"  Estimating K_disp components (n_starts=5)...")
        beam_components = estimate_components_wavesplat_opt(
            loader, apply_reg_amp=True, n_starts=5, lambda_reg=0.0,
            init_tp_factor=2.0, sigma_obs=SIGMA_OBS
        )

        # Pick a representative tracking frame (frame 15 = middle)
        tracking_frames = loader.get_tracking_frames(tp_factor=2.0)
        fidx = min(15, len(tracking_frames) - 1)
        frame = tracking_frames[fidx]
        gt = frame.gt_field.cpu().numpy()
        obs_xy = np.column_stack([frame.x_obs.cpu().numpy(), frame.y_obs.cpu().numpy()])

        # WaveSplat prediction
        print(f"  Running WaveSplat...")
        ws_pred = get_ws_prediction(loader, fidx)

        # GP prediction
        print(f"  Running GP K_disp...")
        gp_pred, gp_std = get_gp_prediction(loader, beam_components, frame, gt_x, gt_y)

        if gp_pred is not None:
            # Single-episode figure
            out_path = os.path.join(DOC_DIR, f'fig_gp_comparison_{tag}.png')
            plot_single_frame_comparison(gt, ws_pred, gp_pred, gp_std, obs_xy,
                                         gt_x, gt_y, Hs, label, out_path)

            all_data.append({
                'label': label, 'tag': tag,
                'gt': gt, 'ws': ws_pred, 'gp': gp_pred, 'gp_std': gp_std,
                'gt_x': gt_x, 'gt_y': gt_y, 'Hs': Hs,
            })

    # Multi-episode grid
    if all_data:
        out_multi = os.path.join(DOC_DIR, 'fig_gp_multi_comparison.png')
        print(f"\nGenerating multi-episode grid...")
        plot_multi_episode_grid(all_data, out_multi)

    # GIF for 2 selected episodes (1 simple, 1 complex)
    gif_episodes = [
        {'episode': 'episode_BF4_5comp_seed0_uniform.h5', 'label': 'BF4 / N=5', 'tag': 'bf4_5c'},
        {'episode': 'episode_BF4_15comp_seed0_uniform.h5', 'label': 'BF4 / N=15', 'tag': 'bf4_15c'},
    ]
    for ep_info in gif_episodes:
        h5_path = os.path.join(DATA_DIR, ep_info['episode'])
        if not os.path.exists(h5_path):
            continue
        print(f"\n[GIF] {ep_info['label']}...")
        loader = EpisodeDataLoader(h5_path)
        gt_x = loader.gt_grid_x.cpu().numpy()
        gt_y = loader.gt_grid_y.cpu().numpy()

        beam_components = estimate_components_wavesplat_opt(
            loader, apply_reg_amp=True, n_starts=5, lambda_reg=0.0,
            init_tp_factor=2.0, sigma_obs=SIGMA_OBS
        )
        generate_gif(loader, beam_components, gt_x, gt_y,
                     ep_info['label'], ep_info['tag'], n_frames=30, fps=4)

    print("\nDone!")


if __name__ == '__main__':
    main()
