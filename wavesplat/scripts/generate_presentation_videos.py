"""
generate_presentation_videos.py

Unified video generator for WaveSplat presentations.

For each episode it produces two GIFs:
  1. wave_<episode>.gif        — 4-panel: LiDAR points / GT / WaveSplat / |GT-WS|
                                 (combines visualisations #1 + #3 from the
                                  presentation plan so GT and reconstruction
                                  can be compared at a glance)
  2. hydrofoil_<scenario>.gif  — PD-only vs PD+FF heave/acceleration time
                                 series with running RMS

The script is parameterised so new episodes (added later) can be rendered
by passing --episode <path>. Defaults cover the 5 representative episodes
shipped in git under wavesplat/data/.

Saves to GIF using matplotlib's pillow writer (no ffmpeg dependency).

Usage:
    python3 wavesplat/scripts/generate_presentation_videos.py
    python3 wavesplat/scripts/generate_presentation_videos.py \\
        --episode wavesplat/data/episode_BF4_5comp_seed0_uniform.h5
    python3 wavesplat/scripts/generate_presentation_videos.py \\
        --skip-hydrofoil --max-frames 80
"""

import argparse
import os
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import TwoSlopeNorm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from wavesplat.data_loader import EpisodeDataLoader
from wavesplat.optimizer.sequential_update import SequentialUpdater
from wavesplat.run import set_seed
from wavesplat.evaluation.metrics import nrmse


# 5 representative episodes shipped in git (matches docs/HANDS_ON.md)
DEFAULT_EPISODES = [
    'episode_BF2_1comp_seed0_uniform.h5',
    'episode_BF3_5comp_seed0_uniform.h5',
    'episode_BF4_5comp_seed0_uniform.h5',
    'episode_BF4_15comp_seed0_uniform.h5',
    'episode_BF5_30comp_seed0_slope.h5',
]


def _ws_config(depth):
    return {
        'n_gaussians': [3, 8, 15],
        'depth': depth,
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


# ---------------------------------------------------------------------------
# Wave field video (LiDAR + GT + WS + Error)
# ---------------------------------------------------------------------------

def render_wave_video(h5_path, output_path, max_frames=120, fps=10):
    """Render 4-panel wave field comparison GIF for a single episode."""
    set_seed(42)
    loader = EpisodeDataLoader(h5_path)

    cfg = _ws_config(loader.depth)
    updater = SequentialUpdater(cfg)
    init_frames = loader.get_init_window(tp_factor=2.0)
    updater.initialize(
        init_frames,
        x_range=(float(loader.gt_grid_x[0]), float(loader.gt_grid_x[-1])),
        y_range=(float(loader.gt_grid_y[0]), float(loader.gt_grid_y[-1])),
    )

    grid_x_np = loader.gt_grid_x.cpu().numpy()
    grid_y_np = loader.gt_grid_y.cpu().numpy()
    grid_x = loader.gt_grid_x
    grid_y = loader.gt_grid_y

    tracking_frames = loader.get_tracking_frames(tp_factor=2.0)
    n_frames = min(len(tracking_frames), max_frames)

    print(f"  Tracking {n_frames} frames for {os.path.basename(h5_path)}...")

    gt_fields, ws_fields, errors = [], [], []
    lidar_pts = []   # list of (x, y, z) arrays
    nrmses, times, n_list = [], [], []

    for i in range(n_frames):
        frame = tracking_frames[i]
        updater.update(frame)
        pred = updater.predict_field(grid_x, grid_y, frame.t)

        gt_np = frame.gt_field.cpu().numpy()
        pred_np = pred.cpu().numpy()
        gt_fields.append(gt_np)
        ws_fields.append(pred_np)
        errors.append(np.abs(gt_np - pred_np))

        lidar_pts.append((
            frame.x_obs.cpu().numpy(),
            frame.y_obs.cpu().numpy(),
            frame.z_obs.cpu().numpy(),
        ))
        nrmses.append(nrmse(pred_np, gt_np, loader.Hs_true))
        times.append(frame.t)
        n_list.append(updater.model.n_gaussians)

        if i % 20 == 0:
            print(f"    frame {i:3d}/{n_frames}  t={frame.t:5.1f}s  "
                  f"nRMSE={nrmses[-1]:.3f}  N={n_list[-1]}")

    vmax = max(max(abs(g.min()), abs(g.max())) for g in gt_fields)
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
    err_max = max(e.max() for e in errors)

    extent = [grid_x_np[0], grid_x_np[-1], grid_y_np[0], grid_y_np[-1]]

    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    ax_l, ax_gt, ax_ws, ax_err = axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]

    # LiDAR scatter
    sc = ax_l.scatter(lidar_pts[0][0], lidar_pts[0][1], c=lidar_pts[0][2],
                      cmap='RdBu_r', norm=norm, s=8)
    ax_l.set_xlim(extent[0], extent[1])
    ax_l.set_ylim(extent[2], extent[3])
    ax_l.set_aspect('equal')
    ax_l.set_title('LiDAR points (input)', fontweight='bold')
    ax_l.set_xlabel('x [m]')
    ax_l.set_ylabel('y [m]')

    im_gt = ax_gt.imshow(gt_fields[0], cmap='RdBu_r', norm=norm,
                         extent=extent, origin='lower', aspect='equal')
    ax_gt.set_title('Ground Truth', fontweight='bold')
    ax_gt.set_xlabel('x [m]')

    im_ws = ax_ws.imshow(ws_fields[0], cmap='RdBu_r', norm=norm,
                         extent=extent, origin='lower', aspect='equal')
    ax_ws.set_title('WaveSplat reconstruction', fontweight='bold')
    ax_ws.set_xlabel('x [m]')
    ax_ws.set_ylabel('y [m]')

    im_err = ax_err.imshow(errors[0], cmap='magma', vmin=0, vmax=err_max,
                           extent=extent, origin='lower', aspect='equal')
    ax_err.set_title('Absolute error |GT − WS|', fontweight='bold')
    ax_err.set_xlabel('x [m]')

    fig.colorbar(im_gt, ax=[ax_l, ax_gt, ax_ws], shrink=0.7,
                 label='η [m]', location='right')
    fig.colorbar(im_err, ax=ax_err, shrink=0.7, label='|Δη| [m]')

    info = ax_ws.text(0.02, 0.98, '', transform=ax_ws.transAxes,
                      fontsize=10, va='top',
                      bbox=dict(boxstyle='round', fc='white', alpha=0.85))
    sea_state = os.path.basename(h5_path).replace('episode_', '').replace('.h5', '')
    suptitle = fig.suptitle(
        f'{sea_state}  |  Hs={loader.Hs_true:.2f}m  Tp={loader.Tp_true:.1f}s  |  t=0.0s',
        fontsize=12, fontweight='bold')

    def update(idx):
        x, y, z = lidar_pts[idx]
        sc.set_offsets(np.column_stack([x, y]))
        sc.set_array(z)
        im_gt.set_data(gt_fields[idx])
        im_ws.set_data(ws_fields[idx])
        im_err.set_data(errors[idx])
        info.set_text(f'nRMSE={nrmses[idx]:.3f}\nN={n_list[idx]}\n'
                      f'M={len(lidar_pts[idx][0])} pts')
        suptitle.set_text(
            f'{sea_state}  |  Hs={loader.Hs_true:.2f}m  '
            f'Tp={loader.Tp_true:.1f}s  |  t={times[idx]:.1f}s')
        return sc, im_gt, im_ws, im_err, info, suptitle

    ani = animation.FuncAnimation(fig, update, frames=n_frames,
                                  interval=1000 // fps, blit=False)
    ani.save(output_path, writer=animation.PillowWriter(fps=fps))
    plt.close(fig)
    print(f"  saved → {output_path}")

    return {
        'episode': sea_state,
        'avg_nRMSE': float(np.mean(nrmses)),
        'final_N': n_list[-1],
        'n_frames': n_frames,
    }


# ---------------------------------------------------------------------------
# Hydrofoil video (PD-only vs PD+FF)
# ---------------------------------------------------------------------------

def render_hydrofoil_video(output_path, scenario_name='BF4 Head Sea',
                           Hs=1.88, Tp=7.5, direction_deg=180.0,
                           duration=30.0, fps=20):
    """Render side-by-side hydrofoil time-series comparison GIF."""
    # Local imports so wave-only mode does not require hydrofoil deps loaded eagerly
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    from wavesim.wave_generator import jonswap_components
    from hydrofoil.config import candela_c8
    from hydrofoil.controller import PDControllerConfig, FeedforwardConfig
    from hydrofoil.sim import run_simulation

    cfg = candela_c8()
    cfg['sim'].duration = duration
    components = jonswap_components(
        Hs=Hs, Tp=Tp, n_components=15,
        depth=cfg['sim'].depth,
        direction_deg=direction_deg,
        direction_spread_deg=20.0, seed=42,
    )
    ctrl = PDControllerConfig()

    np.random.seed(42)
    r_pd = run_simulation(cfg, components, vessel_heading=0.0,
                          controller=ctrl, feedforward=None)
    np.random.seed(42)
    ff = FeedforwardConfig(preview_dt=0.5, K_ff=0.8,
                           noise_std=Hs * 0.20, slope_noise_std=0.02)
    r_ff = run_simulation(cfg, components, vessel_heading=0.0,
                          controller=ctrl, feedforward=ff)

    t = r_pd.time
    z_pd = r_pd.z * 100   # cm
    z_ff = r_ff.z * 100
    a_pd = r_pd.heave_accel
    a_ff = r_ff.heave_accel

    n_total = len(t)
    # Subsample so the GIF stays light
    stride = max(1, n_total // 300)
    idx_arr = np.arange(0, n_total, stride)

    fig, (ax_z, ax_a) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    line_z_pd, = ax_z.plot([], [], 'r-', lw=1.5, label='PD-only')
    line_z_ff, = ax_z.plot([], [], 'b-', lw=1.5, label='PD + WaveSplat FF')
    ax_z.set_ylabel('Heave z [cm]')
    ax_z.set_xlim(t[0], t[-1])
    ax_z.set_ylim(min(z_pd.min(), z_ff.min()) * 1.1,
                  max(z_pd.max(), z_ff.max()) * 1.1)
    ax_z.grid(alpha=0.3)
    ax_z.legend(loc='upper right')
    ax_z.set_title(f'{scenario_name}  Hs={Hs}m  Tp={Tp}s  '
                   f'(WaveSplat ~20% nRMSE noise)', fontweight='bold')

    line_a_pd, = ax_a.plot([], [], 'r-', lw=1.2, label='PD-only')
    line_a_ff, = ax_a.plot([], [], 'b-', lw=1.2, label='PD + FF')
    ax_a.set_ylabel('Heave accel [m/s²]')
    ax_a.set_xlabel('Time [s]')
    ax_a.set_xlim(t[0], t[-1])
    ax_a.set_ylim(min(a_pd.min(), a_ff.min()) * 1.1,
                  max(a_pd.max(), a_ff.max()) * 1.1)
    ax_a.grid(alpha=0.3)

    txt = ax_a.text(0.02, 0.95, '', transform=ax_a.transAxes,
                    va='top', fontsize=10,
                    bbox=dict(boxstyle='round', fc='white', alpha=0.85))

    def running_rms(arr, t_arr, t_now, t_window=5.0):
        m = (t_arr <= t_now) & (t_arr > max(t_arr[0], t_now - t_window))
        if not np.any(m):
            return 0.0
        return float(np.sqrt(np.mean(arr[m] ** 2)))

    def update(k):
        end = idx_arr[k]
        line_z_pd.set_data(t[:end], z_pd[:end])
        line_z_ff.set_data(t[:end], z_ff[:end])
        line_a_pd.set_data(t[:end], a_pd[:end])
        line_a_ff.set_data(t[:end], a_ff[:end])
        t_now = t[end - 1] if end > 0 else t[0]
        rms_pd = running_rms(a_pd, t, t_now)
        rms_ff = running_rms(a_ff, t, t_now)
        red = (1 - rms_ff / rms_pd) * 100 if rms_pd > 0 else 0
        txt.set_text(f't = {t_now:5.1f}s   '
                     f'a_RMS PD={rms_pd:.3f}  FF={rms_ff:.3f}   '
                     f'Δ={red:+.0f}%')
        return line_z_pd, line_z_ff, line_a_pd, line_a_ff, txt

    ani = animation.FuncAnimation(fig, update, frames=len(idx_arr),
                                  interval=1000 // fps, blit=False)
    ani.save(output_path, writer=animation.PillowWriter(fps=fps))
    plt.close(fig)
    print(f"  saved → {output_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--episode', action='append', default=None,
                        help='HDF5 episode path (repeatable). '
                             'Default: 5 representative episodes in wavesplat/data/.')
    parser.add_argument('--data-dir', default=None,
                        help='Directory containing default episodes '
                             '(default: ../data relative to this script).')
    parser.add_argument('--output-dir', default=None,
                        help='Output directory for GIFs '
                             '(default: ../paper/videos).')
    parser.add_argument('--max-frames', type=int, default=120,
                        help='Max tracking frames per wave video.')
    parser.add_argument('--fps', type=int, default=10,
                        help='Wave video FPS (hydrofoil uses 2x).')
    parser.add_argument('--skip-wave', action='store_true',
                        help='Skip wave field videos.')
    parser.add_argument('--skip-hydrofoil', action='store_true',
                        help='Skip hydrofoil comparison video.')
    args = parser.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    data_dir = args.data_dir or os.path.join(here, '..', 'data')
    out_dir = args.output_dir or os.path.join(here, '..', 'paper', 'videos')
    os.makedirs(out_dir, exist_ok=True)

    if args.episode:
        episodes = args.episode
    else:
        episodes = [os.path.join(data_dir, e) for e in DEFAULT_EPISODES]

    summary = []

    if not args.skip_wave:
        print("=" * 78)
        print("WAVE FIELD VIDEOS  (LiDAR + GT + WaveSplat + Error)")
        print("=" * 78)
        for ep in episodes:
            if not os.path.exists(ep):
                print(f"  SKIP (missing): {ep}")
                continue
            tag = os.path.basename(ep).replace('episode_', '').replace('.h5', '')
            out_path = os.path.join(out_dir, f'wave_{tag}.gif')
            try:
                info = render_wave_video(ep, out_path,
                                          max_frames=args.max_frames,
                                          fps=args.fps)
                summary.append(info)
            except Exception as e:
                print(f"  FAILED: {ep}: {e}")

    if not args.skip_hydrofoil:
        print()
        print("=" * 78)
        print("HYDROFOIL VIDEO  (PD-only vs PD + WaveSplat FF)")
        print("=" * 78)
        out_path = os.path.join(out_dir, 'hydrofoil_BF4_head.gif')
        try:
            render_hydrofoil_video(out_path,
                                   scenario_name='BF4 Head Sea',
                                   Hs=1.88, Tp=7.5, direction_deg=180.0,
                                   fps=args.fps * 2)
        except Exception as e:
            print(f"  FAILED: hydrofoil video: {e}")

    if summary:
        print()
        print("=" * 78)
        print("SUMMARY")
        print("=" * 78)
        for s in summary:
            print(f"  {s['episode']:<30}  nRMSE={s['avg_nRMSE']:.3f}  "
                  f"N={s['final_N']}  frames={s['n_frames']}")

    print(f"\nOutput dir: {out_dir}")


if __name__ == '__main__':
    main()
