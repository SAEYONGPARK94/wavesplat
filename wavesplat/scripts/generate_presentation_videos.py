"""
generate_presentation_videos.py

Unified video generator for WaveSplat presentations.

For each episode it produces three GIFs:
  1. wave_<episode>.gif        — 4-panel 2D: LiDAR points / GT / WaveSplat /
                                 |GT-WS| (combines visualisations #1 + #3 so
                                 GT and reconstruction can be compared at a
                                 glance)
  2. wave3d_<episode>.gif      — 3D: GT wave surface + LiDAR scatter | WS
                                 surface + LiDAR scatter (side-by-side)
  3. hydrofoil_<scenario>.gif  — PD-only vs PD+FF heave/acceleration time
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

from scipy.interpolate import RegularGridInterpolator

from wavesplat.data_loader import EpisodeDataLoader
from wavesplat.optimizer.sequential_update import SequentialUpdater
from wavesplat.run import set_seed
from wavesplat.evaluation.metrics import nrmse
from wavesplat.baselines.etkf_grid import ETKFGrid
from wavesplat.baselines.gpr_baseline import GPRBaseline


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
# WaveSplat tracking (shared by 2D and 3D renderers)
# ---------------------------------------------------------------------------

def run_tracking(h5_path, max_frames=120, with_baselines=True):
    """Run WaveSplat (and optionally ETKF + GPR baselines) over an episode."""
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

    grid_x = loader.gt_grid_x
    grid_y = loader.gt_grid_y
    grid_x_np = grid_x.cpu().numpy()
    grid_y_np = grid_y.cpu().numpy()

    # ETKF on a coarser grid (matches run_etkf_comparison.py 'medium')
    etkf = etkf_gx = etkf_gy = None
    gpr = None
    if with_baselines:
        etkf_nx, etkf_ny = 40, 30
        etkf_gx = np.linspace(grid_x_np[0], grid_x_np[-1], etkf_nx)
        etkf_gy = np.linspace(grid_y_np[0], grid_y_np[-1], etkf_ny)
        etkf = ETKFGrid(etkf_gx, etkf_gy, depth=loader.depth, n_ensemble=50)
        gpr = GPRBaseline(kernel_type='matern', length_scale=20.0,
                          noise_sigma=0.05, window_frames=1)

        # Feed init frames to baselines so they have a warm state
        for f in init_frames:
            x = f.x_obs.cpu().numpy()
            y = f.y_obs.cpu().numpy()
            z = f.z_obs.cpu().numpy()
            etkf.update(x, y, z)
            gpr.update(x, y, z)

    tracking_frames = loader.get_tracking_frames(tp_factor=2.0)
    n_frames = min(len(tracking_frames), max_frames)

    print(f"  Tracking {n_frames} frames for {os.path.basename(h5_path)}...")
    gt_fields, ws_fields, etkf_fields, gpr_fields = [], [], [], []
    errors_ws, errors_etkf, errors_gpr = [], [], []
    lidar_pts = []
    nrmses_ws, nrmses_etkf, nrmses_gpr = [], [], []
    times, n_list = [], []

    for i in range(n_frames):
        frame = tracking_frames[i]

        # WaveSplat
        updater.update(frame)
        pred = updater.predict_field(grid_x, grid_y, frame.t)
        gt_np = frame.gt_field.cpu().numpy()
        ws_np = pred.cpu().numpy()
        gt_fields.append(gt_np)
        ws_fields.append(ws_np)
        errors_ws.append(np.abs(gt_np - ws_np))
        nrmses_ws.append(nrmse(ws_np, gt_np, loader.Hs_true))

        x = frame.x_obs.cpu().numpy()
        y = frame.y_obs.cpu().numpy()
        z = frame.z_obs.cpu().numpy()
        lidar_pts.append((x, y, z))
        times.append(frame.t)
        n_list.append(updater.model.n_gaussians)

        # ETKF on coarse grid → interp to GT grid
        if with_baselines:
            etkf.update(x, y, z)
            etkf_pred = etkf.predict_grid()
            interp = RegularGridInterpolator(
                (etkf_gy, etkf_gx), etkf_pred,
                method='linear', bounds_error=False, fill_value=0.0)
            Xg, Yg = np.meshgrid(grid_x_np, grid_y_np)
            etkf_np = interp(np.column_stack([Yg.ravel(), Xg.ravel()])).reshape(
                len(grid_y_np), len(grid_x_np))
            etkf_fields.append(etkf_np)
            errors_etkf.append(np.abs(gt_np - etkf_np))
            nrmses_etkf.append(nrmse(etkf_np, gt_np, loader.Hs_true))

            # GPR per-frame fit + predict on GT grid
            gpr.update(x, y, z)
            gpr_np = gpr.predict_grid(grid_x_np, grid_y_np)
            gpr_fields.append(gpr_np)
            errors_gpr.append(np.abs(gt_np - gpr_np))
            nrmses_gpr.append(nrmse(gpr_np, gt_np, loader.Hs_true))

        if i % 20 == 0:
            extra = ""
            if with_baselines:
                extra = (f"  ETKF={nrmses_etkf[-1]:.3f}  "
                         f"GPR={nrmses_gpr[-1]:.3f}")
            print(f"    frame {i:3d}/{n_frames}  t={frame.t:5.1f}s  "
                  f"WS={nrmses_ws[-1]:.3f}  N={n_list[-1]}{extra}")

    sea_state = os.path.basename(h5_path).replace('episode_', '').replace('.h5', '')
    return {
        'sea_state': sea_state,
        'Hs': loader.Hs_true,
        'Tp': loader.Tp_true,
        'grid_x': grid_x_np,
        'grid_y': grid_y_np,
        'gt_fields': gt_fields,
        'ws_fields': ws_fields,
        'etkf_fields': etkf_fields,
        'gpr_fields': gpr_fields,
        'errors_ws': errors_ws,
        'errors_etkf': errors_etkf,
        'errors_gpr': errors_gpr,
        'lidar_pts': lidar_pts,
        'nrmses_ws': nrmses_ws,
        'nrmses_etkf': nrmses_etkf,
        'nrmses_gpr': nrmses_gpr,
        'times': times,
        'n_list': n_list,
        'n_frames': n_frames,
        'with_baselines': with_baselines,
    }


# ---------------------------------------------------------------------------
# Wave field 2D video (LiDAR + GT + WS + Error)
# ---------------------------------------------------------------------------

def render_wave_video(track, output_path, fps=10):
    """Render 6-panel 2D wave field comparison: LiDAR + GT + 3 methods + nRMSE."""
    grid_x_np = track['grid_x']
    grid_y_np = track['grid_y']
    gt_fields = track['gt_fields']
    ws_fields = track['ws_fields']
    etkf_fields = track['etkf_fields']
    gpr_fields = track['gpr_fields']
    lidar_pts = track['lidar_pts']
    nrmses_ws = track['nrmses_ws']
    nrmses_etkf = track['nrmses_etkf']
    nrmses_gpr = track['nrmses_gpr']
    times = track['times']
    n_list = track['n_list']
    n_frames = track['n_frames']
    sea_state = track['sea_state']
    has_base = track['with_baselines'] and len(etkf_fields) > 0

    vmax = max(max(abs(g.min()), abs(g.max())) for g in gt_fields)
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)

    extent = [grid_x_np[0], grid_x_np[-1], grid_y_np[0], grid_y_np[-1]]

    # 3 rows × 2 cols of imshow + colorbar column.
    # Row 0: LiDAR | GT
    # Row 1: WaveSplat | ETKF
    # Row 2: GPR | running nRMSE line plot
    fig = plt.figure(figsize=(11, 13), constrained_layout=True)
    gs = fig.add_gridspec(3, 3, width_ratios=[1.0, 1.0, 0.04],
                          hspace=0.05, wspace=0.05)
    ax_l = fig.add_subplot(gs[0, 0])
    ax_gt = fig.add_subplot(gs[0, 1])
    ax_ws = fig.add_subplot(gs[1, 0])
    ax_etkf = fig.add_subplot(gs[1, 1])
    ax_gpr = fig.add_subplot(gs[2, 0])
    ax_metric = fig.add_subplot(gs[2, 1])
    cax_eta = fig.add_subplot(gs[:, 2])

    # LiDAR scatter (top-left)
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
    ax_ws.set_title('WaveSplat (ours)', fontweight='bold', color='#1f77b4')
    ax_ws.set_xlabel('x [m]')
    ax_ws.set_ylabel('y [m]')

    if has_base:
        im_etkf = ax_etkf.imshow(etkf_fields[0], cmap='RdBu_r', norm=norm,
                                  extent=extent, origin='lower', aspect='equal')
        ax_etkf.set_title('ETKF-grid', fontweight='bold', color='#d62728')
        ax_etkf.set_xlabel('x [m]')

        im_gpr = ax_gpr.imshow(gpr_fields[0], cmap='RdBu_r', norm=norm,
                                extent=extent, origin='lower', aspect='equal')
        ax_gpr.set_title('GPR (Matérn 5/2)', fontweight='bold', color='#2ca02c')
        ax_gpr.set_xlabel('x [m]')
        ax_gpr.set_ylabel('y [m]')
    else:
        im_etkf = None
        im_gpr = None
        ax_etkf.text(0.5, 0.5, 'baselines disabled',
                     transform=ax_etkf.transAxes, ha='center', va='center')
        ax_gpr.text(0.5, 0.5, 'baselines disabled',
                    transform=ax_gpr.transAxes, ha='center', va='center')

    fig.colorbar(im_gt, cax=cax_eta, label='η [m]')

    # Running nRMSE plot (bottom-right)
    ax_metric.set_title('nRMSE over time', fontweight='bold')
    ax_metric.set_xlabel('t [s]')
    ax_metric.set_ylabel('nRMSE')
    ax_metric.set_xlim(times[0], times[-1])
    all_nrmse = nrmses_ws + (nrmses_etkf if has_base else []) + (nrmses_gpr if has_base else [])
    ax_metric.set_ylim(0, max(all_nrmse) * 1.1)
    ax_metric.grid(alpha=0.3)
    line_ws, = ax_metric.plot([], [], '-', color='#1f77b4', lw=2, label='WaveSplat')
    if has_base:
        line_etkf, = ax_metric.plot([], [], '-', color='#d62728', lw=1.5, label='ETKF')
        line_gpr, = ax_metric.plot([], [], '-', color='#2ca02c', lw=1.5, label='GPR')
    else:
        line_etkf = line_gpr = None
    ax_metric.legend(loc='upper right', fontsize=9)

    # Info box overlaid on WaveSplat panel.
    info = ax_ws.text(0.02, 0.98, '', transform=ax_ws.transAxes,
                      fontsize=9, va='top', ha='left',
                      bbox=dict(boxstyle='round', fc='white', alpha=0.85))
    Hs = track['Hs']
    Tp = track['Tp']
    suptitle = fig.suptitle(
        f'{sea_state}  |  Hs={Hs:.2f}m  Tp={Tp:.1f}s  |  t=0.0s',
        fontsize=13, fontweight='bold')

    def update(idx):
        x, y, z = lidar_pts[idx]
        sc.set_offsets(np.column_stack([x, y]))
        sc.set_array(z)
        im_gt.set_data(gt_fields[idx])
        im_ws.set_data(ws_fields[idx])
        if has_base:
            im_etkf.set_data(etkf_fields[idx])
            im_gpr.set_data(gpr_fields[idx])
            line_etkf.set_data(times[:idx + 1], nrmses_etkf[:idx + 1])
            line_gpr.set_data(times[:idx + 1], nrmses_gpr[:idx + 1])
        line_ws.set_data(times[:idx + 1], nrmses_ws[:idx + 1])

        info.set_text(f'WS nRMSE={nrmses_ws[idx]:.3f}\n'
                      f'N={n_list[idx]}  M={len(x)} pts')
        suptitle.set_text(
            f'{sea_state}  |  Hs={Hs:.2f}m  '
            f'Tp={Tp:.1f}s  |  t={times[idx]:.1f}s')

    ani = animation.FuncAnimation(fig, update, frames=n_frames,
                                  interval=1000 // fps, blit=False)
    ani.save(output_path, writer=animation.PillowWriter(fps=fps))
    plt.close(fig)
    print(f"  saved → {output_path}")


# ---------------------------------------------------------------------------
# Wave field 3D video (LiDAR points + wave surface)
# ---------------------------------------------------------------------------

def render_wave_3d_video(track, output_path, fps=10, stride=2):
    """Render 2x2 3D GT/WaveSplat/ETKF/GPR surfaces with LiDAR scatter.

    `stride` subsamples the surface mesh in both x and y to keep rendering
    fast (3D surface plots cannot be incrementally updated like imshow,
    so each frame redraws).
    """
    grid_x = track['grid_x']
    grid_y = track['grid_y']
    gt_fields = track['gt_fields']
    ws_fields = track['ws_fields']
    etkf_fields = track['etkf_fields']
    gpr_fields = track['gpr_fields']
    lidar_pts = track['lidar_pts']
    nrmses_ws = track['nrmses_ws']
    nrmses_etkf = track['nrmses_etkf']
    nrmses_gpr = track['nrmses_gpr']
    times = track['times']
    n_list = track['n_list']
    n_frames = track['n_frames']
    sea_state = track['sea_state']
    Hs = track['Hs']
    Tp = track['Tp']
    has_base = track['with_baselines'] and len(etkf_fields) > 0

    xs = grid_x[::stride]
    ys = grid_y[::stride]
    X, Y = np.meshgrid(xs, ys)
    gt_sub = [g[::stride, ::stride] for g in gt_fields]
    ws_sub = [w[::stride, ::stride] for w in ws_fields]
    etkf_sub = [e[::stride, ::stride] for e in etkf_fields] if has_base else []
    gpr_sub = [g[::stride, ::stride] for g in gpr_fields] if has_base else []

    vmax = max(max(abs(g.min()), abs(g.max())) for g in gt_fields)
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)

    fig = plt.figure(figsize=(13, 11), constrained_layout=True)
    ax_gt = fig.add_subplot(2, 2, 1, projection='3d')
    ax_ws = fig.add_subplot(2, 2, 2, projection='3d')
    ax_etkf = fig.add_subplot(2, 2, 3, projection='3d')
    ax_gpr = fig.add_subplot(2, 2, 4, projection='3d')

    suptitle = fig.suptitle(
        f'{sea_state}  |  Hs={Hs:.2f}m  Tp={Tp:.1f}s  |  t=0.0s',
        fontsize=13, fontweight='bold')

    def setup_axis(ax, title, color='black'):
        ax.set_xlabel('x [m]')
        ax.set_ylabel('y [m]')
        ax.set_zlabel('η [m]')
        ax.set_xlim(grid_x[0], grid_x[-1])
        ax.set_ylim(grid_y[0], grid_y[-1])
        ax.set_zlim(-vmax * 1.1, vmax * 1.1)
        ax.set_title(title, fontweight='bold', pad=8, color=color)
        ax.view_init(elev=28, azim=-60)

    def draw_surface(ax, Z):
        ax.plot_surface(X, Y, Z, cmap='RdBu_r', norm=norm,
                        rstride=1, cstride=1, alpha=0.85,
                        linewidth=0, antialiased=False, shade=False)

    def draw_frame(idx):
        ax_gt.clear()
        ax_ws.clear()
        ax_etkf.clear()
        ax_gpr.clear()
        setup_axis(ax_gt, 'Ground Truth')
        setup_axis(ax_ws, f'WaveSplat (nRMSE={nrmses_ws[idx]:.3f})',
                   color='#1f77b4')
        if has_base:
            setup_axis(ax_etkf, f'ETKF-grid (nRMSE={nrmses_etkf[idx]:.3f})',
                       color='#d62728')
            setup_axis(ax_gpr, f'GPR (nRMSE={nrmses_gpr[idx]:.3f})',
                       color='#2ca02c')
        else:
            setup_axis(ax_etkf, 'ETKF (disabled)')
            setup_axis(ax_gpr, 'GPR (disabled)')

        draw_surface(ax_gt, gt_sub[idx])
        draw_surface(ax_ws, ws_sub[idx])
        if has_base:
            draw_surface(ax_etkf, etkf_sub[idx])
            draw_surface(ax_gpr, gpr_sub[idx])

        x, y, z = lidar_pts[idx]
        for ax in (ax_gt, ax_ws, ax_etkf, ax_gpr):
            ax.scatter(x, y, z, c='black', s=4, depthshade=False, alpha=0.8)

        suptitle.set_text(
            f'{sea_state}  |  Hs={Hs:.2f}m  '
            f'Tp={Tp:.1f}s  |  t={times[idx]:.1f}s  |  N={n_list[idx]}')

    draw_frame(0)
    ani = animation.FuncAnimation(fig, draw_frame, frames=n_frames,
                                  interval=1000 // fps, blit=False)
    ani.save(output_path, writer=animation.PillowWriter(fps=fps))
    plt.close(fig)
    print(f"  saved → {output_path}")


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
                        help='Skip 2D wave field videos.')
    parser.add_argument('--skip-3d', action='store_true',
                        help='Skip 3D wave + LiDAR videos.')
    parser.add_argument('--skip-hydrofoil', action='store_true',
                        help='Skip hydrofoil comparison video.')
    parser.add_argument('--surface-stride', type=int, default=2,
                        help='3D surface mesh stride (higher = faster, '
                             'but short-wavelength waves may alias).')
    parser.add_argument('--no-baselines', action='store_true',
                        help='Skip ETKF and GPR baselines (faster, '
                             'shows only WaveSplat).')
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
    do_2d = not args.skip_wave
    do_3d = not args.skip_3d

    if do_2d or do_3d:
        print("=" * 78)
        print("WAVE FIELD VIDEOS  (2D 4-panel and/or 3D surface + LiDAR)")
        print("=" * 78)
        for ep in episodes:
            if not os.path.exists(ep):
                print(f"  SKIP (missing): {ep}")
                continue
            tag = os.path.basename(ep).replace('episode_', '').replace('.h5', '')
            try:
                track = run_tracking(ep, max_frames=args.max_frames,
                                     with_baselines=not args.no_baselines)
            except Exception as e:
                print(f"  FAILED tracking {ep}: {e}")
                continue
            summary.append({
                'episode': track['sea_state'],
                'WS_nRMSE': float(np.mean(track['nrmses_ws'])),
                'ETKF_nRMSE': float(np.mean(track['nrmses_etkf']))
                    if track['nrmses_etkf'] else float('nan'),
                'GPR_nRMSE': float(np.mean(track['nrmses_gpr']))
                    if track['nrmses_gpr'] else float('nan'),
                'final_N': track['n_list'][-1],
                'n_frames': track['n_frames'],
            })
            if do_2d:
                try:
                    render_wave_video(track,
                                      os.path.join(out_dir, f'wave_{tag}.gif'),
                                      fps=args.fps)
                except Exception as e:
                    print(f"  FAILED 2D {ep}: {e}")
            if do_3d:
                try:
                    render_wave_3d_video(track,
                                         os.path.join(out_dir, f'wave3d_{tag}.gif'),
                                         fps=args.fps,
                                         stride=args.surface_stride)
                except Exception as e:
                    print(f"  FAILED 3D {ep}: {e}")

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
        print(f"  {'Episode':<30} {'WS':>8} {'ETKF':>8} {'GPR':>8}  N  frames")
        for s in summary:
            print(f"  {s['episode']:<30} "
                  f"{s['WS_nRMSE']:>8.3f} "
                  f"{s['ETKF_nRMSE']:>8.3f} "
                  f"{s['GPR_nRMSE']:>8.3f}  "
                  f"{s['final_N']:>2}  {s['n_frames']}")

    print(f"\nOutput dir: {out_dir}")


if __name__ == '__main__':
    main()
