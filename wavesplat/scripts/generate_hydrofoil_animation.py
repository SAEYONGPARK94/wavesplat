#!/usr/bin/env python3
"""Generate hydrofoil simulation animation: wave + vessel side-view.

Creates a GIF showing:
  - 2D side-view of ocean wave surface
  - Hydrofoil vessel riding on waves (hull + front/rear foils)
  - PD-only vs PD+Feedforward comparison (dual panel)
  - Heave acceleration time-series subplot

Output: docs/hydrofoil_wave_animation.gif
"""

import sys
import os
import numpy as np

# Add project root to path
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import FancyBboxPatch, Polygon
from matplotlib import patheffects

from hydrofoil.config import candela_c8, SimConfig
from hydrofoil.controller import PDControllerConfig, FeedforwardConfig
from hydrofoil.sim import run_simulation
from hydrofoil.disturbance import wave_elevation_at_point


def make_wave_components_bf3():
    """Create BF3 head-sea wave components (2 components)."""
    from wavesim.wave_generator import WaveComponent
    g = 9.81
    # BF3: Hs ~ 0.88-1.25m, Tp ~ 4-6s
    components = [
        WaveComponent(A=0.30, omega=1.20, kx=0.147, ky=0.0, phi=0.0, Q=0.0),
        WaveComponent(A=0.20, omega=1.60, kx=0.261, ky=0.0, phi=1.5, Q=0.0),
    ]
    return components


def draw_vessel(ax, x_cg, z_cg, theta, front_x, rear_x, front_z, rear_z,
                color='#2c3e50', label=None, ride_height=0.0):
    """Draw simplified hydrofoil vessel at given state.

    Args:
        ride_height: visual offset added to z_cg. Foil depths are measured
                     from the actual waterline (z_cg - ride_height), not
                     from the visually elevated CG.
    """
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)

    # Hull body (simplified rectangle, rotated around CG)
    hull_half_len = 2.8
    hull_half_h = 0.15

    # Hull corners in body frame
    corners_body = np.array([
        [-hull_half_len, -hull_half_h],
        [hull_half_len, -hull_half_h],
        [hull_half_len + 0.6, hull_half_h],  # bow taper
        [-hull_half_len - 0.3, hull_half_h],  # stern
    ])

    # Rotate and translate
    R = np.array([[cos_t, -sin_t], [sin_t, cos_t]])
    corners_global = (R @ corners_body.T).T + np.array([x_cg, z_cg])

    hull_patch = Polygon(corners_global, closed=True, facecolor=color,
                         edgecolor='black', linewidth=1.5, alpha=0.85,
                         zorder=5, label=label)
    ax.add_patch(hull_patch)

    # Total strut length: ride_height (visual offset) + foil depth below waterline
    # This ensures foils are drawn at the correct depth below the wave surface
    for foil_x_body, foil_z_depth, strut_color in [
        (front_x, front_z, '#e74c3c'),   # front foil - red
        (rear_x, rear_z, '#3498db'),      # rear foil - blue
    ]:
        total_depth = ride_height + foil_z_depth

        # Strut top (at hull bottom) in global frame
        strut_top_body = np.array([foil_x_body, -hull_half_h])
        strut_top = R @ strut_top_body + np.array([x_cg, z_cg])

        # Strut bottom (at foil depth below waterline) in global frame
        strut_bot_body = np.array([foil_x_body, -total_depth])
        strut_bot = R @ strut_bot_body + np.array([x_cg, z_cg])

        # Draw strut
        ax.plot([strut_top[0], strut_bot[0]], [strut_top[1], strut_bot[1]],
                color=strut_color, linewidth=2.5, zorder=6)

        # Draw foil (horizontal bar at bottom of strut)
        foil_hw = 0.35
        foil_left = R @ np.array([foil_x_body - foil_hw, -total_depth]) + np.array([x_cg, z_cg])
        foil_right = R @ np.array([foil_x_body + foil_hw, -total_depth]) + np.array([x_cg, z_cg])
        ax.plot([foil_left[0], foil_right[0]], [foil_left[1], foil_right[1]],
                color=strut_color, linewidth=4, zorder=6, solid_capstyle='round')


def main():
    print("Setting up hydrofoil simulations...")

    # Configuration
    cfg = candela_c8()
    cfg['sim'] = SimConfig(dt=0.01, duration=15.0, speed=11.3, depth=50.0)
    components = make_wave_components_bf3()

    # Head sea: waves travel in -x direction, vessel in +x
    vessel_heading = 0.0  # vessel goes +x
    # For head sea, wave kx > 0 means wave propagates in +x
    # We flip sign: use vessel_heading = pi to get head-sea encounter
    vessel_heading_head = np.pi

    # Run two simulations
    print("  Running PD-only simulation...")
    ctrl = PDControllerConfig()
    result_pd = run_simulation(cfg, components, vessel_heading=vessel_heading_head,
                               controller=ctrl, feedforward=None)

    print("  Running PD+FF simulation...")
    ff_cfg = FeedforwardConfig(preview_dt=0.1, K_ff=0.8, lp_alpha=0.05)
    result_ff = run_simulation(cfg, components, vessel_heading=vessel_heading_head,
                               controller=ctrl, feedforward=ff_cfg)

    # Compute aRMS
    t_start_idx = int(2.0 / cfg['sim'].dt)  # skip first 2s transient
    arms_pd = np.sqrt(np.mean(result_pd.heave_accel[t_start_idx:]**2))
    arms_ff = np.sqrt(np.mean(result_ff.heave_accel[t_start_idx:]**2))
    improvement = (1 - arms_ff / arms_pd) * 100
    print(f"  aRMS: PD={arms_pd:.3f}, PD+FF={arms_ff:.3f} ({improvement:.1f}% reduction)")

    # --- Animation setup ---
    print("Creating animation frames...")

    fig, axes = plt.subplots(2, 1, figsize=(14, 8),
                             gridspec_kw={'height_ratios': [3, 1]})
    fig.patch.set_facecolor('#f0f4f8')

    ax_wave = axes[0]
    ax_accel = axes[1]

    # Animation parameters
    fps = 15
    skip = int(1.0 / (cfg['sim'].dt * fps))  # frames to skip
    frame_indices = list(range(0, len(result_pd.time), skip))
    # Limit total frames for reasonable GIF size
    max_frames = 180  # 12 seconds at 15fps
    if len(frame_indices) > max_frames:
        frame_indices = frame_indices[:max_frames]

    # Visible window around vessel
    window_half = 25.0  # meters ahead/behind vessel

    front_x = cfg['front_foil'].x_pos
    rear_x = cfg['rear_foil'].x_pos
    front_z = cfg['front_foil'].z_pos
    rear_z = cfg['rear_foil'].z_pos

    def animate(frame_num):
        idx = frame_indices[frame_num]
        t = result_pd.time[idx]

        ax_wave.clear()
        ax_accel.clear()

        # --- Wave surface plot ---
        # Use PD vessel position as camera center
        x_center_pd = result_pd.x_pos[idx]
        x_center_ff = result_ff.x_pos[idx]
        x_center = 0.5 * (x_center_pd + x_center_ff)

        x_wave = np.linspace(x_center - window_half, x_center + window_half, 500)
        y_wave = 0.0
        eta_wave = np.array([wave_elevation_at_point(components, xi, y_wave, t)
                             for xi in x_wave])

        # Draw water body
        ax_wave.fill_between(x_wave, eta_wave, -2.0, color='#a8d8ea', alpha=0.4, zorder=1)
        ax_wave.axhline(y=0, color='#87ceeb', linestyle='--', alpha=0.3, linewidth=0.5)

        # Draw wave surface
        ax_wave.plot(x_wave, eta_wave, color='#2980b9', linewidth=2.0, zorder=3)
        ax_wave.fill_between(x_wave, eta_wave, eta_wave - 0.05,
                             color='#2980b9', alpha=0.3, zorder=2)

        # Foilborne ride height: hull rides above water surface
        # Front foil at 0.8m depth, hull clearance 0.3m → CG is ~0.5m above waterline
        # Add extra visual offset so hull is clearly above wave crests
        ride_height = 0.7  # CG height above calm waterline in foilborne

        # Draw PD-only vessel
        draw_vessel(ax_wave,
                    x_cg=x_center_pd,
                    z_cg=result_pd.z[idx] + ride_height,
                    theta=result_pd.theta[idx],
                    front_x=front_x, rear_x=rear_x,
                    front_z=front_z, rear_z=rear_z,
                    color='#e74c3c',
                    label='PD-only',
                    ride_height=ride_height)

        # Draw PD+FF vessel
        draw_vessel(ax_wave,
                    x_cg=x_center_ff,
                    z_cg=result_ff.z[idx] + ride_height,
                    theta=result_ff.theta[idx],
                    front_x=front_x, rear_x=rear_x,
                    front_z=front_z, rear_z=rear_z,
                    color='#27ae60',
                    label='PD + Feedforward',
                    ride_height=ride_height)

        ax_wave.set_xlim(x_center - window_half, x_center + window_half)
        ax_wave.set_ylim(-1.5, 2.0)
        ax_wave.set_ylabel('Elevation [m]', fontsize=11)
        ax_wave.set_title(f'Hydrofoil Wave Response — BF3 Head Sea  (t = {t:.1f}s)',
                          fontsize=13, fontweight='bold')
        ax_wave.legend(loc='upper right', fontsize=10, framealpha=0.9)
        ax_wave.set_facecolor('#f0f4f8')
        ax_wave.grid(True, alpha=0.2)

        # Direction arrows
        ax_wave.annotate('Vessel →', xy=(x_center + window_half * 0.7, 1.25),
                         fontsize=9, color='#555', ha='center')
        ax_wave.annotate('← Waves', xy=(x_center - window_half * 0.7, 1.25),
                         fontsize=9, color='#2980b9', ha='center')

        # --- Acceleration subplot ---
        t_plot = result_pd.time[:idx+1]
        ax_accel.plot(t_plot, result_pd.heave_accel[:idx+1],
                      color='#e74c3c', linewidth=1.0, alpha=0.8, label='PD-only')
        ax_accel.plot(t_plot, result_ff.heave_accel[:idx+1],
                      color='#27ae60', linewidth=1.0, alpha=0.8, label='PD+FF')

        ax_accel.set_xlim(0, cfg['sim'].duration)
        # Auto-fit y-axis to actual data range (skip transient) for visible difference
        accel_pd_window = result_pd.heave_accel[t_start_idx:]
        accel_ff_window = result_ff.heave_accel[t_start_idx:]
        accel_max = max(np.max(np.abs(accel_pd_window)), np.max(np.abs(accel_ff_window)))
        accel_lim = accel_max * 1.15
        ax_accel.set_ylim(-accel_lim, accel_lim)
        ax_accel.set_xlabel('Time [s]', fontsize=11)
        ax_accel.set_ylabel('Heave Accel [m/s²]', fontsize=11)
        ax_accel.legend(loc='upper right', fontsize=9, framealpha=0.9)
        ax_accel.set_facecolor('#f0f4f8')
        ax_accel.grid(True, alpha=0.2)

        # aRMS text
        if idx > t_start_idx:
            arms_pd_now = np.sqrt(np.mean(result_pd.heave_accel[t_start_idx:idx+1]**2))
            arms_ff_now = np.sqrt(np.mean(result_ff.heave_accel[t_start_idx:idx+1]**2))
            ax_accel.text(0.02, 0.92,
                          f'aRMS: PD={arms_pd_now:.2f}, FF={arms_ff_now:.2f} m/s²',
                          transform=ax_accel.transAxes, fontsize=9,
                          verticalalignment='top',
                          bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                                    alpha=0.8))

        fig.tight_layout(pad=1.5)

    # Create animation
    anim = FuncAnimation(fig, animate, frames=len(frame_indices),
                         interval=1000/fps, repeat=True)

    # Save as GIF
    out_dir = os.path.join(ROOT, 'docs')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'hydrofoil_wave_animation.gif')

    print(f"Saving animation to {out_path} ({len(frame_indices)} frames)...")
    writer = PillowWriter(fps=fps)
    anim.save(out_path, writer=writer, dpi=100)

    file_size_mb = os.path.getsize(out_path) / (1024 * 1024)
    print(f"Done! File size: {file_size_mb:.1f} MB")

    plt.close(fig)


if __name__ == '__main__':
    main()
