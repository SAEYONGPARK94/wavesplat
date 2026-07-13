"""
pipeline.py
End-to-end WaveSim pipeline: wave → LiDAR simulation → save episode.

Usage:
  # Single episode (CLI)
  python -m wavesim.pipeline --config configs/default.yaml --sea_state BF4

  # Full dataset (see generate_dataset.py)
"""

import numpy as np
import argparse
import os

from wavesim import mesh_adequacy
from wavesim.config import WaveSimConfig, load_config
from wavesim.wave_generator import jonswap_components, check_q_constraint
from wavesim.hos_simulator import HOSSimulator, HOSConfig
from wavesim.lidar_simulator import (
    LidarConfig, build_mesh, generate_rays, raycast_surface,
    compute_incidence_angles, interpolate_normals
)
from wavesim.raydrop_uniform import apply_dropout as apply_uniform
from wavesim.raydrop_slope import apply_dropout as apply_slope
from wavesim.raydrop_threshold import (apply_dropout_fixed_T as apply_fixed_T,
                                       T_MANUAL_50)
from wavesim.gt_generator import make_grid, evaluate_field, compute_spectrum_1d
from wavesim.noise_model import add_noise, compute_confidence
from wavesim.data_writer import save_episode


# ── Sea state presets (fallback if no config) ────────────────────────────────
SEA_STATE = {
    'BF2': dict(Hs=0.3, Tp=4.0),
    'BF3': dict(Hs=0.6, Tp=5.5),
    'BF4': dict(Hs=1.2, Tp=7.0),
    'BF5': dict(Hs=2.0, Tp=9.0),
}


def run_episode(sea_state: str = 'BF4',
                n_components: int = 30,
                n_pts: int = 50,
                seed: int = 42,
                raydrop_model: str = 'uniform',
                duration: float = 3.0,
                dt: float = 0.1,
                depth: float = 50.0,
                output_dir: str = 'data',
                verbose: bool = True,
                cfg: WaveSimConfig = None,
                direction_deg: float = 0.0,
                strict_mesh: bool = False) -> str:
    """
    Run one WaveSim episode and save to HDF5.

    If cfg is provided, LiDAR/raycast/mesh/GT settings are taken from it.
    Otherwise, defaults are used (backward compatible).

    Returns:
        filepath: path to saved .h5 file
    """
    if cfg is None:
        cfg = WaveSimConfig()

    rng = np.random.default_rng(seed)
    params = cfg.sea_states.get(sea_state, SEA_STATE.get(sea_state))
    Hs, Tp = params['Hs'], params['Tp']

    # ── 1. Generate wave components ──────────────────────────────────────────
    use_hos = cfg.hos.enabled
    components = jonswap_components(Hs=Hs, Tp=Tp, n_components=n_components,
                                     depth=depth, seed=seed,
                                     direction_deg=direction_deg)
    satisfied, q_val = check_q_constraint(components)
    assert satisfied, f"Q constraint violated: {q_val:.3f}"

    mesh_info = mesh_adequacy.check(
        components, cfg, hos_order=cfg.hos.M if cfg.hos.enabled else None)
    mesh_adequacy.enforce(mesh_info, strict=strict_mesh, verbose=verbose)

    hos_sim = None
    if use_hos:
        hos_cfg = HOSConfig(
            Lx=cfg.hos.Lx, Ly=cfg.hos.Ly,
            Nx=cfg.hos.Nx, Ny=cfg.hos.Ny,
            depth=depth, M=cfg.hos.M,
        )
        hos_sim = HOSSimulator(hos_cfg)
        hos_sim.initialize_jonswap(Hs=Hs, Tp=Tp, seed=seed)
        if verbose:
            print(f"[Wave] HOS M={cfg.hos.M}: Hs={Hs}m, Tp={Tp}s, "
                  f"grid={cfg.hos.Nx}×{cfg.hos.Ny}, "
                  f"Hs_init={hos_sim.get_Hs():.3f}m")
    else:
        if verbose:
            print(f"[Wave] {sea_state}: Hs={Hs}m, Tp={Tp}s, {n_components} components, "
                  f"Q-sum={q_val:.3f}")

    # ── 2. Setup grid & LiDAR from config ────────────────────────────────────
    x_range = cfg.gt_grid.x_range
    y_range = cfg.gt_grid.y_range
    X_gt, Y_gt = make_grid(x_range, y_range, nx=cfg.gt_grid.nx, ny=cfg.gt_grid.ny)

    lidar_cfg = LidarConfig(
        height=cfg.lidar.height,
        channels=cfg.lidar.channels,
        scan_rate=cfg.lidar.scan_rate,
        fov_h_deg=cfg.lidar.fov_h_deg,
        fov_v_min_deg=cfg.lidar.fov_v_min_deg,
        fov_v_max_deg=cfg.lidar.fov_v_max_deg,
        range_min=cfg.lidar.range_min,
        range_max=cfg.lidar.range_max,
        n_h_beams=cfg.lidar.n_h_beams,
    )

    # ── 3. Simulate frames ───────────────────────────────────────────────────
    timestamps = np.arange(0.0, duration + 1e-9, dt)
    frames_data = []

    # Pre-generate rays (same for all frames since ship is stationary)
    origins, ray_dirs = generate_rays(lidar_cfg, ship_x=0.0, ship_y=0.0)

    hos_t_current = 0.0  # track HOS internal time
    for t in timestamps:
        # Evaluate wave surface on dense grid
        if use_hos:
            # Advance HOS to time t
            if t > hos_t_current + 1e-10:
                hos_dt = cfg.hos.dt
                while hos_t_current < t - 1e-10:
                    step_dt = min(hos_dt, t - hos_t_current)
                    hos_sim.step(step_dt)
                    hos_t_current += step_dt
            Z_gt, NX_gt, NY_gt, NZ_gt = hos_sim.evaluate_at(X_gt, Y_gt)
        else:
            Z_gt, NX_gt, NY_gt, NZ_gt = evaluate_field(components, X_gt, Y_gt, t)

        # Build mesh for raycasting
        X_mesh, Y_mesh = make_grid(x_range, y_range,
                                    nx=cfg.mesh.nx, ny=cfg.mesh.ny)
        if use_hos:
            Z_mesh, _, _, _ = hos_sim.evaluate_at(X_mesh, Y_mesh)
        else:
            Z_mesh, _, _, _ = evaluate_field(components, X_mesh, Y_mesh, t)
        verts, tris = build_mesh(X_mesh, Y_mesh, Z_mesh)

        # Raycast
        hit_mask, hit_pts, hit_dist = raycast_surface(
            origins, ray_dirs, verts, tris,
            range_min=lidar_cfg.range_min,
            range_max=lidar_cfg.range_max,
            batch_size=cfg.raycast.batch_size,
            spatial_grid_nx=cfg.raycast.spatial_grid_nx,
            spatial_grid_ny=cfg.raycast.spatial_grid_ny,
        )

        if not np.any(hit_mask):
            if verbose:
                print(f"  [t={t:.1f}] No hits — skipping frame")
            continue

        # Interpolate surface normals at hit points
        hit_idx = np.where(hit_mask)[0]
        hit_normals = interpolate_normals(hit_pts[hit_idx], X_gt, Y_gt,
                                           NX_gt, NY_gt, NZ_gt)

        # Incidence angles
        incidence, _ = compute_incidence_angles(
            hit_pts[hit_idx], ray_dirs[hit_idx], hit_normals
        )

        # Raydrop model
        if raydrop_model == 'uniform':
            return_mask_rel = apply_uniform(
                np.ones(len(hit_idx), dtype=bool),
                hit_pts[hit_idx], hit_dist[hit_idx],
                incidence, n_pts, rng
            )
        elif raydrop_model == 'fixedT':
            # the sensor's actual detection law (F2 verdict; H3-OD 2026-07-11):
            # fixed manual-calibrated threshold, count is a measurement.
            # n_pts is reused as the storage cap per frame (0 = uncapped).
            return_mask_rel = apply_fixed_T(
                np.ones(len(hit_idx), dtype=bool),
                hit_dist[hit_idx], incidence, rng, cap=n_pts
            )
        else:
            return_mask_rel = apply_slope(
                np.ones(len(hit_idx), dtype=bool),
                hit_pts[hit_idx], hit_dist[hit_idx],
                ray_dirs[hit_idx], hit_normals,
                incidence, n_pts, rng
            )

        final_idx = hit_idx[return_mask_rel]
        if len(final_idx) == 0:
            continue

        # Add sensor noise
        noisy_pts = add_noise(hit_pts[final_idx], ray_dirs[final_idx], rng=rng)

        # Confidence weights
        conf = compute_confidence(incidence[return_mask_rel],
                                   hit_dist[final_idx])

        frames_data.append({
            'timestamp': t,
            'lidar_points': noisy_pts,
            'lidar_valid_mask': np.ones(len(final_idx), dtype=bool),
            'lidar_incidence_angles': incidence[return_mask_rel],
            'lidar_confidence': conf,
            'gt_wave_field': Z_gt,
        })

        if verbose:
            print(f"  [t={t:.1f}s] {len(final_idx)} LiDAR points, "
                  f"GT Hs={4*np.std(Z_gt):.3f}m")

    if not frames_data:
        raise RuntimeError("No valid frames generated. Check raydrop parameters.")

    # ── 4. Compute spectrum and save ─────────────────────────────────────────
    spectrum = compute_spectrum_1d(components, n_freq=cfg.episode.n_freq)
    gt_x = X_gt[0, :]
    gt_y = Y_gt[:, 0]

    os.makedirs(output_dir, exist_ok=True)
    dtag = '' if abs(direction_deg) < 1e-9 else f"_dir{int(direction_deg)}"
    fname = (f"episode_{sea_state}_{n_components}comp_seed{seed}"
             f"_{raydrop_model}{dtag}.h5")
    filepath = os.path.join(output_dir, fname)

    save_episode(
        filepath, components, lidar_cfg, frames_data,
        gt_x, gt_y,
        raydrop_model_type=raydrop_model,
        raydrop_params=({'T': T_MANUAL_50, 'cap': n_pts}
                        if raydrop_model == 'fixedT' else {'n_pts': n_pts}),
        sea_state=sea_state, Hs=Hs, Tp=Tp, gamma=3.3,
        depth=depth, spectrum_1d=spectrum, mesh_info=mesh_info
    )

    if verbose:
        actual_pts = [len(f['lidar_points']) for f in frames_data]
        print(f"\n[Saved] {filepath}")
        print(f"  Frames: {len(frames_data)}, "
              f"Pts/frame: {np.mean(actual_pts):.1f}±{np.std(actual_pts):.1f} "
              f"(target={n_pts}), "
              f"Grid: {X_gt.shape}")

    return filepath


def main():
    parser = argparse.ArgumentParser(description='WaveSim pipeline')
    parser.add_argument('--config', type=str, default=None,
                        help='Path to YAML config file')
    parser.add_argument('--sea_state', default='BF4',
                        choices=['BF2', 'BF3', 'BF4', 'BF5'])
    parser.add_argument('--n_components', type=int, default=30)
    parser.add_argument('--n_pts', type=int, default=None)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--raydrop', default='uniform',
                        choices=['uniform', 'slope', 'fixedT'])
    parser.add_argument('--direction', type=float, default=0.0,
                        help='mean wave direction [deg from +x]; 180 = head '
                             'seas (waves toward the sensor, causal encounter)')
    parser.add_argument('--duration', type=float, default=None)
    parser.add_argument('--dt', type=float, default=None)
    parser.add_argument('--output_dir', default=None)
    parser.add_argument('--strict_mesh', action='store_true',
                        help='fail instead of warn when the raycast mesh '
                             'cannot represent the sea (sub-Nyquist)')
    args = parser.parse_args()

    cfg = load_config(args.config)

    # CLI args override config
    duration = args.duration or cfg.episode.duration
    dt = args.dt or cfg.episode.dt
    n_pts = args.n_pts or cfg.episode.n_pts
    output_dir = args.output_dir or cfg.dataset.output_dir

    run_episode(
        sea_state=args.sea_state,
        n_components=args.n_components,
        n_pts=n_pts,
        seed=args.seed,
        raydrop_model=args.raydrop,
        duration=duration,
        dt=dt,
        depth=cfg.episode.depth,
        output_dir=output_dir,
        verbose=True,
        cfg=cfg,
        direction_deg=args.direction,
        strict_mesh=args.strict_mesh,
    )


if __name__ == '__main__':
    main()
