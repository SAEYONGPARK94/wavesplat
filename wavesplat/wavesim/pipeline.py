"""
pipeline.py
End-to-end WaveSim pipeline: wave → LiDAR simulation → save episode.

Usage:
  python pipeline.py --sea_state BF4 --n_components 30 --n_pts 50 --seed 42
"""

import numpy as np
import argparse
import os

from wavesim.wave_generator import jonswap_components, check_q_constraint
from wavesim.lidar_simulator import (
    LidarConfig, build_mesh, generate_rays, raycast_surface,
    compute_incidence_angles, interpolate_normals
)
from wavesim.raydrop_uniform import apply_dropout as apply_uniform
from wavesim.raydrop_slope import apply_dropout as apply_slope
from wavesim.gt_generator import make_grid, evaluate_field, compute_spectrum_1d
from wavesim.noise_model import add_noise, compute_confidence
from wavesim.data_writer import save_episode


# ── Sea state presets ─────────────────────────────────────────────────────────
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
                test_resolution: bool = True,
                mesh_nx: int = 40,
                mesh_ny: int = 30) -> str:
    """
    Run one WaveSim episode and save to HDF5.

    Sensor: Seyond Falcon K1 (1550nm, 120°×25° FOV, 152ch, 10FPS)
      test_resolution=True  → n_h_beams=500  (fast, for development)
      test_resolution=False → n_h_beams=2000 (full 0.06° H-resolution)

    Returns:
        filepath: path to saved .h5 file
    """
    rng = np.random.default_rng(seed)
    params = SEA_STATE[sea_state]
    Hs, Tp = params['Hs'], params['Tp']

    # ── 1. Generate wave components ──────────────────────────────────────────
    components = jonswap_components(Hs=Hs, Tp=Tp, n_components=n_components,
                                     depth=depth, seed=seed)
    satisfied, q_val = check_q_constraint(components)
    assert satisfied, f"Q constraint violated: {q_val:.3f}"
    if verbose:
        print(f"[Wave] {sea_state}: Hs={Hs}m, Tp={Tp}s, {n_components} components, "
              f"Q-sum={q_val:.3f}")

    # ── 2. Setup grid & LiDAR ────────────────────────────────────────────────
    # Grid: forward-looking region, 10-250m in front of ship
    x_range = (10.0, 250.0)
    y_range = (-100.0, 100.0)
    X_gt, Y_gt = make_grid(x_range, y_range, nx=80, ny=60)

    # Seyond Falcon K1: 120°×25° FOV, 152ch, 1550nm, range 1.5~500m (250m@10%)
    # V FOV -25°~0°: covers ~11m (near) to horizon; ships sensor at 5m height
    n_h_beams = 500 if test_resolution else 2000
    lidar_cfg = LidarConfig(
        height=5.0, channels=152, scan_rate=10.0,
        fov_h_deg=120.0, fov_v_min_deg=-25.0, fov_v_max_deg=0.0,
        range_min=1.5, range_max=250.0, n_h_beams=n_h_beams
    )

    # ── 3. Simulate frames ───────────────────────────────────────────────────
    timestamps = np.arange(0.0, duration + 1e-9, dt)
    frames_data = []

    for t in timestamps:
        # Evaluate wave surface on dense grid
        Z_gt, NX_gt, NY_gt, NZ_gt = evaluate_field(components, X_gt, Y_gt, t)

        # Build mesh for raycasting (coarser than GT grid for speed)
        X_mesh, Y_mesh = make_grid(x_range, y_range, nx=mesh_nx, ny=mesh_ny)
        Z_mesh, _, _, _ = evaluate_field(components, X_mesh, Y_mesh, t)
        verts, tris = build_mesh(X_mesh, Y_mesh, Z_mesh)

        # Generate rays
        origins, ray_dirs = generate_rays(lidar_cfg, ship_x=0.0, ship_y=0.0)

        # Raycast
        hit_mask, hit_pts, hit_dist = raycast_surface(
            origins, ray_dirs, verts, tris,
            range_min=lidar_cfg.range_min,
            range_max=lidar_cfg.range_max
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
    spectrum = compute_spectrum_1d(components, n_freq=64)
    gt_x = X_gt[0, :]
    gt_y = Y_gt[:, 0]

    fname = f"episode_{sea_state}_{n_components}comp_seed{seed}_{raydrop_model}.h5"
    filepath = os.path.join(output_dir, fname)

    save_episode(
        filepath, components, lidar_cfg, frames_data,
        gt_x, gt_y,
        raydrop_model_type=raydrop_model,
        raydrop_params={'n_pts': n_pts},
        sea_state=sea_state, Hs=Hs, Tp=Tp, gamma=3.3,
        depth=depth, spectrum_1d=spectrum
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
    parser.add_argument('--sea_state', default='BF4', choices=list(SEA_STATE.keys()))
    parser.add_argument('--n_components', type=int, default=30)
    parser.add_argument('--n_pts', type=int, default=50)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--raydrop', default='uniform', choices=['uniform', 'slope'])
    parser.add_argument('--duration', type=float, default=3.0)
    parser.add_argument('--dt', type=float, default=0.1)
    parser.add_argument('--output_dir', default='data')
    parser.add_argument('--full_resolution', action='store_true',
                        help='Use full Falcon K1 resolution (n_h_beams=2000). '
                             'Default: test mode (n_h_beams=500).')
    args = parser.parse_args()

    run_episode(
        sea_state=args.sea_state,
        n_components=args.n_components,
        n_pts=args.n_pts,
        seed=args.seed,
        raydrop_model=args.raydrop,
        duration=args.duration,
        dt=args.dt,
        output_dir=args.output_dir,
        verbose=True,
        test_resolution=not args.full_resolution,
    )


if __name__ == '__main__':
    main()
