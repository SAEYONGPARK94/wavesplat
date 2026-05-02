"""
lidar_simulator.py
Forward-looking LiDAR simulation on a Gerstner wave surface.

Physics:
  - Beam pattern: elevation angle range based on sensor height & range
  - Ray-mesh intersection via Möller-Trumbore (Möller & Trumbore 1997)
  - Surface normal at hit point
  - Grazing/incidence angle computation for the Fresnel raydrop model
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Tuple


@dataclass
class LidarConfig:
    height: float = 5.0          # sensor height above calm sea [m]
    channels: int = 32           # number of vertical channels
    scan_rate: float = 10.0      # scan rate [Hz]
    fov_h_deg: float = 120.0     # horizontal FOV [degrees]
    fov_v_min_deg: float = -5.0  # minimum vertical angle (below horizontal) [degrees]
    fov_v_max_deg: float = 5.0   # maximum vertical angle [degrees]
    range_min: float = 10.0      # minimum detection range [m]
    range_max: float = 200.0     # maximum detection range [m]
    n_h_beams: int = 360         # number of horizontal beams per scan


def build_mesh(X: np.ndarray, Y: np.ndarray, Z: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert wave surface grid to triangle mesh.
    Grid: (Ny, Nx) arrays X, Y, Z.

    Returns:
        vertices: (N_v, 3) array of vertex positions
        triangles: (N_t, 3) array of vertex indices
    """
    Ny, Nx = Z.shape
    # Flatten vertices
    vx = X.ravel()
    vy = Y.ravel()
    vz = Z.ravel()
    vertices = np.column_stack([vx, vy, vz])

    # Generate triangle indices (2 triangles per grid cell)
    idx = np.arange(Ny * Nx).reshape(Ny, Nx)
    i0 = idx[:-1, :-1].ravel()
    i1 = idx[:-1, 1:].ravel()
    i2 = idx[1:, :-1].ravel()
    i3 = idx[1:, 1:].ravel()

    tri1 = np.column_stack([i0, i1, i2])  # lower-left triangle
    tri2 = np.column_stack([i1, i3, i2])  # upper-right triangle
    triangles = np.vstack([tri1, tri2])

    return vertices, triangles


def moller_trumbore(ray_origin: np.ndarray, ray_dir: np.ndarray,
                     v0: np.ndarray, v1: np.ndarray, v2: np.ndarray,
                     eps: float = 1e-9) -> Tuple[bool, float]:
    """
    Möller-Trumbore ray-triangle intersection test.
    (Möller & Trumbore 1997)

    Args:
        ray_origin: (3,) origin of ray
        ray_dir:    (3,) direction of ray (need not be unit)
        v0, v1, v2: (3,) triangle vertices

    Returns:
        hit: True if intersection found
        t:   ray parameter at intersection (ray_origin + t*ray_dir)
    """
    edge1 = v1 - v0
    edge2 = v2 - v0
    h = np.cross(ray_dir, edge2)
    det = np.dot(edge1, h)

    if abs(det) < eps:
        return False, np.inf

    inv_det = 1.0 / det
    s = ray_origin - v0
    u = inv_det * np.dot(s, h)
    if u < 0.0 or u > 1.0:
        return False, np.inf

    q = np.cross(s, edge1)
    v = inv_det * np.dot(ray_dir, q)
    if v < 0.0 or (u + v) > 1.0:
        return False, np.inf

    t = inv_det * np.dot(edge2, q)
    if t < eps:
        return False, np.inf

    return True, t


def _build_spatial_grid(vertices, triangles, nx_grid, ny_grid):
    """Build 2D spatial grid mapping (x,y) cells → triangle indices."""
    v0 = vertices[triangles[:, 0]]
    v1 = vertices[triangles[:, 1]]
    v2 = vertices[triangles[:, 2]]

    # Bounding box of each triangle in (x,y)
    tri_x_min = np.minimum(np.minimum(v0[:, 0], v1[:, 0]), v2[:, 0])
    tri_x_max = np.maximum(np.maximum(v0[:, 0], v1[:, 0]), v2[:, 0])
    tri_y_min = np.minimum(np.minimum(v0[:, 1], v1[:, 1]), v2[:, 1])
    tri_y_max = np.maximum(np.maximum(v0[:, 1], v1[:, 1]), v2[:, 1])

    x_lo, x_hi = vertices[:, 0].min(), vertices[:, 0].max()
    y_lo, y_hi = vertices[:, 1].min(), vertices[:, 1].max()
    dx = (x_hi - x_lo) / nx_grid
    dy = (y_hi - y_lo) / ny_grid

    # Assign triangles to cells
    grid = [[[] for _ in range(ny_grid)] for _ in range(nx_grid)]
    for ti in range(len(triangles)):
        ix0 = max(0, int((tri_x_min[ti] - x_lo) / dx))
        ix1 = min(nx_grid - 1, int((tri_x_max[ti] - x_lo) / dx))
        iy0 = max(0, int((tri_y_min[ti] - y_lo) / dy))
        iy1 = min(ny_grid - 1, int((tri_y_max[ti] - y_lo) / dy))
        for ix in range(ix0, ix1 + 1):
            for iy in range(iy0, iy1 + 1):
                grid[ix][iy].append(ti)

    return grid, x_lo, x_hi, y_lo, y_hi, dx, dy


def raycast_surface(ray_origins: np.ndarray, ray_dirs: np.ndarray,
                    vertices: np.ndarray, triangles: np.ndarray,
                    range_min: float = 5.0,
                    range_max: float = 250.0,
                    batch_size: int = 4000,
                    spatial_grid_nx: int = 20,
                    spatial_grid_ny: int = 15) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Cast multiple rays against triangle mesh.
    Uses z=0 plane approximation for spatial pre-filtering, then batched
    Möller-Trumbore on candidate triangles only.

    Args:
        ray_origins: (N_rays, 3)
        ray_dirs:    (N_rays, 3) unit direction vectors
        vertices:    (N_v, 3)
        triangles:   (N_t, 3) vertex index triplets
        batch_size:  number of rays per batch (tune for memory)
        spatial_grid_nx: spatial hash grid cells in x
        spatial_grid_ny: spatial hash grid cells in y

    Returns:
        hit_mask:    (N_rays,) bool
        hit_points:  (N_rays, 3) world coordinates (NaN where no hit)
        hit_dist:    (N_rays,) distance to hit (inf where no hit)
    """
    N_rays = ray_origins.shape[0]

    # Pre-filter: only downward rays can hit water surface
    down_mask = ray_dirs[:, 2] < -1e-6
    down_idx = np.where(down_mask)[0]

    hit_mask = np.zeros(N_rays, dtype=bool)
    hit_points = np.full((N_rays, 3), np.nan)
    hit_dist = np.full(N_rays, np.inf)

    if len(down_idx) == 0:
        return hit_mask, hit_points, hit_dist

    # Precompute triangle data
    v0_all = vertices[triangles[:, 0]]
    v1_all = vertices[triangles[:, 1]]
    v2_all = vertices[triangles[:, 2]]
    e1_all = v1_all - v0_all
    e2_all = v2_all - v0_all

    # Build spatial grid (2D grid over mesh extent)
    nx_g, ny_g = spatial_grid_nx, spatial_grid_ny
    grid, x_lo, x_hi, y_lo, y_hi, dx, dy = _build_spatial_grid(
        vertices, triangles, nx_g, ny_g)

    # For each downward ray, find approximate (x,y) hit on z=0 plane
    ro_down = ray_origins[down_idx]
    rd_down = ray_dirs[down_idx]
    t_flat = -ro_down[:, 2] / rd_down[:, 2]
    xy_approx = ro_down[:, :2] + t_flat[:, np.newaxis] * rd_down[:, :2]

    # Map to grid cells (with margin ±1 cell for wave height offset)
    ix_center = np.clip(((xy_approx[:, 0] - x_lo) / dx).astype(int), 0, nx_g - 1)
    iy_center = np.clip(((xy_approx[:, 1] - y_lo) / dy).astype(int), 0, ny_g - 1)

    # Group rays by grid cell for efficient batch processing
    from collections import defaultdict
    cell_rays = defaultdict(list)  # (ix, iy) → list of indices into down_idx
    for k in range(len(down_idx)):
        cell_rays[(ix_center[k], iy_center[k])].append(k)

    # Process each cell group
    for (ix, iy), ray_local_indices in cell_rays.items():
        # Gather candidate triangles from this cell and neighbors
        candidate_set = set()
        for dix in range(-1, 2):
            for diy in range(-1, 2):
                nix, niy = ix + dix, iy + diy
                if 0 <= nix < nx_g and 0 <= niy < ny_g:
                    candidate_set.update(grid[nix][niy])
        if not candidate_set:
            continue

        tri_idx = np.array(sorted(candidate_set))
        v0 = v0_all[tri_idx]
        v1 = v1_all[tri_idx]
        e1 = e1_all[tri_idx]
        e2 = e2_all[tri_idx]
        N_tri = len(tri_idx)

        # Process rays in batches
        ray_local_indices = np.array(ray_local_indices)
        for bstart in range(0, len(ray_local_indices), batch_size):
            b_local = ray_local_indices[bstart:bstart + batch_size]
            b_global = down_idx[b_local]
            B = len(b_local)

            ro = ro_down[b_local]
            rd = rd_down[b_local]

            # Batched Möller-Trumbore: (B, N_tri)
            rd_exp = rd[:, np.newaxis, :]
            e2_exp = e2[np.newaxis, :, :]
            e1_exp = e1[np.newaxis, :, :]
            v0_exp = v0[np.newaxis, :, :]

            h = np.cross(rd_exp, e2_exp)
            det = np.einsum('btj,btj->bt',
                            np.broadcast_to(e1_exp, (B, N_tri, 3)), h)

            valid = np.abs(det) > 1e-9
            safe_det = np.where(valid, det, 1.0)
            inv_det = np.where(valid, 1.0 / safe_det, 0.0)

            s = ro[:, np.newaxis, :] - v0_exp
            u = inv_det * np.einsum('btj,btj->bt', s, h)
            valid &= (u >= 0.0) & (u <= 1.0)

            q = np.cross(s, np.broadcast_to(e1_exp, (B, N_tri, 3)))
            v_param = inv_det * np.einsum('btj,btj->bt',
                                           np.broadcast_to(rd_exp, (B, N_tri, 3)), q)
            valid &= (v_param >= 0.0) & (u + v_param <= 1.0)

            t_param = inv_det * np.einsum('btj,btj->bt',
                                           np.broadcast_to(e2_exp, (B, N_tri, 3)), q)
            valid &= (t_param > 1e-6)
            valid &= (t_param >= range_min) & (t_param <= range_max)

            t_candidates = np.where(valid, t_param, np.inf)
            min_t = np.min(t_candidates, axis=1)
            has_hit = np.isfinite(min_t)

            hit_rays = b_global[has_hit]
            hit_t = min_t[has_hit]

            # Keep closest hit (ray may appear in multiple cells)
            better = hit_t < hit_dist[hit_rays]
            update = hit_rays[better]
            hit_mask[update] = True
            hit_dist[update] = hit_t[better]
            hit_points[update] = (ro[has_hit][better]
                                  + hit_t[better, np.newaxis] * rd[has_hit][better])

    return hit_mask, hit_points, hit_dist


def generate_rays(lidar_cfg: LidarConfig,
                  ship_x: float = 0.0,
                  ship_y: float = 0.0,
                  ship_heading_deg: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate LiDAR ray origins and unit direction vectors.

    Args:
        lidar_cfg: LiDAR configuration
        ship_x, ship_y: ship position [m]
        ship_heading_deg: ship heading [degrees from +x]

    Returns:
        origins: (N_rays, 3) all equal to sensor position
        dirs:    (N_rays, 3) unit direction vectors
    """
    # Sensor position
    sensor_pos = np.array([ship_x, ship_y, lidar_cfg.height])

    # Vertical angles (elevation): positive = above horizontal
    elev_angles = np.linspace(lidar_cfg.fov_v_min_deg, lidar_cfg.fov_v_max_deg,
                               lidar_cfg.channels)  # degrees

    # Horizontal angles: centered on forward direction (heading)
    # FOV centered on ship heading
    heading_rad = np.deg2rad(ship_heading_deg)
    az_half = lidar_cfg.fov_h_deg / 2.0
    az_angles = np.linspace(-az_half, az_half, lidar_cfg.n_h_beams)  # relative to heading

    # Build ray directions
    dirs = []
    for az_rel in az_angles:
        az = np.deg2rad(ship_heading_deg + az_rel)
        for el in np.deg2rad(elev_angles):
            # Ray direction in world frame
            dx = np.cos(el) * np.cos(az)
            dy = np.cos(el) * np.sin(az)
            dz = np.sin(el)  # positive = upward, negative = toward water
            dirs.append([dx, dy, dz])

    dirs = np.array(dirs)  # (N_rays, 3)
    origins = np.tile(sensor_pos, (dirs.shape[0], 1))

    return origins, dirs


def compute_incidence_angles(hit_points: np.ndarray,
                              ray_dirs: np.ndarray,
                              surface_normals: np.ndarray) -> np.ndarray:
    """
    Compute incidence angle θ_i between incoming ray and surface normal.
    θ_i = arccos(|ray_dir · normal|)   (angle between ray and normal)
    Grazing angle θ_g = π/2 - θ_i

    Args:
        hit_points:      (N, 3) — not used directly, kept for interface
        ray_dirs:        (N, 3) unit vectors pointing FROM sensor TO surface
        surface_normals: (N, 3) unit outward normal at hit point

    Returns:
        incidence_angles: (N,) in radians  [0 = normal incidence]
        grazing_angles:   (N,) in radians  [0 = grazing]
    """
    # ray_dir points toward surface; normal points up
    # cos(θ_i) = -dot(ray_dir, normal) for ray going INTO surface
    dot = np.einsum('ij,ij->i', ray_dirs, surface_normals)
    cos_theta = np.clip(-dot, 0.0, 1.0)
    incidence_angles = np.arccos(cos_theta)
    grazing_angles = np.pi / 2.0 - incidence_angles
    return incidence_angles, grazing_angles


def interpolate_normals(hit_points: np.ndarray,
                         X: np.ndarray, Y: np.ndarray,
                         NX: np.ndarray, NY: np.ndarray, NZ: np.ndarray) -> np.ndarray:
    """
    Bilinear interpolation of surface normals at hit points.

    Args:
        hit_points: (N, 3)
        X, Y: (Ny, Nx) grid coordinates
        NX, NY, NZ: (Ny, Nx) normal components

    Returns:
        normals: (N, 3) interpolated unit normals
    """
    from scipy.interpolate import RegularGridInterpolator

    x1d = X[0, :]
    y1d = Y[:, 0]

    interp_nx = RegularGridInterpolator((y1d, x1d), NX, bounds_error=False, fill_value=0.0)
    interp_ny = RegularGridInterpolator((y1d, x1d), NY, bounds_error=False, fill_value=0.0)
    interp_nz = RegularGridInterpolator((y1d, x1d), NZ, bounds_error=False, fill_value=1.0)

    pts_yx = hit_points[:, [1, 0]]  # (y, x) order for RegularGridInterpolator
    nx = interp_nx(pts_yx)
    ny = interp_ny(pts_yx)
    nz = interp_nz(pts_yx)

    normals = np.column_stack([nx, ny, nz])
    # Re-normalize
    mag = np.linalg.norm(normals, axis=1, keepdims=True)
    normals = normals / (mag + 1e-12)
    return normals
