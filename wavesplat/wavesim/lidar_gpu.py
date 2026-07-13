"""GPU Möller-Trumbore, hit-for-hit identical to lidar_simulator.raycast_surface.

("hit-for-hit", not "bit-for-bit": the same rays hit the same triangles, but
float64 reassociation on the GPU moves the intersection coordinates by ~1e-13 m.)

Rendering an adequate mesh (180x150 = 53k triangles) costs 13.9 s/frame on the
CPU -- 70 minutes per episode, 40 hours for the 35-episode E-3 stage.  That is
the only thing standing between us and the re-judgment, so the intersection test
moves to the GPU.

What is deliberately NOT changed: the spatial hash, the z=0 cell lookup, the
+-1 cell margin, the candidate set, the acceptance thresholds, the closest-hit
rule.  Those choices decide *which* triangles a ray is tested against and are
therefore part of the observation operator -- touching them would change the
data.  Only the arithmetic moves.  Verified against the CPU path in
wavesim/tests/test_lidar_gpu.py.
"""
import numpy as np
import torch

from wavesim.lidar_simulator import _build_spatial_grid


def raycast_surface_gpu(ray_origins, ray_dirs, vertices, triangles,
                        range_min=5.0, range_max=250.0, batch_size=4096,
                        spatial_grid_nx=20, spatial_grid_ny=15,
                        device=None, dtype=torch.float64):
    N_rays = ray_origins.shape[0]
    dev = device or ('cuda' if torch.cuda.is_available() else 'cpu')

    down_mask = ray_dirs[:, 2] < -1e-6
    down_idx = np.where(down_mask)[0]

    hit_mask = np.zeros(N_rays, dtype=bool)
    hit_points = np.full((N_rays, 3), np.nan)
    hit_dist = np.full(N_rays, np.inf)
    if len(down_idx) == 0:
        return hit_mask, hit_points, hit_dist

    v0_all = vertices[triangles[:, 0]]
    v1_all = vertices[triangles[:, 1]]
    v2_all = vertices[triangles[:, 2]]
    e1_all = torch.tensor(v1_all - v0_all, device=dev, dtype=dtype)
    e2_all = torch.tensor(v2_all - v0_all, device=dev, dtype=dtype)
    v0_t = torch.tensor(v0_all, device=dev, dtype=dtype)

    grid, x_lo, x_hi, y_lo, y_hi, dx, dy = _build_spatial_grid(
        vertices, triangles, spatial_grid_nx, spatial_grid_ny)

    ro_down = ray_origins[down_idx]
    rd_down = ray_dirs[down_idx]
    t_flat = -ro_down[:, 2] / rd_down[:, 2]
    xy_approx = ro_down[:, :2] + t_flat[:, np.newaxis] * rd_down[:, :2]

    ix_c = np.clip(((xy_approx[:, 0] - x_lo) / dx).astype(int), 0,
                   spatial_grid_nx - 1)
    iy_c = np.clip(((xy_approx[:, 1] - y_lo) / dy).astype(int), 0,
                   spatial_grid_ny - 1)

    from collections import defaultdict
    cell_rays = defaultdict(list)
    for k in range(len(down_idx)):
        cell_rays[(ix_c[k], iy_c[k])].append(k)

    ro_t = torch.tensor(ro_down, device=dev, dtype=dtype)
    rd_t = torch.tensor(rd_down, device=dev, dtype=dtype)
    dist_t = torch.full((N_rays,), float('inf'), device=dev, dtype=dtype)
    mask_t = torch.zeros(N_rays, dtype=torch.bool, device=dev)
    pts_t = torch.full((N_rays, 3), float('nan'), device=dev, dtype=dtype)
    gidx_all = torch.tensor(down_idx, device=dev, dtype=torch.long)

    for (ix, iy), ray_local in cell_rays.items():
        cand = set()
        for dix in (-1, 0, 1):
            for diy in (-1, 0, 1):
                nix, niy = ix + dix, iy + diy
                if 0 <= nix < spatial_grid_nx and 0 <= niy < spatial_grid_ny:
                    cand.update(grid[nix][niy])
        if not cand:
            continue

        tri = torch.tensor(np.array(sorted(cand)), device=dev, dtype=torch.long)
        v0 = v0_t[tri]                                  # (T,3)
        e1 = e1_all[tri]
        e2 = e2_all[tri]

        loc = torch.tensor(np.asarray(ray_local), device=dev, dtype=torch.long)
        for b0 in range(0, len(loc), batch_size):
            bl = loc[b0:b0 + batch_size]
            ro = ro_t[bl]                               # (B,3)
            rd = rd_t[bl]

            h = torch.cross(rd[:, None, :].expand(-1, len(tri), -1),
                            e2[None, :, :].expand(len(bl), -1, -1), dim=-1)
            det = (e1[None, :, :] * h).sum(-1)

            valid = det.abs() > 1e-9
            inv_det = torch.where(valid, 1.0 / torch.where(valid, det,
                                                           torch.ones_like(det)),
                                  torch.zeros_like(det))

            s = ro[:, None, :] - v0[None, :, :]
            u = inv_det * (s * h).sum(-1)
            valid &= (u >= 0.0) & (u <= 1.0)

            q = torch.cross(s, e1[None, :, :].expand(len(bl), -1, -1), dim=-1)
            vp = inv_det * (rd[:, None, :] * q).sum(-1)
            valid &= (vp >= 0.0) & (u + vp <= 1.0)

            tp = inv_det * (e2[None, :, :] * q).sum(-1)
            valid &= (tp > 1e-6) & (tp >= range_min) & (tp <= range_max)

            tc = torch.where(valid, tp, torch.full_like(tp, float('inf')))
            min_t, _ = tc.min(dim=1)
            has = torch.isfinite(min_t)
            if not has.any():
                continue

            g = gidx_all[bl][has]
            th = min_t[has]
            better = th < dist_t[g]
            upd = g[better]
            dist_t[upd] = th[better]
            mask_t[upd] = True
            pts_t[upd] = ro[has][better] + th[better, None] * rd[has][better]

    hit_mask = mask_t.cpu().numpy()
    hit_dist = dist_t.cpu().numpy()
    hit_points = pts_t.cpu().numpy()
    return hit_mask, hit_points, hit_dist
