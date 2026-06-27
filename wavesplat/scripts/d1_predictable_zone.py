"""
d1_predictable_zone.py — Direction I, part 2: the CRB predictable zone.

Given LiDAR observations over a footprint during [0, t_now], how far ahead in
SPACE and TIME can the wave field be reconstructed? We compute the pointwise
field CRB

    σ_z(x,y,t*)² = g(x,y,t*)ᵀ F⁻¹ g(x,y,t*),   g_a = ∂z/∂θ_a

where F is the Fisher information from the observation window. The PREDICTABLE
ZONE is {(x,y) : σ_z/Hs < τ} at forecast time t* = t_now + Δt.

Physical hypothesis: information rides the group-velocity cone — energy seen at
the footprint can only constrain the field within the region reachable by c_g·Δt
downstream. We overlay the geometric c_g-cone (footprint advected by the
component group velocities) on the CRB map to test this.

Reuses the FIM machinery in d1_crb.py. float64 on GPU.
"""

import os
import sys
import json
import math
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from d1_crb import (DEVICE, DT, jacobian, fisher, cg_of_k, load_ep, collect_obs)


def pointwise_field_crb(theta, Finv, gx, gy, t_star, depth, Hs):
    """σ_z(x,y,t*)/Hs over the grid → (Ny, Nx) local nRMSE lower bound."""
    Xg, Yg = torch.meshgrid(gx, gy, indexing='xy')           # (Nx,Ny)
    xf, yf = Xg.reshape(-1), Yg.reshape(-1)
    tf = torch.full_like(xf, float(t_star))
    Jg = jacobian(theta, xf, yf, tf, depth)                  # (P, 4N)
    var = torch.einsum('pa,ab,pb->p', Jg, Finv, Jg)          # g^T F^-1 g
    sig = torch.sqrt(torch.clamp(var, min=0.0)) / Hs
    return sig.reshape(Xg.shape).cpu().numpy()               # (Ny, Nx)


def cg_cone(ep, t_now, dt, depth):
    """Geometric predictable zone: footprint points advected by each component's
    group-velocity vector over [0, dt]. Returns the advected point cloud."""
    m0 = ep['vm'][0]; p0 = ep['lp'][0][m0][:, :2]
    th = ep['theta'].cpu().numpy()
    w, kx, ky = th[:, 0], th[:, 1], th[:, 2]
    k = np.hypot(kx, ky)
    cg = cg_of_k(torch.tensor(k, dtype=DT), depth).cpu().numpy()
    # energy-weighted: advect footprint by every component, union of clouds
    clouds = []
    for i in range(len(w)):
        vx, vy = cg[i] * kx[i] / k[i], cg[i] * ky[i] / k[i]
        clouds.append(p0 + np.array([vx, vy]) * dt)
    return np.concatenate(clouds, 0), cg, kx, ky, k


def run(tag, t_now, horizons, sigma=0.02, depth=50.0):
    ep = load_ep(f'data/episode_{tag}.h5', depth)
    x, y, t = collect_obs(ep, t_now)
    N = ep['theta'].shape[0]
    F = fisher(ep['theta'], x, y, t, depth, sigma)
    n = F.shape[0]
    Finv = torch.linalg.inv(F + 1e-10 * torch.trace(F) / n * torch.eye(n, dtype=DT, device=F.device))
    gx_np, gy_np = ep['gx'].cpu().numpy(), ep['gy'].cpu().numpy()
    maps, areas = {}, {}
    cell = (gx_np[1] - gx_np[0]) * (gy_np[1] - gy_np[0])      # m² per grid cell
    for dt in horizons:
        m = pointwise_field_crb(ep['theta'], Finv, ep['gx'], ep['gy'],
                                t_now + dt, depth, ep['Hs'])
        maps[dt] = m
        areas[dt] = float((m < 0.10).sum() * cell)           # predictable area (τ=0.10)
    return ep, maps, areas, cell


def main():
    out = 'results/direction1'
    os.makedirs(out, exist_ok=True)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    # two regimes: fully-identified vs partially-identified (near the wall)
    configs = [('BF4_5comp_seed0_uniform', 2.0, 'identified (5c, past wall)'),
               ('BF4_15comp_seed0_uniform', 5.0, 'partial (15c, near wall)')]
    horizons = [0.0, 4.0, 8.0]
    total_area = None
    rows_json = {}

    fig, axes = plt.subplots(len(configs), len(horizons),
                             figsize=(4 * len(horizons), 4.2 * len(configs)),
                             sharex=True, sharey=True)
    for ri, (tag, t_now, label) in enumerate(configs):
        ep, maps, areas, cell = run(tag, t_now, horizons)
        foot = ep['lp'][0][ep['vm'][0]][:, :2]
        gx_np, gy_np = ep['gx'].cpu().numpy(), ep['gy'].cpu().numpy()
        if total_area is None:
            total_area = len(gx_np) * len(gy_np) * cell      # consistent w/ cell count
        extent = [gx_np.min(), gx_np.max(), gy_np.min(), gy_np.max()]
        print(f'=== {tag}, t_now={t_now}s ({label}) ===')
        for dt in horizons:
            m = maps[dt]
            print(f'  Δt={dt:.0f}s  area(σ/Hs<0.10)={areas[dt]:7.0f}m² '
                  f'({100*areas[dt]/total_area:4.0f}% grid)  '
                  f'min={m.min():.3f} median={np.median(m):.3f} max={m.max():.2f}')
        rows_json[tag] = dict(t_now=t_now, label=label,
                              areas_m2={str(d): areas[d] for d in horizons},
                              frac_grid={str(d): areas[d] / total_area for d in horizons})
        for ci, dt in enumerate(horizons):
            ax = axes[ri, ci]
            im = ax.imshow(np.clip(maps[dt], 0, 0.3), origin='lower', extent=extent,
                           aspect='auto', cmap='viridis_r', vmin=0, vmax=0.3)
            ax.contour(gx_np, gy_np, maps[dt], levels=[0.10],
                       colors='red', linewidths=1.5)
            ax.scatter(foot[:, 0], foot[:, 1], s=2, c='white', alpha=0.5)
            cloud, *_ = cg_cone(ep, t_now, dt, 50.0)
            ax.scatter(cloud[:, 0], cloud[:, 1], s=2, c='cyan', alpha=0.3)
            ax.set_title(f'{label.split(" (")[0]}: Δt={dt:.0f}s (t*={t_now+dt:.0f}s)\n'
                         f'predictable {100*areas[dt]/total_area:.0f}% grid', fontsize=9)
            if ri == len(configs) - 1:
                ax.set_xlabel('x [m]')
            if ci == 0:
                ax.set_ylabel('y [m]')
    fig.colorbar(im, ax=axes, fraction=0.012, pad=0.01, label='CRB σ_z / Hs')
    fig.suptitle('Direction I — CRB predictable zone: identifiability-limited, not geometric\n'
                 'top: identified → predictable everywhere | bottom: partial → bounded, structured zone\n'
                 'white=footprint, cyan=geometric c_g-cone (does NOT bound the zone), red=σ/Hs=0.10 contour',
                 fontsize=10)
    fig.savefig(f'{out}/fig_predictable_zone.png', dpi=140, bbox_inches='tight')
    plt.close(fig)

    json.dump(dict(horizons=horizons, total_grid_area_m2=total_area, configs=rows_json),
              open(f'{out}/predictable_zone.json', 'w'), indent=2)
    print(f'Saved fig_predictable_zone.png + predictable_zone.json to {out}/')


if __name__ == '__main__':
    main()
