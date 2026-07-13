"""The GPU raycast must be the CPU raycast, only faster.

It is the observation operator: if it disagrees with the CPU path even slightly
about which rays hit, the episodes it renders are not comparable with the ones
already in data/.  So the contract tested here is hit-for-hit identity, not
"close enough".
"""
import numpy as np
import pytest
import torch

from wavesim.config import WaveSimConfig
from wavesim.gt_generator import evaluate_field, make_grid
from wavesim.lidar_gpu import raycast_surface_gpu
from wavesim.lidar_simulator import (LidarConfig, build_mesh, generate_rays,
                                     raycast_surface)
from wavesim.wave_generator import jonswap_components

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(),
                                reason='no GPU available')


def _scene(mesh_nx, mesh_ny):
    cfg = WaveSimConfig()
    cfg.mesh.nx, cfg.mesh.ny = mesh_nx, mesh_ny
    lc = LidarConfig(
        height=cfg.lidar.height, channels=cfg.lidar.channels,
        scan_rate=cfg.lidar.scan_rate, fov_h_deg=cfg.lidar.fov_h_deg,
        fov_v_min_deg=cfg.lidar.fov_v_min_deg,
        fov_v_max_deg=cfg.lidar.fov_v_max_deg,
        range_min=cfg.lidar.range_min, range_max=cfg.lidar.range_max,
        n_h_beams=cfg.lidar.n_h_beams)
    O, D = generate_rays(lc, 0.0, 0.0)
    comps = jonswap_components(Hs=1.2, Tp=7.0, n_components=15, depth=50.0,
                               seed=0)
    XM, YM = make_grid(cfg.gt_grid.x_range, cfg.gt_grid.y_range,
                       nx=cfg.mesh.nx, ny=cfg.mesh.ny)
    ZM, _, _, _ = evaluate_field(comps, XM, YM, 0.0)
    V, T = build_mesh(XM, YM, ZM)
    return cfg, lc, O, D, V, T


@pytest.mark.parametrize('mesh', [
    (40, 30),                                     # the shipped default
    (90, 75),
    pytest.param((180, 150), marks=pytest.mark.slow),  # the mesh E-3 regenerated on
])
def test_gpu_raycast_matches_cpu_hit_for_hit(mesh):
    cfg, lc, O, D, V, T = _scene(*mesh)
    kw = dict(range_min=lc.range_min, range_max=lc.range_max,
              spatial_grid_nx=cfg.raycast.spatial_grid_nx,
              spatial_grid_ny=cfg.raycast.spatial_grid_ny)

    m_cpu, p_cpu, d_cpu = raycast_surface(
        O, D, V, T, batch_size=cfg.raycast.batch_size, **kw)
    m_gpu, p_gpu, d_gpu = raycast_surface_gpu(O, D, V, T, **kw)

    assert m_cpu.sum() > 1000, 'scene should produce plenty of hits'
    assert (m_cpu == m_gpu).all(), 'GPU and CPU disagree on which rays hit'

    hit = m_cpu & m_gpu
    assert np.abs(d_cpu[hit] - d_gpu[hit]).max() < 1e-9
    assert np.abs(p_cpu[hit] - p_gpu[hit]).max() < 1e-9
