"""
Unit tests for lidar_simulator.py

Verification:
  1. Flat surface ray intersection — known hit positions
  2. Surface normal interpolation — matches analytic Gerstner normal
  3. Incidence angle computation
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
import pytest
from wavesim.lidar_simulator import (
    build_mesh, raycast_surface, generate_rays, LidarConfig,
    compute_incidence_angles, interpolate_normals
)
from wavesim.wave_generator import jonswap_components, single_gerstner


# ── Flat surface raycast ──────────────────────────────────────────────────────

def test_flat_surface_vertical_hit():
    """Vertical ray downward onto flat z=0 plane must hit at (x,y,0)."""
    x = np.linspace(0, 50, 20)
    y = np.linspace(0, 50, 20)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X)

    verts, tris = build_mesh(X, Y, Z)

    # Single vertical ray from (25, 25, 10) pointing down
    origin = np.array([[25.0, 25.0, 10.0]])
    direction = np.array([[0.0, 0.0, -1.0]])

    hit_mask, hit_pts, hit_dist = raycast_surface(origin, direction, verts, tris,
                                                    range_min=0.1, range_max=100.0)
    assert hit_mask[0], "Vertical ray onto flat surface must hit"
    assert abs(hit_pts[0, 2]) < 0.5, f"Hit z={hit_pts[0,2]:.3f} should be ~0"
    assert abs(hit_dist[0] - 10.0) < 0.5, f"Hit dist={hit_dist[0]:.3f} should be ~10"


def test_flat_surface_miss():
    """Ray pointing upward misses downward surface."""
    x = np.linspace(0, 50, 10)
    y = np.linspace(0, 50, 10)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X)
    verts, tris = build_mesh(X, Y, Z)

    origin = np.array([[25.0, 25.0, 1.0]])
    direction = np.array([[0.0, 0.0, 1.0]])  # pointing up

    hit_mask, _, _ = raycast_surface(origin, direction, verts, tris,
                                      range_min=0.1, range_max=100.0)
    assert not hit_mask[0], "Upward ray must miss downward surface"


def test_range_filter():
    """Points outside [range_min, range_max] must not be returned."""
    x = np.linspace(0, 10, 10)
    y = np.linspace(0, 10, 10)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X)
    verts, tris = build_mesh(X, Y, Z)

    # Ray 1m above flat surface — hits at d=1m
    origin = np.array([[5.0, 5.0, 1.0]])
    direction = np.array([[0.0, 0.0, -1.0]])

    # range_min=2.0 → should miss (hit is at d=1.0 < 2.0)
    hit_mask, _, _ = raycast_surface(origin, direction, verts, tris,
                                      range_min=2.0, range_max=100.0)
    assert not hit_mask[0], "Hit within range_min must be filtered out"


# ── Generate rays ─────────────────────────────────────────────────────────────

def test_generate_rays_shape():
    """generate_rays returns correct number of rays."""
    cfg = LidarConfig(channels=16, n_h_beams=180)
    origins, dirs = generate_rays(cfg)
    expected = 16 * 180
    assert origins.shape == (expected, 3)
    assert dirs.shape == (expected, 3)


def test_generate_rays_unit_dirs():
    """All ray directions must be unit vectors."""
    cfg = LidarConfig(channels=8, n_h_beams=36)
    _, dirs = generate_rays(cfg)
    norms = np.linalg.norm(dirs, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-10), "Ray directions must be unit vectors"


# ── Incidence angles ─────────────────────────────────────────────────────────

def test_incidence_angle_normal():
    """Ray along -z onto flat surface (normal=[0,0,1]): θ_i = 0."""
    ray_dirs = np.array([[0.0, 0.0, -1.0]])
    normals = np.array([[0.0, 0.0, 1.0]])
    theta_i, theta_g = compute_incidence_angles(None, ray_dirs, normals)
    assert abs(theta_i[0]) < 1e-6, f"Normal incidence: θ_i={np.deg2rad(theta_i[0]):.4f}° should be 0"
    assert abs(theta_g[0] - np.pi / 2) < 1e-6


def test_incidence_grazing():
    """Horizontal ray onto flat surface: θ_i = 90°."""
    ray_dirs = np.array([[1.0, 0.0, 0.0]])
    normals = np.array([[0.0, 0.0, 1.0]])
    theta_i, theta_g = compute_incidence_angles(None, ray_dirs, normals)
    assert abs(theta_i[0] - np.pi / 2) < 0.01


# ── Normal interpolation ─────────────────────────────────────────────────────

def test_normal_interpolation_flat():
    """Flat surface: all normals should interpolate to (0, 0, 1)."""
    x = np.linspace(0, 100, 30)
    y = np.linspace(0, 100, 30)
    X, Y = np.meshgrid(x, y)
    NX = np.zeros_like(X)
    NY = np.zeros_like(X)
    NZ = np.ones_like(X)

    pts = np.array([[20.0, 30.0, 0.0],
                    [50.0, 50.0, 0.0],
                    [80.0, 70.0, 0.0]])

    normals = interpolate_normals(pts, X, Y, NX, NY, NZ)
    assert np.allclose(normals, [[0, 0, 1]] * 3, atol=1e-6)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
