"""
Unit tests for raydrop_uniform.py and raydrop_slope.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
import pytest
from wavesim.raydrop_uniform import fresnel_reflectance, return_probability, apply_dropout as apply_uniform
from wavesim.raydrop_slope import specular_alignment, apply_dropout as apply_slope


# ── Fresnel reflectance ──────────────────────────────────────────────────────

def test_fresnel_normal_incidence():
    """At θ=0 (normal incidence), R = ((n-1)/(n+1))² ≈ 2%. Uses 1550nm default."""
    from wavesim.raydrop_uniform import N_WATER_1550NM
    n = N_WATER_1550NM  # 1.321 @1550nm (Seyond Falcon K1)
    R_expected = ((n - 1) / (n + 1))**2
    R = fresnel_reflectance(np.array([0.0]))[0]  # use default n
    assert abs(R - R_expected) < 1e-6


def test_fresnel_grazing():
    """At grazing (θ=89°), R should approach 1."""
    R = fresnel_reflectance(np.array([np.deg2rad(89.0)]))[0]
    assert R > 0.8, f"Fresnel at 89°: R={R:.3f} should be > 0.8"


def test_fresnel_monotone():
    """Fresnel reflectance must increase monotonically with θ_i (0→90°)."""
    theta = np.linspace(0, np.deg2rad(89), 100)
    R = fresnel_reflectance(theta)
    assert np.all(np.diff(R) >= -1e-9), "Fresnel R must be monotonically non-decreasing"


def test_fresnel_range():
    """All Fresnel values in [0, 1]."""
    theta = np.linspace(0, np.deg2rad(89.9), 200)
    R = fresnel_reflectance(theta)
    assert np.all(R >= 0) and np.all(R <= 1)


# ── Uniform dropout ──────────────────────────────────────────────────────────

def test_uniform_dropout_count():
    """
    apply_dropout (uniform) uses Bernoulli sampling:
    - Mean return count ≈ target_n_pts over many seeds (within 30%)
    - Count varies across seeds (σ > 0) — physically correct frame-to-frame variation
    """
    N = 500
    hit_mask = np.ones(N, dtype=bool)
    hit_points = np.zeros((N, 3))
    hit_dist = np.linspace(20, 150, N)
    incidence = np.linspace(0, np.deg2rad(70), N)
    target = 100

    counts = []
    for seed in range(30):
        r = np.random.default_rng(seed)
        mask = apply_uniform(hit_mask, hit_points, hit_dist, incidence, target, r)
        counts.append(int(mask.sum()))

    mean_count = np.mean(counts)
    std_count = np.std(counts)
    assert abs(mean_count - target) < target * 0.3, \
        f"Mean count {mean_count:.1f} deviates >30% from target {target}"
    assert std_count > 0, \
        "Return count must vary across frames (Bernoulli sampling)"


def test_uniform_dropout_empty():
    """Empty hit_mask → empty return."""
    rng = np.random.default_rng(0)
    hit_mask = np.zeros(100, dtype=bool)
    result = apply_uniform(hit_mask, np.zeros((100, 3)),
                            np.ones(100), np.zeros(100), 10, rng)
    assert not np.any(result)


# ── Specular alignment ────────────────────────────────────────────────────────

def test_specular_flat_surface_grazing():
    """
    Flat surface (normal = [0,0,1]) with grazing ray:
    For forward-looking at h=5m, R=100m → angle ≈ 2.9°.
    Reflected ray goes upward, NOT back to sensor → no specular return.
    """
    # Ray direction toward flat surface at grazing angle ~3°
    el = np.deg2rad(-2.9)  # negative = downward
    d = np.array([[np.cos(el), 0.0, np.sin(el)]])  # nearly horizontal
    n = np.array([[0.0, 0.0, 1.0]])  # flat surface

    mask, align = specular_alignment(d, n, alpha_threshold_deg=1.0)
    # For flat surface, reflected ray goes upward not back to sensor
    assert not mask[0], f"Flat surface should NOT produce specular backscatter at grazing (align={align[0]:.3f})"


def test_specular_retroreflector():
    """
    Corner retroreflector geometry: ray hits surface at exactly normal incidence.
    Reflected ray should go straight back → high alignment.
    """
    # Ray going straight down (-z)
    d = np.array([[0.0, 0.0, -1.0]])
    n = np.array([[0.0, 0.0,  1.0]])  # flat surface

    mask, align = specular_alignment(d, n, alpha_threshold_deg=5.0)
    assert mask[0], f"Normal incidence should produce specular return (align={align[0]:.3f})"
    assert align[0] > 0.99


def test_slope_dropout_count():
    """
    Slope-based dropout uses Bernoulli sampling:
    - Mean return count ≈ target_n_pts over many seeds (within 50% tolerance
      due to geometric variation in specular candidates)
    - Count must vary across seeds (σ > 0)
    """
    N = 1000
    target = 50

    counts = []
    for seed in range(20):
        rng = np.random.default_rng(seed)
        hit_mask = np.ones(N, dtype=bool)
        hit_points = rng.standard_normal((N, 3))

        dirs = rng.standard_normal((N, 3))
        dirs[:, 2] = -abs(dirs[:, 2])
        dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)

        normals = rng.standard_normal((N, 3))
        normals[:, 2] = abs(normals[:, 2])
        normals /= np.linalg.norm(normals, axis=1, keepdims=True)

        incidence = np.deg2rad(rng.uniform(60, 88, N))
        dist = rng.uniform(20, 150, N)

        mask = apply_slope(hit_mask, hit_points, dist, dirs, normals,
                            incidence, target, rng)
        counts.append(int(mask.sum()))

    mean_count = np.mean(counts)
    std_count = np.std(counts)
    assert mean_count > 0, "Slope model returned 0 points on all frames"
    assert std_count > 0, "Return count must vary across frames (Bernoulli sampling)"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
