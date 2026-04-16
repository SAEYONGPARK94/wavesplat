"""
Unit tests for wavesplat model components.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import torch
import numpy as np
import pytest
import math

from wavesplat.model.dispersion import dispersion_omega, wavenumber_from_omega, G
from wavesplat.model.spectral_gaussian import SpectralGaussianModel
from wavesplat.model.confidence import observation_weights


# ── Dispersion tests ─────────────────────────────────────────────────────────

def test_dispersion_deep_water():
    """Deep water: ω ≈ √(gk) when kd >> 1."""
    k = torch.tensor([0.5])
    depth = 100.0  # kd = 50 >> 1
    omega = dispersion_omega(k, depth)
    omega_deep = torch.sqrt(torch.tensor(G) * k)
    assert torch.allclose(omega, omega_deep, rtol=1e-4)


def test_dispersion_matches_numpy():
    """PyTorch dispersion must match wavesim numpy version."""
    from wavesim.wave_generator import full_dispersion

    k_vals = [0.01, 0.05, 0.1, 0.3, 0.5, 1.0]
    depth = 50.0
    for k_val in k_vals:
        omega_np = full_dispersion(k_val, depth)
        omega_pt = dispersion_omega(torch.tensor([k_val]), depth).item()
        assert abs(omega_np - omega_pt) < 1e-5, \
            f"k={k_val}: numpy={omega_np:.6f}, torch={omega_pt:.6f}"


def test_inverse_dispersion_roundtrip():
    """wavenumber_from_omega(dispersion_omega(k)) ≈ k."""
    k_orig = torch.tensor([0.05, 0.1, 0.3, 0.5])
    depth = 50.0
    omega = dispersion_omega(k_orig, depth)
    k_recovered = wavenumber_from_omega(omega, depth)
    assert torch.allclose(k_orig, k_recovered, rtol=1e-4)


def test_dispersion_gradient():
    """Dispersion must be differentiable (autograd)."""
    k = torch.tensor([0.3], requires_grad=True)
    omega = dispersion_omega(k, 50.0)
    omega.backward()
    assert k.grad is not None
    assert k.grad.item() > 0  # dω/dk > 0


# ── Spectral Gaussian model tests ────────────────────────────────────────────

def test_model_single_component():
    """Single component: model output must match analytic linear sinusoidal."""
    depth = 50.0
    model = SpectralGaussianModel(1, depth=depth)

    # Set known parameters
    A = 0.5
    kx_val = 0.1
    ky_val = 0.0
    phi_val = 0.0
    k_mag = abs(kx_val)
    omega_val = math.sqrt(G * k_mag * math.tanh(k_mag * depth))

    with torch.no_grad():
        model.w.copy_(torch.tensor([A]))
        model.kx.copy_(torch.tensor([kx_val]))
        model.ky.copy_(torch.tensor([ky_val]))
        model.phi.copy_(torch.tensor([phi_val]))

    # Evaluate
    x = torch.tensor([0.0, 10.0, 50.0, 100.0])
    y = torch.zeros(4)
    t = 1.0

    z = model(x, y, t)
    z_expected = A * torch.cos(kx_val * x - omega_val * t)

    assert torch.allclose(z, z_expected, atol=1e-5), \
        f"Model: {z.tolist()}, Expected: {z_expected.tolist()}"


def test_model_multi_component():
    """Multi-component: superposition must be sum of individual contributions."""
    model = SpectralGaussianModel(3, depth=50.0)
    x = torch.tensor([20.0, 50.0, 100.0])
    y = torch.tensor([-10.0, 0.0, 10.0])
    t = 2.0

    z_combined = model(x, y, t)
    assert z_combined.shape == (3,)

    # Sum individual contributions
    z_sum = torch.zeros(3)
    for i in range(3):
        single_model = SpectralGaussianModel(1, depth=50.0)
        with torch.no_grad():
            single_model.w.copy_(model.w[i:i+1])
            single_model.kx.copy_(model.kx[i:i+1])
            single_model.ky.copy_(model.ky[i:i+1])
            single_model.phi.copy_(model.phi[i:i+1])
        z_sum += single_model(x, y, t)

    assert torch.allclose(z_combined, z_sum, atol=1e-5)


def test_model_gradient_flow():
    """Loss gradient must flow to all parameters."""
    model = SpectralGaussianModel(5, depth=50.0)
    x = torch.tensor([30.0, 60.0, 90.0])
    y = torch.zeros(3)
    z_obs = torch.tensor([0.1, -0.2, 0.15])

    z_pred = model(x, y, 1.0)
    loss = ((z_pred - z_obs) ** 2).mean()
    loss.backward()

    for name, param in model.named_parameters():
        assert param.grad is not None, f"No gradient for {name}"
        assert not torch.all(param.grad == 0), f"Zero gradient for {name}"


def test_model_predict_grid():
    """Grid prediction must produce correct shape."""
    model = SpectralGaussianModel(5, depth=50.0)
    gx = torch.linspace(10, 250, 20)
    gy = torch.linspace(-100, 100, 15)
    Z = model.predict_grid(gx, gy, 1.0)
    assert Z.shape == (15, 20)


def test_model_add_gaussians():
    """Adding Gaussians preserves existing parameters."""
    model = SpectralGaussianModel(3, depth=50.0)
    w_orig = model.w.data.clone()

    model.add_gaussians(5)
    assert model.n_gaussians == 8
    assert model.w.shape == (8,)
    assert torch.allclose(model.w.data[:3], w_orig)


# ── Confidence tests ─────────────────────────────────────────────────────────

def test_confidence_range():
    """Confidence must be in [0, 1]."""
    angles = torch.linspace(0, math.pi / 2 * 0.95, 50)
    dists = torch.linspace(10, 200, 50)
    conf = observation_weights(angles, dists)
    assert torch.all(conf >= 0) and torch.all(conf <= 1)


def test_confidence_distance_decreasing():
    """Farther observations → lower confidence."""
    angle = torch.tensor([0.5])
    near = observation_weights(angle, torch.tensor([50.0]))
    far = observation_weights(angle, torch.tensor([200.0]))
    assert near > far


def test_confidence_matches_numpy():
    """Must match wavesim.noise_model.compute_confidence."""
    from wavesim.noise_model import compute_confidence

    angles = np.linspace(0.1, 1.2, 10)
    dists = np.linspace(20, 180, 10)

    conf_np = compute_confidence(angles, dists)
    conf_pt = observation_weights(
        torch.tensor(angles, dtype=torch.float32),
        torch.tensor(dists, dtype=torch.float32)
    ).numpy()

    np.testing.assert_allclose(conf_pt, conf_np, atol=1e-5)


# ── Shallow depth (d=3.5m tank) dispersion tests ────────────────────────────

def test_dispersion_shallow_depth():
    """At d=3.5m, finite-depth effects are significant for long waves."""
    depth = 3.5
    # T=3.0s → ω≈2.09, deep-water k≈0.447, kd≈1.56 → tanh(kd)≈0.916
    # Finite depth makes ω smaller than deep-water value
    k = torch.tensor([0.447])
    omega_full = dispersion_omega(k, depth)
    omega_deep = torch.sqrt(torch.tensor(G) * k)  # deep-water

    # Full dispersion ω must be less than deep-water ω
    assert omega_full.item() < omega_deep.item(), \
        f"Finite depth ω={omega_full.item():.4f} should be < deep-water ω={omega_deep.item():.4f}"


def test_dispersion_shallow_deep_convergence():
    """Short waves (kd >> 1) should match deep-water even at d=3.5m."""
    depth = 3.5
    # T=0.5s → very short wave, kd >> 1
    k = torch.tensor([16.0])  # kd = 56 >> 1
    omega_full = dispersion_omega(k, depth)
    omega_deep = torch.sqrt(torch.tensor(G) * k)
    assert torch.allclose(omega_full, omega_deep, rtol=1e-4), \
        "Short waves at d=3.5m should match deep-water dispersion"


def test_inverse_dispersion_shallow():
    """Inverse dispersion roundtrip at d=3.5m."""
    depth = 3.5
    k_orig = torch.tensor([0.3, 0.5, 1.0, 2.0, 5.0])
    omega = dispersion_omega(k_orig, depth)
    k_recovered = wavenumber_from_omega(omega, depth)
    assert torch.allclose(k_orig, k_recovered, rtol=1e-3), \
        f"Roundtrip failed: orig={k_orig.tolist()}, recovered={k_recovered.tolist()}"


def test_dispersion_shallow_wavelength():
    """Wavelength at d=3.5m should be shorter than deep-water prediction."""
    from wavesplat.model.dispersion import wavelength_from_period
    depth = 3.5
    T = torch.tensor([2.0])  # T=2.0s

    lam_shallow = wavelength_from_period(T, depth)
    lam_deep = wavelength_from_period(T, 1000.0)  # effectively infinite

    # Finite depth → shorter wavelength
    assert lam_shallow.item() < lam_deep.item(), \
        f"λ_shallow={lam_shallow.item():.2f} should be < λ_deep={lam_deep.item():.2f}"


def test_dispersion_shallow_known_values():
    """Validate against hand-calculated values at d=3.5m."""
    depth = 3.5
    # For T=1.0s: ω=2π, deep-water k=ω²/g≈4.024
    # kd = 4.024*3.5 = 14.08, tanh(kd)≈1.0 → deep-water applies
    T_short = torch.tensor([1.0])
    omega_short = 2 * math.pi / T_short
    k_short = wavenumber_from_omega(omega_short, depth)
    k_deep = omega_short ** 2 / G
    assert torch.allclose(k_short, k_deep, rtol=1e-2), \
        "T=1.0s at d=3.5m should be nearly deep-water"

    # For T=3.0s: ω=2.094, deep-water k≈0.447, kd≈1.57
    # Finite depth: k should be larger than deep-water k
    T_long = torch.tensor([3.0])
    omega_long = 2 * math.pi / T_long
    k_long = wavenumber_from_omega(omega_long, depth)
    k_deep_long = omega_long ** 2 / G
    assert k_long.item() > k_deep_long.item(), \
        "At T=3.0s, d=3.5m: finite-depth k should exceed deep-water k"


# ── Real data coordinate transform tests ────────────────────────────────────

def test_falcon_coordinate_transform():
    """Falcon K1 → WaveSplat coordinate transform."""
    from wavesplat.real_data import falcon_to_wavesplat, LiDARMountConfig

    mount = LiDARMountConfig(height=3.5, tilt_angle=0.0)
    # Falcon K1: X=up(3.5), Y=right(1.0), Z=forward(10.0)
    pts = np.array([[3.5, 1.0, 10.0]])
    ws_x, ws_y, ws_z = falcon_to_wavesplat(pts, mount)

    # ws_x = k1_z = 10.0 (forward)
    assert abs(ws_x[0] - 10.0) < 1e-6
    # ws_y = k1_y = 1.0 (lateral)
    assert abs(ws_y[0] - 1.0) < 1e-6
    # ws_z = k1_x - height = 3.5 - 3.5 = 0.0 (water surface)
    assert abs(ws_z[0] - 0.0) < 1e-6


def test_falcon_transform_with_tilt():
    """Coordinate transform with downward tilt.

    Positive tilt = sensor looking downward.
    A point on the forward axis in K1 (0, 0, d) should appear
    at a lower elevation after tilt.
    """
    from wavesplat.real_data import falcon_to_wavesplat, LiDARMountConfig

    # No tilt: point at K1 (0, 0, 10) → ws (10, 0, -3.5)
    mount_flat = LiDARMountConfig(height=3.5, tilt_angle=0.0)
    pts = np.array([[0.0, 0.0, 10.0]])
    ws_x0, _, ws_z0 = falcon_to_wavesplat(pts, mount_flat)

    # 30° downward tilt: forward axis points down
    # K1 point (0, 0, 10) should now have lower z (more negative) in WS
    mount_tilt = LiDARMountConfig(height=3.5, tilt_angle=math.radians(30))
    ws_x1, _, ws_z1 = falcon_to_wavesplat(pts, mount_tilt)

    assert ws_z1[0] < ws_z0[0], \
        f"Downward tilt should lower z: flat={ws_z0[0]:.3f}, tilted={ws_z1[0]:.3f}"
    assert not np.isnan(ws_x1[0])
    assert not np.isnan(ws_z1[0])


def test_incidence_angle_nadir():
    """Nadir (straight down): incidence angle should be ~0."""
    from wavesplat.real_data import compute_incidence_angles, LiDARMountConfig

    mount = LiDARMountConfig(height=3.5)
    # Point directly below sensor
    ws_x = np.array([mount.x_offset])
    ws_y = np.array([mount.y_offset])
    ws_z = np.array([0.0])  # water surface

    angles = compute_incidence_angles(ws_x, ws_y, ws_z, mount)
    assert angles[0] < 0.1, f"Nadir angle should be ~0, got {angles[0]:.4f}"


def test_incidence_angle_oblique():
    """Oblique viewing: incidence angle should increase with distance."""
    from wavesplat.real_data import compute_incidence_angles, LiDARMountConfig

    mount = LiDARMountConfig(height=3.5)
    # Points at increasing forward distance
    ws_x = np.array([1.0, 5.0, 20.0, 50.0])
    ws_y = np.zeros(4)
    ws_z = np.zeros(4)

    angles = compute_incidence_angles(ws_x, ws_y, ws_z, mount)

    # Incidence angle should increase with forward distance
    for i in range(len(angles) - 1):
        assert angles[i] < angles[i + 1], \
            f"Angle should increase: {angles[i]:.4f} >= {angles[i+1]:.4f}"


def test_surface_filter():
    """Surface filter should reject walls/floor/out-of-range."""
    from wavesplat.real_data import filter_surface_points, SurfaceFilterConfig

    cfg = SurfaceFilterConfig(z_min=-0.5, z_max=0.5,
                               x_min=0, x_max=50,
                               y_min=-4, y_max=4)

    ws_x = np.array([10.0, -5.0, 60.0, 10.0, 10.0])
    ws_y = np.array([0.0, 0.0, 0.0, 5.0, 0.0])
    ws_z = np.array([0.1, 0.0, 0.0, 0.0, -3.0])  # surface, behind, far, wall, floor
    intensity = np.ones(5) * 128

    mask = filter_surface_points(ws_x, ws_y, ws_z, intensity, cfg)
    assert mask[0] == True   # valid surface
    assert mask[1] == False  # behind (x<0)
    assert mask[2] == False  # beyond tank (x>50)
    assert mask[3] == False  # outside tank width (y>4)
    assert mask[4] == False  # floor (z<-0.5)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
