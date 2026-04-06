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


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
