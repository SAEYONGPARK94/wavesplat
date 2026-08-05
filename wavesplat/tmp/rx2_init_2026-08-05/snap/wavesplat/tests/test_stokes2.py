"""
Tests for Stokes 2nd-order transfer functions and elevation correction.

Key analytical checks:
1. Deep-water single component: L+ = k/2 (classical Stokes 2nd order)
2. Deep-water collinear pair: L+ = (k_n + k_m) / 4
3. Symmetry: L+[n,m] = L+[m,n], L-[n,m] = L-[m,n] (for surface elevation)
4. Self-difference: L-[n,n] = 0 (excluded as constant offset)
5. Superposition: correction vanishes for zero amplitudes
6. Magnitude check: 2nd-order << 1st-order for typical ocean waves
"""

import numpy as np
import torch
import pytest

from wavesplat.model.stokes2 import (
    _compute_transfer_functions_np,
    Stokes2Transfer,
)

G = 9.81


class TestTransferFunctions:
    """Test the L+/L- computation."""

    def test_deep_water_single_component_sum(self):
        """Deep-water self-interaction: L+[0,0] = k/2."""
        k = 0.5  # rad/m
        depth = 100.0  # deep water: kh = 50 >> 1
        kx = np.array([k])
        ky = np.array([0.0])

        L_plus, L_minus = _compute_transfer_functions_np(kx, ky, depth)

        expected = k / 2.0
        assert abs(L_plus[0, 0] - expected) < 0.01 * expected, \
            f"L+[0,0]={L_plus[0,0]:.6f}, expected={expected:.6f}"

    def test_deep_water_collinear_pair_sum(self):
        """Deep-water collinear pair: L+[0,1] ≈ (k0+k1)/4."""
        k0, k1 = 0.3, 0.6
        depth = 100.0
        kx = np.array([k0, k1])
        ky = np.array([0.0, 0.0])

        L_plus, _ = _compute_transfer_functions_np(kx, ky, depth)

        expected = (k0 + k1) / 4.0
        assert abs(L_plus[0, 1] - expected) < 0.01 * expected, \
            f"L+[0,1]={L_plus[0, 1]:.6f}, expected={expected:.6f}"

    def test_symmetry_sum(self):
        """L+[n,m] = L+[m,n]."""
        kx = np.array([0.3, 0.5, 0.1])
        ky = np.array([0.1, -0.2, 0.4])
        depth = 50.0

        L_plus, L_minus = _compute_transfer_functions_np(kx, ky, depth)

        np.testing.assert_allclose(L_plus, L_plus.T, atol=1e-10)
        np.testing.assert_allclose(L_minus, L_minus.T, atol=1e-10)

    def test_self_difference_zero(self):
        """L-[n,n] = 0 (constant set-down excluded)."""
        kx = np.array([0.3, 0.5])
        ky = np.array([0.1, -0.2])
        depth = 50.0

        _, L_minus = _compute_transfer_functions_np(kx, ky, depth)

        assert L_minus[0, 0] == 0.0
        assert L_minus[1, 1] == 0.0

    def test_finite_depth_differs_from_deep(self):
        """L+ at finite depth should differ from deep-water formula."""
        k = 0.3
        kx = np.array([k])
        ky = np.array([0.0])

        # Deep water (kh = 30)
        L_deep, _ = _compute_transfer_functions_np(kx, ky, depth=100.0)
        # Shallow water (kh = 1.5)
        L_shallow, _ = _compute_transfer_functions_np(kx, ky, depth=5.0)

        assert abs(L_deep[0, 0] - L_shallow[0, 0]) > 0.001, \
            "Finite depth should give different L+ than deep water"

    def test_zero_wavenumber_handled(self):
        """Components with k≈0 should have L=0."""
        kx = np.array([0.0, 0.3])
        ky = np.array([0.0, 0.1])
        depth = 50.0

        L_plus, L_minus = _compute_transfer_functions_np(kx, ky, depth)

        assert L_plus[0, 0] == 0.0
        assert L_plus[0, 1] == 0.0
        assert L_plus[1, 0] == 0.0


class TestStokes2Correction:
    """Test the full correction computation."""

    def _make_grid(self, nx=20, ny=20, Lx=100.0, Ly=100.0):
        """Create evaluation grid."""
        xx = np.linspace(0, Lx, nx)
        yy = np.linspace(-Ly/2, Ly/2, ny)
        X, Y = np.meshgrid(xx, yy)
        return X.ravel(), Y.ravel()

    def test_single_component_deep_water_stokes2(self):
        """
        Single component in deep water: known analytical result.
        eta1 = a cos(kx - wt + phi)
        eta2 = (k a^2 / 2) cos(2(kx - wt + phi))

        So the correction should be ka²/2 * cos(2*theta).
        """
        k = 0.4
        a = 0.5  # amplitude
        phi_val = 0.3
        depth = 100.0
        t = 2.0

        kx_arr = np.array([k])
        ky_arr = np.array([0.0])
        w_arr = np.array([a])
        phi_arr = np.array([phi_val])
        omega_arr = np.sqrt(G * k * np.tanh(k * depth))
        omega_arr = np.array([omega_arr])

        transfer = Stokes2Transfer(kx_arr, ky_arr, depth)

        # Evaluate at several x points
        x = np.linspace(0, 2 * np.pi / k, 50)
        y = np.zeros_like(x)

        z2 = transfer.correction_np(x, y, t, w_arr, kx_arr, ky_arr,
                                     phi_arr, omega_arr)

        # Expected: L+[0,0] * a^2 * cos(2*theta)  where theta = kx - wt + phi
        # L+[0,0] ≈ k/2 in deep water
        theta = k * x - omega_arr[0] * t + phi_val
        expected = (k / 2.0) * a**2 * np.cos(2 * theta)

        np.testing.assert_allclose(z2, expected, atol=1e-4,
                                    err_msg="Single component Stokes 2nd order mismatch")

    def test_correction_magnitude_physical(self):
        """
        For typical ocean waves (steepness ~0.05), 2nd order should be
        ~5% of 1st order (epsilon^2 / epsilon = epsilon).
        """
        N = 10
        np.random.seed(42)
        k_mags = np.random.uniform(0.05, 0.3, N)
        directions = np.random.uniform(-np.pi/6, np.pi/6, N)
        kx_arr = k_mags * np.cos(directions)
        ky_arr = k_mags * np.sin(directions)
        depth = 50.0

        # Amplitudes with steepness ~ 0.05
        w_arr = 0.05 / k_mags  # a = epsilon / k
        phi_arr = np.random.uniform(0, 2*np.pi, N)
        omega_arr = np.sqrt(G * k_mags * np.tanh(k_mags * depth))

        transfer = Stokes2Transfer(kx_arr, ky_arr, depth)

        x, y = self._make_grid()
        t = 5.0

        # First order
        phase = (kx_arr[None, :] * x[:, None]
                 + ky_arr[None, :] * y[:, None]
                 - omega_arr[None, :] * t
                 + phi_arr[None, :])
        z1 = np.sum(w_arr[None, :] * np.cos(phase), axis=1)

        # Second order
        z2 = transfer.correction_np(x, y, t, w_arr, kx_arr, ky_arr,
                                     phi_arr, omega_arr)

        ratio = np.std(z2) / np.std(z1)
        assert ratio < 0.15, f"2nd order too large: {ratio:.3f} (expected < 0.15)"
        assert ratio > 0.001, f"2nd order too small: {ratio:.3f} (expected > 0.001)"

    def test_zero_amplitude_gives_zero(self):
        """Zero weights → zero correction."""
        kx_arr = np.array([0.3, 0.5])
        ky_arr = np.array([0.0, 0.1])
        depth = 50.0
        w_arr = np.array([0.0, 0.0])
        phi_arr = np.array([0.0, 1.0])
        omega_arr = np.sqrt(G * np.sqrt(kx_arr**2 + ky_arr**2)
                            * np.tanh(np.sqrt(kx_arr**2 + ky_arr**2) * depth))

        transfer = Stokes2Transfer(kx_arr, ky_arr, depth)
        x = np.linspace(0, 100, 30)
        y = np.zeros(30)

        z2 = transfer.correction_np(x, y, 0.0, w_arr, kx_arr, ky_arr,
                                     phi_arr, omega_arr)
        np.testing.assert_allclose(z2, 0.0, atol=1e-15)

    def test_torch_matches_numpy(self):
        """PyTorch correction should match NumPy version."""
        N = 5
        np.random.seed(123)
        k_mags = np.random.uniform(0.1, 0.4, N)
        dirs = np.random.uniform(-np.pi/4, np.pi/4, N)
        kx_arr = k_mags * np.cos(dirs)
        ky_arr = k_mags * np.sin(dirs)
        depth = 50.0
        w_arr = np.random.uniform(0.1, 0.5, N)
        phi_arr = np.random.uniform(0, 2*np.pi, N)
        omega_arr = np.sqrt(G * k_mags * np.tanh(k_mags * depth))

        transfer = Stokes2Transfer(kx_arr, ky_arr, depth)

        x = np.linspace(0, 200, 100)
        y = np.linspace(-50, 50, 100)
        t = 3.0

        z2_np = transfer.correction_np(x, y, t, w_arr, kx_arr, ky_arr,
                                        phi_arr, omega_arr)

        z2_torch = transfer.correction(
            torch.tensor(x, dtype=torch.float32),
            torch.tensor(y, dtype=torch.float32),
            t,
            torch.tensor(w_arr, dtype=torch.float32),
            torch.tensor(kx_arr, dtype=torch.float32),
            torch.tensor(ky_arr, dtype=torch.float32),
            torch.tensor(phi_arr, dtype=torch.float32),
            torch.tensor(omega_arr, dtype=torch.float32),
        )

        np.testing.assert_allclose(z2_torch.numpy(), z2_np, atol=1e-4)

    def test_torch_gradient_flows(self):
        """Gradients should flow through w and phi."""
        kx_arr = np.array([0.3, 0.5])
        ky_arr = np.array([0.0, 0.1])
        depth = 50.0

        transfer = Stokes2Transfer(kx_arr, ky_arr, depth)

        w = torch.tensor([0.3, 0.2], requires_grad=True)
        phi = torch.tensor([0.0, 1.0], requires_grad=True)
        kx_t = torch.tensor(kx_arr, dtype=torch.float32)
        ky_t = torch.tensor(ky_arr, dtype=torch.float32)
        k_mag = torch.sqrt(kx_t**2 + ky_t**2)
        omega = torch.sqrt(G * k_mag * torch.tanh(k_mag * depth))

        x = torch.linspace(0, 100, 20)
        y = torch.zeros(20)

        z2 = transfer.correction(x, y, 1.0, w, kx_t, ky_t, phi, omega)
        loss = z2.sum()
        loss.backward()

        assert w.grad is not None, "No gradient for w"
        assert phi.grad is not None, "No gradient for phi"
        assert torch.any(w.grad != 0), "w gradient is all zero"
        assert torch.any(phi.grad != 0), "phi gradient is all zero"

    def test_two_component_sum_vs_brute_force(self):
        """
        Verify vectorized trig identity against brute-force N^2 loop.
        """
        kx_arr = np.array([0.2, 0.4])
        ky_arr = np.array([0.05, -0.1])
        depth = 50.0
        w_arr = np.array([0.3, 0.2])
        phi_arr = np.array([0.5, 1.2])
        k_mag = np.sqrt(kx_arr**2 + ky_arr**2)
        omega_arr = np.sqrt(G * k_mag * np.tanh(k_mag * depth))

        transfer = Stokes2Transfer(kx_arr, ky_arr, depth)
        L_plus = transfer.L_plus
        L_minus = transfer.L_minus

        x = np.linspace(10, 150, 40)
        y = np.linspace(-30, 30, 40)
        t = 5.0

        # Brute force
        z2_brute = np.zeros(len(x))
        for n in range(2):
            for m in range(2):
                theta_n = kx_arr[n]*x + ky_arr[n]*y - omega_arr[n]*t + phi_arr[n]
                theta_m = kx_arr[m]*x + ky_arr[m]*y - omega_arr[m]*t + phi_arr[m]
                z2_brute += w_arr[n] * w_arr[m] * L_plus[n, m] * np.cos(theta_n + theta_m)
                z2_brute += w_arr[n] * w_arr[m] * L_minus[n, m] * np.cos(theta_n - theta_m)

        # Vectorized
        z2_vec = transfer.correction_np(x, y, t, w_arr, kx_arr, ky_arr,
                                         phi_arr, omega_arr)

        np.testing.assert_allclose(z2_vec, z2_brute, atol=1e-10,
                                    err_msg="Vectorized != brute force")
