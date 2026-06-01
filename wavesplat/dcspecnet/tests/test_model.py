"""
Tests for DCSpecNet model (grid-based spectral estimator).
"""
import torch
import numpy as np
import pytest
from dcspecnet.model import (
    DCSpecNet, DCSpecNetConfig, DCSpecNetLoss,
    dispersion_omega, MultiHeadAttention, ISAB, PMA,
)


class TestDispersionConstraint:
    """Hard dispersion constraint must be exact."""

    def test_dispersion_values(self):
        """Dispersion omega matches analytical formula."""
        kx = torch.tensor([0.1, 0.2, 0.5, 1.0])
        ky = torch.tensor([0.0, 0.1, 0.3, 0.0])
        depth = 50.0

        omega = dispersion_omega(kx, ky, depth)

        for i in range(len(kx)):
            k = (kx[i]**2 + ky[i]**2).sqrt().item()
            expected = np.sqrt(9.81 * k * np.tanh(k * depth))
            assert abs(omega[i].item() - expected) < 1e-6

    def test_dispersion_gradient_flows(self):
        """Gradients flow through dispersion computation."""
        kx = torch.tensor([0.3], requires_grad=True)
        ky = torch.tensor([0.2], requires_grad=True)
        omega = dispersion_omega(kx, ky, 50.0)
        omega.sum().backward()
        assert kx.grad is not None
        assert ky.grad is not None


class TestModelArchitecture:
    """Test model components."""

    def test_forward_shape(self):
        """Output shapes are correct."""
        cfg = DCSpecNetConfig(embed_dim=32, k_grid_n=5, n_heads=2,
                              n_isab=1, n_inducing=8)
        model = DCSpecNet(cfg)
        K = 25  # 5x5

        B, N = 2, 20
        pts = torch.randn(B, N, 3)
        out = model(pts)

        assert out['c_real'].shape == (B, K)
        assert out['c_imag'].shape == (B, K)
        assert out['A'].shape == (B, K)
        assert out['phi'].shape == (B, K)
        assert out['kx'].shape == (K,)
        assert out['ky'].shape == (K,)
        assert out['omega'].shape == (K,)

    def test_grid_dispersion_exact(self):
        """Grid omega is deterministic from grid kx/ky."""
        cfg = DCSpecNetConfig(embed_dim=32, k_grid_n=7, n_heads=2,
                              n_isab=1, n_inducing=8)
        model = DCSpecNet(cfg)

        omega_check = dispersion_omega(model.kx_grid, model.ky_grid, cfg.depth)
        assert torch.allclose(model.omega_grid, omega_check, atol=1e-7)

    def test_permutation_invariance(self):
        """Model output should not depend on point ordering."""
        cfg = DCSpecNetConfig(embed_dim=32, k_grid_n=5, n_heads=2,
                              n_isab=1, n_inducing=8)
        model = DCSpecNet(cfg)
        model.eval()

        pts = torch.randn(1, 20, 3)
        perm = torch.randperm(20)
        pts_perm = pts[:, perm, :]

        with torch.no_grad():
            out1 = model(pts)
            out2 = model(pts_perm)

        assert torch.allclose(out1['c_real'], out2['c_real'], atol=1e-5)
        assert torch.allclose(out1['c_imag'], out2['c_imag'], atol=1e-5)

    def test_top_components(self):
        """Top-N extraction returns correct shapes."""
        cfg = DCSpecNetConfig(embed_dim=32, k_grid_n=5, n_heads=2,
                              n_isab=1, n_inducing=8)
        model = DCSpecNet(cfg)

        pts = torch.randn(2, 20, 3)
        out = model(pts)
        top = model.top_components(out, n_top=3)

        assert top['kx'].shape == (2, 3)
        assert top['A'].shape == (2, 3)
        assert top['phi'].shape == (2, 3)
        assert top['omega'].shape == (2, 3)


class TestReconstruction:
    """Test wave field reconstruction from coefficients."""

    def test_reconstruction_shape(self):
        """Reconstruction produces correct output shape."""
        cfg = DCSpecNetConfig(embed_dim=32, k_grid_n=5, n_heads=2,
                              n_isab=1, n_inducing=8)
        model = DCSpecNet(cfg)

        pts = torch.randn(2, 20, 3)
        out = model(pts)
        X = torch.randn(2, 50)
        Y = torch.randn(2, 50)
        eta = model.reconstruct(out, X, Y, t=1.0)
        assert eta.shape == (2, 50)

    def test_reconstruction_linearity(self):
        """Reconstruction is linear in coefficients (convex loss)."""
        cfg = DCSpecNetConfig(embed_dim=32, k_grid_n=5, n_heads=2,
                              n_isab=1, n_inducing=8)
        model = DCSpecNet(cfg)
        K = 25

        X = torch.randn(1, 30)
        Y = torch.randn(1, 30)

        p1 = {'c_real': torch.randn(1, K), 'c_imag': torch.randn(1, K),
               'kx': model.kx_grid, 'ky': model.ky_grid, 'omega': model.omega_grid}
        p2 = {'c_real': torch.randn(1, K), 'c_imag': torch.randn(1, K),
               'kx': model.kx_grid, 'ky': model.ky_grid, 'omega': model.omega_grid}

        eta1 = model.reconstruct(p1, X, Y, t=0.0)
        eta2 = model.reconstruct(p2, X, Y, t=0.0)

        alpha = 0.7
        p_mix = {'c_real': alpha * p1['c_real'] + (1-alpha) * p2['c_real'],
                 'c_imag': alpha * p1['c_imag'] + (1-alpha) * p2['c_imag'],
                 'kx': model.kx_grid, 'ky': model.ky_grid, 'omega': model.omega_grid}
        eta_mix = model.reconstruct(p_mix, X, Y, t=0.0)

        expected = alpha * eta1 + (1-alpha) * eta2
        assert torch.allclose(eta_mix, expected, atol=1e-5)

    def test_reconstruction_with_known_wave(self):
        """Single wave component reconstructs correctly."""
        cfg = DCSpecNetConfig(k_grid_n=11, k_max=1.0, depth=50.0)
        model = DCSpecNet(cfg)
        K = 121

        dist = (model.kx_grid - 0.2)**2 + model.ky_grid**2
        idx = dist.argmin().item()

        cr = torch.zeros(1, K)
        ci = torch.zeros(1, K)
        cr[0, idx] = 1.0

        params = {'c_real': cr, 'c_imag': ci,
                  'kx': model.kx_grid, 'ky': model.ky_grid,
                  'omega': model.omega_grid}

        x = torch.linspace(-50, 50, 100).unsqueeze(0)
        y = torch.zeros_like(x)
        eta = model.reconstruct(params, x, y, t=0.0)

        kx_actual = model.kx_grid[idx].item()
        expected = torch.cos(kx_actual * x)
        assert torch.allclose(eta, expected, atol=1e-5)


class TestLoss:
    """Test loss computation."""

    def test_loss_decreases(self):
        """Loss should decrease during training on simple data."""
        cfg = DCSpecNetConfig(embed_dim=32, k_grid_n=5, n_heads=2,
                              n_isab=1, n_inducing=8)
        model = DCSpecNet(cfg)
        loss_fn = DCSpecNetLoss(lambda_sparse=0.0)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        x = torch.linspace(-50, 50, 50).unsqueeze(0).expand(8, -1)
        y = torch.zeros_like(x)
        z = 0.5 * torch.cos(0.2 * x + 1.0)
        pts = torch.stack([x, y, z], dim=-1)

        losses = []
        for _ in range(50):
            params = model(pts)
            loss, info = loss_fn(params, pts)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            losses.append(info['recon'])

        assert losses[-1] < losses[0] * 0.5, \
            f"Loss didn't decrease enough: {losses[0]:.4f} -> {losses[-1]:.4f}"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
