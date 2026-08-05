"""
Tests for Neural Corrector modules.

Tests both NeuralCorrector (per-component) and NeuralFieldCorrector (field-level).
"""

import torch
import numpy as np
import pytest

from wavesplat.model.neural_corrector import (
    NeuralCorrector, NeuralFieldCorrector, NeuralFieldCorrectorV2
)

G = 9.81


class TestNeuralCorrectorArchitecture:

    def test_output_shapes(self):
        model = NeuralCorrector(hidden_dim=32, n_layers=2)
        N = 15
        kx = torch.randn(N) * 0.1
        ky = torch.randn(N) * 0.1
        w = torch.randn(N) * 0.3
        d_omega, d_w = model(kx, ky, w, 50.0)
        assert d_omega.shape == (N,)
        assert d_w.shape == (N,)

    def test_permutation_invariance(self):
        model = NeuralCorrector(hidden_dim=32, n_layers=2)
        N = 5
        torch.manual_seed(42)
        kx = torch.randn(N) * 0.2
        ky = torch.randn(N) * 0.1
        w = torch.randn(N) * 0.3
        d_omega1, d_w1 = model(kx, ky, w, 50.0)
        perm = torch.tensor([3, 1, 4, 0, 2])
        d_omega2, d_w2 = model(kx[perm], ky[perm], w[perm], 50.0)
        torch.testing.assert_close(d_omega1[perm], d_omega2, atol=1e-5, rtol=1e-4)
        torch.testing.assert_close(d_w1[perm], d_w2, atol=1e-5, rtol=1e-4)

    def test_gradient_flow(self):
        model = NeuralCorrector(hidden_dim=32, n_layers=2)
        N = 10
        kx = torch.randn(N) * 0.2
        ky = torch.randn(N) * 0.1
        w = torch.randn(N) * 0.3
        w.requires_grad_(True)
        d_omega, d_w = model(kx, ky, w, 50.0)
        loss = d_omega.sum() + d_w.sum()
        loss.backward()
        assert w.grad is not None
        for p in model.parameters():
            assert p.grad is not None

    def test_model_size(self):
        model = NeuralCorrector(hidden_dim=64, n_layers=3)
        assert model.count_parameters() < 100000

    def test_variable_n(self):
        model = NeuralCorrector(hidden_dim=32, n_layers=2)
        for N in [5, 10, 30, 50]:
            kx = torch.randn(N) * 0.2
            ky = torch.randn(N) * 0.1
            w = torch.randn(N) * 0.3
            d_omega, d_w = model(kx, ky, w, 50.0)
            assert d_omega.shape == (N,)


class TestNeuralFieldCorrector:

    def test_output_shape(self):
        model = NeuralFieldCorrector(hidden_dim=32, n_layers=2)
        N, M = 10, 50
        kx = torch.randn(N) * 0.1
        ky = torch.randn(N) * 0.1
        w = torch.randn(N) * 0.3
        phi = torch.randn(N)
        omega = torch.abs(torch.randn(N))
        x = torch.randn(M) * 100
        y = torch.randn(M) * 100
        delta_eta = model(kx, ky, w, phi, omega, 50.0, x, y, 10.0)
        assert delta_eta.shape == (M,)

    def test_gradient_flow(self):
        model = NeuralFieldCorrector(hidden_dim=32, n_layers=2)
        N, M = 10, 20
        kx = torch.randn(N) * 0.1
        ky = torch.randn(N) * 0.1
        w = torch.randn(N) * 0.3
        w.requires_grad_(True)
        phi = torch.randn(N)
        omega = torch.abs(torch.randn(N))
        x = torch.randn(M) * 100
        y = torch.randn(M) * 100
        delta_eta = model(kx, ky, w, phi, omega, 50.0, x, y, 10.0)
        loss = delta_eta.sum()
        loss.backward()
        assert w.grad is not None
        for p in model.parameters():
            assert p.grad is not None

    def test_model_size(self):
        model = NeuralFieldCorrector(hidden_dim=64, n_layers=2)
        assert model.count_parameters() < 50000

    def test_deterministic(self):
        model = NeuralFieldCorrector(hidden_dim=32, n_layers=2)
        model.eval()
        N, M = 10, 20
        kx = torch.randn(N) * 0.1
        ky = torch.randn(N) * 0.1
        w = torch.randn(N) * 0.3
        phi = torch.randn(N)
        omega = torch.abs(torch.randn(N))
        x = torch.randn(M) * 100
        y = torch.randn(M) * 100
        out1 = model(kx, ky, w, phi, omega, 50.0, x, y, 10.0)
        out2 = model(kx, ky, w, phi, omega, 50.0, x, y, 10.0)
        torch.testing.assert_close(out1, out2)

    def test_single_sample_learnability(self):
        """Model should be able to overfit a single synthetic target."""
        torch.manual_seed(42)
        model = NeuralFieldCorrector(hidden_dim=64, n_layers=2)
        N, M = 10, 100
        kx = torch.randn(N) * 0.1
        ky = torch.randn(N) * 0.05
        w = torch.randn(N) * 0.3
        phi = torch.randn(N)
        omega = torch.abs(torch.randn(N)) * 0.5 + 0.5
        x = torch.linspace(0, 200, M)
        y = torch.linspace(0, 150, M)

        # Create synthetic target (sum of cosines at different k)
        target = 0.01 * torch.cos(0.1 * x + 0.05 * y)

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        initial_loss = None
        for step in range(200):
            pred = model(kx, ky, w, phi, omega, 50.0, x, y, 10.0)
            loss = torch.nn.functional.mse_loss(pred, target)
            if initial_loss is None:
                initial_loss = loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        final_loss = loss.item()
        # Should reduce loss significantly
        assert final_loss < initial_loss * 0.5, \
            f"Model didn't learn: initial={initial_loss:.6f}, final={final_loss:.6f}"


class TestNeuralFieldCorrectorV2:

    def test_output_shape(self):
        model = NeuralFieldCorrectorV2(hidden_dim=32, n_heads=4, n_attn_layers=1)
        N, M = 10, 50
        kx = torch.randn(N) * 0.1
        ky = torch.randn(N) * 0.1
        w = torch.randn(N) * 0.3
        phi = torch.randn(N)
        omega = torch.abs(torch.randn(N))
        x = torch.randn(M) * 100
        y = torch.randn(M) * 100
        delta_eta = model(kx, ky, w, phi, omega, 50.0, x, y, 10.0)
        assert delta_eta.shape == (M,)

    def test_gradient_flow(self):
        model = NeuralFieldCorrectorV2(hidden_dim=32, n_heads=4, n_attn_layers=1)
        N, M = 10, 20
        kx = torch.randn(N) * 0.1
        ky = torch.randn(N) * 0.1
        w = torch.randn(N) * 0.3
        w.requires_grad_(True)
        phi = torch.randn(N)
        omega = torch.abs(torch.randn(N))
        x = torch.randn(M) * 100
        y = torch.randn(M) * 100
        delta_eta = model(kx, ky, w, phi, omega, 50.0, x, y, 10.0)
        loss = delta_eta.sum()
        loss.backward()
        assert w.grad is not None
        for p in model.parameters():
            assert p.grad is not None

    def test_model_size(self):
        model = NeuralFieldCorrectorV2(hidden_dim=64, n_heads=4, n_attn_layers=2)
        n_params = model.count_parameters()
        assert n_params < 200000, f"Too many params: {n_params}"
        assert n_params > 10000, f"Too few params: {n_params}"

    def test_variable_n(self):
        model = NeuralFieldCorrectorV2(hidden_dim=32, n_heads=4, n_attn_layers=1)
        model.eval()
        M = 20
        x = torch.randn(M) * 100
        y = torch.randn(M) * 100
        for N in [5, 10, 30]:
            kx = torch.randn(N) * 0.1
            ky = torch.randn(N) * 0.1
            w = torch.randn(N) * 0.3
            phi = torch.randn(N)
            omega = torch.abs(torch.randn(N))
            delta_eta = model(kx, ky, w, phi, omega, 50.0, x, y, 10.0)
            assert delta_eta.shape == (M,)

    def test_single_sample_learnability(self):
        torch.manual_seed(42)
        model = NeuralFieldCorrectorV2(hidden_dim=64, n_heads=4, n_attn_layers=2)
        N, M = 10, 100
        kx = torch.randn(N) * 0.1
        ky = torch.randn(N) * 0.05
        w = torch.randn(N) * 0.3
        phi = torch.randn(N)
        omega = torch.abs(torch.randn(N)) * 0.5 + 0.5
        x = torch.linspace(0, 200, M)
        y = torch.linspace(0, 150, M)

        target = 0.01 * torch.cos(0.1 * x + 0.05 * y)

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        initial_loss = None
        for step in range(200):
            pred = model(kx, ky, w, phi, omega, 50.0, x, y, 10.0)
            loss = torch.nn.functional.mse_loss(pred, target)
            if initial_loss is None:
                initial_loss = loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        final_loss = loss.item()
        assert final_loss < initial_loss * 0.5, \
            f"V2 didn't learn: initial={initial_loss:.6f}, final={final_loss:.6f}"

    def test_same_interface_as_v1(self):
        """V2 has identical forward signature as V1."""
        v1 = NeuralFieldCorrector(hidden_dim=32)
        v2 = NeuralFieldCorrectorV2(hidden_dim=32, n_heads=4)
        N, M = 10, 20
        kx = torch.randn(N) * 0.1
        ky = torch.randn(N) * 0.1
        w = torch.randn(N) * 0.3
        phi = torch.randn(N)
        omega = torch.abs(torch.randn(N))
        x = torch.randn(M) * 100
        y = torch.randn(M) * 100
        out_v1 = v1(kx, ky, w, phi, omega, 50.0, x, y, 10.0)
        out_v2 = v2(kx, ky, w, phi, omega, 50.0, x, y, 10.0)
        assert out_v1.shape == out_v2.shape == (M,)
