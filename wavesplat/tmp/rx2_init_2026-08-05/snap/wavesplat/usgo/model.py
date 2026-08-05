"""
USGO: Unrolled Spectral Gaussian Optimizer.

Algorithm unrolling applied to WaveSplat's spectral estimation.
Two variants:
  - USGOSimple: learned step sizes only (~40 params for K=10)
  - USGOModel: per-component MLP (~5K-50K params)

Each layer:
  1. Compute analytical gradient of observation loss
  2. Normalize gradient
  3. Apply learned update (step size or MLP)
  4. Soft-clamp to physical range

Reference: Monga et al. 2021 "Algorithm Unrolling" (IEEE SPM)
"""

import torch
import torch.nn as nn
import math

from wavesplat.model.dispersion import dispersion_omega, G


def analytical_grads(kx, ky, w, phi, x_obs, y_obs, z_obs, depth,
                     t, conf=None):
    """Compute observation loss and analytical gradients.

    Returns loss and per-component gradients (N,) each.
    Gradients flow through kx, ky, w, phi for end-to-end training.
    """
    k_mag = torch.sqrt(kx ** 2 + ky ** 2 + 1e-12)
    omega = dispersion_omega(k_mag, depth)

    phase = (kx[None, :] * x_obs[:, None]
             + ky[None, :] * y_obs[:, None]
             - omega[None, :] * t
             + phi[None, :])

    cos_phase = torch.cos(phase)
    sin_phase = torch.sin(phase)

    z_pred = torch.sum(w[None, :] * cos_phase, dim=1)
    residual = z_obs - z_pred

    if conf is not None:
        weighted_r = conf * residual
    else:
        weighted_r = residual

    M = float(x_obs.shape[0])

    g_w = -2.0 / M * (weighted_r[:, None] * cos_phase).sum(dim=0)
    g_kx = 2.0 / M * (weighted_r[:, None] * w[None, :] *
                       x_obs[:, None] * sin_phase).sum(dim=0)
    g_ky = 2.0 / M * (weighted_r[:, None] * w[None, :] *
                       y_obs[:, None] * sin_phase).sum(dim=0)
    g_phi = 2.0 / M * (weighted_r[:, None] * w[None, :] *
                        sin_phase).sum(dim=0)

    loss = torch.mean(weighted_r * residual)
    return loss, g_kx, g_ky, g_w, g_phi


def normalize_grads(g_kx, g_ky, g_w, g_phi):
    """Normalize gradients per-component to prevent explosion."""
    g_stack = torch.stack([g_kx, g_ky, g_w, g_phi], dim=-1)
    g_norm = torch.norm(g_stack, dim=-1, keepdim=True) + 1e-8
    g_normed = g_stack / g_norm
    return g_normed[:, 0], g_normed[:, 1], g_normed[:, 2], g_normed[:, 3]


# ========== USGOSimple: learned step sizes only ==========

class USGOSimple(nn.Module):
    """Minimal USGO: only learned step sizes per layer per param group.

    Total params: K * 4 (e.g., 40 for K=10).
    Cannot overfit — tests fundamental value of algorithm unrolling.
    """

    def __init__(self, n_layers: int = 10, depth: float = 50.0):
        super().__init__()
        self.n_layers = n_layers
        self.depth = depth

        # Learnable step sizes (log-parameterized for positivity)
        # exp(-3.9) ≈ 0.02, exp(-4.6) ≈ 0.01, exp(-3.0) ≈ 0.05
        init_log_ss = torch.tensor([[-3.9, -3.9, -4.6, -3.0]] * n_layers)
        self.log_step_sizes = nn.Parameter(init_log_ss)

        # Learnable momentum (sigmoid-parameterized to [0, 1])
        # logit(0.5) = 0, logit(0.9) ≈ 2.2
        self.momentum_logit = nn.Parameter(torch.zeros(n_layers, 4))

    def forward(self, init_kx, init_ky, init_w, init_phi,
                x_obs, y_obs, z_obs, t, depth, conf=None):
        kx, ky, w, phi = init_kx, init_ky, init_w, init_phi
        prev_delta = torch.zeros(4, len(kx), device=kx.device)
        losses = []

        for k in range(self.n_layers):
            loss, g_kx, g_ky, g_w, g_phi = analytical_grads(
                kx, ky, w, phi, x_obs, y_obs, z_obs, depth, t, conf)
            losses.append(loss)

            g_kx, g_ky, g_w, g_phi = normalize_grads(g_kx, g_ky, g_w, g_phi)

            ss = torch.exp(self.log_step_sizes[k])
            mom = torch.sigmoid(self.momentum_logit[k])

            # Gradient descent with momentum
            delta_kx = -ss[0] * g_kx + mom[0] * prev_delta[0]
            delta_ky = -ss[1] * g_ky + mom[1] * prev_delta[1]
            delta_w = -ss[2] * g_w + mom[2] * prev_delta[2]
            delta_phi = -ss[3] * g_phi + mom[3] * prev_delta[3]

            kx = kx + delta_kx
            ky = ky + delta_ky
            w = w + delta_w
            phi = phi + delta_phi

            prev_delta = torch.stack([delta_kx, delta_ky, delta_w, delta_phi])

            # Soft clamp
            kx = torch.tanh(kx / 5.0) * 5.0
            ky = torch.tanh(ky / 5.0) * 5.0
            w = torch.tanh(w / 3.0) * 3.0

        # Final loss
        final_loss, _, _, _, _ = analytical_grads(
            kx, ky, w, phi, x_obs, y_obs, z_obs, depth, t, conf)
        losses.append(final_loss)

        return kx, ky, w, phi, losses

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ========== USGOModel: per-component MLP ==========

class USGOLayer(nn.Module):
    """One unrolled optimization layer with per-component MLP."""

    def __init__(self, hidden_dim: int = 32):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(8, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 4),
        )
        nn.init.uniform_(self.mlp[-1].weight, -0.01, 0.01)
        nn.init.zeros_(self.mlp[-1].bias)

        self.output_scale = nn.Parameter(
            torch.tensor([0.05, 0.05, 0.02, 0.1]))

    def forward(self, kx, ky, w, phi, grad_kx, grad_ky, grad_w, grad_phi):
        features = torch.stack([kx, ky, w, phi,
                                grad_kx, grad_ky, grad_w, grad_phi], dim=-1)
        delta = self.mlp(features) * self.output_scale

        return (kx + delta[:, 0], ky + delta[:, 1],
                w + delta[:, 2], phi + delta[:, 3])


class USGOModel(nn.Module):
    """K-layer Unrolled Spectral Gaussian Optimizer with MLP."""

    def __init__(self, n_layers: int = 10, hidden_dim: int = 32,
                 depth: float = 50.0):
        super().__init__()
        self.n_layers = n_layers
        self.depth = depth

        self.layers = nn.ModuleList([
            USGOLayer(hidden_dim=hidden_dim) for _ in range(n_layers)
        ])

    def forward(self, init_kx, init_ky, init_w, init_phi,
                x_obs, y_obs, z_obs, t, depth, conf=None):
        kx, ky, w, phi = init_kx, init_ky, init_w, init_phi
        losses = []

        for layer in self.layers:
            loss, g_kx, g_ky, g_w, g_phi = analytical_grads(
                kx, ky, w, phi, x_obs, y_obs, z_obs, depth, t, conf)
            losses.append(loss)

            g_kx, g_ky, g_w, g_phi = normalize_grads(g_kx, g_ky, g_w, g_phi)
            kx, ky, w, phi = layer(kx, ky, w, phi,
                                    g_kx, g_ky, g_w, g_phi)

            kx = torch.tanh(kx / 5.0) * 5.0
            ky = torch.tanh(ky / 5.0) * 5.0
            w = torch.tanh(w / 3.0) * 3.0

        final_loss, _, _, _, _ = analytical_grads(
            kx, ky, w, phi, x_obs, y_obs, z_obs, depth, t, conf)
        losses.append(final_loss)

        return kx, ky, w, phi, losses

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
