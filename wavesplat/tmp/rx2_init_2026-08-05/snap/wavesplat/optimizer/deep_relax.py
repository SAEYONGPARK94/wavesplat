"""
deep_relax.py
Deep Unfolded RELAX (DU-RELAX) for Dispersion-Constrained Spectral Estimation.

Unrolls the RELAX algorithm (Li & Stoica, 1996) into a differentiable
computation graph with learnable parameters per layer.

Architecture (K layers, one per spectral component):
  Layer k:
    1. Compute residual: r = z - sum of fitted components
    2. Dispersion-constrained power map on coarse grid (fixed, not learned)
    3. Soft argmax with learned temperature → initial (kx, ky)
    4. Learned gradient steps for (kx, ky) refinement
    5. Closed-form linear fit for (A, φ)
    6. Learned cyclic refinement with damping

Learnable parameters per layer:
  - step_size: gradient step size for (kx, ky) refinement
  - n_grad_steps: (fixed per layer, but step_size is learned)
  - damping: controls cyclic refinement update rate
  - temperature: soft argmax temperature for coarse search

Physics constraint (HARD, not learned):
  - ω = √(g·|k|·tanh(|k|·d))  always enforced
  - Linear fit for A, φ is closed-form (not learned)

Training:
  - Supervised: minimize ||z_pred - z_true||² on synthetic episodes
  - Self-supervised: minimize reconstruction error on own observations
"""

import torch
import torch.nn as nn
import numpy as np
import math
from typing import List, Tuple, Optional

G = 9.81


class DispersionConstraint:
    """Hard dispersion relation: ω = √(g·|k|·tanh(|k|·d))."""

    @staticmethod
    def omega(kx, ky, depth):
        """Compute angular frequency from wavenumber. Fully differentiable."""
        k_mag = torch.sqrt(kx**2 + ky**2 + 1e-12)
        return torch.sqrt(G * k_mag * torch.tanh(k_mag * depth))

    @staticmethod
    def phase(kx, ky, x, y, t, depth):
        """Compute phase: kx*x + ky*y - ω*t. Shape: (K, N) or (N,)."""
        omega = DispersionConstraint.omega(kx, ky, depth)
        return kx * x + ky * y - omega * t


class PowerMap(nn.Module):
    """
    Differentiable dispersion-constrained power map.

    Computes periodogram power P(kx, ky) = |<z, cos θ>|² + |<z, sin θ>|²
    on a fixed coarse grid. Output is differentiable w.r.t. z.
    """

    def __init__(self, kx_range=(-4.0, 4.0), ky_range=(-4.0, 4.0),
                 n_kx=41, n_ky=41):
        super().__init__()
        kx_axis = torch.linspace(kx_range[0], kx_range[1], n_kx)
        ky_axis = torch.linspace(ky_range[0], ky_range[1], n_ky)
        KX, KY = torch.meshgrid(kx_axis, ky_axis, indexing='ij')
        self.register_buffer('kx_grid', KX.reshape(-1))  # (G,)
        self.register_buffer('ky_grid', KY.reshape(-1))  # (G,)
        self.n_kx = n_kx
        self.n_ky = n_ky

    def forward(self, x, y, z, t, depth):
        """
        Args:
            x, y, z, t: (N,) observation tensors
            depth: scalar

        Returns:
            power: (G,) power at each grid point
            kx_grid, ky_grid: (G,) grid coordinates
        """
        N = x.shape[0]
        G_pts = self.kx_grid.shape[0]

        # Phase: (G, N)
        omega = DispersionConstraint.omega(
            self.kx_grid.unsqueeze(1), self.ky_grid.unsqueeze(1), depth
        )
        phase = (self.kx_grid.unsqueeze(1) * x.unsqueeze(0)
                 + self.ky_grid.unsqueeze(1) * y.unsqueeze(0)
                 - omega * t.unsqueeze(0))  # (G, N)

        cos_corr = torch.mean(z.unsqueeze(0) * torch.cos(phase), dim=1)  # (G,)
        sin_corr = torch.mean(z.unsqueeze(0) * torch.sin(phase), dim=1)  # (G,)
        power = cos_corr**2 + sin_corr**2  # (G,)

        return power


class SoftArgmax(nn.Module):
    """Differentiable soft argmax over power map."""

    def __init__(self):
        super().__init__()
        self.log_temperature = nn.Parameter(torch.tensor(2.0))

    def forward(self, power, kx_grid, ky_grid):
        """
        Args:
            power: (G,) power values
            kx_grid, ky_grid: (G,) grid coordinates

        Returns:
            kx, ky: soft-selected wavenumber (scalar tensors)
        """
        temperature = torch.exp(self.log_temperature) + 0.1
        weights = torch.softmax(power / temperature, dim=0)  # (G,)
        kx = torch.sum(weights * kx_grid)
        ky = torch.sum(weights * ky_grid)
        return kx, ky


class GradientRefinement(nn.Module):
    """
    Learned gradient refinement for (kx, ky).

    Takes gradient of negative power w.r.t. (kx, ky) and applies
    learned step size.
    """

    def __init__(self, n_steps=5):
        super().__init__()
        self.n_steps = n_steps
        # Learnable step size per step (log scale for positivity)
        # Small initial step sizes to prevent divergence
        self.log_step_sizes = nn.Parameter(torch.zeros(n_steps) - 5.0)

    def forward(self, kx, ky, x, y, z, t, depth):
        """Refine (kx, ky) via learned gradient steps.

        Uses analytical gradient of power w.r.t. (kx, ky) to keep
        the computation fully differentiable through the step_size parameter.

        Power P = cc² + ss² where cc = <z, cos(θ)>/N, ss = <z, sin(θ)>/N
        θ = kx*x + ky*y - ω(kx,ky)*t

        dP/dkx = 2*(cc * d_cc/dkx + ss * d_ss/dkx)
        d_cc/dkx = -<z, sin(θ) * dθ/dkx>/N
        d_ss/dkx = <z, cos(θ) * dθ/dkx>/N
        dθ/dkx = x - dω/dkx * t  (via dispersion derivative)
        """
        N = x.shape[0]

        for i in range(self.n_steps):
            step = torch.exp(self.log_step_sizes[i])

            # Compute phase and its derivatives w.r.t. kx, ky
            k_mag = torch.sqrt(kx**2 + ky**2 + 1e-12)
            omega = torch.sqrt(G * k_mag * torch.tanh(k_mag * depth))

            phase = kx * x + ky * y - omega * t

            cos_phase = torch.cos(phase)
            sin_phase = torch.sin(phase)

            cc = torch.dot(z, cos_phase) / N
            ss = torch.dot(z, sin_phase) / N

            # dω/d|k| via chain rule
            tanh_kd = torch.tanh(k_mag * depth)
            sech2_kd = 1.0 - tanh_kd**2
            domega_dk = (G * (tanh_kd + k_mag * depth * sech2_kd)) / (2.0 * omega + 1e-12)

            # dθ/dkx = x - (dω/d|k| * kx/|k|) * t
            dtheta_dkx = x - (domega_dk * kx / (k_mag + 1e-12)) * t
            dtheta_dky = y - (domega_dk * ky / (k_mag + 1e-12)) * t

            # d(cc)/dkx = -<z, sin(θ) * dθ/dkx>/N
            dcc_dkx = -torch.dot(z, sin_phase * dtheta_dkx) / N
            dss_dkx = torch.dot(z, cos_phase * dtheta_dkx) / N
            dcc_dky = -torch.dot(z, sin_phase * dtheta_dky) / N
            dss_dky = torch.dot(z, cos_phase * dtheta_dky) / N

            # dP/dkx = 2*(cc*dcc_dkx + ss*dss_dkx), clamp for stability
            grad_kx = 2.0 * (cc * dcc_dkx + ss * dss_dkx)
            grad_kx = torch.clamp(grad_kx, -1.0, 1.0)
            grad_ky = 2.0 * (cc * dcc_dky + ss * dss_dky)
            grad_ky = torch.clamp(grad_ky, -1.0, 1.0)

            # Gradient ascent (maximize power), clamp kx/ky to valid range
            kx = torch.clamp(kx + step * grad_kx, -6.0, 6.0)
            ky = torch.clamp(ky + step * grad_ky, -6.0, 6.0)

        return kx, ky


class LinearFit(nn.Module):
    """
    Closed-form linear fit for amplitude and phase.

    Not learned — this is the exact solution.
    z ≈ A·cos(θ + φ) = a·cos(θ) + b·sin(θ)
    """

    def forward(self, kx, ky, x, y, z, t, depth):
        """
        Returns:
            amp: amplitude (scalar)
            phi: phase (scalar)
        """
        phase = DispersionConstraint.phase(kx, ky, x, y, t, depth)
        c = torch.cos(phase)
        s = torch.sin(phase)

        cc = torch.dot(c, c)
        ss = torch.dot(s, s)
        cs = torch.dot(c, s)
        cz = torch.dot(c, z)
        sz = torch.dot(s, z)

        det = cc * ss - cs * cs
        det = torch.clamp(det, min=1e-10)

        a = (ss * cz - cs * sz) / det
        b = (cc * sz - cs * cz) / det

        amp = torch.sqrt(a**2 + b**2 + 1e-12)
        phi = torch.atan2(-b, a)

        return amp, phi


class CyclicRefinement(nn.Module):
    """
    Learned cyclic refinement (RELAX core).

    For each existing component, remove it, re-estimate via gradient,
    with learned damping factor.
    """

    def __init__(self, n_cycles=3, n_grad_steps=3):
        super().__init__()
        self.n_cycles = n_cycles
        self.grad_refine = GradientRefinement(n_steps=n_grad_steps)
        self.linear_fit = LinearFit()
        # Learned damping: how much to trust the new estimate vs old
        self.log_damping = nn.Parameter(torch.tensor(0.0))  # sigmoid → (0,1)

    def forward(self, components, x, y, z_centered, t, depth):
        """
        Cyclic refinement of all components.

        Args:
            components: list of (kx, ky, amp, phi) tensors
            x, y, z_centered, t: (N,) observations
            depth: scalar

        Returns:
            refined components: list of (kx, ky, amp, phi)
        """
        if len(components) < 2:
            return components

        damping = torch.sigmoid(self.log_damping)

        for cycle in range(self.n_cycles):
            for j in range(len(components)):
                # Build residual without component j
                z_without_j = z_centered.clone()
                for i, (ckx, cky, camp, cphi) in enumerate(components):
                    if i == j:
                        continue
                    ph = DispersionConstraint.phase(ckx, cky, x, y, t, depth)
                    z_without_j = z_without_j - camp * torch.cos(ph + cphi)

                old_kx, old_ky = components[j][0], components[j][1]

                # Gradient refinement from current position
                new_kx, new_ky = self.grad_refine(
                    old_kx, old_ky, x, y, z_without_j, t, depth
                )

                # Damped update
                kx_updated = old_kx + damping * (new_kx - old_kx)
                ky_updated = old_ky + damping * (new_ky - old_ky)

                # Re-fit amplitude and phase (closed form)
                amp_new, phi_new = self.linear_fit(
                    kx_updated, ky_updated, x, y, z_without_j, t, depth
                )

                components[j] = (kx_updated, ky_updated, amp_new, phi_new)

        return components


class DeepUnfoldedRELAX(nn.Module):
    """
    Deep Unfolded RELAX for dispersion-constrained spectral estimation.

    K layers, each adding one spectral component with learned parameters.
    Dispersion relation is always a hard constraint.

    Learnable parameters:
      - Per-layer soft argmax temperature
      - Per-layer gradient step sizes
      - Per-layer cyclic refinement damping

    Total learnable parameters: ~O(K * n_grad_steps) ≈ 100-300 (very small!)
    """

    def __init__(self, n_components=15, n_grad_steps=5, n_cycles=3,
                 kx_range=(-4.0, 4.0), ky_range=(-4.0, 4.0),
                 n_coarse=41):
        super().__init__()
        self.n_components = n_components

        # Shared modules
        self.power_map = PowerMap(kx_range, ky_range, n_coarse, n_coarse)
        self.linear_fit = LinearFit()

        # Per-layer modules
        self.soft_argmax_layers = nn.ModuleList([
            SoftArgmax() for _ in range(n_components)
        ])
        self.grad_refine_layers = nn.ModuleList([
            GradientRefinement(n_steps=n_grad_steps) for _ in range(n_components)
        ])
        self.cyclic_refine = CyclicRefinement(n_cycles=n_cycles,
                                               n_grad_steps=n_grad_steps)

    def forward(self, x, y, z, t, depth):
        """
        Estimate K spectral components from observations.

        Args:
            x, y, z, t: (N,) observation tensors
            depth: water depth (scalar)

        Returns:
            kx_out: (K,)
            ky_out: (K,)
            amp_out: (K,)
            phi_out: (K,)
            z_pred: (N,) predicted observations
        """
        z_centered = z - z.mean()
        components = []  # list of (kx, ky, amp, phi) tensor tuples

        for k in range(self.n_components):
            # Compute residual
            z_residual = z_centered.clone()
            for ckx, cky, camp, cphi in components:
                ph = DispersionConstraint.phase(ckx, cky, x, y, t, depth)
                z_residual = z_residual - camp * torch.cos(ph + cphi)

            # Coarse power map
            power = self.power_map(x, y, z_residual, t, depth)

            # Soft argmax → initial (kx, ky)
            kx_init, ky_init = self.soft_argmax_layers[k](
                power, self.power_map.kx_grid, self.power_map.ky_grid
            )

            # Learned gradient refinement
            kx_refined, ky_refined = self.grad_refine_layers[k](
                kx_init, ky_init, x, y, z_residual, t, depth
            )

            # Closed-form linear fit
            amp, phi = self.linear_fit(
                kx_refined, ky_refined, x, y, z_residual, t, depth
            )

            components.append((kx_refined, ky_refined, amp, phi))

        # Cyclic refinement (shared across all components)
        components = self.cyclic_refine(
            list(components), x, y, z_centered, t, depth
        )

        # Build output
        kx_out = torch.stack([c[0] for c in components])
        ky_out = torch.stack([c[1] for c in components])
        amp_out = torch.stack([c[2] for c in components])
        phi_out = torch.stack([c[3] for c in components])

        # Predicted observations
        z_pred = torch.zeros_like(z_centered)
        for ckx, cky, camp, cphi in components:
            ph = DispersionConstraint.phase(ckx, cky, x, y, t, depth)
            z_pred = z_pred + camp * torch.cos(ph + cphi)

        return kx_out, ky_out, amp_out, phi_out, z_pred

    def count_parameters(self):
        """Count total learnable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def train_du_relax(model, episodes, n_epochs=100, lr=1e-3, device='cpu'):
    """
    Train Deep Unfolded RELAX on synthetic episodes.

    Uses reconstruction loss: ||z_pred - z_centered||²

    Args:
        model: DeepUnfoldedRELAX instance
        episodes: list of (x, y, z, t, depth) tuples
        n_epochs: training epochs
        lr: learning rate

    Returns:
        losses: list of epoch losses
    """
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, n_epochs)

    losses = []

    for epoch in range(n_epochs):
        epoch_loss = 0.0
        n_samples = 0

        for x, y, z, t, depth in episodes:
            x = torch.tensor(x, dtype=torch.float32, device=device)
            y = torch.tensor(y, dtype=torch.float32, device=device)
            z = torch.tensor(z, dtype=torch.float32, device=device)
            t = torch.tensor(t, dtype=torch.float32, device=device)

            z_centered = z - z.mean()

            optimizer.zero_grad()

            _, _, _, _, z_pred = model(x, y, z, t, float(depth))

            # Reconstruction loss (self-supervised)
            loss = torch.mean((z_pred - z_centered)**2) / (torch.var(z_centered) + 1e-10)

            if torch.isnan(loss) or torch.isinf(loss):
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
            optimizer.step()

            epoch_loss += loss.item()
            n_samples += 1

        scheduler.step()
        avg_loss = epoch_loss / max(n_samples, 1)
        losses.append(avg_loss)

        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}/{n_epochs}: loss={avg_loss:.6f}, "
                  f"lr={scheduler.get_last_lr()[0]:.6f}")

    return losses
