"""
oracle_baseline.py
WaveSplat-oracle: upper-bound baseline using true wave components.

Initializes (kx, ky, w) from GT wave components, then optimizes only phases φᵢ.
This represents the theoretical best performance of the spectral Gaussian model.
"""

import torch
import torch.nn as nn
import numpy as np
import time
import math
from typing import List, Dict

from wavesplat.model.spectral_gaussian import SpectralGaussianModel
from wavesplat.model.dispersion import dispersion_omega
from wavesplat.data_loader import FrameData


class OracleBaseline:
    """
    WaveSplat-oracle: initialize with true (kx, ky, A) from GT components.

    Only φᵢ is optimized via gradient descent. This provides an upper bound
    on what the spectral Gaussian model can achieve with perfect spectral
    knowledge.

    Args:
        wave_components: (N_comp, 6) array [A, omega, kx, ky, phi, Q]
        depth: water depth [m]
        n_gaussians: number of Gaussians to use (top-N by amplitude)
        tracking_steps: gradient steps per frame for phase optimization
        lr: learning rate for phase optimization
        device: 'cpu' or 'cuda'
    """

    def __init__(self, wave_components: np.ndarray,
                 depth: float = 50.0,
                 n_gaussians: int = 15,
                 tracking_steps: int = 10,
                 lr: float = 0.05,
                 device: str = 'cpu'):
        self.depth = depth
        self.tracking_steps = tracking_steps
        self.lr = lr
        self.device = device

        # Select top-N components by amplitude
        amplitudes = wave_components[:, 0]
        n_use = min(n_gaussians, len(amplitudes))
        top_idx = np.argsort(amplitudes)[::-1][:n_use]

        self.true_A = wave_components[top_idx, 0]
        self.true_kx = wave_components[top_idx, 2]
        self.true_ky = wave_components[top_idx, 3]

        # Build model with true spectral parameters
        self.model = SpectralGaussianModel(n_use, depth=depth)
        self.model.to(device)

        with torch.no_grad():
            self.model.w.copy_(torch.tensor(self.true_A, dtype=torch.float32))
            self.model.kx.copy_(torch.tensor(self.true_kx, dtype=torch.float32))
            self.model.ky.copy_(torch.tensor(self.true_ky, dtype=torch.float32))
            # Phase initialized randomly — will be optimized
            self.model.phi.copy_(torch.rand(n_use) * 2 * math.pi)

        # Freeze w, kx, ky — only optimize phi
        self.model.w.requires_grad_(False)
        self.model.kx.requires_grad_(False)
        self.model.ky.requires_grad_(False)

        self.optimizer = torch.optim.Adam([self.model.phi], lr=self.lr)
        self._last_t = 0.0

    def initialize(self, frames: List[FrameData],
                   max_iters: int = 200,
                   verbose: bool = False) -> Dict:
        """
        Phase 1: Optimize phases over initialization window.

        Since w, kx, ky are true values, only φ needs fitting.
        """
        t0 = time.time()
        loss_history = []

        for iteration in range(max_iters):
            self.optimizer.zero_grad()
            total_loss = torch.tensor(0.0, device=self.device)
            n_obs = 0

            for frame in frames:
                if len(frame.x_obs) == 0:
                    continue
                z_pred = self.model(frame.x_obs, frame.y_obs, frame.t)
                residual = z_pred - frame.z_obs
                weighted_sq = frame.confidence * residual ** 2
                total_loss = total_loss + weighted_sq.sum()
                n_obs += len(frame.x_obs)

            if n_obs > 0:
                total_loss = total_loss / n_obs

            total_loss.backward()
            self.optimizer.step()
            loss_history.append(total_loss.item())

            if verbose and iteration % 50 == 0:
                print(f"  [Oracle Init] iter={iteration}, loss={total_loss.item():.6f}")

        if frames:
            self._last_t = frames[-1].t

        return {
            'loss_history': loss_history,
            'n_iters': max_iters,
            'elapsed_s': time.time() - t0,
        }

    def update(self, frame: FrameData) -> Dict:
        """
        Phase 2: Single-frame tracking with phase prediction + optimization.
        """
        t0 = time.time()

        # Phase prediction: φ += ω·Δt
        dt = frame.t - self._last_t
        if abs(dt) > 1e-6:
            with torch.no_grad():
                omega = self.model._omega()
                self.model.phi.add_(omega * dt)
        self._last_t = frame.t

        if len(frame.x_obs) == 0:
            return {'loss': float('nan'), 'n_pts': 0, 'elapsed_ms': 0.0}

        # Gradient update (phase only)
        for step in range(self.tracking_steps):
            self.optimizer.zero_grad()
            z_pred = self.model(frame.x_obs, frame.y_obs, frame.t)
            residual = z_pred - frame.z_obs
            loss = (frame.confidence * residual ** 2).mean()
            loss.backward()
            self.optimizer.step()

        elapsed_ms = (time.time() - t0) * 1000
        return {
            'loss': loss.item(),
            'n_pts': len(frame.x_obs),
            'elapsed_ms': elapsed_ms,
        }

    def predict_grid(self, grid_x: torch.Tensor, grid_y: torch.Tensor,
                     t: float) -> torch.Tensor:
        """Predict wave field on regular grid."""
        with torch.no_grad():
            return self.model.predict_grid(grid_x, grid_y, t)
