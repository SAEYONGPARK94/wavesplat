"""
coarse_to_fine.py
Progressive Gaussian refinement: N=3 → 8 → 15.

At each stage, existing parameters are preserved and new Gaussians
are initialized from residual FFT analysis.
"""

import torch
import numpy as np
from typing import List, Tuple

from wavesplat.model.spectral_gaussian import SpectralGaussianModel
from wavesplat.optimizer.initializer import (
    bin_observations_to_grid, fft_peak_extraction
)


class CoarseToFineSchedule:
    """
    Manages the coarse-to-fine N schedule.

    Usage:
        schedule = CoarseToFineSchedule(stages=[3, 8, 15])
        while schedule.has_next():
            n = schedule.current_n()
            # ... optimize ...
            if loss < schedule.refine_threshold():
                schedule.advance(model, frames, ...)
    """

    def __init__(self, stages: List[int] = None,
                 loss_thresholds: List[float] = None,
                 max_iters_per_stage: List[int] = None):
        """
        Args:
            stages: list of N values [3, 8, 15]
            loss_thresholds: loss below which to advance (None = iterate fully)
            max_iters_per_stage: max optimization iterations per stage
        """
        self.stages = stages or [3, 8, 15]
        self.loss_thresholds = loss_thresholds or [0.1, 0.05, None]
        self.max_iters = max_iters_per_stage or [200, 200, 500]
        self.current_stage = 0

    def current_n(self) -> int:
        return self.stages[self.current_stage]

    def has_next(self) -> bool:
        return self.current_stage < len(self.stages) - 1

    def should_advance(self, loss: float, n_iters: int) -> bool:
        """Check if we should move to next stage."""
        if self.current_stage >= len(self.stages) - 1:
            return False

        # NaN loss → force advance to avoid stuck training
        if np.isnan(loss):
            import warnings
            warnings.warn(f"NaN loss at stage {self.current_stage}, forcing advance")
            return True

        threshold = self.loss_thresholds[self.current_stage]
        max_iter = self.max_iters[self.current_stage]

        if threshold is not None and loss < threshold:
            return True
        if n_iters >= max_iter:
            return True
        return False

    def advance(self, model: SpectralGaussianModel,
                frames, residual_fn,
                x_range=(10.0, 250.0), y_range=(-100.0, 100.0)):
        """
        Move to next stage: add new Gaussians from residual analysis.

        Args:
            model: current model
            frames: observation frames for residual computation
            residual_fn: callable(model, frames) -> (x, y, residual_z) arrays
            x_range, y_range: spatial domain
        """
        if not self.has_next():
            return

        self.current_stage += 1
        n_new = self.stages[self.current_stage] - model.n_gaussians

        if n_new <= 0:
            return

        # Compute residuals
        all_x, all_y, all_res = residual_fn(model, frames)

        if len(all_x) > 0:
            # FFT on residuals to find new peaks
            grid_nx, grid_ny = 32, 24
            dx = (x_range[1] - x_range[0]) / grid_nx
            dy = (y_range[1] - y_range[0]) / grid_ny

            Z_res = bin_observations_to_grid(
                all_x, all_y, all_res, x_range, y_range, grid_nx, grid_ny
            )
            kx_new, ky_new, amp_new = fft_peak_extraction(Z_res, dx, dy, n_new)

            # Pad if needed
            n_found = len(kx_new)
            if n_found < n_new:
                extra = n_new - n_found
                kx_new = np.concatenate([kx_new, np.random.randn(extra) * 0.05])
                ky_new = np.concatenate([ky_new, np.random.randn(extra) * 0.05])
                amp_new = np.concatenate([amp_new, np.ones(extra) * 0.01])

            model.add_gaussians(
                n_new,
                init_kx=torch.tensor(kx_new[:n_new], dtype=torch.float32,
                                     device=model.w.device),
                init_ky=torch.tensor(ky_new[:n_new], dtype=torch.float32,
                                     device=model.w.device),
                init_w=torch.tensor(amp_new[:n_new], dtype=torch.float32,
                                    device=model.w.device),
                init_phi=torch.rand(n_new, device=model.w.device) * 2 * np.pi,
            )
        else:
            model.add_gaussians(n_new)
