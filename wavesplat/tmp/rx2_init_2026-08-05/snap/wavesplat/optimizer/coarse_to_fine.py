"""
coarse_to_fine.py
Progressive Gaussian refinement: N=3 → 8 → 15.

At each stage, existing parameters are preserved and new Gaussians
are initialized from residual analysis. Supports two methods:
  - v1: spatial FFT on binned residuals (fast but Nyquist-limited)
  - v2: dispersion-aware beamforming on raw residuals (accurate)
"""

import torch
import numpy as np
from typing import List, Tuple

# Newly grown components historically get a UNIFORM RANDOM initial phase. This is the
# second site of the same defect investigated in tmp/VARIANCE_MECHANISM_HUNT_2026-07-23.md.
# True preserves shipped behaviour.
GROW_RANDOM_PHI = True

# `should_advance` historically compares the RAW training loss against absolute thresholds
# ([0.1, 0.05, None]). Those constants are scale-dependent: on a low-Hs sea the raw loss can
# already sit below 0.05, so the stage advances after a single gradient step and the residual
# beamforming that follows runs on an essentially untrained residual. Measured in
# tmp/VARIANCE_TRACE_fable_2026-07-23.md (seed 1 advanced after 1 iteration, seed 42 after 4)
# as the ROOT cause of the run-to-run bimodality. When True, the loss is normalised by the
# observation variance so the thresholds mean "fraction of signal variance still unexplained".
SCALE_INVARIANT_ADVANCE = False

from wavesplat.model.spectral_gaussian import SpectralGaussianModel
from wavesplat.optimizer.initializer import (
    bin_observations_to_grid, fft_peak_extraction,
    _beamform_coarse, _beamform_refine, _linear_projection,
    G as _G,
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
                 max_iters_per_stage: List[int] = None,
                 init_method: str = 'v1',
                 depth: float = 50.0):
        """
        Args:
            stages: list of N values [3, 8, 15]
            loss_thresholds: loss below which to advance (None = iterate fully)
            max_iters_per_stage: max optimization iterations per stage
            init_method: 'v1' for spatial FFT, 'v2'/'v2_konly' for beamforming
            depth: water depth [m] (used by beamforming)
        """
        self.stages = stages or [3, 8, 15]
        self.loss_thresholds = loss_thresholds or [0.1, 0.05, None]
        self.max_iters = max_iters_per_stage or [200, 200, 500]
        self.current_stage = 0
        self.init_method = init_method
        self.depth = depth

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

        rel_loss = loss
        if SCALE_INVARIANT_ADVANCE and getattr(self, 'loss_scale', None):
            rel_loss = loss / self.loss_scale

        if threshold is not None and rel_loss < threshold:
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
                         OR callable returning (x, y, residual_z, t) for v2
            x_range, y_range: spatial domain
        """
        if not self.has_next():
            return

        self.current_stage += 1
        n_new = self.stages[self.current_stage] - model.n_gaussians

        if n_new <= 0:
            return

        # Compute residuals
        result = residual_fn(model, frames)

        # 'oracle' = RESEARCH-ONLY S2 arm (2026-08-05): true (kx, ky) at init, everything
        # downstream identical to 'v2_konly', so residual refinement must beamform too.
        use_beamform = self.init_method in ('v2', 'v2_konly', 'relax', 'relax_konly',
                                            'oracle')

        if use_beamform and len(result) == 4:
            # v2: beamforming on residuals with timestamps
            all_x, all_y, all_res, all_t = result
            if len(all_x) > 0:
                kx_new, ky_new, amp_new = self._beamform_residuals(
                    all_x, all_y, all_res, all_t, n_new)
            else:
                kx_new = np.random.randn(n_new) * 0.05
                ky_new = np.random.randn(n_new) * 0.05
                amp_new = np.ones(n_new) * 0.01
        else:
            # v1: spatial FFT on residuals
            all_x, all_y, all_res = result[:3]
            if len(all_x) > 0:
                grid_nx, grid_ny = 32, 24
                dx = (x_range[1] - x_range[0]) / grid_nx
                dy = (y_range[1] - y_range[0]) / grid_ny
                Z_res = bin_observations_to_grid(
                    all_x, all_y, all_res, x_range, y_range,
                    grid_nx, grid_ny)
                kx_new, ky_new, amp_new = fft_peak_extraction(
                    Z_res, dx, dy, n_new)
            else:
                kx_new = np.array([])
                ky_new = np.array([])
                amp_new = np.array([])

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
            init_phi=(torch.rand(n_new, device=model.w.device) * 2 * np.pi
                      if GROW_RANDOM_PHI else
                      torch.zeros(n_new, device=model.w.device)),
        )

    def _beamform_residuals(self, x, y, z_res, t, n_new):
        """Find new peaks in residuals using dispersion-aware beamforming."""
        power, kx_axis, ky_axis = _beamform_coarse(
            x, y, z_res, t, self.depth,
            kx_range=(-4.0, 4.0), ky_range=(-4.0, 4.0),
            n_kx=61, n_ky=61  # coarser grid for speed
        )

        # Zero out DC
        dkx = kx_axis[1] - kx_axis[0]
        cx = np.argmin(np.abs(kx_axis))
        cy = np.argmin(np.abs(ky_axis))
        power[max(0, cy - 1):cy + 2, max(0, cx - 1):cx + 2] = 0.0

        kx_list, ky_list, amp_list = [], [], []
        from wavesplat.optimizer.initializer import MIN_DK
        min_dk = MIN_DK

        for _ in range(n_new):
            flat_idx = np.argmax(power.ravel())
            iy_p, ix_p = np.unravel_index(flat_idx, power.shape)
            if power[iy_p, ix_p] < 1e-12:
                break

            kx_c, ky_c = kx_axis[ix_p], ky_axis[iy_p]
            kx_f, ky_f, _ = _beamform_refine(
                x, y, z_res, t, self.depth, kx_c, ky_c,
                delta=0.15, n_fine=31)
            amp, _ = _linear_projection(x, y, z_res, t, kx_f, ky_f, self.depth)

            kx_list.append(kx_f)
            ky_list.append(ky_f)
            amp_list.append(amp)

            # Suppress neighborhood
            suppress = int(min_dk / dkx) + 1
            y_lo = max(0, iy_p - suppress)
            y_hi = min(power.shape[0], iy_p + suppress + 1)
            x_lo = max(0, ix_p - suppress)
            x_hi = min(power.shape[1], ix_p + suppress + 1)
            power[y_lo:y_hi, x_lo:x_hi] = 0.0

        return np.array(kx_list), np.array(ky_list), np.array(amp_list)
