"""
sequential_update.py
Two-phase sequential wave field estimation.

Phase 1 (Initialization): Joint optimization over first 2*Tp frames
    - Coarse-to-fine: N=3 → 8 → 15
    - FFT-based initial values
    - Full window optimization

Phase 2 (Tracking): Per-frame warm-start update
    - Predict: φᵢ += ωᵢ·Δt
    - Update: 5-10 gradient steps
    - Target: 50ms/frame
"""

import torch
import torch.nn as nn
import numpy as np
import time
from typing import List, Dict, Optional

from wavesplat.model.spectral_gaussian import SpectralGaussianModel
from wavesplat.model.dispersion import dispersion_omega
from wavesplat.optimizer.initializer import initialize_from_frames
from wavesplat.optimizer.coarse_to_fine import CoarseToFineSchedule
from wavesplat.data_loader import FrameData


class SequentialUpdater:
    """
    Two-phase sequential wave field estimator.

    Usage:
        updater = SequentialUpdater(config)
        updater.initialize(init_frames)
        for frame in tracking_frames:
            result = updater.update(frame)
    """

    def __init__(self, config: dict):
        """
        Args:
            config: dict with keys:
                n_gaussians: list [3, 8, 15] or int
                depth: water depth [m]
                lr: learning rate (default 0.01)
                lambda_l1: L1 sparsity weight (default 0.001)
                lambda_smooth: smoothness weight (default 0.0001)
                init_max_iters: max iterations for initialization (default 500)
                tracking_steps: gradient steps per frame (default 10)
                device: 'cpu' or 'cuda'
                loss_thresholds: list of loss thresholds per stage (default [0.1, 0.05, None])
                max_iters_per_stage: list of max iters per stage (default [200, 200, 500])
                tracking_mode: 'phi_only', 'w_phi', or 'all' (default 'phi_only')
                tracking_lr: learning rate for tracking phase (default 0.005)
                max_steepness: wave steepness clamp (default 0.15)
        """
        self.config = config
        self.depth = config.get('depth', 50.0)
        self.lr = config.get('lr', 0.01)
        self.lambda_l1 = config.get('lambda_l1', 0.001)
        self.lambda_smooth = config.get('lambda_smooth', 0.0001)
        self.tracking_steps = config.get('tracking_steps', 10)
        self.device = config.get('device', 'cpu')
        self.loss_thresholds = config.get('loss_thresholds', None)
        self.max_iters_per_stage = config.get('max_iters_per_stage', None)
        self.tracking_mode = config.get('tracking_mode', 'w_phi')
        self.tracking_lr = config.get('tracking_lr', 0.005)

        n_gaussians = config.get('n_gaussians', [3, 8, 15])
        if isinstance(n_gaussians, int):
            self.stages = [n_gaussians]
        else:
            self.stages = list(n_gaussians)

        self.model = None
        self.optimizer = None
        self.tracking_optimizer = None
        self._last_dt = 0.1

    def _build_model(self, n: int) -> SpectralGaussianModel:
        model = SpectralGaussianModel(n, depth=self.depth)
        model.to(self.device)
        return model

    def _build_optimizer(self):
        """Build optimizer for initialization phase (all params unfrozen)."""
        param_groups = [
            {'params': [self.model.w, self.model.phi], 'lr': self.lr},
            {'params': [self.model.kx, self.model.ky], 'lr': self.lr * 0.5},
        ]
        self.optimizer = torch.optim.Adam(param_groups)

    def _build_tracking_optimizer(self):
        """Build optimizer for tracking phase based on tracking_mode.

        Freezes parameters not being updated and creates a fresh Adam optimizer
        with tracking_lr for the remaining parameters.
        """
        mode = self.tracking_mode

        if mode == 'phi_only':
            self.model.kx.requires_grad_(False)
            self.model.ky.requires_grad_(False)
            self.model.w.requires_grad_(False)
            self.model.phi.requires_grad_(True)
            params = [{'params': [self.model.phi], 'lr': self.tracking_lr}]
        elif mode == 'w_phi':
            self.model.kx.requires_grad_(False)
            self.model.ky.requires_grad_(False)
            self.model.w.requires_grad_(True)
            self.model.phi.requires_grad_(True)
            params = [{'params': [self.model.w, self.model.phi],
                        'lr': self.tracking_lr}]
        else:  # 'all' — original behavior
            self.model.kx.requires_grad_(True)
            self.model.ky.requires_grad_(True)
            self.model.w.requires_grad_(True)
            self.model.phi.requires_grad_(True)
            params = [
                {'params': [self.model.w, self.model.phi],
                 'lr': self.tracking_lr},
                {'params': [self.model.kx, self.model.ky],
                 'lr': self.tracking_lr * 0.5},
            ]

        self.tracking_optimizer = torch.optim.Adam(params)

    def _compute_loss(self, frames: List[FrameData]) -> torch.Tensor:
        """Compute weighted MSE + regularization over multiple frames."""
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

        # L1 sparsity on weights
        if self.lambda_l1 > 0:
            total_loss = total_loss + self.lambda_l1 * torch.sum(torch.abs(self.model.w))

        # Spectral smoothness: penalize weight variation (sorted by |k|)
        if self.lambda_smooth > 0 and self.model.n_gaussians > 1:
            with torch.no_grad():
                k_mag = torch.sqrt(self.model.kx ** 2 + self.model.ky ** 2)
                sort_idx = torch.argsort(k_mag)
            w_sorted = self.model.w[sort_idx]
            smooth_penalty = torch.sum((w_sorted[1:] - w_sorted[:-1]) ** 2)
            total_loss = total_loss + self.lambda_smooth * smooth_penalty

        return total_loss

    def _compute_residuals(self, model, frames):
        """Compute observation residuals for coarse-to-fine refinement."""
        all_x, all_y, all_res = [], [], []
        with torch.no_grad():
            for frame in frames:
                if len(frame.x_obs) == 0:
                    continue
                z_pred = model(frame.x_obs, frame.y_obs, frame.t)
                res = (frame.z_obs - z_pred).cpu().numpy()
                all_x.append(frame.x_obs.cpu().numpy())
                all_y.append(frame.y_obs.cpu().numpy())
                all_res.append(res)

        if all_x:
            return np.concatenate(all_x), np.concatenate(all_y), np.concatenate(all_res)
        return np.array([]), np.array([]), np.array([])

    def initialize(self, frames: List[FrameData],
                   x_range=(10.0, 250.0), y_range=(-100.0, 100.0),
                   verbose: bool = False) -> Dict:
        """
        Phase 1: Joint optimization over initialization window.

        Args:
            frames: list of FrameData from init window
            x_range, y_range: spatial domain
            verbose: print progress

        Returns:
            dict with 'loss_history', 'n_iters', 'elapsed_s'
        """
        t0 = time.time()
        loss_history = []

        # Coarse-to-fine schedule (configurable thresholds)
        default_thresholds = [0.1, 0.05, None] if len(self.stages) == 3 else [None]
        default_max_iters = [200, 200, 500] if len(self.stages) == 3 else [500]
        schedule = CoarseToFineSchedule(
            stages=self.stages,
            loss_thresholds=self.loss_thresholds or default_thresholds,
            max_iters_per_stage=self.max_iters_per_stage or default_max_iters,
        )

        # Initialize first stage
        n_init = schedule.current_n()
        init_kx, init_ky, init_w, init_phi = initialize_from_frames(
            frames, n_init, self.depth, x_range, y_range
        )

        self.model = self._build_model(n_init)
        with torch.no_grad():
            self.model.kx.copy_(init_kx.to(self.device))
            self.model.ky.copy_(init_ky.to(self.device))
            self.model.w.copy_(init_w.to(self.device))
            self.model.phi.copy_(init_phi.to(self.device))
        self._build_optimizer()

        stage_iters = 0
        total_iters = 0
        max_total = sum(schedule.max_iters)

        for iteration in range(max_total):
            self.optimizer.zero_grad()
            loss = self._compute_loss(frames)
            loss.backward()
            self.optimizer.step()
            self.model.clamp_steepness()

            loss_val = loss.item()
            loss_history.append(loss_val)
            stage_iters += 1
            total_iters += 1

            if verbose and iteration % 50 == 0:
                print(f"  [Init] iter={iteration}, N={self.model.n_gaussians}, "
                      f"loss={loss_val:.6f}")

            # Check coarse-to-fine advancement
            if schedule.should_advance(loss_val, stage_iters):
                if verbose:
                    print(f"  [Refine] N={schedule.current_n()} → "
                          f"{schedule.stages[schedule.current_stage + 1]}")
                schedule.advance(
                    self.model, frames, self._compute_residuals,
                    x_range, y_range
                )
                self._build_optimizer()
                stage_iters = 0

                if not schedule.has_next():
                    # Final stage: continue until max_iters
                    continue

        elapsed = time.time() - t0
        if verbose:
            print(f"  [Init complete] {total_iters} iters, "
                  f"N={self.model.n_gaussians}, "
                  f"loss={loss_history[-1]:.6f}, {elapsed:.1f}s")

        # Set _last_t for tracking phase
        if frames:
            self._last_t = frames[-1].t

        # Build tracking-specific optimizer (freezes params per tracking_mode)
        self._build_tracking_optimizer()

        return {
            'loss_history': loss_history,
            'n_iters': total_iters,
            'elapsed_s': elapsed,
            'final_n': self.model.n_gaussians,
        }

    def update(self, frame: FrameData) -> Dict:
        """
        Phase 2: Single-frame tracking update.

        1. Predict: φᵢ += ωᵢ·Δt
        2. Update: gradient descent on single-frame loss
        3. Return timing and loss info

        Args:
            frame: single FrameData

        Returns:
            dict with 'loss', 'n_pts', 'elapsed_ms'
        """
        t0 = time.time()

        if self.model is None:
            raise RuntimeError("Must call initialize() before update()")

        # Phase prediction: advance phases by ω·Δt
        dt = frame.t - getattr(self, '_last_t', frame.t)
        if abs(dt) > 1e-6:
            with torch.no_grad():
                omega = self.model._omega()
                self.model.phi.add_(omega * dt)
        self._last_t = frame.t

        if len(frame.x_obs) == 0:
            return {'loss': float('nan'), 'n_pts': 0, 'elapsed_ms': 0.0}

        # Gradient update using tracking optimizer
        opt = self.tracking_optimizer if self.tracking_optimizer is not None \
            else self.optimizer
        for step in range(self.tracking_steps):
            opt.zero_grad()
            loss = self._compute_loss([frame])
            loss.backward()
            opt.step()

        self.model.clamp_steepness()
        elapsed_ms = (time.time() - t0) * 1000

        return {
            'loss': loss.item(),
            'n_pts': len(frame.x_obs),
            'elapsed_ms': elapsed_ms,
        }

    def predict_field(self, grid_x: torch.Tensor, grid_y: torch.Tensor,
                      t: float) -> torch.Tensor:
        """Predict wave field on regular grid at time t."""
        if self.model is None:
            raise RuntimeError("Model not initialized")
        with torch.no_grad():
            return self.model.predict_grid(grid_x, grid_y, t)
