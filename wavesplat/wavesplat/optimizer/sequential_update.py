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


# `_adaptive_prune_grow` historically compares an ABSOLUTE residual RMS in metres against
# `grow_residual_threshold` (0.05 m) with no normalisation by sea state. This is the same
# defect class as coarse_to_fine.SCALE_INVARIANT_ADVANCE, but it lives in a different
# function, so that flag never reached it. Measured over 8 episodes spanning Hs 0.3-5.0 m
# (tmp/SCALE_AUDIT_fable_2026-07-23.md): residual_rms rises monotonically with Hs
# (0.069 -> 0.107 -> 0.175 m), so on crowded high-Hs episodes the threshold is *guaranteed*
# to fire and the 30% rollback guard fires along with it -- an outcome previously read as
# incidental rather than structural.
# When True, the residual RMS is divided by the observation RMS of the same frames, so the
# threshold means "fraction of observed signal amplitude still unexplained". That normaliser
# needs no ground truth and is therefore deployable. False preserves shipped behaviour.
SCALE_INVARIANT_GROW = False

# The L1 sparsity term adds `lambda_l1 * sum|w|`, a quantity in metres, to a mean-squared
# loss in metres squared. lambda_l1 therefore carries a hidden unit, and its effective
# strength is sea-state dependent: measured 3-62% of the total loss across Hs 0.3-5.0 m,
# non-monotonically, with a 58% swing in total recovered amplitude in a BF3-2comp ablation
# (tmp/SCALE_AUDIT_fable_2026-07-23.md; the nRMSE impact was not isolated there).
# When True the penalty is multiplied by the observation RMS so both terms are [m^2].
# False preserves shipped behaviour.
SCALE_INVARIANT_L1 = False

# The tracking warm-start historically advances every component's phase by omega*dt before the
# per-frame gradient update (`self.model.phi.add_(omega * dt)` in `update`). This DOUBLE-counts
# time: the forward model already evolves the wave through its own -omega*t term
# (spectral_gaussian.forward), so evaluating forward(t) with the init-time phase already gives
# the correctly-propagated field. Adding omega*dt on top cancels the -omega*t evolution and
# FREEZES the field at the initialization instant; the 10 Adam steps/frame then have to re-slew
# the phase back every frame, and under Adam momentum + L1 that per-frame tug-of-war bleeds
# amplitude toward zero. Measured (tmp/TRACKER_AMPDECAY_2026-07-25.md): starting from a
# parameter-exact init (field nRMSE 0.000), shipped tracking decorrelates to nRMSE ~0.31 with no
# gradient (pure frozen-wave propagation) and to ~0.16 with gradient while sum|w| decays -32%;
# setting lambda_l1=0 does NOT stop the decay (0.1631 vs 0.1633), so L1 is not the cause.
# When False, the redundant advance is skipped and forward(t) tracks the true time evolution
# directly, leaving only the genuine slow drift from any wavenumber error for the gradient to
# correct. True preserves shipped behaviour bit-for-bit.
TRACK_PHASE_PREDICT = True


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
        self.init_method = config.get('init_method', 'v2_konly')
        self.use_dispersion = config.get('use_dispersion', True)
        self.use_confidence = config.get('use_confidence', True)
        # Was documented above and set in configs/estimation.yaml, but never read: every
        # clamp_steepness() call site used the function default. The shipped YAML value
        # happens to equal that default, so behaviour is unchanged -- but the knob was
        # silently inert and would not have taken effect if changed. (2026-07-24)
        self.max_steepness = config.get('max_steepness', 0.15)
        self.seed = config.get('seed', None)

        # Neural corrector (nonlinear correction)
        self.neural_corrector_path = config.get('neural_corrector_path', None)

        n_gaussians = config.get('n_gaussians', [3, 8, 15])
        if isinstance(n_gaussians, int):
            self.stages = [n_gaussians]
        else:
            self.stages = list(n_gaussians)

        # Adaptive N parameters
        self.adaptive_n = config.get('adaptive_n', True)
        self.prune_threshold = config.get('prune_threshold', 0.10)  # energy fraction to discard
        self.grow_residual_threshold = config.get('grow_residual_threshold', 0.05)
        self.max_n = config.get('max_n', 30)

        # Multi-frame tracking
        self.tracking_window = config.get('tracking_window', 3)
        self._frame_buffer = []

        self.model = None
        self.optimizer = None
        self.tracking_optimizer = None
        self._last_dt = 0.1

    def _build_model(self, n: int) -> SpectralGaussianModel:
        # Build model WITHOUT neural corrector (optimization should be clean)
        # NN corrector is loaded but disabled during optimization, enabled only for prediction
        model = SpectralGaussianModel(
            n, depth=self.depth, use_dispersion=self.use_dispersion,
            use_neural_corrector=False)
        if self.neural_corrector_path is not None:
            # Preserve RNG state: NN corrector creation changes torch random state
            rng_state = torch.random.get_rng_state()
            model.load_neural_corrector(self.neural_corrector_path)
            torch.random.set_rng_state(rng_state)
            # Disable during optimization — will be re-enabled for prediction
            model.use_neural_corrector = False
        model.to(self.device)
        return model

    def _build_optimizer(self):
        """Build optimizer for initialization phase (all params unfrozen)."""
        param_groups = [
            {'params': [self.model.w, self.model.phi], 'lr': self.lr},
            {'params': [self.model.kx, self.model.ky], 'lr': self.lr * 0.5},
        ]
        if not self.use_dispersion:
            param_groups.append(
                {'params': [self.model.omega_free], 'lr': self.lr})
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

        # A1 ablation: omega_free needs gradient in tracking too
        if not self.use_dispersion:
            self.model.omega_free.requires_grad_(True)
            params.append({'params': [self.model.omega_free],
                           'lr': self.tracking_lr})

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
            weights = frame.confidence if self.use_confidence \
                else torch.ones_like(frame.confidence)
            weighted_sq = weights * residual ** 2
            total_loss = total_loss + weighted_sq.sum()
            n_obs += len(frame.x_obs)

        if n_obs > 0:
            total_loss = total_loss / n_obs

        # L1 sparsity on weights. Note the shipped form adds a [m] quantity to an [m^2] loss,
        # so lambda_l1 carries a hidden unit and its effective strength moves with sea state:
        # measured contribution ranges over 3-62% of the total loss across Hs, non-monotonically
        # (tmp/SCALE_AUDIT_fable_2026-07-23.md). SCALE_INVARIANT_L1 multiplies by the
        # observation RMS to make the penalty [m^2] like the data term; False keeps shipped
        # behaviour. The scale is computed under no_grad -- it is a normaliser, not a target.
        if self.lambda_l1 > 0:
            l1 = torch.sum(torch.abs(self.model.w))
            if SCALE_INVARIANT_L1:
                with torch.no_grad():
                    zs = [f.z_obs for f in frames if len(f.z_obs) > 0]
                    scale = torch.sqrt(torch.mean(torch.cat(zs) ** 2)) if zs else None
                if scale is not None and float(scale) > 1e-9:
                    l1 = l1 * scale
            total_loss = total_loss + self.lambda_l1 * l1

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
        """Compute observation residuals for coarse-to-fine refinement.

        Returns (x, y, residual_z, t) when init_method is v2/v2_konly,
        or (x, y, residual_z) for v1 (backward compatible).
        """
        all_x, all_y, all_res, all_t = [], [], [], []
        with torch.no_grad():
            for frame in frames:
                if len(frame.x_obs) == 0:
                    continue
                z_pred = model(frame.x_obs, frame.y_obs, frame.t)
                res = (frame.z_obs - z_pred).cpu().numpy()
                n = len(frame.x_obs)
                all_x.append(frame.x_obs.cpu().numpy())
                all_y.append(frame.y_obs.cpu().numpy())
                all_res.append(res)
                all_t.append(np.full(n, frame.t))

        if all_x:
            x = np.concatenate(all_x)
            y = np.concatenate(all_y)
            r = np.concatenate(all_res)
            if self.init_method in ('v2', 'v2_konly'):
                return x, y, r, np.concatenate(all_t)
            return x, y, r
        if self.init_method in ('v2', 'v2_konly'):
            return np.array([]), np.array([]), np.array([]), np.array([])
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
            init_method=self.init_method,
            depth=self.depth,
        )

        # Observation-variance scale for the (optional) scale-invariant stage-advance test.
        # Set unconditionally; it is only consulted when
        # coarse_to_fine.SCALE_INVARIANT_ADVANCE is True.
        try:
            _z = np.concatenate([f.z_obs.detach().cpu().numpy().ravel() for f in frames])
            schedule.loss_scale = float(np.mean(_z ** 2)) if _z.size else None
        except Exception:
            schedule.loss_scale = None

        # Initialize first stage
        n_init = schedule.current_n()
        init_kx, init_ky, init_w, init_phi = initialize_from_frames(
            frames, n_init, self.depth, x_range, y_range,
            init_method=self.init_method, seed=self.seed
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
            self.model.clamp_steepness(self.max_steepness)

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

        # Adaptive N: prune weak components, grow if residual high
        if self.adaptive_n:
            self._adaptive_prune_grow(frames, x_range, y_range, verbose)

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

        # Phase prediction: advance phases by ω·Δt.
        # NOTE: this is redundant with the forward model's own -ω·t term and, when enabled,
        # double-counts time (see TRACK_PHASE_PREDICT note above). Gated so shipped behaviour
        # (advance ON) is the default; TRACK_PHASE_PREDICT=False skips it.
        dt = frame.t - getattr(self, '_last_t', frame.t)
        if TRACK_PHASE_PREDICT and abs(dt) > 1e-6:
            with torch.no_grad():
                omega = self.model._omega()
                self.model.phi.add_(omega * dt)
        self._last_t = frame.t

        if len(frame.x_obs) == 0:
            return {'loss': float('nan'), 'n_pts': 0, 'elapsed_ms': 0.0}

        # Multi-frame window: keep recent frames for more stable gradients
        self._frame_buffer.append(frame)
        if len(self._frame_buffer) > self.tracking_window:
            self._frame_buffer.pop(0)

        # Gradient update using tracking optimizer on window
        opt = self.tracking_optimizer if self.tracking_optimizer is not None \
            else self.optimizer
        for step in range(self.tracking_steps):
            opt.zero_grad()
            loss = self._compute_loss(self._frame_buffer)
            loss.backward()
            opt.step()

        self.model.clamp_steepness(self.max_steepness)

        elapsed_ms = (time.time() - t0) * 1000

        return {
            'loss': loss.item(),
            'n_pts': len(frame.x_obs),
            'elapsed_ms': elapsed_ms,
        }

    def _adaptive_prune_grow(self, frames, x_range, y_range, verbose=False):
        """Adaptive N: prune weak components, grow if residual is still high.

        Pruning uses energy-based selection: keep fewest components that
        capture (1 - prune_threshold) fraction of total energy (Σ wᵢ²).
        """

        # --- Prune: energy-based selection ---
        with torch.no_grad():
            energy = self.model.w ** 2  # energy per component
            total_energy = energy.sum().item()

            if total_energy < 1e-12:
                return

            # Sort by energy descending, keep enough for target fraction
            target_fraction = 1.0 - self.prune_threshold  # e.g., keep 95%
            sort_idx = torch.argsort(energy, descending=True)
            cumulative = torch.cumsum(energy[sort_idx], dim=0)
            n_keep = int((cumulative / total_energy < target_fraction).sum().item()) + 1
            n_keep = max(1, min(n_keep, self.model.n_gaussians))

            n_before = self.model.n_gaussians
            if n_keep < n_before:
                keep_idx = sort_idx[:n_keep]
                keep_idx = keep_idx.sort()[0]

                # Safety check: compute residual before and after pruning
                # Rollback if pruning significantly increases residual
                result_before = self._compute_residuals(self.model, frames)
                res_before = result_before[2] if len(result_before) >= 3 else np.array([])
                rms_before = float(np.sqrt(np.mean(res_before ** 2))) if len(res_before) > 0 else 0.0

                # Temporarily prune
                import copy
                saved_state = {
                    'w': self.model.w.data.clone(),
                    'kx': self.model.kx.data.clone(),
                    'ky': self.model.ky.data.clone(),
                    'phi': self.model.phi.data.clone(),
                    'n': self.model.n_gaussians,
                }
                if not self.model.use_dispersion:
                    saved_state['omega_free'] = self.model.omega_free.data.clone()

                self.model.remove_gaussians(keep_idx)

                result_after = self._compute_residuals(self.model, frames)
                res_after = result_after[2] if len(result_after) >= 3 else np.array([])
                rms_after = float(np.sqrt(np.mean(res_after ** 2))) if len(res_after) > 0 else 0.0

                # Rollback if residual increases by more than 30%
                if rms_before > 1e-12 and rms_after > 1.3 * rms_before:
                    # Restore original model
                    self.model.n_gaussians = saved_state['n']
                    self.model.w = torch.nn.Parameter(saved_state['w'])
                    self.model.kx = torch.nn.Parameter(saved_state['kx'])
                    self.model.ky = torch.nn.Parameter(saved_state['ky'])
                    self.model.phi = torch.nn.Parameter(saved_state['phi'])
                    if not self.model.use_dispersion:
                        self.model.omega_free = torch.nn.Parameter(saved_state['omega_free'])
                    if verbose:
                        print(f"  [Adaptive] Pruning rolled back: residual would increase "
                              f"{rms_before:.4f} → {rms_after:.4f}")
                else:
                    if verbose:
                        frac = cumulative[n_keep-1].item() / total_energy * 100
                        print(f"  [Adaptive] Pruned: {n_before} → {self.model.n_gaussians} "
                              f"(kept {frac:.1f}% energy)")

        # --- Grow: add components if residual energy is high ---
        if self.model.n_gaussians >= self.max_n:
            return

        # Compute residual energy
        result = self._compute_residuals(self.model, frames)
        if self.init_method in ('v2', 'v2_konly') and len(result) == 4:
            all_x, all_y, all_res, all_t = result
        else:
            all_x, all_y, all_res = result[:3]
            all_t = None

        if len(all_res) == 0:
            return

        residual_rms = float(np.sqrt(np.mean(all_res ** 2)))

        # Quantity actually compared against grow_residual_threshold. Absolute metres by
        # default; a dimensionless fraction of observed signal amplitude when
        # SCALE_INVARIANT_GROW is set (see the module-level note).
        grow_metric = residual_rms
        if SCALE_INVARIANT_GROW:
            obs = [f.z_obs.detach().cpu().numpy() for f in frames if len(f.z_obs) > 0]
            if obs:
                obs_rms = float(np.sqrt(np.mean(np.concatenate(obs) ** 2)))
                if obs_rms > 1e-9:
                    grow_metric = residual_rms / obs_rms

        if grow_metric < self.grow_residual_threshold:
            if verbose:
                print(f"  [Adaptive] Residual RMS={residual_rms:.4f} "
                      f"(metric={grow_metric:.4f}) < threshold "
                      f"{self.grow_residual_threshold}, no growth needed")
            return

        # Decide how many to add (proportional to residual, capped)
        n_add = min(
            max(3, int(grow_metric / self.grow_residual_threshold * 3)),
            self.max_n - self.model.n_gaussians
        )

        if n_add <= 0:
            return

        # Use beamforming on residuals to find new peaks
        if self.init_method in ('v2', 'v2_konly') and all_t is not None:
            from wavesplat.optimizer.initializer import (
                _beamform_coarse, _beamform_refine, _linear_projection,
            )
            power, kx_axis, ky_axis = _beamform_coarse(
                all_x, all_y, all_res, all_t, self.depth,
                kx_range=(-4.0, 4.0), ky_range=(-4.0, 4.0),
                n_kx=61, n_ky=61)

            # Zero DC
            cx = np.argmin(np.abs(kx_axis))
            cy = np.argmin(np.abs(ky_axis))
            power[max(0, cy-1):cy+2, max(0, cx-1):cx+2] = 0.0

            kx_list, ky_list, amp_list = [], [], []
            dkx = kx_axis[1] - kx_axis[0]
            from wavesplat.optimizer.initializer import MIN_DK
            min_dk = MIN_DK

            for _ in range(n_add):
                flat_idx = np.argmax(power.ravel())
                iy_p, ix_p = np.unravel_index(flat_idx, power.shape)
                if power[iy_p, ix_p] < 1e-12:
                    break
                kx_c, ky_c = kx_axis[ix_p], ky_axis[iy_p]
                kx_f, ky_f, _ = _beamform_refine(
                    all_x, all_y, all_res, all_t, self.depth,
                    kx_c, ky_c, delta=0.15, n_fine=31)
                amp, _ = _linear_projection(
                    all_x, all_y, all_res, all_t, kx_f, ky_f, self.depth)
                kx_list.append(kx_f)
                ky_list.append(ky_f)
                amp_list.append(amp)

                suppress = int(min_dk / dkx) + 1
                y_lo = max(0, iy_p - suppress)
                y_hi = min(power.shape[0], iy_p + suppress + 1)
                x_lo = max(0, ix_p - suppress)
                x_hi = min(power.shape[1], ix_p + suppress + 1)
                power[y_lo:y_hi, x_lo:x_hi] = 0.0

            n_found = len(kx_list)
        else:
            kx_list, ky_list, amp_list = [], [], []
            n_found = 0

        # Pad with random if needed
        if n_found < n_add:
            extra = n_add - n_found
            kx_list.extend(np.random.randn(extra) * 0.1)
            ky_list.extend(np.random.randn(extra) * 0.1)
            amp_list.extend(np.ones(extra) * 0.01)

        n_actual = min(len(kx_list), n_add)
        if n_actual > 0:
            device = self.model.w.device
            self.model.add_gaussians(
                n_actual,
                init_kx=torch.tensor(kx_list[:n_actual], dtype=torch.float32, device=device),
                init_ky=torch.tensor(ky_list[:n_actual], dtype=torch.float32, device=device),
                init_w=torch.tensor(amp_list[:n_actual], dtype=torch.float32, device=device),
                init_phi=torch.rand(n_actual, device=device) * 2 * np.pi,
            )

            # Fine-tune the new components with a few iterations
            self._build_optimizer()
            for _ in range(50):
                self.optimizer.zero_grad()
                loss = self._compute_loss(frames)
                loss.backward()
                self.optimizer.step()
                self.model.clamp_steepness(self.max_steepness)

            if verbose:
                print(f"  [Adaptive] Grew: +{n_actual} → N={self.model.n_gaussians} "
                      f"(residual RMS={residual_rms:.4f})")

    def _ls_w_update(self, frames: List[FrameData]):
        """Analytically update w via least-squares with current kx/ky/phi/omega.

        Builds design matrix A[row,i] = cos(kx_i*x + ky_i*y - omega_i*t + phi_i)
        and solves w = argmin ||A w - z||^2.
        """
        n = self.model.n_gaussians
        if n == 0:
            return

        with torch.no_grad():
            kx = self.model.kx.cpu().numpy()
            ky = self.model.ky.cpu().numpy()
            phi = self.model.phi.cpu().numpy()
            omega = self.model._omega().cpu().numpy()

        rows_A = []
        rows_z = []
        for frame in frames:
            if len(frame.x_obs) == 0:
                continue
            x = frame.x_obs.cpu().numpy()
            y = frame.y_obs.cpu().numpy()
            z = frame.z_obs.cpu().numpy()
            t = frame.t
            phase = (kx[None, :] * x[:, None]
                     + ky[None, :] * y[:, None]
                     - omega[None, :] * t
                     + phi[None, :])
            rows_A.append(np.cos(phase))
            rows_z.append(z)

        if not rows_A:
            return

        A = np.vstack(rows_A)
        z_vec = np.concatenate(rows_z)

        w_ls, _, _, _ = np.linalg.lstsq(A, z_vec, rcond=None)

        with torch.no_grad():
            self.model.w.copy_(torch.tensor(w_ls, dtype=torch.float32,
                                            device=self.model.w.device))

    def amplitude_refit(self, frames: List[FrameData],
                        verbose: bool = False) -> Dict:
        """
        Phase 3: Post-hoc least-squares amplitude+phase calibration.

        Freezes kx, ky and solves for optimal w and phi via linear LS
        using cos/sin decomposition:
            z = Σ [a_i cos(θ_i) + b_i sin(θ_i)]
        where θ_i = kx_i·x + ky_i·y - ω_i·t, then
            w_i = sqrt(a_i² + b_i²), phi_i = atan2(b_i, a_i)

        This corrects both amplitude decay and phase drift caused by
        tracking oscillation in Phase 2.

        Args:
            frames: list of recent FrameData (typically last 5-10 frames)
            verbose: print progress

        Returns:
            dict with 'w_before', 'w_after', 'residual_before', 'residual_after'
        """
        if self.model is None:
            raise RuntimeError("Model not initialized")

        n = self.model.n_gaussians
        if n == 0:
            return {}

        with torch.no_grad():
            kx = self.model.kx.cpu().numpy()
            ky = self.model.ky.cpu().numpy()
            omega = self.model._omega().cpu().numpy()
            w_before = self.model.w.cpu().numpy().copy()

        # Build design matrix with cos/sin pairs:
        # A = [cos(θ_1) sin(θ_1) cos(θ_2) sin(θ_2) ... cos(θ_N) sin(θ_N)]
        # where θ_i = kx_i·x + ky_i·y - ω_i·t  (no phi — we solve for it)
        rows_A = []
        rows_z = []
        for frame in frames:
            if len(frame.x_obs) == 0:
                continue
            x = frame.x_obs.cpu().numpy()
            y = frame.y_obs.cpu().numpy()
            z = frame.z_obs.cpu().numpy()
            t = frame.t

            # base phase without phi: (M, N)
            theta = (kx[None, :] * x[:, None]
                     + ky[None, :] * y[:, None]
                     - omega[None, :] * t)
            # Interleave cos/sin columns: (M, 2N)
            cos_theta = np.cos(theta)
            sin_theta = np.sin(theta)
            A_frame = np.empty((len(x), 2 * n), dtype=np.float64)
            A_frame[:, 0::2] = cos_theta
            A_frame[:, 1::2] = sin_theta
            rows_A.append(A_frame)
            rows_z.append(z)

        if not rows_A:
            return {}

        A = np.vstack(rows_A)   # (total_obs, 2N)
        z_vec = np.concatenate(rows_z)  # (total_obs,)

        # Solve [a_1, b_1, a_2, b_2, ...] = argmin ||A c - z||^2
        coeffs, _, _, _ = np.linalg.lstsq(A, z_vec, rcond=None)

        a = coeffs[0::2]  # cos coefficients
        b = coeffs[1::2]  # sin coefficients
        w_refit = np.sqrt(a ** 2 + b ** 2)
        phi_refit = np.arctan2(b, a)

        # Compute residuals for reporting
        # Before: use old w, phi
        phi_before = self.model.phi.detach().cpu().numpy()
        A_before = np.empty((len(z_vec), n))
        idx = 0
        for frame in frames:
            if len(frame.x_obs) == 0:
                continue
            m = len(frame.x_obs)
            x = frame.x_obs.cpu().numpy()
            y = frame.y_obs.cpu().numpy()
            theta = (kx[None, :] * x[:, None]
                     + ky[None, :] * y[:, None]
                     - omega[None, :] * frame.t
                     + phi_before[None, :])
            A_before[idx:idx + m] = np.cos(theta)
            idx += m
        residual_before = float(np.sqrt(np.mean((A_before @ w_before - z_vec) ** 2)))
        residual_after = float(np.sqrt(np.mean((A @ coeffs - z_vec) ** 2)))

        # Update model weights and phases
        with torch.no_grad():
            self.model.w.copy_(torch.tensor(w_refit, dtype=torch.float32,
                                            device=self.model.w.device))
            self.model.phi.copy_(torch.tensor(phi_refit, dtype=torch.float32,
                                              device=self.model.phi.device))

        if verbose:
            print(f"  [Refit] LS refit (w+phi): "
                  f"residual {residual_before:.4f} → {residual_after:.4f}")
            print(f"  [Refit] |w| mean: {np.mean(np.abs(w_before)):.4f} → "
                  f"{np.mean(np.abs(w_refit)):.4f}")

        return {
            'w_before': w_before.tolist(),
            'w_after': w_refit.tolist(),
            'residual_before': residual_before,
            'residual_after': residual_after,
        }

    def predict_field(self, grid_x: torch.Tensor, grid_y: torch.Tensor,
                      t: float) -> torch.Tensor:
        """Predict wave field on regular grid at time t.

        Temporarily enables neural corrector (if loaded) for prediction only.
        """
        if self.model is None:
            raise RuntimeError("Model not initialized")
        with torch.no_grad():
            # Enable NN corrector for prediction if loaded
            had_nn = self.model.use_neural_corrector
            if self.neural_corrector_path and self.model._neural_corrector is not None:
                self.model.use_neural_corrector = True
            result = self.model.predict_grid(grid_x, grid_y, t)
            self.model.use_neural_corrector = had_nn
            return result
