"""Dual-actuator PD + feedforward controller for hydrofoil stabilization.

Control architecture: collective + differential + wave feedforward
  - Collective δ_c: same sign on both foils → controls total lift → heave
  - Differential δ_d: opposite sign → controls pitch moment → pitch
  - Feedforward δ_ff: pre-compensates wave-induced AoA changes using
    spatial wave preview from WaveSplat (or GT for upper-bound analysis)

Front flap: δ_front = δ_c - δ_d + δ_ff_front
Rear  flap: δ_rear  = δ_c + δ_d + δ_ff_rear

The feedforward uses a spatial preview: the wave field at (x_foil + V·Δt, y)
evaluated at current time t. Since the wave propagates and the vessel advances,
this approximates what the foil will encounter Δt seconds in the future.

Sign conventions:
  - z > 0 (above trim) → δ_c < 0 → reduce total lift → descend
  - theta > 0 (bow up) → δ_d > 0 → more rear lift, less front → nose-down
  - Wave slope up-ahead → δ_ff < 0 → reduce alpha before wave arrives
"""

from dataclasses import dataclass, field
from typing import Optional
import numpy as np

from .vehicle import HydrofoilState


@dataclass
class PDControllerConfig:
    """PD controller gains for hydrofoil stabilization."""
    # Heave (collective mode)
    Kp_z: float = 0.8
    Kd_z: float = 0.3

    # Pitch (differential mode)
    Kp_theta: float = 4.0
    Kd_theta: float = 1.5

    # Actuator limits per foil
    delta_max: float = 0.26  # [rad] ~15°


@dataclass
class FeedforwardConfig:
    """Feedforward controller configuration."""
    # Preview time horizon
    preview_dt: float = 0.5       # [s] how far ahead to look

    # Feedforward gain (0 = off, 1 = full compensation)
    K_ff: float = 0.8             # feedforward gain (< 1 for robustness)

    # Low-pass filter for feedforward signal (suppresses estimation noise)
    # IIR: y[n] = alpha * x[n] + (1-alpha) * y[n-1]
    # alpha=1.0 means no filtering, alpha=0.1 is strong smoothing
    lp_alpha: float = 0.05        # IIR filter coefficient (~0.8 Hz cutoff at dt=0.01s)

    # Noise level for simulating WaveSplat estimation error
    noise_std: float = 0.0        # [m] std of elevation noise (0 = GT)
    slope_noise_std: float = 0.0  # [rad] std of slope noise


@dataclass
class FeedforwardWaveInfo:
    """Wave preview information at foil positions."""
    slope_front: float = 0.0
    slope_rear: float = 0.0
    w_front: float = 0.0
    w_rear: float = 0.0


class FeedforwardFilter:
    """IIR low-pass filter state for feedforward signals.

    Prevents high-frequency estimation noise from reaching the actuators.
    """
    def __init__(self):
        self.delta_ff_front = 0.0
        self.delta_ff_rear = 0.0

    def filter(self, raw_front: float, raw_rear: float,
               alpha: float) -> tuple:
        """Apply first-order IIR low-pass filter.

        Args:
            raw_front, raw_rear: unfiltered feedforward deltas
            alpha: filter coefficient (1.0 = no filter, 0.0 = frozen)

        Returns:
            (filtered_front, filtered_rear)
        """
        self.delta_ff_front = alpha * raw_front + (1 - alpha) * self.delta_ff_front
        self.delta_ff_rear = alpha * raw_rear + (1 - alpha) * self.delta_ff_rear
        return self.delta_ff_front, self.delta_ff_rear


@dataclass
class FlapCommand:
    """Flap deflections for front and rear foils."""
    front: float = 0.0
    rear: float = 0.0


def compute_flap_command(state: HydrofoilState,
                         ctrl: PDControllerConfig,
                         ff_info: Optional[FeedforwardWaveInfo] = None,
                         ff_cfg: Optional[FeedforwardConfig] = None,
                         speed: float = 11.3,
                         ff_filter: Optional[FeedforwardFilter] = None) -> FlapCommand:
    """Compute front and rear foil flap deflections.

    Args:
        state: current vehicle state (relative to trim)
        ctrl: PD controller gains
        ff_info: feedforward wave preview (None = PD only)
        ff_cfg: feedforward configuration (None = PD only)
        speed: vessel speed [m/s]
        ff_filter: low-pass filter state for feedforward (None = no filter)

    Returns:
        FlapCommand with front/rear flap deflections
    """
    # --- PD Feedback ---
    delta_c = -ctrl.Kp_z * state.z - ctrl.Kd_z * state.z_dot
    delta_d = ctrl.Kp_theta * state.theta + ctrl.Kd_theta * state.theta_dot

    delta_front = delta_c - delta_d
    delta_rear = delta_c + delta_d

    # --- Feedforward (if enabled) ---
    if ff_info is not None and ff_cfg is not None and ff_cfg.K_ff > 0:
        K = ff_cfg.K_ff

        # Wave-induced AoA change prediction
        alpha_wave_front = -ff_info.slope_front - ff_info.w_front / speed
        alpha_wave_rear = -ff_info.slope_rear - ff_info.w_rear / speed

        # Raw feedforward: counteract predicted wave alpha
        ff_front = -K * alpha_wave_front
        ff_rear = -K * alpha_wave_rear

        # Low-pass filter to suppress estimation noise
        if ff_filter is not None:
            ff_front, ff_rear = ff_filter.filter(
                ff_front, ff_rear, ff_cfg.lp_alpha)

        delta_front += ff_front
        delta_rear += ff_rear

    # Saturate
    delta_front = np.clip(delta_front, -ctrl.delta_max, ctrl.delta_max)
    delta_rear = np.clip(delta_rear, -ctrl.delta_max, ctrl.delta_max)

    return FlapCommand(front=delta_front, rear=delta_rear)
