"""Single-point bow sensor feedforward — comparison baseline.

Models the approach used by commercial hydrofoils (e.g., Candela C-8):
a single downward-facing sensor at the bow measures wave elevation at
one spatial point and uses it for feedforward control on both foils.

Two variants
-----------
SP-1pt (best-case)
    Evaluate GT wave components analytically at ONE spatial preview
    point. Apply the SAME signal to both front and rear foils.
    No estimation noise. This is the theoretical upper bound for any
    single-point sensor regardless of measurement quality.

SP-temporal (realistic)
    Temporal differentiation of bow elevation (sensor with additive
    noise), plus head-sea assumption for slope:
        dη/dx ≈ -(1/c_phase) · dη/dt
    This is what a Candela-style bow sensor likely computes.
    Fails systematically for beam/oblique seas where wave propagation
    direction differs from vessel heading.

Key limitation of both variants vs WaveSplat-FF
------------------------------------------------
- Both foils receive the IDENTICAL feedforward signal.
  WaveSplat evaluates slope/w independently at each foil's preview
  position, capturing the per-foil phase difference.
  For BF3 (Tp=5 s), λ ≈ 39 m and foil separation = 4.5 m gives a
  phase difference of 2π·4.5/39 ≈ 0.72 rad ≈ 41° — significant.
- SP-temporal assumes head-sea direction; introduces systematic
  sign error on slope for beam/oblique seas, potentially inverting
  the feedforward and degrading performance.
"""

import numpy as np

from .controller import FeedforwardWaveInfo
from .disturbance import (
    wave_elevation_at_point,
    wave_slope_x_at_point,
    wave_slope_y_at_point,
    wave_vertical_velocity_at_point,
)


# ─────────────────────────────────────────────────────────────────────────────
# Variant 1: SP-1pt  (best-case single-point, analytic GT)
# ─────────────────────────────────────────────────────────────────────────────

def compute_single_point_ff_info(
    x_cg: float,
    components: list,
    t: float,
    vessel_heading: float,
    speed: float,
    preview_dt: float,
) -> FeedforwardWaveInfo:
    """Best-case single-point feedforward: analytic GT at one preview point.

    Evaluates wave slope and orbital velocity at a single spatial point
    (CG + V·preview_dt along vessel heading).  The SAME signal is applied
    to both front and rear foils — no per-foil discrimination.

    Args:
        x_cg:          longitudinal CG position in global frame [m]
        components:    list of GT WaveComponent / WaveSplatComponent
        t:             current simulation time [s]
        vessel_heading: vessel heading angle [rad] (0 = +x direction)
        speed:         forward speed [m/s]
        preview_dt:    preview time horizon [s]

    Returns:
        FeedforwardWaveInfo with identical slope/w for front and rear.
    """
    cos_h = np.cos(vessel_heading)
    sin_h = np.sin(vessel_heading)

    dx = speed * preview_dt
    x_p = x_cg + dx * cos_h
    y_p =        dx * sin_h

    deta_dx = wave_slope_x_at_point(components, x_p, y_p, t)
    deta_dy = wave_slope_y_at_point(components, x_p, y_p, t)
    slope   = deta_dx * cos_h + deta_dy * sin_h   # projected onto heading
    w       = wave_vertical_velocity_at_point(components, x_p, y_p, t)

    return FeedforwardWaveInfo(
        slope_front=slope, slope_rear=slope,
        w_front=w,         w_rear=w,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Variant 2: SP-temporal  (realistic bow sensor + head-sea assumption)
# ─────────────────────────────────────────────────────────────────────────────

class TemporalBowSensor:
    """Simulates a downward-facing bow elevation sensor.

    Estimates wave slope from temporal derivative under the head-sea
    assumption.  For a wave η = A cos(kx - ωt + φ):
        dη/dx = -(k/ω) · dη/dt = -(1/c_phase) · dη/dt   [head sea]

    This identity holds only when the wave propagates along the vessel
    heading.  For beam/oblique seas the true dη/dx_body ≠ -(1/c) · dη/dt,
    so the slope estimate carries a systematic direction error.

    Attributes:
        dt:         simulation timestep [s]
        c_phase:    estimated wave phase velocity [m/s]
        noise_std:  sensor elevation noise standard deviation [m]
        bow_offset: bow distance ahead of CG [m] (Candela C-8 ≈ 4 m)
    """

    def __init__(
        self,
        dt: float,
        c_phase: float = 8.0,
        noise_std: float = 0.02,
        bow_offset: float = 4.0,
    ) -> None:
        """
        Args:
            dt:         integration timestep [s]
            c_phase:    phase velocity estimate [m/s].
                        Deep-water: c = gTp / (2π).
                        BF3 Tp=5 s → c ≈ 7.8 m/s.
                        BF4 Tp=7.5 s → c ≈ 11.7 m/s.
                        Default 8.0 m/s is a reasonable fixed estimate.
            noise_std:  elevation sensor noise [m] (Candela uses ultrasonic,
                        typical σ ≈ 0.01–0.03 m in calm water).
            bow_offset: bow position ahead of CG [m].
        """
        self.dt = dt
        self.c_phase = c_phase
        self.noise_std = noise_std
        self.bow_offset = bow_offset
        self._eta_prev: float | None = None

    def compute_ff_info(
        self,
        x_cg: float,
        components: list,
        t: float,
        vessel_heading: float,
        speed: float,
        preview_dt: float,
    ) -> FeedforwardWaveInfo:
        """Compute FF signal from temporal derivative of bow measurement.

        The bow sensor is located at x_cg + bow_offset ahead of CG.
        The preview adds V·preview_dt in the heading direction.

        Returns:
            FeedforwardWaveInfo with identical (estimated) slope/w
            for both foils.
        """
        cos_h = np.cos(vessel_heading)
        sin_h = np.sin(vessel_heading)

        # Bow preview position in global frame
        dx_total = (self.bow_offset + speed * preview_dt) * cos_h
        dy_total = (self.bow_offset + speed * preview_dt) * sin_h
        x_bow = x_cg + dx_total
        y_bow = dy_total

        # Noisy elevation measurement
        eta_now = (wave_elevation_at_point(components, x_bow, y_bow, t)
                   + np.random.normal(0.0, self.noise_std))

        # First step: no previous measurement — return zero
        if self._eta_prev is None:
            self._eta_prev = eta_now
            return FeedforwardWaveInfo()

        # Temporal derivative → orbital velocity estimate
        w_est = (eta_now - self._eta_prev) / self.dt
        self._eta_prev = eta_now

        # Slope estimate via head-sea assumption:
        #   head sea: dη/dx = -(1/c) · dη/dt
        # For beam sea the true body-frame slope is ~0, but this formula
        # yields slope_est = -w/c ≠ 0  → systematic over-correction.
        slope_est = -w_est / self.c_phase

        return FeedforwardWaveInfo(
            slope_front=slope_est, slope_rear=slope_est,
            w_front=w_est,         w_rear=w_est,
        )
