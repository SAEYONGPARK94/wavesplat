"""Wave disturbance module: wavesim GT → encounter wave forces at foil positions.

This module bridges wavesim's Gerstner wave field with the hydrofoil dynamics.
It evaluates the wave field at the instantaneous foil positions and computes:
  - Wave elevation η(x, y, t)
  - Wave slope ∂η/∂x (longitudinal)
  - Wave vertical velocity ∂η/∂t

For oblique waves (arbitrary heading), the encounter quantities are computed
in the vessel's body frame.

Encounter frequency: ω_e = ω - k · V · cos(μ)
  where μ = angle between wave propagation and vessel heading
  - μ = 0 (following sea): ω_e < ω
  - μ = π (head sea): ω_e > ω
"""

import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from wavesim.wave_generator import WaveComponent

from .config import FoilConfig, SimConfig
from .vehicle import HydrofoilState, WaveDisturbanceInput


def wave_elevation_at_point(components: list, x: float, y: float,
                            t: float) -> float:
    """Evaluate Gerstner wave elevation at a single (x, y, t) point.

    Uses the linear (vertical) component of Gerstner superposition:
      η(x,y,t) = Σ Aᵢ · cos(kxᵢ·x + kyᵢ·y - ωᵢ·t + φᵢ)

    Note: We use the linear approximation here (no horizontal displacement)
    for consistency with WaveSplat's estimation model. The model mismatch
    with Gerstner GT is O(ε²) — same as documented in the WaveSplat paper.

    Args:
        components: list of WaveComponent from wavesim
        x, y: spatial position [m]
        t: time [s]

    Returns:
        η: wave surface elevation [m]
    """
    eta = 0.0
    for comp in components:
        phase = comp.kx * x + comp.ky * y - comp.omega * t + comp.phi
        eta += comp.A * np.cos(phase)
    return eta


def wave_slope_x_at_point(components: list, x: float, y: float,
                          t: float) -> float:
    """Longitudinal wave slope ∂η/∂x at a single point.

    ∂η/∂x = -Σ Aᵢ · kxᵢ · sin(kxᵢ·x + kyᵢ·y - ωᵢ·t + φᵢ)
    """
    slope = 0.0
    for comp in components:
        phase = comp.kx * x + comp.ky * y - comp.omega * t + comp.phi
        slope += -comp.A * comp.kx * np.sin(phase)
    return slope


def wave_slope_y_at_point(components: list, x: float, y: float,
                          t: float) -> float:
    """Lateral wave slope ∂η/∂y at a single point.

    ∂η/∂y = -Σ Aᵢ · kyᵢ · sin(kxᵢ·x + kyᵢ·y - ωᵢ·t + φᵢ)
    """
    slope = 0.0
    for comp in components:
        phase = comp.kx * x + comp.ky * y - comp.omega * t + comp.phi
        slope += -comp.A * comp.ky * np.sin(phase)
    return slope


def wave_vertical_velocity_at_point(components: list, x: float, y: float,
                                    t: float) -> float:
    """Wave vertical velocity ∂η/∂t at a single point.

    ∂η/∂t = Σ Aᵢ · ωᵢ · sin(kxᵢ·x + kyᵢ·y - ωᵢ·t + φᵢ)
    """
    w = 0.0
    for comp in components:
        phase = comp.kx * x + comp.ky * y - comp.omega * t + comp.phi
        w += comp.A * comp.omega * np.sin(phase)
    return w


def compute_disturbance(state: HydrofoilState,
                        front_foil: FoilConfig,
                        rear_foil: FoilConfig,
                        sim: SimConfig,
                        components: list,
                        t: float,
                        vessel_heading: float = 0.0) -> WaveDisturbanceInput:
    """Compute wave disturbance at both foil positions.

    The vessel travels along the x-axis by default. For oblique waves,
    the wave field is evaluated in the global frame at foil positions
    that account for vessel heading.

    Args:
        state: current vehicle state
        front_foil: front foil config
        rear_foil: rear foil config
        sim: simulation config
        components: wavesim WaveComponent list
        t: current simulation time [s]
        vessel_heading: vessel heading angle [rad] (0 = positive x direction)

    Returns:
        WaveDisturbanceInput with wave quantities at each foil
    """
    cos_h = np.cos(vessel_heading)
    sin_h = np.sin(vessel_heading)

    # Vessel CG position in global frame
    # (vessel moves along heading direction at constant speed)
    x_cg = state.x
    y_cg = 0.0  # 2DOF: no lateral motion

    # Foil positions in global frame
    # Front foil is ahead of CG along vessel heading
    x_front = x_cg + front_foil.x_pos * cos_h
    y_front = y_cg + front_foil.x_pos * sin_h
    x_rear = x_cg + rear_foil.x_pos * cos_h
    y_rear = y_cg + rear_foil.x_pos * sin_h

    # Evaluate wave field at foil positions
    eta_f = wave_elevation_at_point(components, x_front, y_front, t)
    eta_r = wave_elevation_at_point(components, x_rear, y_rear, t)

    # Wave slope in vessel longitudinal direction
    # Project global gradients onto vessel heading
    deta_dx_f = wave_slope_x_at_point(components, x_front, y_front, t)
    deta_dy_f = wave_slope_y_at_point(components, x_front, y_front, t)
    slope_f = deta_dx_f * cos_h + deta_dy_f * sin_h

    deta_dx_r = wave_slope_x_at_point(components, x_rear, y_rear, t)
    deta_dy_r = wave_slope_y_at_point(components, x_rear, y_rear, t)
    slope_r = deta_dx_r * cos_h + deta_dy_r * sin_h

    # Vertical velocity
    w_f = wave_vertical_velocity_at_point(components, x_front, y_front, t)
    w_r = wave_vertical_velocity_at_point(components, x_rear, y_rear, t)

    return WaveDisturbanceInput(
        eta_front=eta_f,
        eta_rear=eta_r,
        slope_front=slope_f,
        slope_rear=slope_r,
        w_front=w_f,
        w_rear=w_r,
    )


def compute_disturbance_gerstner(state: HydrofoilState,
                                 front_foil: FoilConfig,
                                 rear_foil: FoilConfig,
                                 sim: SimConfig,
                                 components: list,
                                 t: float,
                                 vessel_heading: float = 0.0,
                                 fd_eps: float = 0.01) -> WaveDisturbanceInput:
    """Compute wave disturbance using wavesim's full Gerstner model.

    Unlike compute_disturbance() which uses linear wave theory, this
    calls wavesim's single_gerstner() for full nonlinear evaluation.
    Slope and vertical velocity are computed via finite differences
    on the Gerstner surface.

    Args:
        state, front_foil, rear_foil, sim, components, t, vessel_heading:
            same as compute_disturbance()
        fd_eps: spatial/temporal step for finite differences [m or s]

    Returns:
        WaveDisturbanceInput with Gerstner-accurate wave quantities
    """
    from wavesim.wave_generator import single_gerstner

    cos_h = np.cos(vessel_heading)
    sin_h = np.sin(vessel_heading)

    x_cg = state.x
    y_cg = 0.0

    points = []
    for foil in [front_foil, rear_foil]:
        xf = x_cg + foil.x_pos * cos_h
        yf = y_cg + foil.x_pos * sin_h
        points.append((xf, yf))

    results = []
    for (xf, yf) in points:
        # Evaluate at point and nearby for finite differences
        x_arr = np.array([[xf - fd_eps, xf, xf + fd_eps]])
        y_arr = np.array([[yf, yf, yf]])

        _, _, Z, _, _, _ = single_gerstner(components, x_arr, y_arr, t)
        _, _, Z_dt, _, _, _ = single_gerstner(components, x_arr, y_arr, t + fd_eps)

        eta = Z[0, 1]
        slope_x = (Z[0, 2] - Z[0, 0]) / (2 * fd_eps)
        w = (Z_dt[0, 1] - Z[0, 1]) / fd_eps

        # For y-slope (needed for oblique heading projection)
        y_arr2 = np.array([[yf - fd_eps, yf, yf + fd_eps]])
        x_arr2 = np.array([[xf, xf, xf]])
        _, _, Zy, _, _, _ = single_gerstner(components, x_arr2, y_arr2, t)
        slope_y = (Zy[0, 2] - Zy[0, 0]) / (2 * fd_eps)

        # Project slope onto vessel heading
        slope_body = slope_x * cos_h + slope_y * sin_h

        results.append((eta, slope_body, w))

    return WaveDisturbanceInput(
        eta_front=results[0][0],
        eta_rear=results[1][0],
        slope_front=results[0][1],
        slope_rear=results[1][1],
        w_front=results[0][2],
        w_rear=results[1][2],
    )
