"""Hydrofoil simulation runner with PD feedback + wave feedforward control.

Integrates 2DOF dynamics with wavesim GT wave disturbance.
Supports three control modes:
  1. PD-only (feedback): baseline
  2. PD + GT feedforward: upper bound (perfect wave knowledge)
  3. PD + noisy feedforward: simulated WaveSplat estimation
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional

from .config import FoilConfig, HullConfig, SimConfig
from .controller import (
    PDControllerConfig, FeedforwardConfig, FeedforwardWaveInfo,
    FeedforwardFilter, FlapCommand, compute_flap_command,
)
from .vehicle import (
    HydrofoilState, WaveDisturbanceInput, compute_derivatives,
)
from .disturbance import (
    compute_disturbance,
    wave_slope_x_at_point, wave_slope_y_at_point,
    wave_vertical_velocity_at_point,
)


@dataclass
class SimResult:
    """Time history of simulation results."""
    time: np.ndarray
    z: np.ndarray
    theta: np.ndarray
    z_dot: np.ndarray
    theta_dot: np.ndarray
    x_pos: np.ndarray
    eta_front: np.ndarray
    eta_rear: np.ndarray
    flap_front: Optional[np.ndarray] = None
    flap_rear: Optional[np.ndarray] = None
    heave_accel: Optional[np.ndarray] = None
    pitch_accel: Optional[np.ndarray] = None


def _compute_feedforward_info(state: HydrofoilState,
                              front_foil: FoilConfig,
                              rear_foil: FoilConfig,
                              sim: SimConfig,
                              components: list,
                              t: float,
                              vessel_heading: float,
                              ff_cfg: FeedforwardConfig) -> FeedforwardWaveInfo:
    """Compute wave preview at future foil positions.

    Evaluates wave slope and vertical velocity at positions
    (x_foil + V * preview_dt) — where the foil will be in the future.
    """
    cos_h = np.cos(vessel_heading)
    sin_h = np.sin(vessel_heading)

    x_cg = state.x
    y_cg = 0.0
    dx_preview = sim.speed * ff_cfg.preview_dt

    # Preview positions: current foil pos + forward travel
    results = {}
    for name, foil in [('front', front_foil), ('rear', rear_foil)]:
        x_foil = x_cg + foil.x_pos * cos_h + dx_preview * cos_h
        y_foil = y_cg + foil.x_pos * sin_h + dx_preview * sin_h

        # Evaluate wave at preview position at CURRENT time
        # (spatial preview: look ahead in space, not time)
        deta_dx = wave_slope_x_at_point(components, x_foil, y_foil, t)
        deta_dy = wave_slope_y_at_point(components, x_foil, y_foil, t)
        slope = deta_dx * cos_h + deta_dy * sin_h
        w = wave_vertical_velocity_at_point(components, x_foil, y_foil, t)

        # Add estimation noise (simulates WaveSplat error)
        if ff_cfg.slope_noise_std > 0:
            slope += np.random.normal(0, ff_cfg.slope_noise_std)
        if ff_cfg.noise_std > 0:
            # Convert elevation noise to velocity noise: dη/dt ~ ω * noise
            w += np.random.normal(0, ff_cfg.noise_std * 1.0)  # ~1 rad/s

        results[name] = (slope, w)

    return FeedforwardWaveInfo(
        slope_front=results['front'][0],
        slope_rear=results['rear'][0],
        w_front=results['front'][1],
        w_rear=results['rear'][1],
    )


def _get_flaps(state: HydrofoilState,
               ctrl: Optional[PDControllerConfig],
               ff_info: Optional[FeedforwardWaveInfo] = None,
               ff_cfg: Optional[FeedforwardConfig] = None,
               speed: float = 11.3,
               ff_filter: Optional[FeedforwardFilter] = None) -> FlapCommand:
    """Get flap command from controller."""
    if ctrl:
        return compute_flap_command(state, ctrl, ff_info, ff_cfg, speed, ff_filter)
    return FlapCommand()


def _rk4_step(state, front_foil, rear_foil, hull, sim, components,
              t, dt, vessel_heading, ctrl, ff_cfg=None,
              ff_delta_front=0.0, ff_delta_rear=0.0):
    """Single RK4 integration step with feedback + feedforward.

    The feedforward deltas (ff_delta_front/rear) are pre-computed and filtered
    by the caller. Inside RK4, only PD feedback recomputes per substep.
    This ensures the filter advances exactly once per main timestep.

    Args:
        ff_delta_front/rear: pre-filtered feedforward flap deltas [rad].
            These are added to the PD command as a fixed offset.
    """
    y = state.as_array()
    x_pos = state.x

    def deriv(y_local, t_local, x_local):
        s = HydrofoilState.from_array(y_local, x=x_local)
        wave = compute_disturbance(s, front_foil, rear_foil, sim,
                                   components, t_local, vessel_heading)
        # PD-only (no feedforward) — FF is added as fixed pre-filtered offset
        flaps = _get_flaps(s, ctrl, speed=sim.speed)
        return compute_derivatives(s, front_foil, rear_foil, hull, sim,
                                   wave,
                                   flap_front=flaps.front + ff_delta_front,
                                   flap_rear=flaps.rear + ff_delta_rear)

    dx = sim.speed * dt
    k1 = deriv(y, t, x_pos)
    k2 = deriv(y + 0.5 * dt * k1, t + 0.5 * dt, x_pos + 0.5 * dx)
    k3 = deriv(y + 0.5 * dt * k2, t + 0.5 * dt, x_pos + 0.5 * dx)
    k4 = deriv(y + dt * k3, t + dt, x_pos + dx)

    y_new = y + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
    return HydrofoilState.from_array(y_new, x=x_pos + dx)


def find_trim(front_foil, rear_foil, hull, sim,
              ctrl=None, max_iter=5000, tol=1e-6):
    """Find static trim in calm water."""
    trim_sim = SimConfig(
        dt=sim.dt, duration=50.0, speed=sim.speed, depth=sim.depth,
        rho=sim.rho, g=sim.g,
        B_heave=sim.B_heave * 5, B_pitch=sim.B_pitch * 5,
    )
    state = HydrofoilState()
    no_wave = WaveDisturbanceInput()

    for i in range(max_iter):
        flaps = _get_flaps(state, ctrl, speed=sim.speed)
        deriv = compute_derivatives(state, front_foil, rear_foil,
                                    hull, trim_sim, no_wave,
                                    flap_front=flaps.front,
                                    flap_rear=flaps.rear)
        y = state.as_array() + trim_sim.dt * deriv
        state = HydrofoilState.from_array(y, x=0.0)
        if (abs(state.z_dot) < tol and abs(state.theta_dot) < tol
                and abs(deriv[2]) < tol and abs(deriv[3]) < tol):
            break
    return state


def run_simulation(cfg, components, vessel_heading=0.0,
                   initial_state=None, find_trim_first=True,
                   controller=None, use_default_controller=True,
                   feedforward=None, ff_info_provider=None):
    """Run hydrofoil simulation.

    Args:
        cfg: configuration dict from candela_c8()
        components: wavesim WaveComponent list
        vessel_heading: vessel heading [rad]
        initial_state: starting state (default: find trim)
        find_trim_first: find trim before simulation
        controller: PDControllerConfig (None = use default)
        use_default_controller: use default PD if no controller given
        feedforward: FeedforwardConfig (None = PD only)
        ff_info_provider: optional callable(state, front_foil, rear_foil,
            sim, components, t, vessel_heading, ff_cfg) -> FeedforwardWaveInfo.
            If provided, replaces the default _compute_feedforward_info.
            Used for single-point sensor baselines and custom FF models.

    Returns:
        SimResult
    """
    front_foil = cfg['front_foil']
    rear_foil = cfg['rear_foil']
    hull = cfg['hull']
    sim = cfg['sim']

    ctrl = controller
    if ctrl is None and use_default_controller:
        ctrl = PDControllerConfig()

    ff_cfg = feedforward
    ff_filter = FeedforwardFilter() if ff_cfg and ff_cfg.K_ff > 0 else None

    if initial_state is not None:
        state = initial_state
    elif find_trim_first:
        state = find_trim(front_foil, rear_foil, hull, sim, ctrl)
    else:
        state = HydrofoilState()

    n_steps = int(sim.duration / sim.dt) + 1
    time_arr = np.linspace(0.0, sim.duration, n_steps)

    result = SimResult(
        time=time_arr,
        z=np.zeros(n_steps), theta=np.zeros(n_steps),
        z_dot=np.zeros(n_steps), theta_dot=np.zeros(n_steps),
        x_pos=np.zeros(n_steps),
        eta_front=np.zeros(n_steps), eta_rear=np.zeros(n_steps),
        flap_front=np.zeros(n_steps), flap_rear=np.zeros(n_steps),
    )

    for i, t in enumerate(time_arr):
        result.z[i] = state.z
        result.theta[i] = state.theta
        result.z_dot[i] = state.z_dot
        result.theta_dot[i] = state.theta_dot
        result.x_pos[i] = state.x

        wave = compute_disturbance(state, front_foil, rear_foil, sim,
                                   components, t, vessel_heading)
        result.eta_front[i] = wave.eta_front
        result.eta_rear[i] = wave.eta_rear

        # Compute feedforward with filter (once per timestep)
        ff_delta_front = 0.0
        ff_delta_rear = 0.0
        if ff_cfg and ff_cfg.K_ff > 0:
            if ff_info_provider is not None:
                ff_info = ff_info_provider(
                    state, front_foil, rear_foil, sim, components,
                    t, vessel_heading, ff_cfg)
            else:
                ff_info = _compute_feedforward_info(
                    state, front_foil, rear_foil, sim, components,
                    t, vessel_heading, ff_cfg)
            # Compute raw FF deltas
            K = ff_cfg.K_ff
            raw_front = -K * (-ff_info.slope_front - ff_info.w_front / sim.speed)
            raw_rear = -K * (-ff_info.slope_rear - ff_info.w_rear / sim.speed)
            # Apply low-pass filter
            if ff_filter is not None:
                ff_delta_front, ff_delta_rear = ff_filter.filter(
                    raw_front, raw_rear, ff_cfg.lp_alpha)
            else:
                ff_delta_front, ff_delta_rear = raw_front, raw_rear

        # Record flap commands (PD + filtered FF)
        if ctrl:
            flaps = _get_flaps(state, ctrl, speed=sim.speed)
            result.flap_front[i] = np.clip(
                flaps.front + ff_delta_front,
                -ctrl.delta_max, ctrl.delta_max)
            result.flap_rear[i] = np.clip(
                flaps.rear + ff_delta_rear,
                -ctrl.delta_max, ctrl.delta_max)

        # Integrate (PD recomputes per substep, FF is fixed pre-filtered value)
        if i < n_steps - 1:
            dt = time_arr[i + 1] - time_arr[i]
            state = _rk4_step(state, front_foil, rear_foil, hull, sim,
                              components, t, dt, vessel_heading, ctrl, ff_cfg,
                              ff_delta_front, ff_delta_rear)

    result.heave_accel = np.gradient(result.z_dot, result.time)
    result.pitch_accel = np.gradient(result.theta_dot, result.time)
    return result
