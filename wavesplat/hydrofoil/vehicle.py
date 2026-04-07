"""2DOF hydrofoil vehicle dynamics: heave (z) + pitch (theta).

State vector: [z, theta, z_dot, theta_dot]
  z:     heave displacement [m] (positive up from equilibrium)
  theta: pitch angle [rad] (positive bow-up)
  z_dot: heave velocity [m/s]
  theta_dot: pitch rate [rad/s]

Equations of motion (Newton-Euler in body-vertical plane):
  (m + a33) * z_ddot     = L_front + L_rear - m*g - B_heave * z_dot + F_wave_z
  (I_yy + a55) * th_ddot = L_front*l_f - L_rear*l_r - B_pitch * th_dot + M_wave

where:
  a33, a55: added mass terms (simplified frequency-independent approximation)
  L_front, L_rear: foil lift forces (from foil.py)
  l_f, l_r: moment arms from CG to foils
  B_heave, B_pitch: linear damping coefficients
  F_wave_z, M_wave: wave-induced forces (from disturbance.py)

Coordinate convention:
  - x: forward (direction of travel)
  - z: upward (opposite to gravity)
  - theta: pitch, positive bow-up (right-hand rule about y-axis pointing port)
"""

import numpy as np
from dataclasses import dataclass

from .config import FoilConfig, HullConfig, SimConfig
from .foil import foil_forces


@dataclass
class HydrofoilState:
    """Instantaneous state of the hydrofoil vessel."""
    z: float = 0.0            # heave [m]
    theta: float = 0.0        # pitch [rad]
    z_dot: float = 0.0        # heave rate [m/s]
    theta_dot: float = 0.0    # pitch rate [rad/s]
    x: float = 0.0            # longitudinal position [m] (for wave encounter)

    def as_array(self) -> np.ndarray:
        return np.array([self.z, self.theta, self.z_dot, self.theta_dot])

    @classmethod
    def from_array(cls, arr: np.ndarray, x: float = 0.0) -> 'HydrofoilState':
        return cls(z=arr[0], theta=arr[1], z_dot=arr[2], theta_dot=arr[3], x=x)


@dataclass
class WaveDisturbanceInput:
    """Wave-induced quantities at each foil location.

    Provided by disturbance.py at each timestep.
    """
    # Wave elevation at front/rear foil x-positions [m]
    eta_front: float = 0.0
    eta_rear: float = 0.0

    # Wave slope dη/dx at front/rear foil x-positions [rad]
    slope_front: float = 0.0
    slope_rear: float = 0.0

    # Wave vertical velocity dη/dt at front/rear foil x-positions [m/s]
    w_front: float = 0.0
    w_rear: float = 0.0


def effective_alpha(state: HydrofoilState, foil: FoilConfig,
                    wave: WaveDisturbanceInput, sim: SimConfig,
                    is_front: bool) -> float:
    """Compute effective angle of attack at a foil.

    The effective AoA combines:
    1. Geometric incidence from pitch angle
    2. Induced angle from heave rate (z_dot / V)
    3. Induced angle from pitch rate (theta_dot * l / V)
    4. Wave-induced orbital velocity (w / V)
    5. Wave slope (dη/dx)

    Args:
        state: current vehicle state
        foil: foil configuration
        wave: wave disturbance at this timestep
        sim: simulation parameters
        is_front: True for front foil, False for rear

    Returns:
        alpha_eff [rad]
    """
    V = sim.speed

    # Select wave quantities for this foil
    if is_front:
        wave_slope = wave.slope_front
        wave_w = wave.w_front
    else:
        wave_slope = wave.slope_rear
        wave_w = wave.w_rear

    # 1. Geometric: pitch angle contributes to AoA
    alpha_geom = state.theta

    # 2. Heave rate: upward velocity reduces AoA
    alpha_heave = -state.z_dot / V

    # 3. Pitch rate: rotation induces velocity at foil location
    #    v_induced = theta_dot * arm (positive arm = forward of CG)
    alpha_pitch = -state.theta_dot * foil.x_pos / V

    # 4. Wave orbital velocity: upward wave motion reduces AoA
    alpha_wave_w = -wave_w / V

    # 5. Wave slope: tilted surface changes effective incidence
    alpha_wave_slope = -wave_slope

    return alpha_geom + alpha_heave + alpha_pitch + alpha_wave_w + alpha_wave_slope


def foil_submergence(state: HydrofoilState, foil: FoilConfig,
                     wave_eta: float) -> float:
    """Compute instantaneous submergence depth of a foil.

    h = (design depth) - (heave) - (pitch effect) - (wave elevation)

    Positive h means foil is submerged below the water surface.

    Args:
        state: current vehicle state
        foil: foil configuration
        wave_eta: local wave elevation at foil x-position [m]

    Returns:
        h: submergence depth [m] (positive = submerged)
    """
    # Design depth below calm waterline
    h0 = foil.z_pos

    # Heave moves the whole vessel up/down
    dz_heave = state.z

    # Pitch rotates foil about CG
    # Forward foil (x_pos > 0): bow-up (theta > 0) lifts it → reduces submergence
    dz_pitch = foil.x_pos * np.sin(state.theta)

    # Wave elevation raises local water surface → increases submergence
    h = h0 - dz_heave - dz_pitch + wave_eta

    return h


def added_mass_heave(hull: HullConfig, rho: float = 1025.0) -> float:
    """Simplified added mass in heave.

    Approximation: a33 ≈ (2/3) * rho * pi * (B/2)^2 * L
    For a slender hull with B~2.4m, L~8.5m.
    In foilborne, hull is out of water — added mass dominated by foils.
    Use ~5% of vessel mass as a conservative estimate for foilborne.
    """
    return 0.05 * hull.mass


def added_mass_pitch(hull: HullConfig) -> float:
    """Simplified added moment of inertia in pitch.

    Similar reasoning: foilborne added inertia is small.
    Use ~5% of I_yy.
    """
    return 0.05 * hull.I_yy


def compute_derivatives(state: HydrofoilState,
                        front_foil: FoilConfig,
                        rear_foil: FoilConfig,
                        hull: HullConfig,
                        sim: SimConfig,
                        wave: WaveDisturbanceInput,
                        flap_front: float = 0.0,
                        flap_rear: float = 0.0) -> np.ndarray:
    """Compute state derivatives [z_dot, theta_dot, z_ddot, theta_ddot].

    This is the core dynamics function called by the integrator.

    Args:
        flap_front: front foil flap deflection [rad] from controller.
        flap_rear: rear foil flap deflection [rad] from controller.
    """
    g = sim.g
    rho = sim.rho

    # --- Effective angle of attack at each foil ---
    alpha_f = effective_alpha(state, front_foil, wave, sim, is_front=True)
    alpha_r = effective_alpha(state, rear_foil, wave, sim, is_front=False)

    # Add controller flap deflections
    alpha_f += flap_front
    alpha_r += flap_rear

    # --- Submergence ---
    h_f = foil_submergence(state, front_foil, wave.eta_front)
    h_r = foil_submergence(state, rear_foil, wave.eta_rear)

    # --- Foil forces ---
    L_f, D_f = foil_forces(alpha_f, sim.speed, h_f, front_foil, rho)
    L_r, D_r = foil_forces(alpha_r, sim.speed, h_r, rear_foil, rho)

    # --- Heave equation ---
    # (m + a33) * z_ddot = L_f + L_r - mg - B_heave * z_dot
    m_eff = hull.mass + added_mass_heave(hull, rho)
    F_z = L_f + L_r - hull.mass * g - sim.B_heave * state.z_dot
    z_ddot = F_z / m_eff

    # --- Pitch equation ---
    # (I + a55) * theta_ddot = L_f * l_f + L_r * l_r - B_pitch * theta_dot
    # Note: l_r = rear_foil.x_pos is negative → rear lift creates nose-down moment
    I_eff = hull.I_yy + added_mass_pitch(hull)
    M_theta = (L_f * front_foil.x_pos + L_r * rear_foil.x_pos
               - sim.B_pitch * state.theta_dot)
    theta_ddot = M_theta / I_eff

    return np.array([state.z_dot, state.theta_dot, z_ddot, theta_ddot])
