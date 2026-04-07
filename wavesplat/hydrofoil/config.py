"""Hydrofoil simulation configuration.

Reference vessel: Candela C-8 (Swedish electric hydrofoil)
- LOA: 8.5 m, Beam: 2.4 m, Displacement: ~1700 kg
- Tandem foil configuration: front (inverted-T) + rear (T-foil)
- Cruise speed: ~22 knots foilborne
"""

from dataclasses import dataclass, field
import numpy as np


@dataclass
class FoilConfig:
    """Single hydrofoil geometry and hydrodynamic parameters."""
    span: float          # [m] foil span
    chord: float         # [m] foil chord length
    cl_alpha: float      # [1/rad] lift curve slope (2D)
    cl_max: float        # max CL before stall
    cd0: float           # zero-lift drag coefficient
    alpha_zl: float      # [rad] zero-lift angle of attack
    x_pos: float         # [m] longitudinal position from CG (positive = forward)
    z_pos: float         # [m] vertical position below waterline (positive = downward)


@dataclass
class HullConfig:
    """Vessel hull parameters."""
    mass: float          # [kg] total mass (vessel + payload)
    I_yy: float          # [kg·m²] pitch moment of inertia about CG
    cg_x: float          # [m] CG position from bow
    cg_z: float          # [m] CG height above keel
    draft_foilborne: float  # [m] hull clearance above waterline in foilborne


@dataclass
class SimConfig:
    """Simulation parameters."""
    dt: float = 0.01          # [s] integration timestep
    duration: float = 30.0    # [s] total simulation time
    speed: float = 11.3       # [m/s] forward speed (~22 knots)
    depth: float = 50.0       # [m] water depth
    wave_heading: float = 0.0  # [rad] wave propagation direction (0=head sea)
    rho: float = 1025.0       # [kg/m³] seawater density
    g: float = 9.81           # [m/s²] gravitational acceleration

    # Structural/parasitic damping only — foil lift-slope already provides
    # natural hydrodynamic damping via alpha = -z_dot/V term in effective_alpha.
    # These values represent strut drag + structural dissipation.
    B_heave: float = 50.0     # [N·s/m] structural heave damping (NOT hydro)
    B_pitch: float = 20.0     # [N·m·s/rad] structural pitch damping (NOT hydro)


def candela_c8() -> dict:
    """Default configuration based on Candela C-8 class hydrofoil.

    Returns dict with 'front_foil', 'rear_foil', 'hull', 'sim' keys.
    """
    # Front foil: inverted-T, main lift provider (~60-70% of weight)
    front_foil = FoilConfig(
        span=1.8,           # slightly narrower than beam
        chord=0.20,         # ~200mm chord
        cl_alpha=2 * np.pi * 0.85,  # thin airfoil theory * aspect ratio correction
        cl_max=1.2,
        cd0=0.008,
        alpha_zl=0.0,
        x_pos=2.0,          # 2m forward of CG
        z_pos=0.8,          # 0.8m below waterline
    )

    # Rear foil: T-foil, pitch stabilizer (~30-40% of weight)
    rear_foil = FoilConfig(
        span=1.2,           # smaller span
        chord=0.15,         # 150mm chord
        cl_alpha=2 * np.pi * 0.80,
        cl_max=1.2,
        cd0=0.010,
        alpha_zl=0.0,
        x_pos=-2.5,         # 2.5m aft of CG
        z_pos=0.6,          # 0.6m below waterline
    )

    hull = HullConfig(
        mass=1700.0,
        I_yy=4900.0,        # ~ m * (L/5)^2 = 1700 * 1.7^2
        cg_x=3.5,           # mid-ship
        cg_z=0.5,
        draft_foilborne=0.3,
    )

    sim = SimConfig()

    cfg = {
        'front_foil': front_foil,
        'rear_foil': rear_foil,
        'hull': hull,
        'sim': sim,
    }

    # Compute design alpha_zl so that trim exists at zero pitch/heave.
    # At trim: L_front + L_rear = m*g, and moment balance about CG.
    _set_trim_alpha(cfg)

    return cfg


def _set_trim_alpha(cfg: dict) -> None:
    """Compute and set alpha_zl for each foil so that static trim exists.

    At trim (z=0, theta=0, calm water):
      - required total lift = m*g
      - moment balance: L_f * x_f + L_r * x_r = 0
      - L_i = 0.5 * rho * V^2 * S_i * cl_alpha_i * (alpha_trim_i - alpha_zl_i)

    We solve for alpha_trim at each foil, then set alpha_zl = -alpha_trim
    so that CL(alpha=0) = cl_alpha * (0 - alpha_zl) = cl_alpha * alpha_trim.
    """
    hull = cfg['hull']
    sim = cfg['sim']
    ff = cfg['front_foil']
    rf = cfg['rear_foil']

    W = hull.mass * sim.g
    rho = sim.rho
    V = sim.speed

    q = 0.5 * rho * V ** 2
    S_f = ff.span * ff.chord
    S_r = rf.span * rf.chord

    # From moment balance: L_f * x_f + L_r * x_r = 0
    # → L_f / L_r = -x_r / x_f
    # From force balance: L_f + L_r = W
    # → L_f = W * (-x_r) / (x_f - x_r)
    #   L_r = W * x_f / (x_f - x_r)   [note: x_r < 0]
    lever = ff.x_pos - rf.x_pos
    L_f_req = W * (-rf.x_pos) / lever
    L_r_req = W * ff.x_pos / lever

    # Required CL at each foil
    cl_f_req = L_f_req / (q * S_f)
    cl_r_req = L_r_req / (q * S_r)

    # Required alpha at each foil: CL = cl_alpha * (alpha - alpha_zl)
    # At trim, alpha_geom = 0, so we need alpha_zl = -CL_req / cl_alpha
    alpha_f_req = cl_f_req / ff.cl_alpha
    alpha_r_req = cl_r_req / rf.cl_alpha

    # Set alpha_zl so that alpha=0 produces the required CL
    # CL = cl_alpha * (0 - alpha_zl) = CL_req → alpha_zl = -alpha_req
    ff.alpha_zl = -alpha_f_req
    rf.alpha_zl = -alpha_r_req
