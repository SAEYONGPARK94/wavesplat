"""Hydrofoil lift/drag model with submergence and free-surface corrections.

Physics model:
  L = 0.5 * rho * V² * S * CL(alpha)
  D = 0.5 * rho * V² * S * CD(alpha)

  where CL = cl_alpha * (alpha - alpha_zl),  clamped at cl_max
        CD = cd0 + CL² / (pi * AR * e)       (drag polar)

Corrections:
  - Submergence: CL degrades when foil approaches free surface (h/c < 1.0)
  - Ventilation: CL drops sharply when h/c < 0.3 (air entrainment)
"""

from typing import Tuple

import numpy as np
from .config import FoilConfig


def aspect_ratio(foil: FoilConfig) -> float:
    """Geometric aspect ratio AR = span / chord."""
    return foil.span / foil.chord


def planform_area(foil: FoilConfig) -> float:
    """Planform area S = span * chord [m²]."""
    return foil.span * foil.chord


def submergence_factor(h: float, chord: float) -> float:
    """Free-surface proximity correction factor.

    Based on free-surface proximity effects (Faltinsen, 2005, Ch.7).
    Models lift degradation as foil approaches the surface.

    Args:
        h: submergence depth of foil quarter-chord below calm water [m].
           Positive means submerged.
        chord: foil chord length [m]

    Returns:
        factor in [0, 1]. Multiply with CL to get corrected lift.
    """
    if h <= 0.0:
        # Foil breached surface — catastrophic lift loss
        return 0.0

    ratio = h / chord

    if ratio >= 2.0:
        # Deep enough — no free-surface effect
        return 1.0
    elif ratio >= 0.5:
        # Gradual degradation (approximately linear in log-space)
        # At h/c=2.0: factor=1.0, at h/c=0.5: factor~0.7
        return 0.7 + 0.3 * (ratio - 0.5) / 1.5
    else:
        # Ventilation risk zone — rapid degradation
        # At h/c=0.5: factor=0.7, at h/c=0: factor=0
        return 0.7 * (ratio / 0.5)


def compute_cl(alpha: float, foil: FoilConfig) -> float:
    """Lift coefficient from angle of attack.

    Linear lift curve with stall clamp. Post-stall behavior is simplified
    as a plateau (no CL drop) since hydrofoils typically operate well
    below stall in foilborne regime.

    Args:
        alpha: effective angle of attack [rad]
        foil: foil configuration

    Returns:
        CL (unsigned convention: positive alpha → positive CL)
    """
    cl = foil.cl_alpha * (alpha - foil.alpha_zl)
    return np.clip(cl, -foil.cl_max, foil.cl_max)


def compute_cd(cl: float, foil: FoilConfig, oswald_e: float = 0.85) -> float:
    """Drag coefficient from drag polar.

    CD = CD0 + CL² / (pi * AR * e)

    Args:
        cl: current lift coefficient
        foil: foil configuration
        oswald_e: Oswald efficiency factor

    Returns:
        CD (always positive)
    """
    ar = aspect_ratio(foil)
    cd_induced = cl ** 2 / (np.pi * ar * oswald_e)
    return foil.cd0 + cd_induced


def foil_forces(alpha: float, V: float, h: float, foil: FoilConfig,
                rho: float = 1025.0) -> Tuple[float, float]:
    """Compute lift and drag forces on a single foil.

    Args:
        alpha: effective angle of attack [rad]
        V: flow velocity magnitude [m/s]
        h: submergence depth below instantaneous water surface [m]
        foil: foil configuration
        rho: water density [kg/m³]

    Returns:
        (L, D): lift force [N] (perpendicular to flow, positive up),
                drag force [N] (along flow, positive aft)
    """
    S = planform_area(foil)
    q = 0.5 * rho * V ** 2  # dynamic pressure

    cl = compute_cl(alpha, foil)
    cd = compute_cd(cl, foil)

    # Apply submergence correction to lift only
    # (drag is less affected by free-surface — conservative assumption)
    sf = submergence_factor(h, foil.chord)
    cl_eff = cl * sf

    L = q * S * cl_eff
    D = q * S * cd

    return L, D
