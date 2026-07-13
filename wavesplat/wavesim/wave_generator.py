"""
wave_generator.py
Gerstner wave generation with JONSWAP spectrum.

Physical basis:
  - Individual Gerstner wave: rotational exact solution (Gerstner 1802, Dean & Dalrymple 1991)
  - Superposition: approximate solution, valid for ε << 1 (BF3~4)
  - JONSWAP spectrum: Hasselmann et al. (1973)
  - Q constraint (particle crossing prevention): Tessendorf (2001), Finch (2004)
  - Full dispersion: ω = √(g·k·tanh(k·d))  [no deep-water approximation]
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional


G = 9.81  # gravitational acceleration [m/s²]


@dataclass
class WaveComponent:
    """Single Gerstner wave component."""
    A: float        # amplitude [m]
    omega: float    # angular frequency [rad/s]
    kx: float       # wavenumber x-component [rad/m]
    ky: float       # wavenumber y-component [rad/m]
    phi: float      # initial phase [rad]
    Q: float        # Gerstner steepness parameter (0=sinusoidal, 1/ωA=max)

    @property
    def k(self) -> float:
        return np.sqrt(self.kx**2 + self.ky**2)

    @property
    def direction(self) -> float:
        return np.arctan2(self.ky, self.kx)


def full_dispersion(k: float, depth: float = 50.0) -> float:
    """
    Full linear dispersion relation.
    ω = √(g · k · tanh(k · d))

    Args:
        k: wavenumber magnitude [rad/m]
        depth: water depth [m]
    Returns:
        angular frequency ω [rad/s]
    """
    return np.sqrt(G * k * np.tanh(k * depth))


def wavenumber_from_omega(omega: float, depth: float = 50.0,
                           tol: float = 1e-8, max_iter: int = 50) -> float:
    """
    Inverse dispersion: solve ω² = g·k·tanh(k·d) for k.
    Uses Newton-Raphson iteration.
    """
    # Deep-water initial guess: k0 = ω²/g
    k = omega**2 / G
    for _ in range(max_iter):
        th = np.tanh(k * depth)
        f = G * k * th - omega**2
        df = G * (th + k * depth * (1.0 - th**2))
        dk = -f / df
        k += dk
        if abs(dk) < tol:
            break
    return k


def single_gerstner(components: List[WaveComponent],
                    x: np.ndarray, y: np.ndarray,
                    t: float) -> tuple:
    """
    Evaluate Gerstner wave superposition at grid points (x, y) at time t.

    Returns:
        X_disp: x-displacement from rest position [m]  shape=(*x.shape)
        Y_disp: y-displacement from rest position [m]
        Z:      surface elevation [m]
        nx, ny, nz: surface normal (unit vector components)
    """
    X_disp = np.zeros_like(x, dtype=float)
    Y_disp = np.zeros_like(y, dtype=float)
    Z = np.zeros_like(x, dtype=float)

    # Jacobian terms for normal computation
    dZdx = np.zeros_like(x, dtype=float)
    dZdy = np.zeros_like(x, dtype=float)
    dXdx = np.ones_like(x, dtype=float)
    dYdy = np.ones_like(x, dtype=float)

    for c in components:
        phase = c.kx * x + c.ky * y - c.omega * t + c.phi
        cos_p = np.cos(phase)
        sin_p = np.sin(phase)

        Z += c.A * cos_p
        X_disp += -c.Q * c.A * (c.kx / c.k) * sin_p
        Y_disp += -c.Q * c.A * (c.ky / c.k) * sin_p

        # Partial derivatives for surface normal
        dZdx += -c.A * c.kx * sin_p
        dZdy += -c.A * c.ky * sin_p
        dXdx += -c.Q * c.A * c.k * (c.kx / c.k)**2 * cos_p
        dYdy += -c.Q * c.A * c.k * (c.ky / c.k)**2 * cos_p

    # Normal vector: n = (-∂z/∂x, -∂z/∂y, 1) normalized
    n_unnorm = np.stack([-dZdx, -dZdy, np.ones_like(dZdx)], axis=-1)
    n_mag = np.linalg.norm(n_unnorm, axis=-1, keepdims=True)
    n_unit = n_unnorm / (n_mag + 1e-12)
    nx, ny, nz = n_unit[..., 0], n_unit[..., 1], n_unit[..., 2]

    return X_disp, Y_disp, Z, nx, ny, nz


def check_q_constraint(components: List[WaveComponent]) -> tuple:
    """
    Check Particle crossing prevention condition:
      ΣQᵢ·ωᵢ·Aᵢ ≤ 1

    Returns:
        satisfied: bool
        value: actual ΣQᵢ·ωᵢ·Aᵢ
    """
    value = sum(c.Q * c.omega * c.A for c in components)
    return value <= 1.0, value


def jonswap_components(Hs: float, Tp: float,
                        n_components: int = 30,
                        depth: float = 50.0,
                        gamma: float = 3.3,
                        direction_deg: float = 0.0,
                        direction_spread_deg: float = 30.0,
                        seed: Optional[int] = None,
                        Q_max: float = 0.8) -> List[WaveComponent]:
    """
    Generate wave components from JONSWAP spectrum (Hasselmann et al. 1973).

    Args:
        Hs:         significant wave height [m]
        Tp:         peak period [s]
        n_components: number of spectral components
        depth:      water depth [m]
        gamma:      JONSWAP peak enhancement factor (default 3.3)
        direction_deg: mean wave direction [degrees from +x axis]
        direction_spread_deg: directional spreading [degrees, std dev]
        seed:       random seed for reproducibility
        Q_max:      maximum Q parameter value (< 1/(ω·A) for each component)

    Returns:
        List of WaveComponent
    """
    rng = np.random.default_rng(seed)

    omega_p = 2 * np.pi / Tp  # peak angular frequency

    # Frequency range: 0.5·ωp ~ 3·ωp
    omega_min = 0.5 * omega_p
    omega_max = 3.0 * omega_p
    omegas = np.linspace(omega_min, omega_max, n_components + 1)
    omega_centers = 0.5 * (omegas[:-1] + omegas[1:])
    d_omega = np.diff(omegas)

    if n_components == 1:
        # The lone bin spans the whole band, so its centre lands at 1.75*omega_p
        # -- NOT the peak.  A "1-component BF2" is a T = 0.57*Tp wave carrying
        # the full Hs, with a wavelength ~3x shorter than the peak's.  Changing
        # this would break reproducibility of every stored episode, so it stands
        # as documented behaviour (docs/E2_MESH_AUDIT_prereg_2026-07-13.md).
        print(f"[wave][WARNING] n_components=1 places the single component at "
              f"1.75*omega_p (T={2 * np.pi / omega_centers[0]:.2f} s), not at "
              f"the peak (Tp={Tp:.1f} s). This is a spectral-flank wave, not a "
              f"monochromatic peak wave.", flush=True)

    # JONSWAP spectrum S(ω)
    def jonswap_spectrum(omega):
        sigma = np.where(omega <= omega_p, 0.07, 0.09)
        alpha_j = np.exp(-0.5 * ((omega / omega_p - 1) / sigma)**2)
        S = (0.3125 * Hs**2 * omega_p**4 / omega**5
             * np.exp(-1.25 * (omega_p / omega)**4)
             * gamma**alpha_j)
        return S

    S = jonswap_spectrum(omega_centers)

    # Variance → amplitude: A = √(2·S·dω)
    variances = S * d_omega
    amplitudes = np.sqrt(2 * variances)

    # Normalize amplitudes so that Hs matches exactly:
    # Hs = 4·√(Σ Aᵢ²/2)  →  scale = Hs / (4·√(Σ Aᵢ²/2))
    Hs_raw = 4.0 * np.sqrt(np.sum(amplitudes**2 / 2.0))
    if Hs_raw > 0:
        amplitudes = amplitudes * (Hs / Hs_raw)

    # Wavenumbers from full dispersion
    ks = np.array([wavenumber_from_omega(w, depth) for w in omega_centers])

    # Random phases
    phases = rng.uniform(0, 2 * np.pi, n_components)

    # Directional spreading (Gaussian)
    dir_rad = np.deg2rad(direction_deg)
    spread_rad = np.deg2rad(direction_spread_deg)
    directions = rng.normal(dir_rad, spread_rad, n_components)

    # Build components
    components = []
    for i in range(n_components):
        A = amplitudes[i]
        omega = omega_centers[i]
        k = ks[i]
        kx = k * np.cos(directions[i])
        ky = k * np.sin(directions[i])
        phi = phases[i]

        # Q parameter: use Q_max as fraction of maximum allowable
        Q_max_i = min(Q_max / (omega * A + 1e-12), 1.0)
        Q = Q_max_i * 0.5  # conservative: 50% of maximum

        components.append(WaveComponent(A=A, omega=omega,
                                         kx=kx, ky=ky,
                                         phi=phi, Q=Q))

    # Enforce global Q constraint: scale down Qs if needed
    satisfied, q_val = check_q_constraint(components)
    if not satisfied:
        scale = 0.95 / q_val
        for c in components:
            c.Q *= scale

    return components


def superpose(components: List[WaveComponent],
              x: np.ndarray, y: np.ndarray,
              t: float) -> tuple:
    """
    Convenience wrapper: evaluate superposed Gerstner field.
    Returns (Z, nx, ny, nz) — surface height and normals.
    """
    _, _, Z, nx, ny, nz = single_gerstner(components, x, y, t)
    return Z, nx, ny, nz


def compute_Hs(Z: np.ndarray) -> float:
    """Estimate Hs from surface elevation field: Hs ≈ 4·σ_η"""
    return 4.0 * np.std(Z)
