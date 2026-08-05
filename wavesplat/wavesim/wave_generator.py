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
                        Q_max: float = 0.8,
                        peak_locked: bool = False,
                        dirs_per_freq: int = 1) -> List[WaveComponent]:
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
        dirs_per_freq: directions placed at each frequency bin (default 1).

            With the default of 1 this generator puts exactly one direction on
            each frequency, so the WITHIN-frequency directional spread is
            identically zero at every n_components and every
            direction_spread_deg -- an algebraic identity, not a tuning
            artefact.  Measured against 6,263 QC-good CDIP directional-moment
            records, that quantity is positive in 100% of them (median 29.5
            deg), so a faithful discretisation of a measured directional
            spectrum is unreachable on this path: total spread is bounded by
            the across-frequency term alone, median 21.8 deg (18.4 deg over
            this generator's own [0.5, 3]*omega_p band) against a measured
            36.8 deg, and that bound already saturates by N=120.
            See tmp/depth3_ceiling_2026-08-05/FINDING.md.

            dirs_per_freq > 1 splits each bin's energy across that many
            directions (A_i = A_bin / sqrt(M), so sum A_i^2 is preserved),
            which is the only way to make the within-frequency term non-zero.
            dirs_per_freq=1 is bit-identical to the previous behaviour.

    Returns:
        List of WaveComponent
    """
    rng = np.random.default_rng(seed)

    omega_p = 2 * np.pi / Tp  # peak angular frequency

    # Frequency range: 0.5·ωp ~ 3·ωp
    omega_min = 0.5 * omega_p
    omega_max = 3.0 * omega_p
    if dirs_per_freq < 1:
        raise ValueError(f'dirs_per_freq must be >= 1, got {dirs_per_freq}')
    if dirs_per_freq > 1:
        if peak_locked:
            raise ValueError('peak_locked and dirs_per_freq>1 are not combined '
                             '(out of scope; see tmp/s4_within_spread_2026-08-05/PREREG.md)')
        if dirs_per_freq > n_components:
            raise ValueError(f'dirs_per_freq={dirs_per_freq} exceeds '
                             f'n_components={n_components}')
    # Frequency-bin count.  At dirs_per_freq=1 this is n_components and every
    # array below is bit-identical to the previous code path.
    n_freq = n_components // dirs_per_freq
    # Remainder directions are folded into the last bin (declared in the prereg
    # before implementation, so the rule is not chosen after seeing results).
    per_bin = np.full(n_freq, dirs_per_freq, dtype=int)
    per_bin[-1] += n_components - int(per_bin.sum())

    omegas = np.linspace(omega_min, omega_max, n_freq + 1)
    omega_centers = 0.5 * (omegas[:-1] + omegas[1:])
    d_omega = np.diff(omegas)

    if peak_locked:
        # What a "1-component sea" is supposed to mean: the component sits at
        # the spectral peak.  Bin-centring does not do that -- with one bin
        # spanning [0.5, 3]*omega_p the centre lands at 1.75*omega_p, i.e.
        # T = 0.57*Tp and a wavelength 3x shorter than the peak's.  That is how
        # every stored 1-component episode was built, so the correct placement
        # is opt-in rather than a silent change of the dataset
        # (docs/E10_PATTERN_prereg_2026-07-13.md §5, user directive 2026-07-13).
        if n_components != 1:
            raise ValueError(
                'peak_locked only means anything for a single component; with '
                f'n_components={n_components} the bins already tile the band '
                'and one of them holds the peak. Silently ignoring it would '
                'have produced a dataset that looks corrected and is not.')
        omega_centers = np.full(1, omega_p)
        d_omega = np.array([omega_max - omega_min])
    elif n_components == 1:
        print(f"[wave][WARNING] n_components=1 places the single component at "
              f"1.75*omega_p (T={2 * np.pi / omega_centers[0]:.2f} s), not at "
              f"the peak (Tp={Tp:.1f} s). This is a spectral-flank wave, not a "
              f"monochromatic peak wave. Pass peak_locked=True for the peak.",
              flush=True)

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

    if dirs_per_freq > 1:
        # Split each bin's energy across per_bin[i] directions.  A_i =
        # A_bin/sqrt(m) keeps sum(A_i^2) -- and therefore Hs -- unchanged.
        amplitudes = np.repeat(amplitudes / np.sqrt(per_bin), per_bin)
        omega_centers = np.repeat(omega_centers, per_bin)
        ks = np.repeat(ks, per_bin)

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
