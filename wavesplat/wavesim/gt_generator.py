"""
gt_generator.py
Ground truth wave field generation on a regular grid.

Evaluates Gerstner superposition on a fixed spatial grid.
This serves as the reference for algorithm evaluation.
"""

import numpy as np
from typing import List, Tuple
from wavesim.wave_generator import WaveComponent, single_gerstner


def make_grid(x_range: Tuple[float, float],
              y_range: Tuple[float, float],
              nx: int = 100, ny: int = 100) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create evaluation grid.

    Returns:
        X, Y: (ny, nx) meshgrid arrays
    """
    x = np.linspace(x_range[0], x_range[1], nx)
    y = np.linspace(y_range[0], y_range[1], ny)
    return np.meshgrid(x, y)


def evaluate_field(components: List[WaveComponent],
                   X: np.ndarray, Y: np.ndarray,
                   t: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Evaluate wave field at grid (X, Y) at time t.

    Returns:
        Z:       (ny, nx) surface elevation [m]
        NX, NY, NZ: (ny, nx) surface normal components
    """
    _, _, Z, NX, NY, NZ = single_gerstner(components, X, Y, t)
    return Z, NX, NY, NZ


def compute_spectrum_1d(components: List[WaveComponent],
                         n_freq: int = 64) -> np.ndarray:
    """
    Compute 1D energy spectrum from wave components.

    Returns:
        spectrum: (n_freq, 2) array — [omega, S(omega)]
    """
    omegas = np.array([c.omega for c in components])
    energies = np.array([0.5 * c.A**2 for c in components])

    omega_min = omegas.min() * 0.9
    omega_max = omegas.max() * 1.1
    omega_bins = np.linspace(omega_min, omega_max, n_freq + 1)

    S = np.zeros(n_freq)
    for i in range(n_freq):
        mask = (omegas >= omega_bins[i]) & (omegas < omega_bins[i + 1])
        d_omega = omega_bins[i + 1] - omega_bins[i]
        if d_omega > 0:
            S[i] = energies[mask].sum() / d_omega

    omega_centers = 0.5 * (omega_bins[:-1] + omega_bins[1:])
    return np.column_stack([omega_centers, S])
