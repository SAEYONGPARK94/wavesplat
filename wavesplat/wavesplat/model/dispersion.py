"""
dispersion.py
Differentiable dispersion relation for PyTorch.

Physics:
    ω = √(g · |k| · tanh(|k| · d))

    Full dispersion (no deep-water approximation), following principle #3.
    This is the projection constraint: ω is NOT a free parameter, it is
    deterministically computed from wavenumber k and water depth d.

Reference: Airy wave theory; Dean & Dalrymple (1991).
"""

import torch
import math

G = 9.81  # gravitational acceleration [m/s²]


def dispersion_omega(k_mag: torch.Tensor, depth: float) -> torch.Tensor:
    """
    Full dispersion relation: ω = √(g · |k| · tanh(|k| · d)).

    Args:
        k_mag: (N,) wavenumber magnitudes [rad/m], must be >= 0
        depth: water depth [m]

    Returns:
        omega: (N,) angular frequencies [rad/s]
    """
    kd = k_mag * depth
    return torch.sqrt(G * k_mag * torch.tanh(kd) + 1e-12)


def wavenumber_from_omega(omega: torch.Tensor, depth: float,
                          n_iter: int = 20) -> torch.Tensor:
    """
    Inverse dispersion: given ω, find |k| via Newton-Raphson.
    Used for FFT-based initialization (not in gradient path).

    Args:
        omega: (N,) angular frequencies [rad/s]
        depth: water depth [m]
        n_iter: Newton iterations

    Returns:
        k_mag: (N,) wavenumber magnitudes [rad/m]
    """
    # Initial guess: deep-water approximation k = ω²/g
    k = omega ** 2 / G

    for _ in range(n_iter):
        kd = k * depth
        tanh_kd = torch.tanh(kd)
        f = G * k * tanh_kd - omega ** 2
        # f' = g * (tanh(kd) + kd * (1 - tanh²(kd)))
        fp = G * (tanh_kd + k * depth * (1.0 - tanh_kd ** 2))
        k = k - f / (fp + 1e-12)
        k = torch.clamp(k, min=1e-6)

    return k


def wavelength_from_period(period: torch.Tensor, depth: float) -> torch.Tensor:
    """Convert wave period T [s] to wavelength λ [m]."""
    omega = 2.0 * math.pi / period
    k = wavenumber_from_omega(omega, depth)
    return 2.0 * math.pi / k
