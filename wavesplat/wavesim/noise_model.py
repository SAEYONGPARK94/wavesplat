"""
noise_model.py
LiDAR sensor noise model.

Based on Gaussian noise from LiDAR datasheets:
  - Range noise:   σ_range ~ 2-3 cm (typical 1550nm solid-state LiDAR, e.g. Seyond Falcon K1)
  - Angular noise: σ_angle ~ 0.05° (beam divergence)
"""

import numpy as np


def add_noise(hit_points: np.ndarray,
              ray_dirs: np.ndarray,
              sigma_range: float = 0.02,
              sigma_angle_deg: float = 0.05,
              rng: np.random.Generator = None) -> np.ndarray:
    """
    Add Gaussian range and angular noise to hit points.

    Range noise:  perturbation along ray direction
    Angular noise: small lateral deflection perpendicular to ray

    Args:
        hit_points:       (N, 3) clean hit positions
        ray_dirs:         (N, 3) unit direction vectors
        sigma_range:      range noise std dev [m]
        sigma_angle_deg:  angular noise std dev [degrees]
        rng:              numpy Generator

    Returns:
        noisy_points: (N, 3)
    """
    if rng is None:
        rng = np.random.default_rng()

    N = hit_points.shape[0]

    # Range noise: along ray direction
    delta_range = rng.normal(0.0, sigma_range, N)
    noisy = hit_points + delta_range[:, np.newaxis] * ray_dirs

    # Angular noise: perpendicular to ray
    sigma_angle = np.deg2rad(sigma_angle_deg)
    # Build arbitrary perpendicular vectors
    up = np.array([0.0, 0.0, 1.0])
    perp1 = np.cross(ray_dirs, up)
    mag1 = np.linalg.norm(perp1, axis=1, keepdims=True)
    # Handle nearly vertical rays
    nearly_vertical = (mag1 < 1e-6).ravel()
    right = np.array([1.0, 0.0, 0.0])
    perp1[nearly_vertical] = np.cross(ray_dirs[nearly_vertical], right)
    mag1 = np.linalg.norm(perp1, axis=1, keepdims=True)
    perp1 = perp1 / (mag1 + 1e-12)
    perp2 = np.cross(ray_dirs, perp1)

    # Small angular perturbation
    da1 = rng.normal(0.0, sigma_angle, N)
    da2 = rng.normal(0.0, sigma_angle, N)

    # Approximate: lateral displacement ≈ d * angle (small angle)
    dist = np.linalg.norm(hit_points - noisy, axis=1) + 1.0  # rough scale
    noisy += (da1 * dist)[:, np.newaxis] * perp1
    noisy += (da2 * dist)[:, np.newaxis] * perp2

    return noisy


def compute_confidence(incidence_angles: np.ndarray,
                        distances: np.ndarray,
                        d_ref: float = 125.0) -> np.ndarray:
    """
    Geometry-aware confidence weight for each LiDAR point.
    Higher at near-normal incidence and close range.

    c = cos(θ_i) / (d/d_ref)

    The cos(θ_i) factor combines the Lambertian projected-area term
    and a monotone surrogate for the angle-dependent specular loss
    (the exact Fresnel R(θ) is used in the raydrop model, not here).
    The 1/(d/d_ref) factor is a normalised inverse-range falloff.

    Args:
        incidence_angles: (N,) [radians]
        distances:        (N,) [m]

    Returns:
        confidence: (N,) in (0, 1]
    """
    cos_i = np.cos(np.clip(incidence_angles, 0, np.pi / 2))
    d_norm = np.clip(distances / d_ref, 0.1, 10.0)
    conf = cos_i / d_norm
    return np.clip(conf, 0.0, 1.0)
