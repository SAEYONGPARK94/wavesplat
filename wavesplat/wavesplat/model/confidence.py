"""
confidence.py
Geometry-aware observation weighting for LiDAR wave measurements.

The confidence weight cⱼ reflects observation noise heteroscedasticity:
    cⱼ = cos(θᵢ) / (d / d_ref)

cos(θᵢ) acts as a Lambertian projected-area factor and as a monotone
surrogate for angle-dependent specular loss; (d/d_ref) is a normalised
inverse-range falloff. This is NOT the exact Fresnel reflectance R(θ)
(which is used only in the raydrop model in wavesim.raydrop_uniform),
but a parameter-free geometric simplification.

This is NOT the wave model accuracy — it's how accurate the z-measurement
is at each observation point. Higher incidence angle and larger distance
both degrade measurement precision.

This matches wavesim.noise_model.compute_confidence exactly.
"""

import torch


def observation_weights(incidence_angles: torch.Tensor,
                        distances: torch.Tensor,
                        d_ref: float = 125.0) -> torch.Tensor:
    """
    Compute per-observation confidence weights.

    Args:
        incidence_angles: (N,) incidence angles [rad]
        distances:        (N,) sensor-to-surface distances [m]
        d_ref:            reference distance [m]

    Returns:
        weights: (N,) confidence in [0, 1]
    """
    cos_theta = torch.cos(incidence_angles)
    d_norm = distances / d_ref
    conf = cos_theta / (d_norm + 1e-9)
    return torch.clamp(conf, 0.0, 1.0)
