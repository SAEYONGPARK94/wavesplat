"""
raydrop_slope.py
Model B: Slope-based specular backscatter dropout.

Physics:
  A specular return occurs when the surface normal at the hit point
  is aligned with the direction BACK toward the sensor.
  Condition: the reflection of the incoming ray off the surface
             points toward the sensor.

  Reflected ray: r = d - 2·(d·n)·n  where d=ray direction, n=surface normal
  Condition for return: r · (-d) ≥ cos(α_threshold)
  i.e., the reflected direction is within α_threshold of the back-direction.

  Among specular candidates, each beam returns independently (Bernoulli)
  with probability proportional to Fresnel·alignment/d².  η is scaled so
  that E[N_specular_returns] = target_n_pts.  This produces physically
  realistic frame-to-frame variation in return count.

  On a flat (still) surface: reflected ray goes upward → no specular return
  at forward-looking grazing geometry (correct).
  Fallback for calm sea (no specular candidates): uniform Bernoulli sampling
  with P = target_n_pts / N_hits, modelling diffuse capillary-wave scattering.

Reference: Standard specular reflection geometry (reflection law).
"""

import numpy as np


def specular_alignment(ray_dirs: np.ndarray,
                        surface_normals: np.ndarray,
                        alpha_threshold_deg: float = 2.0) -> np.ndarray:
    """
    Compute whether each ray achieves specular backscatter.

    Args:
        ray_dirs:        (N, 3) unit direction vectors (sensor → surface)
        surface_normals: (N, 3) unit outward normals at surface
        alpha_threshold_deg: angular threshold for specular backscatter [degrees]

    Returns:
        specular_mask: (N,) bool — True if specular condition met
        alignment:     (N,) float — dot(reflected_ray, -ray_dir) in [-1,1]
    """
    d_dot_n = np.einsum('ij,ij->i', ray_dirs, surface_normals)
    reflected = ray_dirs - 2.0 * d_dot_n[:, np.newaxis] * surface_normals

    reflected_mag = np.linalg.norm(reflected, axis=1, keepdims=True)
    reflected = reflected / (reflected_mag + 1e-12)

    back_dir = -ray_dirs
    alignment = np.einsum('ij,ij->i', reflected, back_dir)

    cos_thresh = np.cos(np.deg2rad(alpha_threshold_deg))
    specular_mask = alignment >= cos_thresh

    return specular_mask, alignment


def apply_dropout(hit_mask: np.ndarray,
                   hit_points: np.ndarray,
                   hit_dist: np.ndarray,
                   ray_dirs: np.ndarray,
                   surface_normals: np.ndarray,
                   incidence_angles: np.ndarray,
                   target_n_pts: int,
                   rng: np.random.Generator,
                   alpha_threshold_deg: float = 5.0,
                   d_ref: float = 125.0) -> np.ndarray:
    """
    Apply slope-based specular dropout via independent Bernoulli sampling.

    Steps:
      1. Check specular condition for each hit beam.
      2. Among specular candidates, weight W_i = Fresnel · alignment · 1/d².
      3. Scale η so E[N_returns] = target_n_pts, clip P_i = clip(η·W_i, 0, 1).
      4. Independent Bernoulli draw per candidate beam.

    If no specular candidates (calm sea / BF2-3), fall back to uniform
    Bernoulli with P = target_n_pts / N_hits (diffuse capillary scattering).

    The actual return count varies naturally across frames, reflecting the
    evolving wave surface geometry.

    Args:
        hit_mask:         (N_rays,) bool
        hit_points:       (N_rays, 3)
        hit_dist:         (N_rays,)
        ray_dirs:         (N_rays, 3)
        surface_normals:  (N_rays, 3)
        incidence_angles: (N_rays,)
        target_n_pts:     expected number of returns per frame
        rng:              numpy random Generator
        alpha_threshold_deg: specular cone half-angle [degrees]
        d_ref:            reference distance [m]

    Returns:
        return_mask: (N_rays,) bool — variable count per call
    """
    from wavesim.raydrop_uniform import fresnel_reflectance

    return_mask = np.zeros(len(hit_mask), dtype=bool)
    if not np.any(hit_mask):
        return return_mask

    hit_idx = np.where(hit_mask)[0]

    spec_mask, alignment = specular_alignment(
        ray_dirs[hit_idx], surface_normals[hit_idx], alpha_threshold_deg
    )

    specular_idx = hit_idx[spec_mask]

    if len(specular_idx) == 0:
        # Fallback: calm sea — uniform Bernoulli (diffuse capillary scattering)
        P_uniform = min(target_n_pts / len(hit_idx), 1.0)
        draws = rng.random(len(hit_idx))
        return_mask[hit_idx[draws < P_uniform]] = True
        return return_mask

    # Weight: Fresnel × alignment quality × 1/d²
    R = fresnel_reflectance(incidence_angles[specular_idx])
    d_norm = hit_dist[specular_idx] / d_ref
    W = R * alignment[spec_mask] / (d_norm**2 + 1e-9)
    W = np.clip(W, 0.0, None)

    W_sum = W.sum()
    if W_sum < 1e-12:
        return return_mask

    # Scale so E[N_returns] = target_n_pts, cap at 1
    eta_scaled = target_n_pts / W_sum
    P = np.clip(eta_scaled * W, 0.0, 1.0)

    # Independent Bernoulli draw per specular candidate
    draws = rng.random(len(specular_idx))
    selected = specular_idx[draws < P]
    return_mask[selected] = True

    return return_mask
