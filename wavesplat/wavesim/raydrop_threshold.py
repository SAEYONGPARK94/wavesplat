"""raydrop_threshold.py — threshold-on-received-power dropout (Model C).

Why this model exists. The Falcon manual specifies detection as threshold
behaviour: "Detection range (10% reflectivity) 250 m" (p.13) is the range at
which a stated target's return crosses the receiver's detection threshold, and
the point cloud carries a graded confidence_level 0-3 with strongest/furthest
multi-return selection (p.105) — peak-over-threshold language throughout. See
docs/F2_DETECTION_PHYSICS_2026-07-10.md. The existing models are different
families: raydrop_uniform is Bernoulli-with-gain (P = eta*A, multiplicative),
raydrop_slope's specular gate never fires at this geometry (required facet
slope r/h = 3..40 vs wave-slope sd ~0.08) and falls back to constant-P.

Detection law:
    A_i = R_fresnel(theta_i) * cos(theta_i) / (d_i/d_ref)^2      (received power)
    X_i ~ Gamma(M, 1/M)   E[X]=1                                 (speckle, M looks)
    return  iff  A_i * X_i > T
M = 1 is fully-developed (exponential) speckle — the standard model for a
coherent 1550 nm return from a rough water surface. Pulse/aperture averaging
raises M and steepens the tail toward the multiplicative limit; M is therefore
the honest knob for "how much speckle diversity does the real receiver have",
which the manual does not specify.

Why the information content differs from raydrop_uniform: with exponential
speckle p(s) = exp(-T/A(s)), so d ln p / ds = (-ln p) * d ln A / ds — the
selection's slope-sensitivity is AMPLIFIED by (-ln p) at sparse operation,
instead of being rate-independent. Sparser sensing makes the mask MORE
informative per bit. This is the F2 channel; measured in
scripts/d3_s2e_threshold_accounting.py.

Threshold calibration: per frame, T is set so the return count equals
target_n_pts exactly (top-k on A*X). This mirrors an AGC/CFAR-like receiver
holding its false-alarm/return budget, matches the "strongest return"
selection mode, and parallels raydrop_uniform's per-frame eta recalibration.
Like that recalibration it couples the bits weakly through the frame
normalizer; the accounting script treats bits as independent, which is the
conservative direction for information.
"""
import numpy as np

from .raydrop_uniform import fresnel_reflectance


def received_power(incidence_angles: np.ndarray,
                   distances: np.ndarray,
                   d_ref: float = 125.0) -> np.ndarray:
    """A_i = R_fresnel(theta) * cos(theta) / (d/d_ref)^2, the radiometric part."""
    R = fresnel_reflectance(incidence_angles)
    d_norm = distances / d_ref
    return R * np.cos(incidence_angles) / (d_norm ** 2 + 1e-9)


def apply_dropout(hit_mask: np.ndarray,
                  hit_points: np.ndarray,
                  hit_dist: np.ndarray,
                  incidence_angles: np.ndarray,
                  target_n_pts: int,
                  rng: np.random.Generator,
                  m_looks: int = 1,
                  d_ref: float = 125.0) -> np.ndarray:
    """Threshold detection with M-look speckle; exact per-frame return count.

    Same interface as raydrop_uniform.apply_dropout (eta replaced by m_looks).

    Returns: (N_rays,) bool return mask, exactly min(target_n_pts, N_hits) True.
    """
    return_mask = np.zeros(len(hit_mask), dtype=bool)
    hit_idx = np.where(hit_mask)[0]
    if len(hit_idx) == 0:
        return return_mask

    A = received_power(incidence_angles[hit_idx], hit_dist[hit_idx], d_ref)
    X = rng.gamma(shape=m_looks, scale=1.0 / m_looks, size=len(hit_idx))
    power = A * X

    k = min(int(target_n_pts), len(hit_idx))
    top = np.argpartition(power, -k)[-k:]        # T = k-th largest A*X
    return_mask[hit_idx[top]] = True
    return return_mask


# Manual link-budget calibration for the FIXED threshold (H3-OD 2026-07-11).
# "Detection range (10% reflectivity) 250 m" (p.13): the stated target's
# received power in THIS normalisation is A_m = 0.10 / (250/d_ref)^2 = 0.025;
# 50% detection under exponential speckle means T = A_m ln 2. This is a
# RELATIVE calibration on top of the water-Fresnel radiometry, not an
# absolute link budget (docs/H3OD_FIXEDT_OPERATING_DOMAIN_2026-07-11.md §4).
T_MANUAL_50 = 0.10 / (250.0 / 125.0) ** 2 * np.log(2.0)


def apply_dropout_fixed_T(hit_mask: np.ndarray,
                          hit_dist: np.ndarray,
                          incidence_angles: np.ndarray,
                          rng: np.random.Generator,
                          T: float = T_MANUAL_50,
                          m_looks: int = 1,
                          d_ref: float = 125.0,
                          cap: int = 0) -> np.ndarray:
    """FIXED-threshold detection — the sensor's actual law (F2 verdict).

    Unlike apply_dropout there is no per-frame count budget: return iff
    A*X > T, so the count is a measurement (~95% of hits at T_MANUAL_50).
    cap > 0 subsamples the returns uniformly (random-within-selection keeps
    the spatial distribution; storage/compute control only — record the raw
    count from the caller if it matters).
    """
    return_mask = np.zeros(len(hit_mask), dtype=bool)
    hit_idx = np.where(hit_mask)[0]
    if len(hit_idx) == 0:
        return return_mask

    A = received_power(incidence_angles[hit_idx], hit_dist[hit_idx], d_ref)
    X = rng.gamma(shape=m_looks, scale=1.0 / m_looks, size=len(hit_idx))
    sel = np.where(A * X > T)[0]
    if cap and len(sel) > cap:
        sel = rng.choice(sel, cap, replace=False)
    return_mask[hit_idx[sel]] = True
    return return_mask
