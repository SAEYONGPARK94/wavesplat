"""
relax_estimator.py
Gridless Dispersion-Constrained RELAX Spectral Estimator.

RELAX (Li & Stoica, 1996) adapted for ocean wave spectral estimation:
  - Works with non-uniform (irregular) spatial-temporal observations
  - Dispersion relation ω(k) is a hard constraint
  - Cyclic joint refinement of all components after each addition
  - Continuous parameter optimization (no grid discretization)

Algorithm:
  1. Greedy component addition via coarse grid + Newton refinement
  2. After adding k-th component, cycle through ALL k components:
     - Remove component j from model
     - Re-estimate (kx_j, ky_j, A_j, φ_j) from residual
     - Repeat until convergence
  3. This achieves a jointly optimal K-component decomposition

Key advantage over beamforming:
  - Beamforming: greedy extraction with residual error accumulation
  - RELAX: joint cyclic refinement eliminates inter-component interference
"""

import numpy as np
import torch
import math
from typing import List, Tuple, Optional
from scipy.optimize import minimize

G = 9.81


def _dispersion_omega(kx, ky, depth):
    """Compute ω from dispersion relation: ω = √(g·|k|·tanh(|k|·d))."""
    k_mag = np.sqrt(kx**2 + ky**2 + 1e-12)
    return np.sqrt(G * k_mag * np.tanh(k_mag * depth))


def _compute_phase(kx, ky, x, y, t, depth):
    """Compute phase array: kx*x + ky*y - ω(kx,ky)*t."""
    omega = _dispersion_omega(kx, ky, depth)
    return kx * x + ky * y - omega * t


def _linear_fit(phase, z):
    """
    Fit amplitude and phase from z ≈ A*cos(phase + φ).

    Returns (amplitude, phi).
    """
    c = np.cos(phase)
    s = np.sin(phase)

    cc = np.dot(c, c)
    ss = np.dot(s, s)
    cs = np.dot(c, s)
    cz = np.dot(c, z)
    sz = np.dot(s, z)

    det = cc * ss - cs * cs
    if abs(det) < 1e-12:
        return 1e-4, 0.0

    a = (ss * cz - cs * sz) / det
    b = (cc * sz - cs * cz) / det

    amplitude = np.sqrt(a**2 + b**2)
    phi = np.arctan2(-b, a)

    return max(float(amplitude), 1e-6), float(phi)


def _component_power(kx, ky, x, y, z, t, depth):
    """
    Compute periodogram power at (kx, ky) with dispersion constraint.

    Power = |<z, cos(θ)>|² + |<z, sin(θ)>|²
    where θ = kx*x + ky*y - ω*t
    """
    phase = _compute_phase(kx, ky, x, y, t, depth)
    n = len(z)
    cos_corr = np.dot(z, np.cos(phase)) / n
    sin_corr = np.dot(z, np.sin(phase)) / n
    return cos_corr**2 + sin_corr**2


def _coarse_search(x, y, z, t, depth,
                   kx_range=(-4.0, 4.0), ky_range=(-4.0, 4.0),
                   n_kx=81, n_ky=81):
    """
    Coarse grid search for strongest spectral component.

    Returns (kx_best, ky_best, power_best).
    """
    kx_axis = np.linspace(kx_range[0], kx_range[1], n_kx)
    ky_axis = np.linspace(ky_range[0], ky_range[1], n_ky)

    n_obs = len(z)
    best_power = -1.0
    best_kx, best_ky = 0.0, 0.0

    for ky_val in ky_axis:
        kx_arr = kx_axis[:, None]  # (n_kx, 1)
        k_mag = np.sqrt(kx_arr**2 + ky_val**2 + 1e-12)
        omega = np.sqrt(G * k_mag * np.tanh(k_mag * depth))

        phase = kx_arr * x[None, :] + ky_val * y[None, :] - omega * t[None, :]
        cos_corr = np.mean(z[None, :] * np.cos(phase), axis=1)
        sin_corr = np.mean(z[None, :] * np.sin(phase), axis=1)
        pw = cos_corr**2 + sin_corr**2

        max_idx = np.argmax(pw)
        if pw[max_idx] > best_power:
            best_power = pw[max_idx]
            best_kx = kx_axis[max_idx]
            best_ky = ky_val

    return best_kx, best_ky, best_power


def _refine_newton(x, y, z, t, depth, kx0, ky0):
    """
    Refine (kx, ky) via L-BFGS-B to maximize periodogram power.

    Uses negative power as objective (minimization).
    """
    n = len(z)

    def neg_power(params):
        kx, ky = params
        phase = _compute_phase(kx, ky, x, y, t, depth)
        cos_corr = np.dot(z, np.cos(phase)) / n
        sin_corr = np.dot(z, np.sin(phase)) / n
        return -(cos_corr**2 + sin_corr**2)

    result = minimize(neg_power, [kx0, ky0], method='L-BFGS-B',
                      bounds=[(-6.0, 6.0), (-6.0, 6.0)],
                      options={'maxiter': 50, 'ftol': 1e-12})

    return result.x[0], result.x[1]


def _build_model(x, y, t, depth, components):
    """
    Build the full linear model from component list.

    components: list of (kx, ky, amp, phi)
    Returns: predicted z values
    """
    z_pred = np.zeros(len(x))
    for kx, ky, amp, phi in components:
        phase = _compute_phase(kx, ky, x, y, t, depth)
        z_pred += amp * np.cos(phase + phi)
    return z_pred


def relax_estimate(x, y, z, t, depth, n_components,
                   kx_range=(-4.0, 4.0), ky_range=(-4.0, 4.0),
                   n_coarse=81,
                   max_relax_cycles=5,
                   convergence_tol=1e-6):
    """
    RELAX spectral estimation with dispersion constraint.

    Args:
        x, y, z, t: (N,) observation arrays
        depth: water depth [m]
        n_components: number of spectral components to extract
        kx_range, ky_range: search bounds [rad/m]
        n_coarse: coarse grid resolution per axis
        max_relax_cycles: max cyclic refinement iterations per component addition
        convergence_tol: convergence threshold for cyclic refinement

    Returns:
        components: list of (kx, ky, amplitude, phase) tuples
    """
    z_centered = z - z.mean()
    components = []  # list of [kx, ky, amp, phi]

    for k in range(n_components):
        # --- Step 1: Compute residual ---
        z_residual = z_centered - _build_model(x, y, t, depth, components)

        # --- Step 2: Find new component from residual ---
        kx_coarse, ky_coarse, pw = _coarse_search(
            x, y, z_residual, t, depth,
            kx_range, ky_range, n_coarse, n_coarse
        )

        if pw < 1e-14:
            break

        # Newton refinement
        kx_fine, ky_fine = _refine_newton(
            x, y, z_residual, t, depth, kx_coarse, ky_coarse
        )

        # Linear fit for amplitude and phase
        phase = _compute_phase(kx_fine, ky_fine, x, y, t, depth)
        amp, phi = _linear_fit(phase, z_residual)

        components.append([kx_fine, ky_fine, amp, phi])

        # --- Step 3: Cyclic RELAX refinement ---
        # Skip cyclic refinement until we have enough components
        if len(components) < 2:
            continue

        for cycle in range(max_relax_cycles):
            max_change = 0.0

            for j in range(len(components)):
                old_kx, old_ky = components[j][0], components[j][1]

                # Remove component j from model
                z_without_j = z_centered.copy()
                for i, (ckx, cky, camp, cphi) in enumerate(components):
                    if i == j:
                        continue
                    ph = _compute_phase(ckx, cky, x, y, t, depth)
                    z_without_j -= camp * np.cos(ph + cphi)

                # Re-estimate component j via Newton only (fast)
                kx_new, ky_new = _refine_newton(
                    x, y, z_without_j, t, depth, old_kx, old_ky
                )

                # Re-fit amplitude and phase
                ph = _compute_phase(kx_new, ky_new, x, y, t, depth)
                amp_new, phi_new = _linear_fit(ph, z_without_j)

                # Track change
                change = np.sqrt((kx_new - old_kx)**2 + (ky_new - old_ky)**2)
                max_change = max(max_change, change)

                components[j] = [kx_new, ky_new, amp_new, phi_new]

            if max_change < convergence_tol:
                break

    return components


def initialize_relax(frames, n_gaussians: int, depth: float,
                     kx_range: Tuple[float, float] = (-4.0, 4.0),
                     ky_range: Tuple[float, float] = (-4.0, 4.0),
                     n_coarse: int = 81,
                     max_relax_cycles: int = 5,
                     seed: int = None):
    """
    RELAX spectral initialization for WaveSplat.

    Drop-in replacement for initialize_from_frames_v2.

    Args:
        frames: list of FrameData
        n_gaussians: target number of components
        depth: water depth [m]
        kx_range, ky_range: search bounds [rad/m]
        n_coarse: coarse grid resolution
        max_relax_cycles: RELAX cyclic refinement iterations
        seed: random seed (for padding if fewer components found)

    Returns:
        init_kx, init_ky, init_w, init_phi: (n_gaussians,) tensors
    """
    rng = np.random.RandomState(seed) if seed is not None else np.random

    # Collect observations
    all_x, all_y, all_z, all_t = [], [], [], []
    for f in frames:
        n = len(f.x_obs)
        if n == 0:
            continue
        all_x.append(f.x_obs.cpu().numpy())
        all_y.append(f.y_obs.cpu().numpy())
        all_z.append(f.z_obs.cpu().numpy())
        all_t.append(np.full(n, f.t))

    if not all_x:
        return _random_init(n_gaussians, rng)

    x = np.concatenate(all_x)
    y = np.concatenate(all_y)
    z = np.concatenate(all_z)
    t = np.concatenate(all_t)

    if len(x) < 10:
        return _random_init(n_gaussians, rng)

    # Run RELAX
    components = relax_estimate(
        x, y, z, t, depth, n_gaussians,
        kx_range, ky_range, n_coarse,
        max_relax_cycles
    )

    n_found = len(components)

    kx_arr = np.array([c[0] for c in components]) if n_found > 0 else np.array([])
    ky_arr = np.array([c[1] for c in components]) if n_found > 0 else np.array([])
    w_arr = np.array([c[2] for c in components]) if n_found > 0 else np.array([])
    phi_arr = np.array([c[3] for c in components]) if n_found > 0 else np.array([])

    # Pad if needed
    if n_found < n_gaussians:
        extra = n_gaussians - n_found
        kx_arr = np.concatenate([kx_arr, rng.randn(extra) * 0.05])
        ky_arr = np.concatenate([ky_arr, rng.randn(extra) * 0.05])
        w_arr = np.concatenate([w_arr, np.ones(extra) * 0.01])
        phi_arr = np.concatenate([phi_arr, rng.uniform(0, 2*math.pi, extra)])

    return (torch.tensor(kx_arr[:n_gaussians], dtype=torch.float32),
            torch.tensor(ky_arr[:n_gaussians], dtype=torch.float32),
            torch.tensor(w_arr[:n_gaussians], dtype=torch.float32),
            torch.tensor(phi_arr[:n_gaussians], dtype=torch.float32))


def _random_init(n, rng):
    """Fallback random initialization."""
    return (torch.tensor(rng.randn(n) * 0.05, dtype=torch.float32),
            torch.tensor(rng.randn(n) * 0.05, dtype=torch.float32),
            torch.ones(n, dtype=torch.float32) * 0.1,
            torch.tensor(rng.uniform(0, 2*math.pi, n), dtype=torch.float32))
