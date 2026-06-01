"""
cs_initializer.py
Compressed Sensing spectral initialization for WaveSplat.

Uses L1-regularized (LASSO) estimation on a fixed wavenumber grid
to recover sparse spectral components from irregular LiDAR observations.

Key advantage over beamforming (v2):
  - Handles severely underdetermined systems (K_grid >> N_obs) via sparsity
  - Multi-frame temporal stacking increases effective observations
  - Dispersion relation ω(k) is built into the design matrix (hard constraint)
  - Can recover K=50-200+ components vs K=5-30 from beamforming

Forward model (linear in unknowns):
    z_i = Σ_j [c_j cos(θ_ij) + s_j sin(θ_ij)]
    where θ_ij = kx_j·x_i + ky_j·y_i - ω_j·t_i
    and ω_j = √(g·|k_j|·tanh(|k_j|·d))

Unknowns: (c_j, s_j) for each grid point j
Amplitude: A_j = √(c_j² + s_j²)
Phase: φ_j = atan2(-s_j, c_j)
"""

import numpy as np
import torch
import math
from typing import List, Tuple, Optional

G = 9.81


def _build_design_matrix(x, y, t, kx_grid, ky_grid, omega_grid):
    """
    Build the design matrix A for LASSO.

    A has shape (N_obs, 2*K) where K = len(kx_grid).
    Column 2j   = cos(kx_j*x + ky_j*y - omega_j*t)  (for all obs)
    Column 2j+1 = sin(kx_j*x + ky_j*y - omega_j*t)

    Args:
        x, y, t: (N,) observation coordinates
        kx_grid, ky_grid, omega_grid: (K,) wavenumber grid

    Returns:
        A: (N, 2K) design matrix
    """
    N = len(x)
    K = len(kx_grid)

    # Phase: (N, K)
    phase = (kx_grid[None, :] * x[:, None]
             + ky_grid[None, :] * y[:, None]
             - omega_grid[None, :] * t[:, None])

    A = np.empty((N, 2 * K), dtype=np.float64)
    A[:, 0::2] = np.cos(phase)
    A[:, 1::2] = np.sin(phase)

    return A


def _build_wavenumber_grid(kx_range=(-3.0, 3.0), ky_range=(-3.0, 3.0),
                           n_kx=31, n_ky=31, depth=50.0,
                           k_min=0.01, k_max=5.0):
    """
    Build dispersion-constrained wavenumber grid.

    Only includes grid points within physically relevant wavenumber range.

    Returns:
        kx_grid, ky_grid, omega_grid: (K,) arrays
    """
    kx_1d = np.linspace(kx_range[0], kx_range[1], n_kx)
    ky_1d = np.linspace(ky_range[0], ky_range[1], n_ky)
    KX, KY = np.meshgrid(kx_1d, ky_1d, indexing='xy')
    kx_flat = KX.ravel()
    ky_flat = KY.ravel()

    k_mag = np.sqrt(kx_flat**2 + ky_flat**2)

    # Filter: keep only physically relevant wavenumbers
    mask = (k_mag >= k_min) & (k_mag <= k_max)
    kx_grid = kx_flat[mask]
    ky_grid = ky_flat[mask]
    k_mag = k_mag[mask]

    # Dispersion: ω = √(g·|k|·tanh(|k|·d))
    omega_grid = np.sqrt(G * k_mag * np.tanh(k_mag * depth))

    return kx_grid, ky_grid, omega_grid


def _ista_lasso(A, z, alpha, max_iter=500, tol=1e-6):
    """
    ISTA (Iterative Shrinkage-Thresholding Algorithm) for LASSO.

    Solves: min 0.5 ||z - A·c||² + alpha ||c||₁

    Faster than sklearn for small problems and avoids dependency.

    Args:
        A: (N, P) design matrix
        z: (N,) observations
        alpha: L1 regularization strength
        max_iter: maximum iterations
        tol: convergence tolerance

    Returns:
        c: (P,) solution
    """
    N, P = A.shape

    # Precompute
    AtA = A.T @ A
    Atz = A.T @ z

    # Step size: 1 / L where L = largest eigenvalue of A'A
    # Use power iteration estimate
    L = np.linalg.norm(AtA, ord=2)
    if L < 1e-12:
        return np.zeros(P)
    step = 1.0 / L

    c = np.zeros(P)
    for iteration in range(max_iter):
        # Gradient step
        grad = AtA @ c - Atz
        c_new = c - step * grad

        # Soft thresholding (proximal operator for L1)
        threshold = alpha * step
        c_new = np.sign(c_new) * np.maximum(np.abs(c_new) - threshold, 0.0)

        # Check convergence
        if np.max(np.abs(c_new - c)) < tol:
            c = c_new
            break
        c = c_new

    return c


def _extract_components(c, kx_grid, ky_grid, omega_grid, min_amp=1e-4):
    """
    Extract spectral components from LASSO solution.

    Args:
        c: (2K,) solution [c0_cos, c0_sin, c1_cos, c1_sin, ...]
        kx_grid, ky_grid, omega_grid: (K,) grid
        min_amp: minimum amplitude threshold

    Returns:
        kx, ky, w, phi: arrays of extracted components
    """
    K = len(kx_grid)
    c_cos = c[0::2]  # (K,)
    c_sin = c[1::2]  # (K,)

    amplitudes = np.sqrt(c_cos**2 + c_sin**2)
    phases = np.arctan2(-c_sin, c_cos)

    # Filter by amplitude
    mask = amplitudes > min_amp
    if not np.any(mask):
        # Return single strongest
        idx = np.argmax(amplitudes)
        return (np.array([kx_grid[idx]]),
                np.array([ky_grid[idx]]),
                np.array([max(amplitudes[idx], 0.01)]),
                np.array([phases[idx]]))

    return (kx_grid[mask], ky_grid[mask],
            amplitudes[mask], phases[mask])


def initialize_cs(frames, n_gaussians: int, depth: float,
                  kx_range: Tuple[float, float] = (-3.0, 3.0),
                  ky_range: Tuple[float, float] = (-3.0, 3.0),
                  n_kx: int = 31, n_ky: int = 31,
                  alpha: float = 0.001,
                  max_components: int = 100,
                  seed: int = None):
    """
    Compressed sensing spectral initialization.

    Args:
        frames: list of FrameData
        n_gaussians: target number of components (can return more or fewer)
        depth: water depth [m]
        kx_range, ky_range: wavenumber grid bounds [rad/m]
        n_kx, n_ky: grid resolution per axis
        alpha: LASSO regularization strength
        max_components: maximum components to return
        seed: random seed

    Returns:
        init_kx, init_ky, init_w, init_phi: tensors
    """
    rng = np.random.RandomState(seed) if seed is not None else np.random

    # Collect observations from all frames
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

    # Center observations
    z = z - z.mean()

    # Build wavenumber grid
    kx_grid, ky_grid, omega_grid = _build_wavenumber_grid(
        kx_range, ky_range, n_kx, n_ky, depth)
    K = len(kx_grid)

    if K == 0:
        return _random_init(n_gaussians, rng)

    # Build design matrix
    A = _build_design_matrix(x, y, t, kx_grid, ky_grid, omega_grid)

    # Normalize columns for better LASSO conditioning
    col_norms = np.sqrt(np.sum(A**2, axis=0))
    col_norms = np.maximum(col_norms, 1e-10)
    A_norm = A / col_norms[None, :]

    # Solve LASSO
    c_norm = _ista_lasso(A_norm, z, alpha=alpha, max_iter=1000)

    # Unnormalize
    c = c_norm / col_norms

    # Extract components
    kx_comp, ky_comp, w_comp, phi_comp = _extract_components(
        c, kx_grid, ky_grid, omega_grid, min_amp=1e-4)

    # Sort by amplitude (descending)
    order = np.argsort(w_comp)[::-1]
    kx_comp = kx_comp[order]
    ky_comp = ky_comp[order]
    w_comp = w_comp[order]
    phi_comp = phi_comp[order]

    # Limit to max_components
    n_found = min(len(kx_comp), max_components)
    kx_comp = kx_comp[:n_found]
    ky_comp = ky_comp[:n_found]
    w_comp = w_comp[:n_found]
    phi_comp = phi_comp[:n_found]

    # If fewer than n_gaussians, pad; if more, truncate
    if n_found < n_gaussians:
        extra = n_gaussians - n_found
        kx_comp = np.concatenate([kx_comp, rng.randn(extra) * 0.1])
        ky_comp = np.concatenate([ky_comp, rng.randn(extra) * 0.1])
        w_comp = np.concatenate([w_comp, np.ones(extra) * 0.01])
        phi_comp = np.concatenate([phi_comp, rng.uniform(0, 2*math.pi, extra)])
    elif n_found > n_gaussians:
        kx_comp = kx_comp[:n_gaussians]
        ky_comp = ky_comp[:n_gaussians]
        w_comp = w_comp[:n_gaussians]
        phi_comp = phi_comp[:n_gaussians]

    return (torch.tensor(kx_comp, dtype=torch.float32),
            torch.tensor(ky_comp, dtype=torch.float32),
            torch.tensor(w_comp, dtype=torch.float32),
            torch.tensor(phi_comp, dtype=torch.float32))


def _random_init(n, rng):
    """Fallback random initialization."""
    return (torch.tensor(rng.randn(n) * 0.05, dtype=torch.float32),
            torch.tensor(rng.randn(n) * 0.05, dtype=torch.float32),
            torch.ones(n, dtype=torch.float32) * 0.1,
            torch.tensor(rng.uniform(0, 2*math.pi, n), dtype=torch.float32))
