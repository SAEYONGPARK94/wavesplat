"""
initializer.py
FFT-based initialization for spectral Gaussian parameters.

Seeds initial (kx, ky, w, phi) from observed LiDAR data by:
1. Binning irregular observations onto a coarse regular grid
2. Applying 2D FFT to identify dominant wavenumber peaks
3. Extracting top-N peaks as initial Gaussian parameters
"""

import torch
import numpy as np
import math
from typing import List, Tuple

from wavesplat.model.dispersion import wavenumber_from_omega


def bin_observations_to_grid(x: np.ndarray, y: np.ndarray, z: np.ndarray,
                              x_range: Tuple[float, float],
                              y_range: Tuple[float, float],
                              nx: int = 32, ny: int = 24) -> np.ndarray:
    """
    Bin irregular observations onto a regular grid via averaging.

    Args:
        x, y, z: (M,) observation coordinates and values
        x_range, y_range: grid bounds
        nx, ny: grid resolution

    Returns:
        Z_grid: (ny, nx) binned values (NaN where no data)
    """
    x_edges = np.linspace(x_range[0], x_range[1], nx + 1)
    y_edges = np.linspace(y_range[0], y_range[1], ny + 1)

    Z_sum = np.zeros((ny, nx))
    Z_count = np.zeros((ny, nx))

    ix = np.clip(np.digitize(x, x_edges) - 1, 0, nx - 1)
    iy = np.clip(np.digitize(y, y_edges) - 1, 0, ny - 1)

    for i in range(len(x)):
        Z_sum[iy[i], ix[i]] += z[i]
        Z_count[iy[i], ix[i]] += 1

    Z_grid = np.full((ny, nx), np.nan)
    mask = Z_count > 0
    Z_grid[mask] = Z_sum[mask] / Z_count[mask]

    return Z_grid


def fft_peak_extraction(Z_grid: np.ndarray,
                        dx: float, dy: float,
                        n_peaks: int = 15) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract dominant wavenumber peaks from 2D FFT of gridded data.

    Args:
        Z_grid: (ny, nx) gridded wave field (NaN filled with 0)
        dx, dy: grid spacing [m]
        n_peaks: number of peaks to extract

    Returns:
        kx_peaks: (n_peaks,) x-wavenumber [rad/m]
        ky_peaks: (n_peaks,) y-wavenumber [rad/m]
        amplitudes: (n_peaks,) estimated amplitudes [m]
    """
    # Replace NaN with 0 and apply Hann window
    Z = np.nan_to_num(Z_grid, nan=0.0)
    ny, nx = Z.shape

    win_x = np.hanning(nx)
    win_y = np.hanning(ny)
    window = win_y[:, None] * win_x[None, :]
    Z_windowed = Z * window

    # 2D FFT
    F = np.fft.fft2(Z_windowed)
    F_shifted = np.fft.fftshift(F)
    power = np.abs(F_shifted) ** 2

    # Wavenumber axes
    kx_axis = 2 * np.pi * np.fft.fftshift(np.fft.fftfreq(nx, d=dx))
    ky_axis = 2 * np.pi * np.fft.fftshift(np.fft.fftfreq(ny, d=dy))

    # Zero out DC component
    cy, cx = ny // 2, nx // 2
    power[cy, cx] = 0.0

    # Find top peaks
    flat_idx = np.argsort(power.ravel())[::-1]
    kx_peaks = []
    ky_peaks = []
    amplitudes = []

    for idx in flat_idx:
        iy, ix = np.unravel_index(idx, power.shape)
        kx_val = kx_axis[ix]
        ky_val = ky_axis[iy]

        # Skip if too close to existing peak (merge protection)
        too_close = False
        for kx_p, ky_p in zip(kx_peaks, ky_peaks):
            if abs(kx_val - kx_p) < 0.02 and abs(ky_val - ky_p) < 0.02:
                too_close = True
                break
        if too_close:
            continue

        amp = 2.0 * np.sqrt(power[iy, ix]) / (nx * ny)
        kx_peaks.append(kx_val)
        ky_peaks.append(ky_val)
        amplitudes.append(amp)

        if len(kx_peaks) >= n_peaks:
            break

    return (np.array(kx_peaks), np.array(ky_peaks), np.array(amplitudes))


def initialize_from_frames(frames, n_gaussians: int, depth: float,
                           x_range: Tuple[float, float] = (10.0, 250.0),
                           y_range: Tuple[float, float] = (-100.0, 100.0),
                           grid_nx: int = 32, grid_ny: int = 24,
                           seed: int = None):
    """
    Initialize spectral Gaussian parameters from a window of frames.

    Accumulates all observations, bins them, applies 2D FFT,
    and returns initial (kx, ky, w, phi) tensors.

    Args:
        frames: list of FrameData objects
        n_gaussians: number of components to initialize
        depth: water depth [m]
        x_range, y_range: spatial domain
        grid_nx, grid_ny: binning resolution

    Returns:
        init_kx, init_ky, init_w, init_phi: (n_gaussians,) tensors
    """
    # Accumulate all observations
    all_x, all_y, all_z = [], [], []
    for f in frames:
        all_x.append(f.x_obs.cpu().numpy())
        all_y.append(f.y_obs.cpu().numpy())
        all_z.append(f.z_obs.cpu().numpy())

    all_x = np.concatenate(all_x)
    all_y = np.concatenate(all_y)
    all_z = np.concatenate(all_z)

    if seed is not None:
        rng = np.random.RandomState(seed)
    else:
        rng = np.random

    if len(all_x) == 0:
        # Fallback: random initialization
        return (torch.randn(n_gaussians) * 0.05,
                torch.randn(n_gaussians) * 0.05,
                torch.ones(n_gaussians) * 0.1,
                torch.rand(n_gaussians) * 2 * math.pi)

    # Bin and FFT
    dx = (x_range[1] - x_range[0]) / grid_nx
    dy = (y_range[1] - y_range[0]) / grid_ny
    Z_grid = bin_observations_to_grid(all_x, all_y, all_z,
                                       x_range, y_range, grid_nx, grid_ny)
    kx_peaks, ky_peaks, amplitudes = fft_peak_extraction(
        Z_grid, dx, dy, n_peaks=n_gaussians
    )

    # Pad if fewer peaks found than requested
    n_found = len(kx_peaks)
    if n_found < n_gaussians:
        extra = n_gaussians - n_found
        kx_peaks = np.concatenate([kx_peaks, rng.randn(extra) * 0.05])
        ky_peaks = np.concatenate([ky_peaks, rng.randn(extra) * 0.05])
        amplitudes = np.concatenate([amplitudes, np.ones(extra) * 0.01])

    init_kx = torch.tensor(kx_peaks[:n_gaussians], dtype=torch.float32)
    init_ky = torch.tensor(ky_peaks[:n_gaussians], dtype=torch.float32)
    init_w = torch.tensor(amplitudes[:n_gaussians], dtype=torch.float32)
    init_phi = torch.rand(n_gaussians) * 2 * math.pi  # random phase init

    return init_kx, init_ky, init_w, init_phi
