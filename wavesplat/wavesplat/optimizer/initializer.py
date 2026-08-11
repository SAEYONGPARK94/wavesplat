"""
initializer.py
FFT-based initialization for spectral Gaussian parameters.

Two initialization strategies:
  v1 (legacy): Spatial-only 2D FFT — bins all frames into one grid.
  v2 (default): Temporal-first — uses time-series per spatial bin to find ω,
                 then dispersion for |k|, direction sweep for θ,
                 and linear projection for amplitude + phase.

v2 overcomes the spatial Nyquist limitation that prevents v1 from resolving
high-wavenumber components (e.g., ky_Nyq=0.38 vs actual ky=1.3-2.7 rad/m).
"""

import os
import torch
import numpy as np
import math
from typing import List, Tuple

from wavesplat.model.dispersion import wavenumber_from_omega, dispersion_omega


def _rx2_gate(name: str) -> bool:
    """Rx-2 (2026-08-05) RESEARCH gate. Unset or '0' == shipped behaviour, bit-for-bit.

    The environment is read at CALL time, not import time, because the Rx-2 driver reuses
    multiprocessing pool workers across cells; an import-time read would freeze whichever
    cell the worker happened to run first. Same contract as
    sequential_update._rx1_gate.

    Gates (all off by default, none reachable from any shipped config path):

    RX2_GROW_SUBTRACT=1
        At the two RESIDUAL placement sites (coarse_to_fine._beamform_residuals and
        sequential_update._adaptive_prune_grow), subtract each newly fitted component from
        the working residual before the next peak is refined/projected. The shipped code
        does NOT do this -- it refines and least-squares-projects every new component
        against the SAME untouched residual and only zeroes a box in the power map. Note
        initialize_from_frames_v2 already subtracts, so the deployment ladder [3,8,15] uses
        the subtracting greedy for 3 atoms and the non-subtracting greedy for the other 12.
    RX2_GROW_RECOMPUTE=1
        Implies RX2_GROW_SUBTRACT. Additionally recompute the coarse beamform power map
        from the updated residual after each component (previously suppressed cells stay
        zeroed), i.e. the 'recompute_residual' policy ported to the residual sites.
    RX2_GROW_TIGHT_SUPPRESS=1
        Use RECOMPUTE_MIN_DK instead of MIN_DK as the suppression radius at the residual
        sites. Only meaningful together with RX2_GROW_RECOMPUTE, where the box stops being
        load-bearing.
    RX2_GROW_FITPHI=1
        Seed newly grown components with the phase that _linear_projection already
        computed and the shipped code throws away (`amp, _ = _linear_projection(...)`),
        instead of a fresh U(0, 2pi) draw.
    """
    return os.environ.get(name, '0') == '1'

# Minimum separation enforced between beamform peaks [rad/m]. Hardcoded identically at three
# sites historically; hoisted here so it can be varied. fable's census
# (tmp/SCALE_AUDIT_fable_2026-07-23.md) found 126/195 shipped episodes -- and 100% of the
# 15/20/30-component ones -- contain true component pairs closer than this, i.e. the value
# imposes a resolution floor on exactly the crowded regime the information-wall argument
# relies on.
MIN_DK = 0.15

# How the greedy peak search decides where the *next* component goes.
#
#   'static_suppress' (shipped, default) -- the coarse power map is computed ONCE from the
#       original observations. After each component is fitted and subtracted from the
#       residual, the map is not recomputed; instead a MIN_DK-radius box around the chosen
#       peak is zeroed as a stand-in. This is why MIN_DK cannot simply be lowered: with a
#       small radius the search re-selects the same true component from a power map that
#       still contains it (measured, tmp/MIN_DK_RESULT_2026-07-23.md: at 0.05 the crowded
#       regime improves 5/5 but the mid-density regime degrades 0/5 via duplicate peaks).
#   'recompute_residual' -- recompute the coarse map on the current residual each iteration.
#       The subtraction then removes the peak on its own, so the suppression radius is no
#       longer load-bearing and can be small. This is the principled version of the
#       "adaptive radius" idea: rather than guessing a separation scale, stop relying on one.
#       Costs one extra coarse beamform per component.
PEAK_SELECT_MODE = 'static_suppress'

# Suppression radius used when PEAK_SELECT_MODE == 'recompute_residual'. Only needs to stop
# the search re-picking the identical grid cell, so it is deliberately near the grid step.
RECOMPUTE_MIN_DK = 0.02

# Amplitude given to padding (random-k, random-phase) components in
# initialize_from_frames_v2. Module-level so experiments can vary it without touching call
# sites. Default 0.01 preserves historical behaviour.
PAD_W = 0.01

# 'v2_konly' historically discards the (w, phi) that v2 fits by linear projection and
# re-draws phi uniformly at random. True preserves that behaviour; False keeps the fitted
# values, which removes the only stochastic input to the estimator.
KONLY_RANDOM_WPHI = True

# ---------------------------------------------------------------------------
# RESEARCH-ONLY (2026-08-05, experiment S2). NOT a deployable init_method.
# ---------------------------------------------------------------------------
# `init_method='oracle'` seeds (kx, ky) from the episode's TRUE wave components and leaves
# (w, phi) handled exactly as in 'v2_konly' (w = 0.01 uniform, phi ~ U(0, 2pi)), so that
# oracle-vs-v2_konly is a single-factor comparison of *where the atoms are placed*.
# The components are handed in through this module-level slot because
# `initialize_from_frames` has no access to the episode file. The research driver sets it
# immediately before each run and clears it afterwards. When it is None (always, in every
# shipped code path) the 'oracle' branch raises rather than silently degrading.
#
# Expected value: an (N_gt, >=4) array with column order [A, omega, kx, ky, ...], i.e. the
# episode's `config/wave_components` dataset verbatim.
ORACLE_COMPONENTS = None


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


def initialize_from_frames_v1(frames, n_gaussians: int, depth: float,
                              x_range: Tuple[float, float] = (10.0, 250.0),
                              y_range: Tuple[float, float] = (-100.0, 100.0),
                              grid_nx: int = 32, grid_ny: int = 24,
                              seed: int = None):
    """
    v1 (legacy): Spatial-only 2D FFT initialization.

    Accumulates all observations, bins them, applies 2D FFT,
    and returns initial (kx, ky, w, phi) tensors.
    Limited by spatial Nyquist — cannot resolve high-k components.
    """
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
        return (torch.randn(n_gaussians) * 0.05,
                torch.randn(n_gaussians) * 0.05,
                torch.ones(n_gaussians) * 0.1,
                torch.rand(n_gaussians) * 2 * math.pi)

    dx = (x_range[1] - x_range[0]) / grid_nx
    dy = (y_range[1] - y_range[0]) / grid_ny
    Z_grid = bin_observations_to_grid(all_x, all_y, all_z,
                                       x_range, y_range, grid_nx, grid_ny)
    kx_peaks, ky_peaks, amplitudes = fft_peak_extraction(
        Z_grid, dx, dy, n_peaks=n_gaussians
    )

    n_found = len(kx_peaks)
    if n_found < n_gaussians:
        extra = n_gaussians - n_found
        kx_peaks = np.concatenate([kx_peaks, rng.randn(extra) * 0.05])
        ky_peaks = np.concatenate([ky_peaks, rng.randn(extra) * 0.05])
        amplitudes = np.concatenate([amplitudes, np.ones(extra) * 0.01])

    init_kx = torch.tensor(kx_peaks[:n_gaussians], dtype=torch.float32)
    init_ky = torch.tensor(ky_peaks[:n_gaussians], dtype=torch.float32)
    init_w = torch.tensor(amplitudes[:n_gaussians], dtype=torch.float32)
    init_phi = torch.rand(n_gaussians) * 2 * math.pi

    return init_kx, init_ky, init_w, init_phi


# ---------------------------------------------------------------------------
# v2: Dispersion-aware beamforming initialization
# ---------------------------------------------------------------------------

G = 9.81


def _collect_observations(frames):
    """Collect all observations from frames into flat arrays."""
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
        return None, None, None, None

    return (np.concatenate(all_x), np.concatenate(all_y),
            np.concatenate(all_z), np.concatenate(all_t))


def _beamform_coarse(x, y, z, t, depth,
                     kx_range=(-4.0, 4.0), ky_range=(-4.0, 4.0),
                     n_kx=81, n_ky=81):
    """
    Coarse dispersion-aware beamforming over (kx, ky) grid.

    For each candidate (kx, ky), ω is computed via dispersion relation.
    Power = |<z, cos(phase)>|² + |<z, sin(phase)>|²  (periodogram)

    Args:
        x, y, z, t: (N,) observation arrays
        depth: water depth [m]
        kx_range, ky_range: search bounds [rad/m]
        n_kx, n_ky: grid resolution

    Returns:
        power: (n_ky, n_kx) beamform power map
        kx_axis: (n_kx,) kx values
        ky_axis: (n_ky,) ky values
    """
    kx_axis = np.linspace(kx_range[0], kx_range[1], n_kx)
    ky_axis = np.linspace(ky_range[0], ky_range[1], n_ky)

    z_centered = z - z.mean()
    n_obs = len(z_centered)
    power = np.zeros((n_ky, n_kx))

    # Process row by row (ky) for memory efficiency
    for iy, ky_val in enumerate(ky_axis):
        # (n_kx, n_obs)
        kx_arr = kx_axis[:, None]  # (n_kx, 1)
        k_mag = np.sqrt(kx_arr ** 2 + ky_val ** 2 + 1e-12)
        omega = np.sqrt(G * k_mag * np.tanh(k_mag * depth))

        phase = kx_arr * x[None, :] + ky_val * y[None, :] - omega * t[None, :]

        cos_corr = np.mean(z_centered[None, :] * np.cos(phase), axis=1)
        sin_corr = np.mean(z_centered[None, :] * np.sin(phase), axis=1)
        power[iy, :] = cos_corr ** 2 + sin_corr ** 2

    return power, kx_axis, ky_axis


def beamform_peak_ratio(frames, depth, kx_range=(-4.0, 4.0), ky_range=(-4.0, 4.0),
                        n_coarse=81, n_extract=30):
    """GT-free 'dominant-peak structure' scalar for the WavePrior UNIFIED routing guard.

    Runs the v2 coarse dispersion beamformer on the collected init-window observations, zeroes
    the DC block, and greedily extracts up to ``n_extract`` peaks with the same MIN_DK box
    suppression the v2 initializer uses. Returns ``power[2nd peak] / power[1st peak]`` (0.0 if
    fewer than two peaks are found). A single clean swell -> one dominant beamform peak -> ratio
    near zero; a crowded sea -> comparable secondary peaks -> larger ratio. Computed directly
    from observations, so it is independent of (and robust to) the diffusion proposal's
    over-proposal on OOD single-swell seas. ADDITIVE: only called by ``_auto_select_init``.
    """
    x, y, z, t = _collect_observations(frames)
    if x is None or len(x) < 10:
        return 0.0
    power, kx_axis, ky_axis = _beamform_coarse(
        x, y, z, t, depth, kx_range, ky_range, n_coarse, n_coarse)
    dkx = kx_axis[1] - kx_axis[0]
    dky = ky_axis[1] - ky_axis[0]
    cx = np.argmin(np.abs(kx_axis))
    cy = np.argmin(np.abs(ky_axis))
    power[max(0, cy - 1):cy + 2, max(0, cx - 1):cx + 2] = 0.0
    supx = int(MIN_DK / dkx) + 1
    supy = int(MIN_DK / dky) + 1
    peaks = []
    P = power.copy()
    for _ in range(n_extract):
        idx = int(np.argmax(P.ravel()))
        iy, ix = np.unravel_index(idx, P.shape)
        pk = float(P[iy, ix])
        if pk < 1e-12:
            break
        peaks.append(pk)
        yl, yh = max(0, iy - supy), min(P.shape[0], iy + supy + 1)
        xl, xh = max(0, ix - supx), min(P.shape[1], ix + supx + 1)
        P[yl:yh, xl:xh] = 0.0
    if len(peaks) < 2 or peaks[0] <= 0:
        return 0.0
    return float(peaks[1] / peaks[0])


def _beamform_refine(x, y, z, t, depth,
                     kx_center, ky_center,
                     delta=0.2, n_fine=41):
    """
    Fine refinement around a coarse peak.

    Returns:
        best_kx, best_ky: refined wavenumber
        best_power: power at refined peak
    """
    kx_fine = np.linspace(kx_center - delta, kx_center + delta, n_fine)
    ky_fine = np.linspace(ky_center - delta, ky_center + delta, n_fine)

    z_centered = z - z.mean()
    best_power = -1
    best_kx, best_ky = kx_center, ky_center

    for ky_val in ky_fine:
        kx_arr = kx_fine[:, None]
        k_mag = np.sqrt(kx_arr ** 2 + ky_val ** 2 + 1e-12)
        omega = np.sqrt(G * k_mag * np.tanh(k_mag * depth))

        phase = kx_arr * x[None, :] + ky_val * y[None, :] - omega * t[None, :]
        cos_corr = np.mean(z_centered[None, :] * np.cos(phase), axis=1)
        sin_corr = np.mean(z_centered[None, :] * np.sin(phase), axis=1)
        pw = cos_corr ** 2 + sin_corr ** 2

        max_idx = np.argmax(pw)
        if pw[max_idx] > best_power:
            best_power = pw[max_idx]
            best_kx = kx_fine[max_idx]
            best_ky = ky_val

    return best_kx, best_ky, best_power


def _linear_projection(x, y, z, t, kx, ky, depth):
    """
    Estimate amplitude and phase for a single component via linear projection.

    Model: z ≈ a*cos(kx*x + ky*y - ω*t) + b*sin(kx*x + ky*y - ω*t)
    Then: A = sqrt(a² + b²), φ = atan2(-b, a)

    Returns:
        amplitude, phase
    """
    k_mag = np.sqrt(kx ** 2 + ky ** 2 + 1e-12)
    omega = np.sqrt(G * k_mag * np.tanh(k_mag * depth))

    ph = kx * x + ky * y - omega * t
    c = np.cos(ph)
    s = np.sin(ph)

    cc = np.dot(c, c)
    ss = np.dot(s, s)
    cs = np.dot(c, s)
    cz = np.dot(c, z)
    sz = np.dot(s, z)

    det = cc * ss - cs * cs
    if abs(det) < 1e-12:
        return 0.01, 0.0

    a = (ss * cz - cs * sz) / det
    b = (cc * sz - cs * cz) / det

    amplitude = np.sqrt(a ** 2 + b ** 2)
    phase = np.arctan2(-b, a)

    return float(max(amplitude, 1e-4)), float(phase)


def initialize_from_frames_v2(frames, n_gaussians: int, depth: float,
                              x_range: Tuple[float, float] = (10.0, 250.0),
                              y_range: Tuple[float, float] = (-100.0, 100.0),
                              kx_range: Tuple[float, float] = (-4.0, 4.0),
                              ky_range: Tuple[float, float] = (-4.0, 4.0),
                              n_coarse: int = 81,
                              seed: int = None):
    """
    v2: Dispersion-aware beamforming initialization.

    Overcomes spatial Nyquist limitation by working directly on raw
    irregular observations without spatial binning. For each candidate
    (kx, ky), ω is determined by the dispersion relation, making the
    search dispersion-constrained.

    Algorithm:
    1. Coarse beamforming: sweep (kx, ky) grid, compute periodogram power
    2. Greedy peak extraction: suppress neighborhood, subtract from residual
    3. Fine refinement around each coarse peak
    4. Linear projection → amplitude (w) and phase (φ)

    Args:
        frames: list of FrameData from init window
        n_gaussians: number of components to initialize
        depth: water depth [m]
        x_range, y_range: spatial domain (unused, kept for API compat)
        kx_range, ky_range: beamforming search bounds [rad/m]
        n_coarse: coarse grid resolution per axis
        seed: random seed

    Returns:
        init_kx, init_ky, init_w, init_phi: (n_gaussians,) tensors
    """
    if seed is not None:
        rng = np.random.RandomState(seed)
    else:
        rng = np.random

    # Collect all observations
    x, y, z, t = _collect_observations(frames)
    if x is None or len(x) < 10:
        return (torch.randn(n_gaussians) * 0.05,
                torch.randn(n_gaussians) * 0.05,
                torch.ones(n_gaussians) * 0.1,
                torch.rand(n_gaussians) * 2 * math.pi)

    # Step 1: Single coarse beamforming on original data
    power, kx_axis, ky_axis = _beamform_coarse(
        x, y, z, t, depth, kx_range, ky_range, n_coarse, n_coarse
    )

    # Zero out DC neighborhood
    dkx = kx_axis[1] - kx_axis[0]
    dky = ky_axis[1] - ky_axis[0]
    cx = np.argmin(np.abs(kx_axis))
    cy = np.argmin(np.abs(ky_axis))
    power[max(0, cy - 1):cy + 2, max(0, cx - 1):cx + 2] = 0.0

    # Step 2: Greedy extraction — suppress + refine + subtract from residual
    z_residual = z.copy()
    kx_list, ky_list, w_list, phi_list = [], [], [], []
    recompute = (PEAK_SELECT_MODE == 'recompute_residual')
    min_dk = RECOMPUTE_MIN_DK if recompute else MIN_DK
    # Cells zeroed so far, kept so they stay zeroed across recomputations of the map.
    _suppressed = []

    for comp_idx in range(n_gaussians):
        # Find peak in power map
        flat_idx = np.argmax(power.ravel())
        iy_peak, ix_peak = np.unravel_index(flat_idx, power.shape)

        if power[iy_peak, ix_peak] < 1e-12:
            break

        kx_coarse = kx_axis[ix_peak]
        ky_coarse = ky_axis[iy_peak]

        # Step 3: Fine refinement on residual
        kx_fine, ky_fine, _ = _beamform_refine(
            x, y, z_residual, t, depth, kx_coarse, ky_coarse
        )

        # Step 4: Linear projection for amplitude and phase
        amp, phi = _linear_projection(x, y, z_residual, t, kx_fine, ky_fine, depth)

        kx_list.append(float(kx_fine))
        ky_list.append(float(ky_fine))
        w_list.append(amp)
        phi_list.append(phi)

        # Subtract fitted component from residual
        k_mag = np.sqrt(kx_fine ** 2 + ky_fine ** 2 + 1e-12)
        omega = np.sqrt(G * k_mag * np.tanh(k_mag * depth))
        ph = kx_fine * x + ky_fine * y - omega * t + phi
        z_residual = z_residual - amp * np.cos(ph)

        # Suppress neighborhood in power map. In the shipped mode this stands in for
        # recomputing the map; in 'recompute_residual' the map is rebuilt from the residual
        # below and this only stops the search re-picking the identical cell.
        suppress_kx = int(min_dk / dkx) + 1
        suppress_ky = int(min_dk / dky) + 1
        y_lo = max(0, iy_peak - suppress_ky)
        y_hi = min(power.shape[0], iy_peak + suppress_ky + 1)
        x_lo = max(0, ix_peak - suppress_kx)
        x_hi = min(power.shape[1], ix_peak + suppress_kx + 1)
        power[y_lo:y_hi, x_lo:x_hi] = 0.0
        _suppressed.append((y_lo, y_hi, x_lo, x_hi))

        if recompute and comp_idx + 1 < n_gaussians:
            power, kx_axis, ky_axis = _beamform_coarse(
                x, y, z_residual, t, depth, kx_range, ky_range, n_coarse, n_coarse)
            power[max(0, cy - 1):cy + 2, max(0, cx - 1):cx + 2] = 0.0
            for (a, b, c_, d) in _suppressed:
                power[a:b, c_:d] = 0.0

    # Pad if fewer peaks found.
    # PAD_W is module-level so it can be varied without touching call sites: padding
    # components carry RANDOM (kx, ky, phi), so a non-zero PAD_W injects active spurious
    # components whose survival is then decided by the energy prune -- a candidate cause of
    # the bimodal run-to-run variance measured in
    # tmp/ESTIMATOR_VARIANCE_RESULT_2026-07-23.md. PAD_W = 0 makes them inert at init while
    # still leaving them free to grow if the residual demands it (zero-init warm start).
    n_found = len(kx_list)
    if n_found < n_gaussians:
        extra = n_gaussians - n_found
        kx_list.extend(rng.randn(extra).tolist())
        ky_list.extend(rng.randn(extra).tolist())
        w_list.extend([PAD_W] * extra)
        phi_list.extend(rng.uniform(0, 2 * math.pi, extra).tolist())

    init_kx = torch.tensor(kx_list[:n_gaussians], dtype=torch.float32)
    init_ky = torch.tensor(ky_list[:n_gaussians], dtype=torch.float32)
    init_w = torch.tensor(w_list[:n_gaussians], dtype=torch.float32)
    init_phi = torch.tensor(phi_list[:n_gaussians], dtype=torch.float32)

    return init_kx, init_ky, init_w, init_phi


def initialize_oracle(frames, n_gaussians: int, depth: float, seed: int = None):
    """RESEARCH-ONLY: seed (kx, ky) from the TRUE components (module global
    ORACLE_COMPONENTS); (w, phi) exactly as 'v2_konly' does.

    Selection rule when n_gaussians != N_gt, FIXED IN ADVANCE
    (tmp/s1s2_2026-08-05/PREREG.md, section S2):
      * n_gaussians <  N_gt : the n_gaussians components with the LARGEST amplitude A
                              (equivalently largest energy A^2), i.e. the same top-N rule
                              wavesplat/baselines/oracle_baseline.py already uses.
      * n_gaussians >  N_gt : all N_gt true components, then padded with random (kx, ky)
                              drawn from the SAME RandomState(seed) and the same
                              `rng.randn()` call that pads initialize_from_frames_v2, so the
                              padding is identical in law to the control arm's padding.
      * n_gaussians == N_gt : all true components, order as stored.

    (w, phi) are NOT taken from ground truth: w = 0.01 uniform and phi ~ U(0, 2pi) via the
    global torch RNG, byte-for-byte the same two lines the 'v2_konly' branch runs. Only the
    wavenumber placement differs, which is the single factor S2 manipulates.
    """
    if ORACLE_COMPONENTS is None:
        raise RuntimeError(
            "init_method='oracle' requires initializer.ORACLE_COMPONENTS to be set to the "
            "episode's true wave_components array. This is a research-only path.")
    comp = np.asarray(ORACLE_COMPONENTS, dtype=np.float64)
    A = comp[:, 0]
    kx_true = comp[:, 2]
    ky_true = comp[:, 3]
    n_gt = len(A)

    rng = np.random.RandomState(seed) if seed is not None else np.random

    if n_gaussians <= n_gt:
        order = np.argsort(A)[::-1][:n_gaussians]
        kx_list = kx_true[order].tolist()
        ky_list = ky_true[order].tolist()
    else:
        kx_list = kx_true.tolist()
        ky_list = ky_true.tolist()
        extra = n_gaussians - n_gt
        kx_list.extend(rng.randn(extra).tolist())
        ky_list.extend(rng.randn(extra).tolist())

    init_kx = torch.tensor(kx_list[:n_gaussians], dtype=torch.float32)
    init_ky = torch.tensor(ky_list[:n_gaussians], dtype=torch.float32)
    init_w = torch.ones(n_gaussians) * 0.01
    init_phi = torch.rand(n_gaussians) * 2 * math.pi

    # (S4 2026-08-08) AMPLITUDE/PHASE ORACLE.  Opt-in via S4_ORACLE_WPHI=1; with the gate
    # off the two lines above stand and this function is byte-for-byte what it was.
    #
    # Why this exists: S3 measured that with oracle (kx, ky) AND the tracker off, the
    # crowded field still decorrelates 48% from truth (decorr 0.4829 = 99.1% of the total
    # error, shrink 0.0045).  Placement is exhausted and amplitude shrinkage is ~0, so the
    # residual lives in phase -- but every "oracle" arm run so far still had to FIND (w, phi)
    # itself, exactly as the docstring above says.  Seeding them from GT is the measurement
    # that separates "we cannot search for phase" from "the representation cannot fit".
    #
    # The model is z = sum w*cos(kx*x + ky*y - omega*t + phi) (spectral_gaussian.py:6) and
    # the GT is the same expression with the same sign convention, so A -> w and phi -> phi
    # transfer directly.  Padding beyond N_gt keeps the control arm's law (0.01 / U(0,2pi)).
    if os.environ.get('S4_ORACLE_WPHI') == '1':
        A_all = comp[:, 0]
        phi_all = comp[:, 4]
        if n_gaussians <= n_gt:
            w_src = A_all[order]
            phi_src = phi_all[order]
        else:
            w_src = np.concatenate([A_all, init_w.numpy()[n_gt:]])
            phi_src = np.concatenate([phi_all, init_phi.numpy()[n_gt:]])
        init_w = torch.tensor(np.asarray(w_src[:n_gaussians]), dtype=torch.float32)
        init_phi = torch.tensor(np.asarray(phi_src[:n_gaussians]), dtype=torch.float32)
    return init_kx, init_ky, init_w, init_phi


def initialize_from_frames(frames, n_gaussians: int, depth: float,
                           x_range: Tuple[float, float] = (10.0, 250.0),
                           y_range: Tuple[float, float] = (-100.0, 100.0),
                           grid_nx: int = 32, grid_ny: int = 24,
                           seed: int = None, init_method: str = 'v2'):
    """
    Unified entry point for initialization.

    Args:
        init_method:
            'v1' — spatial-only 2D FFT (legacy)
            'v2' — beamforming with full linear projection (kx,ky,w,phi)
            'v2_konly' — beamforming for kx/ky only, random w/phi (default)
        Other args: see initialize_from_frames_v1 / v2

    Returns:
        init_kx, init_ky, init_w, init_phi: (n_gaussians,) tensors
    """
    if init_method == 'v1':
        return initialize_from_frames_v1(
            frames, n_gaussians, depth, x_range, y_range,
            grid_nx, grid_ny, seed
        )
    elif init_method == 'v2':
        return initialize_from_frames_v2(
            frames, n_gaussians, depth, x_range, y_range,
            seed=seed
        )
    elif init_method == 'oracle':
        # RESEARCH-ONLY (experiment S2). See initialize_oracle.
        return initialize_oracle(frames, n_gaussians, depth, seed=seed)
    elif init_method == 'relax':
        from wavesplat.optimizer.relax_estimator import initialize_relax
        return initialize_relax(frames, n_gaussians, depth, seed=seed)
    elif init_method == 'relax_konly':
        from wavesplat.optimizer.relax_estimator import initialize_relax
        kx, ky, w, phi = initialize_relax(frames, n_gaussians, depth, seed=seed)
        init_w = torch.ones(n_gaussians) * 0.01
        init_phi = torch.rand(n_gaussians) * 2 * math.pi
        return kx, ky, init_w, init_phi
    else:  # 'v2_konly'
        kx, ky, w, phi = initialize_from_frames_v2(
            frames, n_gaussians, depth, x_range, y_range,
            seed=seed
        )
        if KONLY_RANDOM_WPHI:
            # Keep beamforming kx/ky but randomize w/phi
            # so optimizer explores the amplitude/phase landscape freely.
            # NOTE: this random phase draw is the measured source of the bimodal
            # run-to-run variance (CV up to 53% on a fixed episode) documented in
            # tmp/ESTIMATOR_VARIANCE_RESULT_2026-07-23.md -- v2 already computed a
            # data-derived (w, phi) by linear projection, and this discards it.
            init_w = torch.ones(n_gaussians) * 0.01
            init_phi = torch.rand(n_gaussians) * 2 * math.pi
            return kx, ky, init_w, init_phi
        # Deterministic alternative: keep the least-squares (w, phi) v2 already fitted.
        return kx, ky, w, phi
