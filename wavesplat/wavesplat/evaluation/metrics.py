"""
metrics.py
Evaluation metrics for wave field estimation.

Primary (5): RMSE_z, nRMSE, Hs_error, Tp_error, Spectral_R²
Secondary (6): RMSE_near/far, RMSE_unobserved, T_converge, T_inference,
               Directional_R², Dispersion_concentration
"""

import numpy as np
from typing import Dict, Optional, Tuple
from scipy import signal


def rmse(pred: np.ndarray, gt: np.ndarray) -> float:
    """Root mean square error."""
    return float(np.sqrt(np.mean((pred - gt) ** 2)))


def nrmse(pred: np.ndarray, gt: np.ndarray, Hs: float) -> float:
    """Normalized RMSE: RMSE / Hs."""
    return rmse(pred, gt) / max(Hs, 1e-6)


def hs_error(pred_field: np.ndarray, Hs_true: float) -> float:
    """Relative Hs error: |Hs_pred - Hs_true| / Hs_true."""
    Hs_pred = 4.0 * np.std(pred_field)
    return abs(Hs_pred - Hs_true) / max(Hs_true, 1e-6)


def tp_error(pred_timeseries: np.ndarray, dt: float, Tp_true: float) -> float:
    """Peak period error: |Tp_pred - Tp_true| [s]."""
    if len(pred_timeseries) < 10:
        return float('nan')

    freqs, psd = signal.welch(pred_timeseries, fs=1.0/dt, nperseg=min(256, len(pred_timeseries)))
    if len(psd) == 0 or np.max(psd) == 0:
        return float('nan')

    peak_idx = np.argmax(psd[1:]) + 1  # skip DC
    Tp_pred = 1.0 / max(freqs[peak_idx], 1e-6)
    return abs(Tp_pred - Tp_true)


def spectral_r2(pred_spectrum: np.ndarray, gt_spectrum: np.ndarray) -> float:
    """R² between predicted and GT 1D spectra."""
    if len(pred_spectrum) != len(gt_spectrum):
        min_len = min(len(pred_spectrum), len(gt_spectrum))
        pred_spectrum = pred_spectrum[:min_len]
        gt_spectrum = gt_spectrum[:min_len]

    ss_res = np.sum((pred_spectrum - gt_spectrum) ** 2)
    ss_tot = np.sum((gt_spectrum - np.mean(gt_spectrum)) ** 2)
    if ss_tot < 1e-12:
        return 1.0 if ss_res < 1e-12 else 0.0
    return float(1.0 - ss_res / ss_tot)


def rmse_spatial_split(pred: np.ndarray, gt: np.ndarray,
                       grid_x: np.ndarray,
                       split_distance: float = 80.0) -> Tuple[float, float]:
    """
    Split RMSE into near (<split) and far (>=split) regions.

    Args:
        pred, gt: (Ny, Nx) fields
        grid_x: (Nx,) x-coordinates
        split_distance: split threshold [m]

    Returns:
        (rmse_near, rmse_far)
    """
    near_mask = grid_x < split_distance
    far_mask = grid_x >= split_distance

    near_pred = pred[:, near_mask]
    near_gt = gt[:, near_mask]
    far_pred = pred[:, far_mask]
    far_gt = gt[:, far_mask]

    rmse_near = rmse(near_pred, near_gt) if near_pred.size > 0 else float('nan')
    rmse_far = rmse(far_pred, far_gt) if far_pred.size > 0 else float('nan')

    return rmse_near, rmse_far


def rmse_unobserved(pred: np.ndarray, gt: np.ndarray,
                    grid_x: np.ndarray, grid_y: np.ndarray,
                    obs_x: np.ndarray, obs_y: np.ndarray,
                    min_dist: float = 10.0) -> float:
    """
    RMSE only at grid points far from any observation.
    This is the key evidence for physics prior contribution.

    Args:
        pred, gt: (Ny, Nx) fields
        grid_x: (Nx,), grid_y: (Ny,)
        obs_x, obs_y: (M,) observation locations
        min_dist: minimum distance from any observation [m]
    """
    Xg, Yg = np.meshgrid(grid_x, grid_y)

    # Distance from each grid point to nearest observation
    if len(obs_x) == 0:
        return rmse(pred, gt)

    min_dists = np.full(Xg.shape, np.inf)
    # Batch computation to avoid memory explosion
    batch = 1000
    for i in range(0, len(obs_x), batch):
        ox = obs_x[i:i+batch]
        oy = obs_y[i:i+batch]
        for j in range(len(ox)):
            d = np.sqrt((Xg - ox[j]) ** 2 + (Yg - oy[j]) ** 2)
            min_dists = np.minimum(min_dists, d)

    unobs_mask = min_dists >= min_dist
    if not np.any(unobs_mask):
        return float('nan')

    return rmse(pred[unobs_mask], gt[unobs_mask])


def directional_r2(pred_field: np.ndarray, gt_field: np.ndarray) -> float:
    """
    Directional R² from 2D wavenumber spectra.

    Compares the 2D power spectrum (kx, ky) of predicted vs GT fields.
    Measures how well directional wave energy distribution is captured.

    Args:
        pred_field, gt_field: (Ny, Nx) spatial fields

    Returns:
        R² between 2D power spectra (excluding DC)
    """
    # 2D FFT → power spectrum
    F_pred = np.abs(np.fft.fft2(pred_field)) ** 2
    F_gt = np.abs(np.fft.fft2(gt_field)) ** 2

    # Zero DC component
    F_pred[0, 0] = 0.0
    F_gt[0, 0] = 0.0

    # Flatten and compute R²
    ss_res = np.sum((F_pred - F_gt) ** 2)
    ss_tot = np.sum((F_gt - np.mean(F_gt)) ** 2)

    if ss_tot < 1e-12:
        return 1.0 if ss_res < 1e-12 else 0.0
    return float(1.0 - ss_res / ss_tot)


def dispersion_concentration(pred_kx: np.ndarray, pred_ky: np.ndarray,
                              pred_omega: np.ndarray,
                              depth: float = 50.0,
                              g: float = 9.81) -> float:
    """
    Dispersion concentration: how well estimated (k, ω) pairs
    satisfy the dispersion relation ω² = g|k|tanh(|k|d).

    Returns mean relative error |ω_est - ω_disp| / ω_disp.
    Lower = better. 0.0 means perfect dispersion compliance.
    Used only for ablation E2 (dispersion constraint removed).

    Args:
        pred_kx, pred_ky: (N,) estimated wavenumber components
        pred_omega: (N,) estimated angular frequencies
        depth: water depth [m]

    Returns:
        Mean relative dispersion error (0 = perfect)
    """
    k_mag = np.sqrt(pred_kx ** 2 + pred_ky ** 2 + 1e-12)
    omega_disp = np.sqrt(g * k_mag * np.tanh(k_mag * depth) + 1e-12)

    rel_error = np.abs(pred_omega - omega_disp) / (omega_disp + 1e-12)
    return float(np.mean(rel_error))


def compute_all_metrics(pred_field: np.ndarray,
                        gt_field: np.ndarray,
                        grid_x: np.ndarray,
                        grid_y: np.ndarray,
                        Hs_true: float,
                        Tp_true: float = None,
                        dt: float = None,
                        pred_timeseries: np.ndarray = None,
                        pred_spectrum: np.ndarray = None,
                        gt_spectrum: np.ndarray = None,
                        obs_x: np.ndarray = None,
                        obs_y: np.ndarray = None) -> Dict[str, float]:
    """
    Compute all applicable metrics for a single frame or averaged field.

    Returns dict with metric names as keys.
    """
    metrics = {}

    # Primary
    metrics['RMSE_z'] = rmse(pred_field, gt_field)
    metrics['nRMSE'] = nrmse(pred_field, gt_field, Hs_true)
    metrics['Hs_error'] = hs_error(pred_field, Hs_true)

    if pred_timeseries is not None and dt is not None and Tp_true is not None:
        metrics['Tp_error'] = tp_error(pred_timeseries, dt, Tp_true)

    if pred_spectrum is not None and gt_spectrum is not None:
        metrics['Spectral_R2'] = spectral_r2(pred_spectrum, gt_spectrum)

    # Secondary — spatial
    rmse_n, rmse_f = rmse_spatial_split(pred_field, gt_field, grid_x)
    metrics['RMSE_near'] = rmse_n
    metrics['RMSE_far'] = rmse_f

    if obs_x is not None and obs_y is not None:
        metrics['RMSE_unobserved'] = rmse_unobserved(
            pred_field, gt_field, grid_x, grid_y, obs_x, obs_y
        )

    # Directional R² (2D spectral comparison)
    metrics['Directional_R2'] = directional_r2(pred_field, gt_field)

    return metrics
