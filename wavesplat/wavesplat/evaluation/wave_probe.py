"""
wave_probe.py
Wave probe time-series comparison for tank experiment validation.

Compares WaveSplat prediction at the probe location against
single-point ultrasonic wave probe measurements.

Metrics:
  - nRMSE at probe location
  - Pearson correlation
  - Spectrum comparison (PSD)
  - Phase lag (cross-correlation)
"""

import numpy as np
from typing import Dict, Optional, Tuple
from scipy import signal


def load_wave_probe_csv(filepath: str,
                         time_col: int = 0,
                         eta_col: int = 1,
                         skip_header: int = 1,
                         delimiter: str = ',') -> Tuple[np.ndarray, np.ndarray]:
    """
    Load wave probe time series from CSV.

    Expected format: time [s], elevation [m]

    Args:
        filepath: CSV file path
        time_col: column index for time
        eta_col: column index for surface elevation
        skip_header: number of header lines to skip
        delimiter: CSV delimiter

    Returns:
        t: (N,) timestamps [s]
        eta: (N,) surface elevation [m]
    """
    data = np.genfromtxt(filepath, delimiter=delimiter,
                          skip_header=skip_header)
    t = data[:, time_col]
    eta = data[:, eta_col]

    # Remove NaN
    valid = ~(np.isnan(t) | np.isnan(eta))
    return t[valid], eta[valid]


def extract_prediction_at_probe(pred_fields: np.ndarray,
                                  timestamps: np.ndarray,
                                  grid_x: np.ndarray,
                                  grid_y: np.ndarray,
                                  probe_x: float,
                                  probe_y: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract predicted time series at the wave probe location.

    Uses bilinear interpolation on the predicted grid field.

    Args:
        pred_fields: (N_frames, Ny, Nx) predicted wave fields
        timestamps: (N_frames,) frame timestamps
        grid_x: (Nx,) x-coordinates
        grid_y: (Ny,) y-coordinates
        probe_x, probe_y: probe location [m]

    Returns:
        t: (N_frames,) timestamps
        eta_pred: (N_frames,) predicted elevation at probe
    """
    # Find nearest grid indices for bilinear interpolation
    ix = np.searchsorted(grid_x, probe_x) - 1
    iy = np.searchsorted(grid_y, probe_y) - 1

    ix = np.clip(ix, 0, len(grid_x) - 2)
    iy = np.clip(iy, 0, len(grid_y) - 2)

    # Bilinear weights
    dx = grid_x[ix + 1] - grid_x[ix]
    dy = grid_y[iy + 1] - grid_y[iy]
    wx = (probe_x - grid_x[ix]) / max(dx, 1e-12)
    wy = (probe_y - grid_y[iy]) / max(dy, 1e-12)
    wx = np.clip(wx, 0, 1)
    wy = np.clip(wy, 0, 1)

    # Interpolate for each frame
    eta = (
        (1 - wx) * (1 - wy) * pred_fields[:, iy, ix] +
        wx * (1 - wy) * pred_fields[:, iy, ix + 1] +
        (1 - wx) * wy * pred_fields[:, iy + 1, ix] +
        wx * wy * pred_fields[:, iy + 1, ix + 1]
    )

    return timestamps, eta


def synchronize_timeseries(t_pred: np.ndarray, eta_pred: np.ndarray,
                            t_probe: np.ndarray, eta_probe: np.ndarray
                            ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Synchronize prediction and probe time series to common timestamps.

    Interpolates probe data to prediction timestamps.

    Returns:
        t_common: (N,) common timestamps
        eta_pred_sync: (N,) prediction at common times
        eta_probe_sync: (N,) probe at common times
    """
    # Find overlapping time range
    t_start = max(t_pred[0], t_probe[0])
    t_end = min(t_pred[-1], t_probe[-1])

    mask_pred = (t_pred >= t_start) & (t_pred <= t_end)
    t_common = t_pred[mask_pred]
    eta_pred_sync = eta_pred[mask_pred]

    # Interpolate probe to common timestamps
    eta_probe_sync = np.interp(t_common, t_probe, eta_probe)

    return t_common, eta_pred_sync, eta_probe_sync


def compute_probe_metrics(eta_pred: np.ndarray,
                           eta_probe: np.ndarray,
                           dt: float,
                           Hs_nominal: float = None) -> Dict[str, float]:
    """
    Compute comparison metrics between prediction and wave probe.

    Args:
        eta_pred: (N,) predicted elevation [m]
        eta_probe: (N,) measured elevation [m]
        dt: sampling interval [s]
        Hs_nominal: nominal Hs for normalization [m]

    Returns:
        Dictionary of metrics
    """
    metrics = {}

    # Basic statistics
    rmse = float(np.sqrt(np.mean((eta_pred - eta_probe) ** 2)))
    metrics['RMSE'] = rmse

    if Hs_nominal and Hs_nominal > 0:
        metrics['nRMSE'] = rmse / Hs_nominal
    else:
        Hs_meas = 4.0 * np.std(eta_probe)
        metrics['nRMSE'] = rmse / max(Hs_meas, 1e-6)

    # Pearson correlation
    p = eta_pred - np.mean(eta_pred)
    g = eta_probe - np.mean(eta_probe)
    denom = np.sqrt(np.sum(p**2) * np.sum(g**2))
    metrics['Pearson_rho'] = float(np.sum(p * g) / max(denom, 1e-12))

    # Variance ratio
    var_probe = float(np.var(eta_probe))
    metrics['Variance_Ratio'] = float(np.var(eta_pred) / max(var_probe, 1e-12))

    # Hs comparison
    Hs_pred = 4.0 * np.std(eta_pred)
    Hs_meas = 4.0 * np.std(eta_probe)
    metrics['Hs_pred'] = Hs_pred
    metrics['Hs_meas'] = Hs_meas
    metrics['Hs_rel_error'] = abs(Hs_pred - Hs_meas) / max(Hs_meas, 1e-6)

    # Spectrum comparison
    nperseg = min(256, len(eta_pred) // 2)
    if nperseg >= 16:
        f_pred, psd_pred = signal.welch(eta_pred, fs=1.0/dt, nperseg=nperseg)
        f_probe, psd_probe = signal.welch(eta_probe, fs=1.0/dt, nperseg=nperseg)

        # Tp comparison
        peak_idx_pred = np.argmax(psd_pred[1:]) + 1
        peak_idx_probe = np.argmax(psd_probe[1:]) + 1
        Tp_pred = 1.0 / max(f_pred[peak_idx_pred], 1e-6)
        Tp_meas = 1.0 / max(f_probe[peak_idx_probe], 1e-6)
        metrics['Tp_pred'] = Tp_pred
        metrics['Tp_meas'] = Tp_meas
        metrics['Tp_error'] = abs(Tp_pred - Tp_meas)

        # Spectral R²
        ss_res = np.sum((psd_pred - psd_probe) ** 2)
        ss_tot = np.sum((psd_probe - np.mean(psd_probe)) ** 2)
        metrics['Spectral_R2'] = float(1.0 - ss_res / max(ss_tot, 1e-12))

    # Phase lag via cross-correlation
    corr = np.correlate(eta_pred - np.mean(eta_pred),
                         eta_probe - np.mean(eta_probe), mode='full')
    lags = np.arange(-len(eta_pred) + 1, len(eta_pred))
    peak_lag = lags[np.argmax(corr)]
    metrics['phase_lag_samples'] = int(peak_lag)
    metrics['phase_lag_seconds'] = float(peak_lag * dt)

    return metrics


def compare_spectra(eta_pred: np.ndarray,
                     eta_probe: np.ndarray,
                     dt: float,
                     nperseg: int = 256
                     ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute PSD for both signals for plotting.

    Returns:
        freq: (N_freq,) frequencies [Hz]
        psd_pred: (N_freq,) predicted PSD
        psd_probe: (N_freq,) measured PSD
    """
    nperseg = min(nperseg, len(eta_pred) // 2)
    freq, psd_pred = signal.welch(eta_pred, fs=1.0/dt, nperseg=nperseg)
    _, psd_probe = signal.welch(eta_probe, fs=1.0/dt, nperseg=nperseg)
    return freq, psd_pred, psd_probe
