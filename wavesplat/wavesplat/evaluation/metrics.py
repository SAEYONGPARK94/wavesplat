"""
metrics.py
Evaluation metrics for wave field estimation.

Primary (5): RMSE_z, nRMSE, Hs_error, Tp_error, Spectral_R²
Secondary (6): RMSE_near/far, RMSE_unobserved, T_converge, T_inference,
               Directional_R², Dispersion_concentration

Reviewer-defense (3): Variance Ratio, Pearson correlation, Skill Score
    These metrics expose under-fitting (e.g., over-smoothed predictors)
    that nRMSE alone cannot distinguish from accurate estimation.
    A trivial zero predictor on a zero-mean wave field with std ≈ Hs/4
    achieves nRMSE ≈ 0.25, so methods reporting nRMSE in this range
    must be checked for actual reconstruction quality.

    NOTE: These metrics are defined as the "pre-registered metric panel"
    for the upcoming full-dataset re-evaluation. They MUST be applied
    uniformly to all methods (WaveSplat, ETKF, GPR, baselines) — not
    cherry-picked to favor any particular method.
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


def variance_ratio(pred: np.ndarray, gt: np.ndarray) -> float:
    """
    Variance ratio: Var(pred) / Var(gt).

    Reviewer-defense metric for exposing over-smoothing.
    - VR ≈ 1.0: amplitude preserved (good)
    - VR << 1: over-smoothed (under-fitting; e.g., GPR with too-large
      length scale, or any predictor that flattens to the mean)
    - VR > 1.2: over-fitting (tracking observation noise)

    Critical for distinguishing "low nRMSE due to accurate estimation"
    from "low nRMSE due to predicting near-zero".

    Args:
        pred, gt: arrays of equal shape (1D, 2D, or higher)

    Returns:
        Variance ratio (positive, dimensionless)
    """
    var_gt = float(np.var(gt))
    if var_gt < 1e-12:
        return float('nan')
    return float(np.var(pred) / var_gt)


def pearson_correlation(pred: np.ndarray, gt: np.ndarray) -> float:
    """
    Pearson correlation coefficient between predicted and GT fields.

    Measures shape match independent of amplitude scaling.
    - ρ → 1: perfect shape agreement
    - ρ → 0: no linear relationship (predictor captures no structure)
    - ρ < 0: inverted prediction (sign error)

    Complementary to nRMSE: ρ captures whether the prediction "looks like"
    the truth even if amplitude is wrong, while nRMSE captures absolute
    error. Together they distinguish four regimes:
    - High ρ + low nRMSE: accurate
    - High ρ + high nRMSE: right shape, wrong amplitude
    - Low ρ + low nRMSE: under-fitting (predicting near-mean)
    - Low ρ + high nRMSE: total failure

    Args:
        pred, gt: arrays of equal shape

    Returns:
        Pearson correlation in [-1, 1]
    """
    p = pred.flatten()
    g = gt.flatten()
    if len(p) < 2:
        return float('nan')

    p_c = p - np.mean(p)
    g_c = g - np.mean(g)
    denom = np.sqrt(np.sum(p_c ** 2) * np.sum(g_c ** 2))
    if denom < 1e-12:
        return float('nan')
    return float(np.sum(p_c * g_c) / denom)


def skill_score(pred: np.ndarray, gt: np.ndarray,
                reference: Optional[np.ndarray] = None) -> float:
    """
    Skill score relative to a trivial baseline.

    SS = 1 - MSE(pred) / MSE(reference)

    - SS = 1: perfect prediction
    - SS = 0: equivalent to reference (no skill)
    - SS < 0: worse than reference (negative skill)

    Default reference: zero predictor (constant zero output).
    For zero-mean ocean waves, a zero predictor achieves
    MSE = Var(z), so SS = 1 - MSE(pred)/Var(gt).

    This metric makes the "trivial baseline" explicit, preventing
    the misinterpretation of nRMSE values near the zero-baseline level
    (≈ 0.25 for typical wave fields) as meaningful accuracy.

    Args:
        pred, gt: arrays of equal shape
        reference: optional reference predictor (default: zero array)

    Returns:
        Skill score (1 = perfect, 0 = baseline, <0 = worse than baseline)
    """
    if reference is None:
        reference = np.zeros_like(gt)

    mse_pred = float(np.mean((pred - gt) ** 2))
    mse_ref = float(np.mean((reference - gt) ** 2))

    if mse_ref < 1e-12:
        return float('nan')
    return float(1.0 - mse_pred / mse_ref)


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

    # Reviewer-defense metrics — pre-registered panel for full re-evaluation
    metrics['Variance_Ratio'] = variance_ratio(pred_field, gt_field)
    metrics['Pearson_rho'] = pearson_correlation(pred_field, gt_field)
    metrics['Skill_Score'] = skill_score(pred_field, gt_field)

    return metrics
