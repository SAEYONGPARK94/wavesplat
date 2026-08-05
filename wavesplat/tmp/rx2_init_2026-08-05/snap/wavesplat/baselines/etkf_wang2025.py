"""
etkf_wang2025.py
ETKF baseline faithful to Wang et al. 2025 (Coastal Eng 197, 104690).

"From points to waves: Fast ocean wave spatial-temporal fields estimation
 using ensemble transform Kalman filter with optical measurement"

Key differences from our original etkf_grid.py:
  1. Interpolated measurements: sparse 3D points -> Delaunay interp -> grid (H=I)
  2. Fuzzy logic: adaptive R based on innovation + innovation difference
  3. Ensemble size: 300 (optimal per Table 2)
  4. Sub-state localization: state split into L sub-states for parallel update

Reference: https://github.com/fengHEU/Wave-ETKF
"""

import numpy as np
import time
from scipy.interpolate import griddata


class ETKFWang2025:
    """
    ETKF with fuzzy-logic outlier handling, following Wang et al. 2025.

    State: flattened elevation grid (Ny*Nx x 1).
    Propagation: FFT-based spectral (linear dispersion).
    Measurement: interpolated sparse points onto grid -> H = identity.
    """

    def __init__(self, grid_x: np.ndarray, grid_y: np.ndarray,
                 depth: float = 50.0,
                 n_ensemble: int = 300,
                 dt: float = 0.1,
                 obs_noise: float = 0.05,
                 n_substates: int = 4,
                 use_fuzzy: bool = True):
        self.grid_x = grid_x
        self.grid_y = grid_y
        self.nx = len(grid_x)
        self.ny = len(grid_y)
        self.n_grid = self.nx * self.ny
        self.depth = depth
        self.n_ensemble = n_ensemble
        self.dt = dt
        self.obs_noise = obs_noise
        self.n_substates = n_substates
        self.use_fuzzy = use_fuzzy
        self.g = 9.81
        self.max_eta = 5.0

        # Grid spacings
        self.dx = grid_x[1] - grid_x[0] if self.nx > 1 else 1.0
        self.dy = grid_y[1] - grid_y[0] if self.ny > 1 else 1.0

        # Meshgrid for interpolation targets
        self._gx, self._gy = np.meshgrid(grid_x, grid_y)
        self._grid_pts = np.column_stack([self._gx.ravel(), self._gy.ravel()])

        # Precompute spectral propagation (same as etkf_grid.py)
        kx = 2 * np.pi * np.fft.fftfreq(self.nx, d=self.dx)
        ky = 2 * np.pi * np.fft.fftfreq(self.ny, d=self.dy)
        KX, KY = np.meshgrid(kx[:self.nx // 2 + 1], ky)
        K = np.sqrt(KX ** 2 + KY ** 2 + 1e-12)
        omega = np.sqrt(self.g * K * np.tanh(K * self.depth))

        self._cos_wdt = np.cos(omega * dt)
        self._sin_wdt = np.sin(omega * dt)
        self._sinc_wdt = np.where(omega > 1e-10,
                                   np.sin(omega * dt) / omega, dt)
        self._omega = omega
        self._damp = np.exp(-0.002 * K ** 2 * dt)

        # State: elevation only (Wang 2025 uses elevation as state)
        # But we keep eta + deta for spectral propagation
        self.n_state = self.n_grid * 2
        self.ensemble = np.random.randn(n_ensemble, self.n_state) * 0.05

        # Previous innovation for fuzzy logic
        self._prev_innovation = np.zeros(self.n_grid)

    def _forecast_all(self):
        """Batch spectral propagation for all ensemble members at once."""
        Ne = self.n_ensemble
        eta_all = self.ensemble[:, :self.n_grid].reshape(Ne, self.ny, self.nx)
        deta_all = self.ensemble[:, self.n_grid:].reshape(Ne, self.ny, self.nx)

        eta_hat = np.fft.rfft2(eta_all, axes=(-2, -1))
        deta_hat = np.fft.rfft2(deta_all, axes=(-2, -1))

        eta_hat_new = eta_hat * self._cos_wdt[None] + deta_hat * self._sinc_wdt[None]
        deta_hat_new = (-eta_hat * self._omega[None] * self._sin_wdt[None]
                        + deta_hat * self._cos_wdt[None])

        eta_hat_new *= self._damp[None]
        deta_hat_new *= self._damp[None]

        self.ensemble[:, :self.n_grid] = np.fft.irfft2(
            eta_hat_new, s=(self.ny, self.nx), axes=(-2, -1)).reshape(Ne, -1)
        self.ensemble[:, self.n_grid:] = np.fft.irfft2(
            deta_hat_new, s=(self.ny, self.nx), axes=(-2, -1)).reshape(Ne, -1)

        np.clip(self.ensemble[:, :self.n_grid], -self.max_eta, self.max_eta,
                out=self.ensemble[:, :self.n_grid])
        np.clip(self.ensemble[:, self.n_grid:], -self.max_eta * 2, self.max_eta * 2,
                out=self.ensemble[:, self.n_grid:])

    def _interpolate_to_grid(self, obs_x, obs_y, obs_z):
        """
        Interpolate sparse observations onto the estimation grid.
        Wang 2025 uses Delaunay triangulation (barycentric interpolation).
        Returns: (n_grid,) interpolated elevation, (n_grid,) valid mask.
        """
        if len(obs_x) < 3:
            return np.zeros(self.n_grid), np.zeros(self.n_grid, dtype=bool)

        pts = np.column_stack([obs_x, obs_y])
        z_interp = griddata(pts, obs_z, self._grid_pts,
                            method='linear', fill_value=np.nan)

        valid = ~np.isnan(z_interp)
        z_interp = np.nan_to_num(z_interp, nan=0.0)
        return z_interp, valid

    def _fuzzy_r_adaptation(self, innovation, valid_mask):
        """
        Fuzzy logic R adaptation (Wang 2025, Table 1, Eq. 33-36).

        Uses current innovation d and innovation difference d_diff
        to compute per-grid-point R scaling factor.
        """
        if not self.use_fuzzy:
            return np.ones(self.n_grid)

        d = np.abs(innovation)
        d_diff = np.abs(innovation - self._prev_innovation)
        self._prev_innovation = innovation.copy()

        # Thresholds for fuzzy membership (adaptive based on data spread)
        d_std = max(np.std(d[valid_mask]) if valid_mask.any() else 0.1, 0.01)
        T_d_small = 0.5 * d_std
        T_d_large = 2.0 * d_std
        T_diff_small = 0.5 * d_std
        T_diff_large = 2.0 * d_std

        # Membership functions (triangular/trapezoidal)
        def membership_small(x, t_s, t_l):
            return np.clip(1.0 - x / t_s, 0, 1)

        def membership_large(x, t_s, t_l):
            return np.clip((x - t_s) / (t_l - t_s), 0, 1)

        def membership_medium(x, t_s, t_l):
            return np.clip(np.minimum(x / t_s, (t_l - x) / (t_l - t_s)), 0, 1)

        # Input memberships
        d_S = membership_small(d, T_d_small, T_d_large)
        d_M = membership_medium(d, T_d_small, T_d_large)
        d_L = membership_large(d, T_d_small, T_d_large)
        dd_S = membership_small(d_diff, T_diff_small, T_diff_large)
        dd_M = membership_medium(d_diff, T_diff_small, T_diff_large)
        dd_L = membership_large(d_diff, T_diff_small, T_diff_large)

        # Output membership values: Small=0.3, Medium=1.0, Large=3.0
        # Table 1 from Wang 2025 (9 rules)
        r_scale = np.zeros_like(d)
        weights = np.zeros_like(d)

        rules = [
            # (d_membership, dd_membership, output_value)
            (d_L, dd_L, 3.0),    # Rule 1: Large, Large -> Large
            (d_L, dd_M, 3.0),    # Rule 2: Large, Medium -> Large
            (d_L, dd_S, 1.0),    # Rule 3: Large, Small -> Medium
            (d_M, dd_L, 3.0),    # Rule 4: Medium, Large -> Large
            (d_M, dd_M, 1.0),    # Rule 5: Medium, Medium -> Medium
            (d_M, dd_S, 1.0),    # Rule 6: Medium, Small -> Medium (was Small in paper but conservative)
            (d_S, dd_L, 1.0),    # Rule 7: Small, Large -> Medium
            (d_S, dd_M, 0.3),    # Rule 8: Small, Medium -> Small
            (d_S, dd_S, 0.3),    # Rule 9: Small, Small -> Small
        ]

        for d_mem, dd_mem, out_val in rules:
            w = np.minimum(d_mem, dd_mem)  # AND operation = min
            r_scale += w * out_val
            weights += w

        # Centroid defuzzification
        r_scale = np.where(weights > 1e-10, r_scale / weights, 1.0)
        return r_scale

    def _etkf_analysis_substate(self, ensemble_sub, z_obs, R_diag_vec):
        """
        ETKF analysis for a sub-state block.
        H = identity (interpolated measurements match state dimensions).
        """
        Ne = ensemble_sub.shape[0]
        n = ensemble_sub.shape[1]

        x_mean = np.mean(ensemble_sub, axis=0)
        X_pert = ensemble_sub - x_mean[None, :]

        # H = I, so HX = ensemble_sub (elevation part)
        HX_mean = x_mean
        HX_pert = X_pert

        # Innovation
        d = z_obs - HX_mean

        # Normalize by R
        R_inv_sqrt = 1.0 / np.sqrt(R_diag_vec + 1e-12)
        Y = HX_pert * R_inv_sqrt[None, :]  # (Ne, n)
        d_norm = d * R_inv_sqrt

        # ETKF core (Hunt 2007)
        C = (Ne - 1) * np.eye(Ne) + Y @ Y.T

        try:
            C_inv = np.linalg.inv(C)
        except np.linalg.LinAlgError:
            return ensemble_sub, d

        # Mean update
        w_mean = C_inv @ Y @ d_norm
        x_mean_new = x_mean + X_pert.T @ w_mean

        # Perturbation update
        eigvals, eigvecs = np.linalg.eigh(C_inv)
        eigvals = np.maximum(eigvals, 0)
        W = eigvecs @ np.diag(np.sqrt(eigvals * (Ne - 1))) @ eigvecs.T
        X_pert_new = W @ X_pert

        ensemble_new = x_mean_new[None, :] + X_pert_new
        return ensemble_new, d

    def update(self, obs_x: np.ndarray, obs_y: np.ndarray,
               obs_z: np.ndarray) -> float:
        """One ETKF cycle: forecast + interpolate + fuzzy + localized analysis."""
        t0 = time.time()

        if len(obs_x) == 0:
            return (time.time() - t0) * 1000

        Ne = self.n_ensemble

        # 1. Forecast all ensemble members (batch FFT)
        self._forecast_all()

        # 2. Interpolate sparse observations onto grid
        z_grid, valid = self._interpolate_to_grid(obs_x, obs_y, obs_z)

        # 3. Compute innovation (for elevation part only)
        eta_mean = np.mean(self.ensemble[:, :self.n_grid], axis=0)
        innovation = z_grid - eta_mean

        # 4. Fuzzy logic R adaptation
        r_scale = self._fuzzy_r_adaptation(innovation, valid)
        R_diag = (self.obs_noise ** 2) * r_scale
        # Invalid (non-interpolated) grid points get very large R -> ignored
        R_diag[~valid] = 1e6

        # 5. Localized ETKF analysis (sub-state decomposition)
        # Split grid into L sub-states
        L = min(self.n_substates, self.ny)
        rows_per_sub = self.ny // L
        remainder = self.ny % L

        all_innovation = np.zeros(self.n_grid)

        for l in range(L):
            r_start = l * rows_per_sub + min(l, remainder)
            r_end = r_start + rows_per_sub + (1 if l < remainder else 0)

            # Indices for this sub-state (both eta and deta)
            eta_idx = []
            deta_idx = []
            for r in range(r_start, r_end):
                for c in range(self.nx):
                    idx = r * self.nx + c
                    eta_idx.append(idx)
                    deta_idx.append(self.n_grid + idx)

            eta_idx = np.array(eta_idx)
            deta_idx = np.array(deta_idx)
            sub_idx = np.concatenate([eta_idx, deta_idx])

            # Extract sub-state ensemble
            ensemble_sub_eta = self.ensemble[:, eta_idx].copy()
            z_sub = z_grid[eta_idx]
            R_sub = R_diag[eta_idx]

            # ETKF analysis on sub-state (eta only for obs)
            ensemble_sub_new, d_sub = self._etkf_analysis_substate(
                ensemble_sub_eta, z_sub, R_sub)

            # Update eta part of ensemble
            self.ensemble[:, eta_idx] = ensemble_sub_new
            all_innovation[eta_idx] = d_sub

        # Clamp
        np.clip(self.ensemble[:, :self.n_grid],
                -self.max_eta, self.max_eta,
                out=self.ensemble[:, :self.n_grid])

        return (time.time() - t0) * 1000

    def predict_grid(self) -> np.ndarray:
        """Return ensemble mean eta: (Ny, Nx)."""
        x_mean = np.mean(self.ensemble, axis=0)
        return x_mean[:self.n_grid].reshape(self.ny, self.nx)

    def reset(self):
        """Reset ensemble."""
        self.ensemble = np.random.randn(
            self.n_ensemble, self.n_state) * 0.05
        self._prev_innovation = np.zeros(self.n_grid)
