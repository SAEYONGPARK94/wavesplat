"""
etkf_grid.py
Ensemble Transform Kalman Filter on a grid — baseline from "From Points to Waves".

State: wave elevation η(grid) + ∂η/∂t on regular grid
Prediction: spectral propagation (exact dispersion in Fourier space)
Observation: bilinear interpolation from grid to observation points

This demonstrates the grid interpolation limitation that WaveSplat overcomes.
"""

import numpy as np
import time
from scipy.interpolate import RegularGridInterpolator


class ETKFGrid:
    """
    ETKF-grid baseline for wave field estimation.

    Uses spectral (FFT-based) propagation for unconditionally stable forecasting,
    observation thinning to prevent ensemble collapse (M >> Ne),
    and additive inflation to maintain ensemble spread.
    """

    def __init__(self, grid_x: np.ndarray, grid_y: np.ndarray,
                 depth: float = 50.0,
                 n_ensemble: int = 50,
                 dt: float = 0.1,
                 mult_inflation: float = 1.01,
                 add_inflation: float = 0.005,
                 obs_noise: float = 0.05,
                 max_obs: int = 40):
        self.grid_x = grid_x
        self.grid_y = grid_y
        self.nx = len(grid_x)
        self.ny = len(grid_y)
        self.n_grid = self.nx * self.ny
        self.n_state = self.n_grid * 2  # η + ∂η/∂t
        self.depth = depth
        self.n_ensemble = n_ensemble
        self.dt = dt
        self.mult_inflation = mult_inflation
        self.add_inflation = add_inflation
        self.obs_noise = obs_noise
        self.max_obs = max_obs
        self.g = 9.81
        self.max_eta = 5.0

        # Grid spacings
        self.dx = grid_x[1] - grid_x[0] if self.nx > 1 else 1.0
        self.dy = grid_y[1] - grid_y[0] if self.ny > 1 else 1.0

        # Precompute spectral propagation coefficients
        kx = 2 * np.pi * np.fft.fftfreq(self.nx, d=self.dx)
        ky = 2 * np.pi * np.fft.fftfreq(self.ny, d=self.dy)
        KX, KY = np.meshgrid(kx[:self.nx // 2 + 1], ky)
        K = np.sqrt(KX ** 2 + KY ** 2 + 1e-12)
        omega = np.sqrt(self.g * K * np.tanh(K * self.depth))

        self._cos_wdt = np.cos(omega * dt)
        self._sin_wdt = np.sin(omega * dt)
        self._sinc_wdt = np.where(omega > 1e-10,
                                   self._sin_wdt / omega, dt)
        self._omega = omega
        # Spectral damping for high-k modes
        self._damp = np.exp(-0.002 * K ** 2 * dt)

        # Initialize ensemble with meaningful perturbations
        self.ensemble = np.random.randn(n_ensemble, self.n_state) * 0.1

    def _forecast_all(self):
        """Batch spectral propagation for all ensemble members at once."""
        Ne = self.n_ensemble
        eta_all = self.ensemble[:, :self.n_grid].reshape(Ne, self.ny, self.nx)
        deta_all = self.ensemble[:, self.n_grid:].reshape(Ne, self.ny, self.nx)

        eta_hat = np.fft.rfft2(eta_all, axes=(-2, -1))
        deta_hat = np.fft.rfft2(deta_all, axes=(-2, -1))

        eta_hat_new = (eta_hat * self._cos_wdt[None]
                       + deta_hat * self._sinc_wdt[None])
        deta_hat_new = (-eta_hat * self._omega[None] * self._sin_wdt[None]
                        + deta_hat * self._cos_wdt[None])

        eta_hat_new *= self._damp[None]
        deta_hat_new *= self._damp[None]

        self.ensemble[:, :self.n_grid] = np.fft.irfft2(
            eta_hat_new, s=(self.ny, self.nx), axes=(-2, -1)).reshape(Ne, -1)
        self.ensemble[:, self.n_grid:] = np.fft.irfft2(
            deta_hat_new, s=(self.ny, self.nx), axes=(-2, -1)).reshape(Ne, -1)

        np.clip(self.ensemble[:, :self.n_grid],
                -self.max_eta, self.max_eta,
                out=self.ensemble[:, :self.n_grid])
        np.clip(self.ensemble[:, self.n_grid:],
                -self.max_eta * 2, self.max_eta * 2,
                out=self.ensemble[:, self.n_grid:])

    def _thin_observations(self, obs_x, obs_y, obs_z):
        """Subsample observations to at most max_obs for ensemble stability."""
        M = len(obs_x)
        if M <= self.max_obs:
            return obs_x, obs_y, obs_z

        # Stratified subsampling: pick spatially spread points
        idx = np.random.choice(M, self.max_obs, replace=False)
        return obs_x[idx], obs_y[idx], obs_z[idx]

    def _observation_operator(self, state, obs_x, obs_y):
        """Bilinear interpolation from grid to observation points."""
        eta = state[:self.n_grid].reshape(self.ny, self.nx)
        try:
            interp = RegularGridInterpolator(
                (self.grid_y, self.grid_x), eta,
                method='linear', bounds_error=False, fill_value=0.0)
            return interp(np.column_stack([obs_y, obs_x]))
        except ValueError:
            return np.zeros(len(obs_x))

    def update(self, obs_x: np.ndarray, obs_y: np.ndarray,
               obs_z: np.ndarray) -> float:
        """One ETKF cycle: forecast + analysis."""
        t0 = time.time()
        M_orig = len(obs_x)

        if M_orig == 0:
            return (time.time() - t0) * 1000

        # Thin observations to prevent M >> Ne issues
        obs_x, obs_y, obs_z = self._thin_observations(obs_x, obs_y, obs_z)
        M = len(obs_x)
        Ne = self.n_ensemble

        # Forecast all ensemble members (batch FFT)
        self._forecast_all()

        # Ensemble mean and perturbations
        x_mean = np.mean(self.ensemble, axis=0)
        X_pert = self.ensemble - x_mean[None, :]

        # Multiplicative inflation
        X_pert *= self.mult_inflation

        # Additive inflation: add random noise to maintain spread
        if self.add_inflation > 0:
            X_pert += np.random.randn(Ne, self.n_state) * self.add_inflation

        self.ensemble = x_mean[None, :] + X_pert

        # Observation operator for each member
        HX = np.zeros((Ne, M))
        for i in range(Ne):
            HX[i] = self._observation_operator(self.ensemble[i], obs_x, obs_y)

        HX_mean = np.mean(HX, axis=0)
        HX_pert = HX - HX_mean[None, :]

        # Innovation
        d = obs_z - HX_mean

        # ETKF analysis (Hunt et al. 2007)
        R_diag = self.obs_noise ** 2
        Y = HX_pert / np.sqrt(R_diag)  # (Ne, M)
        d_norm = d / np.sqrt(R_diag)

        C = (Ne - 1) * np.eye(Ne) + Y @ Y.T

        try:
            C_inv = np.linalg.inv(C)
        except np.linalg.LinAlgError:
            return (time.time() - t0) * 1000

        # Mean update
        w_mean = C_inv @ Y @ d_norm
        x_mean_new = x_mean + X_pert.T @ w_mean

        # Perturbation update (square root)
        eigvals, eigvecs = np.linalg.eigh(C_inv)
        eigvals = np.maximum(eigvals, 0)
        W = eigvecs @ np.diag(np.sqrt(eigvals * (Ne - 1))) @ eigvecs.T

        X_pert_new = W @ X_pert

        self.ensemble = x_mean_new[None, :] + X_pert_new

        # Clamp all members
        for i in range(Ne):
            np.clip(self.ensemble[i, :self.n_grid],
                    -self.max_eta, self.max_eta,
                    out=self.ensemble[i, :self.n_grid])
            np.clip(self.ensemble[i, self.n_grid:],
                    -self.max_eta * 2, self.max_eta * 2,
                    out=self.ensemble[i, self.n_grid:])

        return (time.time() - t0) * 1000

    def predict_grid(self) -> np.ndarray:
        """Return ensemble mean η field: (Ny, Nx)."""
        x_mean = np.mean(self.ensemble, axis=0)
        return x_mean[:self.n_grid].reshape(self.ny, self.nx)

    def reset(self):
        """Reset ensemble."""
        self.ensemble = np.random.randn(
            self.n_ensemble, self.n_state) * 0.1
