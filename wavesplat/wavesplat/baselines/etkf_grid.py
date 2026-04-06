"""
etkf_grid.py
Ensemble Transform Kalman Filter on a grid — baseline from "From Points to Waves".

State: wave elevation η(grid) + ∂η/∂t on regular grid
Prediction: dispersion-based linear wave propagation
Observation: bilinear interpolation from grid to observation points

This demonstrates the grid interpolation limitation that WaveSplat overcomes.
"""

import numpy as np
import time
from typing import Tuple
from scipy.interpolate import RegularGridInterpolator


class ETKFGrid:
    """
    ETKF-grid baseline for wave field estimation.

    Args:
        grid_x, grid_y: (Nx,), (Ny,) grid coordinates
        depth: water depth [m]
        n_ensemble: ensemble size
        dt: timestep [s]
        inflation: covariance inflation factor
    """

    def __init__(self, grid_x: np.ndarray, grid_y: np.ndarray,
                 depth: float = 50.0,
                 n_ensemble: int = 50,
                 dt: float = 0.1,
                 inflation: float = 1.05,
                 obs_noise: float = 0.02):
        self.grid_x = grid_x
        self.grid_y = grid_y
        self.nx = len(grid_x)
        self.ny = len(grid_y)
        self.n_state = self.nx * self.ny * 2  # η + ∂η/∂t
        self.depth = depth
        self.n_ensemble = n_ensemble
        self.dt = dt
        self.inflation = inflation
        self.obs_noise = obs_noise
        self.g = 9.81

        # Initialize ensemble: small random perturbations
        self.ensemble = np.random.randn(self.n_ensemble, self.n_state) * 0.01

    def _get_eta(self, state: np.ndarray) -> np.ndarray:
        """Extract η from state vector → (Ny, Nx)."""
        n_grid = self.nx * self.ny
        return state[:n_grid].reshape(self.ny, self.nx)

    def _get_deta_dt(self, state: np.ndarray) -> np.ndarray:
        """Extract ∂η/∂t from state vector → (Ny, Nx)."""
        n_grid = self.nx * self.ny
        return state[n_grid:].reshape(self.ny, self.nx)

    def _forecast_member(self, state: np.ndarray) -> np.ndarray:
        """
        Propagate one ensemble member by dt.
        Simple linear wave equation: ∂²η/∂t² = -g·d·∇²η (shallow water limit)
        or use spectral propagation for full dispersion.

        For simplicity, use leapfrog with Laplacian:
            η(t+dt) = η(t) + ∂η/∂t · dt
            ∂η/∂t(t+dt) = ∂η/∂t(t) + g·d·∇²η(t) · dt
        """
        eta = self._get_eta(state)
        deta = self._get_deta_dt(state)

        # Laplacian via finite differences
        dx = self.grid_x[1] - self.grid_x[0] if self.nx > 1 else 1.0
        dy = self.grid_y[1] - self.grid_y[0] if self.ny > 1 else 1.0

        laplacian = np.zeros_like(eta)
        if self.nx > 2:
            laplacian[:, 1:-1] += (eta[:, 2:] - 2*eta[:, 1:-1] + eta[:, :-2]) / dx**2
        if self.ny > 2:
            laplacian[1:-1, :] += (eta[2:, :] - 2*eta[1:-1, :] + eta[:-2, :]) / dy**2

        # Phase speed: c² = g·d·tanh(k·d)/(k·d) ≈ g·d for shallow
        c2 = self.g * min(self.depth, 100.0)  # cap for stability

        new_eta = eta + deta * self.dt
        new_deta = deta + c2 * laplacian * self.dt

        # Damping at boundaries
        new_eta[0, :] *= 0.9
        new_eta[-1, :] *= 0.9
        new_eta[:, 0] *= 0.9
        new_eta[:, -1] *= 0.9

        new_state = state.copy()
        n_grid = self.nx * self.ny
        new_state[:n_grid] = new_eta.ravel()
        new_state[n_grid:] = new_deta.ravel()
        return new_state

    def _observation_operator(self, state: np.ndarray,
                               obs_x: np.ndarray,
                               obs_y: np.ndarray) -> np.ndarray:
        """
        Bilinear interpolation from grid to observation points.
        This is the key limitation — requires regular grid.
        """
        eta = self._get_eta(state)

        try:
            interp = RegularGridInterpolator(
                (self.grid_y, self.grid_x), eta,
                method='linear', bounds_error=False, fill_value=0.0
            )
            pts = np.column_stack([obs_y, obs_x])
            return interp(pts)
        except ValueError as e:
            import warnings
            warnings.warn(f"ETKF observation interpolation failed: {e}")
            return np.zeros(len(obs_x))

    def update(self, obs_x: np.ndarray, obs_y: np.ndarray,
               obs_z: np.ndarray) -> float:
        """
        One ETKF cycle: forecast + analysis.

        Args:
            obs_x, obs_y, obs_z: (M,) observations

        Returns:
            elapsed_ms
        """
        t0 = time.time()
        M = len(obs_x)

        if M == 0:
            return (time.time() - t0) * 1000

        # Forecast
        for i in range(self.n_ensemble):
            self.ensemble[i] = self._forecast_member(self.ensemble[i])

        # Ensemble mean and perturbations
        x_mean = np.mean(self.ensemble, axis=0)
        X_pert = self.ensemble - x_mean[None, :]

        # Inflation
        X_pert *= self.inflation
        self.ensemble = x_mean[None, :] + X_pert

        # Observation operator applied to each member
        HX = np.zeros((self.n_ensemble, M))
        for i in range(self.n_ensemble):
            HX[i] = self._observation_operator(self.ensemble[i], obs_x, obs_y)

        HX_mean = np.mean(HX, axis=0)
        HX_pert = HX - HX_mean[None, :]

        # Innovation
        d = obs_z - HX_mean

        # ETKF analysis (Bishop et al. 2001)
        R_inv = np.eye(M) / (self.obs_noise ** 2)

        # C = HX_pert^T R^{-1} HX_pert + (Ne-1) I
        # Use economy SVD for stability
        S = HX_pert.T  # (M, Ne)
        C = S.T @ R_inv @ S + (self.n_ensemble - 1) * np.eye(self.n_ensemble)

        try:
            C_inv = np.linalg.inv(C)
        except np.linalg.LinAlgError:
            import warnings
            warnings.warn("ETKF analysis skipped: singular matrix C")
            return (time.time() - t0) * 1000

        # Mean update
        K_eff = X_pert.T @ C_inv @ S.T @ R_inv  # (n_state, M)
        x_mean_new = x_mean + K_eff @ d

        # Perturbation update (square root)
        eigvals, eigvecs = np.linalg.eigh(C_inv * (self.n_ensemble - 1))
        eigvals = np.maximum(eigvals, 0)
        T = eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T
        X_pert_new = X_pert @ T

        self.ensemble = x_mean_new[None, :] + X_pert_new

        return (time.time() - t0) * 1000

    def predict_grid(self) -> np.ndarray:
        """Return ensemble mean η field: (Ny, Nx)."""
        x_mean = np.mean(self.ensemble, axis=0)
        return self._get_eta(x_mean)

    def reset(self):
        """Reset ensemble."""
        self.ensemble = np.random.randn(self.n_ensemble, self.n_state) * 0.01
