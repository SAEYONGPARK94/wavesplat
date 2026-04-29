"""
letkf_grid.py
Local Ensemble Transform Kalman Filter with Gaspari-Cohn localization.

Stronger baseline than vanilla ETKF: spatial localization prevents
spurious long-range correlations from contaminating the analysis.

Implementation uses a "serial observation" approach for efficiency:
observations are assimilated one at a time with localized covariance
tapering, avoiding the per-gridpoint eigendecomposition bottleneck.
"""

import numpy as np
import time
from scipy.interpolate import RegularGridInterpolator


def gaspari_cohn(r, c):
    """Gaspari-Cohn (1999) compactly supported correlation function."""
    r = np.asarray(r, dtype=float)
    c = float(c)
    result = np.zeros_like(r)
    z = r / c

    mask1 = (z >= 0) & (z <= 1)
    z1 = z[mask1]
    result[mask1] = (1.0 - 5.0/3.0*z1**2 + 5.0/8.0*z1**3
                     + 1.0/2.0*z1**4 - 1.0/4.0*z1**5)

    mask2 = (z > 1) & (z <= 2)
    z2 = z[mask2]
    result[mask2] = (4.0 - 5.0*z2 + 5.0/3.0*z2**2 + 5.0/8.0*z2**3
                     - 1.0/2.0*z2**4 + 1.0/12.0*z2**5 - 2.0/(3.0*z2))

    return np.clip(result, 0.0, 1.0)


class LETKFGrid:
    """
    LETKF with Gaspari-Cohn localization.

    Uses the "localized gain" approach: compute the global ETKF gain
    then taper it element-wise with the Gaspari-Cohn function.
    This is O(Ne^3 + Ne^2*M + n_grid*M) — much faster than
    per-gridpoint eigendecomposition.
    """

    def __init__(self, grid_x, grid_y, depth=50.0, n_ensemble=100,
                 dt=0.1, mult_inflation=1.0, add_inflation=0.005,
                 obs_noise=0.05, max_obs=40, loc_radius=50.0):
        self.grid_x = grid_x
        self.grid_y = grid_y
        self.nx = len(grid_x)
        self.ny = len(grid_y)
        self.n_grid = self.nx * self.ny
        self.n_state = self.n_grid * 2
        self.depth = depth
        self.n_ensemble = n_ensemble
        self.dt = dt
        self.mult_inflation = mult_inflation
        self.add_inflation = add_inflation
        self.obs_noise = obs_noise
        self.max_obs = max_obs
        self.loc_radius = loc_radius
        self.g = 9.81
        self.max_eta = 5.0

        self.dx = grid_x[1] - grid_x[0] if self.nx > 1 else 1.0
        self.dy = grid_y[1] - grid_y[0] if self.ny > 1 else 1.0

        # Grid coordinates for localization
        Xg, Yg = np.meshgrid(grid_x, grid_y)
        self._grid_xy = np.column_stack([Xg.ravel(), Yg.ravel()])  # (n_grid, 2)

        # Spectral propagation coefficients
        kx = 2 * np.pi * np.fft.fftfreq(self.nx, d=self.dx)
        ky = 2 * np.pi * np.fft.fftfreq(self.ny, d=self.dy)
        KX, KY = np.meshgrid(kx[:self.nx // 2 + 1], ky)
        K = np.sqrt(KX**2 + KY**2 + 1e-12)
        omega = np.sqrt(self.g * K * np.tanh(K * self.depth))

        self._cos_wdt = np.cos(omega * dt)
        self._sin_wdt = np.sin(omega * dt)
        self._sinc_wdt = np.where(omega > 1e-10, self._sin_wdt / omega, dt)
        self._omega = omega
        self._damp = np.exp(-0.002 * K**2 * dt)

        self.ensemble = np.random.randn(n_ensemble, self.n_state) * 0.1

    def _forecast_member(self, state):
        eta = state[:self.n_grid].reshape(self.ny, self.nx)
        deta = state[self.n_grid:].reshape(self.ny, self.nx)

        eta_hat = np.fft.rfft2(eta)
        deta_hat = np.fft.rfft2(deta)

        eta_hat_new = eta_hat * self._cos_wdt + deta_hat * self._sinc_wdt
        deta_hat_new = (-eta_hat * self._omega * self._sin_wdt
                        + deta_hat * self._cos_wdt)

        eta_hat_new *= self._damp
        deta_hat_new *= self._damp

        new_state = state.copy()
        new_state[:self.n_grid] = np.fft.irfft2(
            eta_hat_new, s=(self.ny, self.nx)).ravel()
        new_state[self.n_grid:] = np.fft.irfft2(
            deta_hat_new, s=(self.ny, self.nx)).ravel()

        np.clip(new_state[:self.n_grid], -self.max_eta, self.max_eta,
                out=new_state[:self.n_grid])
        np.clip(new_state[self.n_grid:], -self.max_eta * 2, self.max_eta * 2,
                out=new_state[self.n_grid:])
        return new_state

    def _thin_observations(self, obs_x, obs_y, obs_z):
        M = len(obs_x)
        if M <= self.max_obs:
            return obs_x, obs_y, obs_z
        idx = np.random.choice(M, self.max_obs, replace=False)
        return obs_x[idx], obs_y[idx], obs_z[idx]

    def _observation_operator(self, state, obs_x, obs_y):
        eta = state[:self.n_grid].reshape(self.ny, self.nx)
        try:
            interp = RegularGridInterpolator(
                (self.grid_y, self.grid_x), eta,
                method='linear', bounds_error=False, fill_value=0.0)
            return interp(np.column_stack([obs_y, obs_x]))
        except ValueError:
            return np.zeros(len(obs_x))

    def update(self, obs_x, obs_y, obs_z):
        """
        One LETKF cycle using localized gain approach.

        Steps:
        1. Forecast all ensemble members
        2. Compute global ETKF gain K_global
        3. Compute localization matrix L (Gaspari-Cohn between grid and obs)
        4. Apply element-wise: K_local = L * K_global (Schur product)
        5. Update: x_new = x_f + K_local * (y - H*x_mean)
        """
        t0 = time.time()

        if len(obs_x) == 0:
            return (time.time() - t0) * 1000

        obs_x, obs_y, obs_z = self._thin_observations(obs_x, obs_y, obs_z)
        M = len(obs_x)
        Ne = self.n_ensemble
        obs_coords = np.column_stack([obs_x, obs_y])  # (M, 2)

        # Forecast
        for i in range(Ne):
            self.ensemble[i] = self._forecast_member(self.ensemble[i])

        # Ensemble stats
        x_mean = np.mean(self.ensemble, axis=0)
        X_pert = self.ensemble - x_mean[None, :]

        # Inflation
        X_pert *= self.mult_inflation
        if self.add_inflation > 0:
            X_pert += np.random.randn(Ne, self.n_state) * self.add_inflation
        self.ensemble = x_mean[None, :] + X_pert

        # Observation predictions
        HX = np.zeros((Ne, M))
        for i in range(Ne):
            HX[i] = self._observation_operator(self.ensemble[i], obs_x, obs_y)

        HX_mean = np.mean(HX, axis=0)
        HX_pert = HX - HX_mean[None, :]  # (Ne, M)

        # Innovation
        d = obs_z - HX_mean  # (M,)
        R_diag = self.obs_noise ** 2

        # --- Global ETKF transform ---
        Y = HX_pert / np.sqrt(R_diag)  # (Ne, M)
        d_norm = d / np.sqrt(R_diag)   # (M,)

        C = (Ne - 1) * np.eye(Ne) + Y @ Y.T  # (Ne, Ne)
        try:
            C_inv = np.linalg.inv(C)
        except np.linalg.LinAlgError:
            return (time.time() - t0) * 1000

        # Global Kalman gain in ensemble space
        # K_ens = C_inv @ Y  (Ne, M)
        K_ens = C_inv @ Y  # (Ne, M)

        # Global mean update weights
        w_mean = K_ens @ d_norm  # (Ne,)

        # Mean state update (before localization)
        # x_mean_update = X_pert.T @ w_mean  (n_state,)
        # But we need to localize this per grid point.

        # --- Localization ---
        # Distance from each grid point to each observation
        # grid_xy: (n_grid, 2), obs_coords: (M, 2)
        # dists: (n_grid, M)
        dists = np.sqrt(
            (self._grid_xy[:, 0:1] - obs_coords[:, 0:1].T)**2 +
            (self._grid_xy[:, 1:2] - obs_coords[:, 1:2].T)**2
        )  # (n_grid, M)

        # Localization weights
        L = gaspari_cohn(dists, self.loc_radius)  # (n_grid, M)

        # Localized mean update for eta states
        # For each grid point gp:
        #   w_local[gp] = C_inv @ (L[gp,:] * Y).T @ (L[gp,:] * d_norm)
        # This is expensive if done per gp, so we use a vectorized approach:
        #
        # The localized Kalman gain contribution to grid point gp is:
        #   delta_x[gp] = sum_m L[gp,m] * (sum_e X_pert[e, gp] * K_ens[e, m]) * d_norm[m]

        # X_pert for eta: (Ne, n_grid)
        X_pert_eta = X_pert[:, :self.n_grid]    # (Ne, n_grid)
        X_pert_deta = X_pert[:, self.n_grid:]   # (Ne, n_grid)

        # Ensemble-space gain applied to innovation: (Ne, M) @ (M,) = (Ne,)
        # But we need per-observation contributions for localization.
        # K_ens @ diag(d_norm) = (Ne, M), each column is the weight from obs m
        Kd = K_ens * d_norm[None, :]  # (Ne, M)

        # For eta grid point gp:
        #   delta_eta[gp] = sum_m L[gp,m] * sum_e X_pert_eta[e,gp] * Kd[e,m]
        #                 = sum_e X_pert_eta[e,gp] * sum_m L[gp,m] * Kd[e,m]
        # Let W = Kd @ L.T → (Ne, n_grid), where W[e,gp] = sum_m Kd[e,m]*L[gp,m]
        W_local = Kd @ L.T  # (Ne, n_grid)

        # delta_eta[gp] = sum_e X_pert_eta[e,gp] * W_local[e,gp]
        delta_eta = np.sum(X_pert_eta * W_local, axis=0)   # (n_grid,)
        delta_deta = np.sum(X_pert_deta * W_local, axis=0)  # (n_grid,)

        x_mean_new = x_mean.copy()
        x_mean_new[:self.n_grid] += delta_eta
        x_mean_new[self.n_grid:] += delta_deta

        # Perturbation update (use global square root — approximate)
        eigvals, eigvecs = np.linalg.eigh(C_inv)
        eigvals = np.maximum(eigvals, 0)
        W_sqrt = eigvecs @ np.diag(np.sqrt(eigvals * (Ne - 1))) @ eigvecs.T
        X_pert_new = W_sqrt @ X_pert

        self.ensemble = x_mean_new[None, :] + X_pert_new

        # Clamp
        for i in range(Ne):
            np.clip(self.ensemble[i, :self.n_grid],
                    -self.max_eta, self.max_eta,
                    out=self.ensemble[i, :self.n_grid])
            np.clip(self.ensemble[i, self.n_grid:],
                    -self.max_eta * 2, self.max_eta * 2,
                    out=self.ensemble[i, self.n_grid:])

        return (time.time() - t0) * 1000

    def predict_grid(self):
        x_mean = np.mean(self.ensemble, axis=0)
        return x_mean[:self.n_grid].reshape(self.ny, self.nx)

    def reset(self):
        self.ensemble = np.random.randn(
            self.n_ensemble, self.n_state) * 0.1
