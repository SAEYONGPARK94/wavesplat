"""
gpr_dispersion.py
Dispersion-aware GPR baseline: spatio-temporal kernel using wave physics.

Stronger baseline than vanilla GPR: instead of treating each frame
independently, this GPR uses a spatio-temporal kernel that accounts
for wave propagation via the dispersion relation.

Key idea:
  - Standard GPR kernel: k(x1, x2) = Matern(||x1 - x2||)
  - Dispersion-aware kernel: k((x1,t1), (x2,t2)) incorporates the
    expected phase shift between observations at different times.

We approximate this by phase-shifting past observations to the current
time using a dominant wavenumber estimated from the data, then applying
standard spatial GPR on the phase-corrected observations.

This is simpler than a full spatio-temporal GP but captures the key
physics: observations from past frames carry information about the
current wave field through dispersion-governed propagation.
"""

import numpy as np
import time
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel


class GPRDispersion:
    """
    Dispersion-aware GPR baseline.

    Uses multi-frame observations with phase correction based on
    estimated dominant wave parameters from the dispersion relation.

    Args:
        length_scale: spatial kernel length scale [m]
        noise_sigma: observation noise std [m]
        window_frames: number of frames to accumulate
        depth: water depth [m]
        dt: frame interval [s]
        n_k_candidates: number of wavenumber candidates for dominant k estimation
    """

    def __init__(self, length_scale=20.0, noise_sigma=0.02,
                 window_frames=3, depth=50.0, dt=0.1,
                 n_k_candidates=50):
        self.length_scale = length_scale
        self.noise_sigma = noise_sigma
        self.window_frames = window_frames
        self.depth = depth
        self.dt = dt
        self.g = 9.81
        self.n_k_candidates = n_k_candidates

        self._buffer = []  # list of (x, y, z, t_frame)
        self._gp = None
        self._frame_count = 0
        self._dominant_k = None  # estimated dominant |k|

    def _estimate_dominant_k(self):
        """
        Estimate dominant wavenumber from accumulated observations
        using a simplified beamforming approach.
        """
        if len(self._buffer) < 2:
            return 0.3  # default for BF4

        # Combine all observations
        all_x = np.concatenate([b[0] for b in self._buffer])
        all_y = np.concatenate([b[1] for b in self._buffer])
        all_z = np.concatenate([b[2] for b in self._buffer])

        if len(all_x) < 10:
            return 0.3

        # Simple spatial frequency estimation via autocorrelation
        # Use pairwise distances and correlations
        n_sample = min(200, len(all_x))
        idx = np.random.choice(len(all_x), n_sample, replace=False)
        x_s, y_s, z_s = all_x[idx], all_y[idx], all_z[idx]

        # Pairwise distances
        dx = x_s[:, None] - x_s[None, :]
        dy = y_s[:, None] - y_s[None, :]
        dists = np.sqrt(dx**2 + dy**2)

        # Pairwise products (proxy for correlation)
        zz = z_s[:, None] * z_s[None, :]

        # Scan k candidates
        k_candidates = np.linspace(0.05, 2.0, self.n_k_candidates)
        best_power = -np.inf
        best_k = 0.3

        for k in k_candidates:
            # Expected correlation pattern for plane wave with |k|
            cos_pattern = np.cos(k * dists)
            power = np.sum(zz * cos_pattern)
            if power > best_power:
                best_power = power
                best_k = k

        return best_k

    def _phase_correct(self, x, y, z, dt_frames):
        """
        Phase-correct observations from dt_frames ago to current time.

        For a wave with wavenumber k and dispersion omega(k),
        the elevation at time t+dt is approximately:
        z(x, t+dt) ≈ z(x, t) * cos(omega * dt) + noise

        This is a crude approximation (assumes single dominant frequency),
        but better than ignoring temporal information entirely.
        """
        if self._dominant_k is None or dt_frames == 0:
            return x, y, z

        k = self._dominant_k
        omega = np.sqrt(self.g * k * np.tanh(k * self.depth))
        dt = dt_frames * self.dt

        # Phase shift factor
        phase_shift = np.cos(omega * dt)

        # Scale observations by expected amplitude decay
        # (crude: just multiply by cos(omega*dt))
        z_corrected = z * phase_shift

        # Weight older observations less (exponential decay)
        weight = np.exp(-0.5 * (dt_frames * self.dt / 1.0)**2)
        z_corrected *= weight

        return x, y, z_corrected

    def update(self, x_obs, y_obs, z_obs):
        """Add observations and fit dispersion-aware GP."""
        t0 = time.time()

        self._buffer.append((x_obs.copy(), y_obs.copy(), z_obs.copy(),
                             self._frame_count))
        self._frame_count += 1

        # Keep only last window_frames
        if len(self._buffer) > self.window_frames:
            self._buffer = self._buffer[-self.window_frames:]

        # Re-estimate dominant k periodically
        if self._frame_count % 10 == 1 or self._dominant_k is None:
            self._dominant_k = self._estimate_dominant_k()

        # Phase-correct all buffered observations to current time
        current_frame = self._frame_count - 1
        all_x, all_y, all_z = [], [], []

        for buf_x, buf_y, buf_z, buf_t in self._buffer:
            dt_frames = current_frame - buf_t
            cx, cy, cz = self._phase_correct(buf_x, buf_y, buf_z, dt_frames)
            all_x.append(cx)
            all_y.append(cy)
            all_z.append(cz)

        all_x = np.concatenate(all_x)
        all_y = np.concatenate(all_y)
        all_z = np.concatenate(all_z)

        if len(all_x) < 3:
            self._gp = None
            return (time.time() - t0) * 1000

        X = np.column_stack([all_x, all_y])

        kernel = Matern(
            length_scale=self.length_scale, nu=2.5,
            length_scale_bounds=(1.0, 200.0)
        ) + WhiteKernel(noise_level=self.noise_sigma**2)

        self._gp = GaussianProcessRegressor(
            kernel=kernel,
            alpha=self.noise_sigma**2,
            normalize_y=True,
            n_restarts_optimizer=2,
        )
        self._gp.fit(X, all_z)

        return (time.time() - t0) * 1000

    def predict_grid(self, grid_x, grid_y):
        """Predict wave field on regular grid."""
        if self._gp is None:
            return np.zeros((len(grid_y), len(grid_x)))

        Xg, Yg = np.meshgrid(grid_x, grid_y)
        X_pred = np.column_stack([Xg.ravel(), Yg.ravel()])

        Z_pred = self._gp.predict(X_pred)
        return Z_pred.reshape(len(grid_y), len(grid_x))

    def reset(self):
        self._buffer = []
        self._gp = None
        self._frame_count = 0
        self._dominant_k = None
