"""
gpr_baseline.py
Gaussian Process Regression baseline for wave field estimation.

Two modes:
  1. Single-frame: GP fit on one frame's observations
  2. Window: accumulate last W frames, fit GP on combined observations

No physics, no temporal model. Pure spatial interpolation.
Kernel: Matérn ν=5/2 (physically motivated — C² smooth, suitable for waves).
"""

import numpy as np
import time
from typing import Dict, List, Optional
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel


class GPRBaseline:
    """
    GPR baseline for wave field estimation.

    Args:
        kernel_type: 'matern' (default) or 'rbf'
        length_scale: initial length scale [m]
        noise_sigma: observation noise std [m]
        window_frames: number of frames to accumulate (1 = single frame)
    """

    def __init__(self, kernel_type: str = 'matern',
                 length_scale: float = 20.0,
                 noise_sigma: float = 0.02,
                 window_frames: int = 1):
        self.kernel_type = kernel_type
        self.length_scale = length_scale
        self.noise_sigma = noise_sigma
        self.window_frames = window_frames

        self._buffer_x = []
        self._buffer_y = []
        self._buffer_z = []
        self._gp = None

    def _build_kernel(self):
        if self.kernel_type == 'matern':
            kernel = Matern(
                length_scale=self.length_scale,
                nu=2.5,
                length_scale_bounds=(1.0, 200.0)
            )
        else:
            from sklearn.gaussian_process.kernels import RBF
            kernel = RBF(
                length_scale=self.length_scale,
                length_scale_bounds=(1.0, 200.0)
            )
        kernel = kernel + WhiteKernel(noise_level=self.noise_sigma ** 2)
        return kernel

    def update(self, x_obs: np.ndarray, y_obs: np.ndarray,
               z_obs: np.ndarray) -> float:
        """
        Add observations and fit GP.

        Args:
            x_obs, y_obs, z_obs: (M,) observation arrays

        Returns:
            elapsed_ms: fitting time
        """
        t0 = time.time()

        self._buffer_x.append(x_obs)
        self._buffer_y.append(y_obs)
        self._buffer_z.append(z_obs)

        # Keep only last window_frames
        if len(self._buffer_x) > self.window_frames:
            self._buffer_x = self._buffer_x[-self.window_frames:]
            self._buffer_y = self._buffer_y[-self.window_frames:]
            self._buffer_z = self._buffer_z[-self.window_frames:]

        # Combine all buffered observations
        all_x = np.concatenate(self._buffer_x)
        all_y = np.concatenate(self._buffer_y)
        all_z = np.concatenate(self._buffer_z)

        if len(all_x) < 3:
            self._gp = None
            return (time.time() - t0) * 1000

        X = np.column_stack([all_x, all_y])

        kernel = self._build_kernel()
        self._gp = GaussianProcessRegressor(
            kernel=kernel,
            alpha=self.noise_sigma ** 2,
            normalize_y=True,
            n_restarts_optimizer=2,
        )
        self._gp.fit(X, all_z)

        return (time.time() - t0) * 1000

    def predict_grid(self, grid_x: np.ndarray,
                     grid_y: np.ndarray) -> np.ndarray:
        """
        Predict wave field on regular grid.

        Args:
            grid_x: (Nx,), grid_y: (Ny,)

        Returns:
            Z_pred: (Ny, Nx)
        """
        if self._gp is None:
            return np.zeros((len(grid_y), len(grid_x)))

        Xg, Yg = np.meshgrid(grid_x, grid_y)
        X_pred = np.column_stack([Xg.ravel(), Yg.ravel()])

        Z_pred = self._gp.predict(X_pred)
        return Z_pred.reshape(len(grid_y), len(grid_x))

    def reset(self):
        """Clear observation buffer."""
        self._buffer_x = []
        self._buffer_y = []
        self._buffer_z = []
        self._gp = None
