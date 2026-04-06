"""
data_loader.py
Load WaveSim HDF5 episodes and provide frame-by-frame PyTorch tensors.

Wraps wavesim.data_writer.load_episode for use with the estimation algorithm.
"""

import torch
import numpy as np
from dataclasses import dataclass
from typing import List, Optional

from wavesim.data_writer import load_episode


@dataclass
class FrameData:
    """Single frame of LiDAR observations + GT."""
    t: float
    x_obs: torch.Tensor  # (M,) x-coordinates of observations
    y_obs: torch.Tensor  # (M,) y-coordinates
    z_obs: torch.Tensor  # (M,) z-values (noisy elevation)
    confidence: torch.Tensor  # (M,) observation weights
    incidence_angles: torch.Tensor  # (M,) for analysis
    gt_field: Optional[torch.Tensor] = None  # (Ny, Nx) GT wave elevation


class EpisodeDataLoader:
    """
    Load an HDF5 episode and iterate over frames.

    Usage:
        loader = EpisodeDataLoader('data/episode_BF4_30comp_seed42_uniform.h5')
        init_frames = loader.get_init_window()
        for frame in loader:
            # use frame.x_obs, frame.z_obs, etc.
    """

    def __init__(self, h5_path: str, device: str = 'cpu'):
        self.device = torch.device(device)
        self.data = load_episode(h5_path)

        # Extract metadata
        self.Hs_true = float(self.data['Hs_true'])
        self.Tp_true = float(self.data['Tp_true'])
        self.depth = float(self.data.get('depth', 50.0))
        self.sea_state = self.data.get('sea_state', 'unknown')
        self.raydrop_model = self.data.get('raydrop_model_type', 'unknown')

        # GT grid coordinates
        self.gt_grid_x = torch.tensor(self.data['gt_grid_x'],
                                       dtype=torch.float32, device=self.device)
        self.gt_grid_y = torch.tensor(self.data['gt_grid_y'],
                                       dtype=torch.float32, device=self.device)

        # Wave components (for oracle baseline)
        self.wave_components = self.data.get('wave_components', None)

        # Spectrum
        self.spectrum_1d = self.data.get('spectrum_1d', None)

        # Frame arrays
        self.timestamps = self.data['timestamps']
        self.n_frames = len(self.timestamps)

        # Padded arrays
        self._lidar_points = self.data['lidar_points']  # (N_frames, max_pts, 3)
        self._valid_mask = self.data['lidar_valid_mask']  # (N_frames, max_pts)
        self._confidence = self.data['lidar_confidence']  # (N_frames, max_pts)
        self._incidence = self.data['lidar_incidence_angles']  # (N_frames, max_pts)
        self._gt_fields = self.data['gt_wave_field']  # (N_frames, Ny, Nx)

    def __len__(self):
        return self.n_frames

    def __getitem__(self, idx: int) -> FrameData:
        mask = self._valid_mask[idx].astype(bool)
        pts = self._lidar_points[idx][mask]  # (M, 3)
        conf = self._confidence[idx][mask]
        inc = self._incidence[idx][mask]
        gt = self._gt_fields[idx]

        dev = self.device
        return FrameData(
            t=float(self.timestamps[idx]),
            x_obs=torch.tensor(pts[:, 0], dtype=torch.float32, device=dev),
            y_obs=torch.tensor(pts[:, 1], dtype=torch.float32, device=dev),
            z_obs=torch.tensor(pts[:, 2], dtype=torch.float32, device=dev),
            confidence=torch.tensor(conf, dtype=torch.float32, device=dev),
            incidence_angles=torch.tensor(inc, dtype=torch.float32, device=dev),
            gt_field=torch.tensor(gt, dtype=torch.float32, device=dev),
        )

    def __iter__(self):
        for i in range(self.n_frames):
            yield self[i]

    def get_init_window(self, tp_factor: float = 2.0) -> List[FrameData]:
        """
        Get frames for the initialization window (first tp_factor * Tp seconds).

        Args:
            tp_factor: number of peak periods for initialization window

        Returns:
            List of FrameData for initialization
        """
        t_init = tp_factor * self.Tp_true
        frames = []
        for i in range(self.n_frames):
            if self.timestamps[i] > t_init:
                break
            frames.append(self[i])
        return frames

    def get_tracking_frames(self, tp_factor: float = 2.0) -> List[FrameData]:
        """Get frames after initialization window."""
        t_init = tp_factor * self.Tp_true
        frames = []
        for i in range(self.n_frames):
            if self.timestamps[i] > t_init:
                frames.append(self[i])
        return frames
