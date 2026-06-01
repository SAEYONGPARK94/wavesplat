"""
Dataset for DCSpecNet training.

Generates (point_cloud, gt_components) pairs from:
  1. Existing HDF5 episodes (wavesim pipeline output)
  2. On-the-fly generation using wavesim + HOS
"""

import numpy as np
import torch
from torch.utils.data import Dataset
import h5py
from pathlib import Path
from typing import Optional, List
from dataclasses import dataclass


@dataclass
class DataConfig:
    """Configuration for dataset."""
    max_points: int = 200       # max LiDAR points per sample
    max_components: int = 16    # max GT components (padded with zeros)
    depth: float = 50.0         # water depth [m]


class EpisodeDataset(Dataset):
    """
    Load training data from existing HDF5 episodes.

    Each episode contains multiple frames. Each frame provides:
      - LiDAR point cloud (x, y, z)
      - GT wave components (A, omega, kx, ky, phi)
    """

    def __init__(self, episode_paths: List[str], config: DataConfig):
        self.config = config
        self.samples = []  # list of (points, components, t)

        for path in episode_paths:
            self._load_episode(path)

    def _load_episode(self, path: str):
        with h5py.File(path, 'r') as f:
            lidar_pts = f['lidar_points'][:]          # (T, N, 3)
            valid_mask = f['lidar_valid_mask'][:]      # (T, N)
            wave_comps = f['wave_components'][:]       # (C, 6): A, omega, kx, ky, phi, Q
            timestamps = f['timestamps'][:]            # (T,)

            for t_idx in range(lidar_pts.shape[0]):
                mask = valid_mask[t_idx]
                pts = lidar_pts[t_idx][mask]           # (n_valid, 3)
                if len(pts) < 3:
                    continue
                self.samples.append({
                    'points': pts,
                    'components': wave_comps[:, :5],   # drop Q column
                    't': timestamps[t_idx],
                })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        pts = sample['points']
        comps = sample['components']
        t = sample['t']

        # Subsample or pad points to max_points
        n = len(pts)
        max_pts = self.config.max_points
        if n > max_pts:
            indices = np.random.choice(n, max_pts, replace=False)
            pts = pts[indices]
        elif n < max_pts:
            pad = np.zeros((max_pts - n, 3), dtype=pts.dtype)
            pts = np.vstack([pts, pad])
            # Create mask for valid points
        else:
            pass

        n_valid = min(n, max_pts)

        return {
            'points': torch.tensor(pts, dtype=torch.float32),
            'components': torch.tensor(comps, dtype=torch.float32),
            't': torch.tensor(t, dtype=torch.float32),
            'n_valid': n_valid,
        }


class OnTheFlyDataset(Dataset):
    """
    Generate training data on-the-fly using wavesim pipeline.

    Each sample:
      1. Sample random sea state (Hs, Tp, N_comp)
      2. Generate wave components
      3. Evaluate field + simulate LiDAR
      4. Return (point_cloud, gt_components)
    """

    def __init__(self, n_samples: int = 10000, config: DataConfig = None,
                 seed: int = 0):
        self.n_samples = n_samples
        self.config = config or DataConfig()
        self.seed = seed

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        rng = np.random.default_rng(self.seed + idx)

        # Random sea state
        Hs = rng.uniform(0.3, 3.0)
        Tp = rng.uniform(4.0, 12.0)
        n_comp = rng.integers(2, 8)
        depth = self.config.depth

        # Generate components
        from wavesim.wave_generator import jonswap_components
        components = jonswap_components(
            Hs=Hs, Tp=Tp, n_components=n_comp,
            depth=depth, seed=self.seed + idx)

        # Random time
        t = rng.uniform(0, 20.0)

        # Generate observation points (simplified: random positions)
        n_pts = rng.integers(20, self.config.max_points + 1)
        x = rng.uniform(-100, 100, n_pts)
        y = rng.uniform(-100, 100, n_pts)

        # Evaluate true surface at observation points
        z = np.zeros(n_pts)
        for c in components:
            z += c.A * np.cos(c.kx * x + c.ky * y - c.omega * t + c.phi)

        # Add noise (simulating LiDAR measurement noise)
        noise_std = 0.02 * Hs  # 2% of Hs
        z += rng.normal(0, noise_std, n_pts)

        # Pad to max_points
        pts = np.stack([x, y, z], axis=-1).astype(np.float32)
        if n_pts < self.config.max_points:
            pad = np.zeros((self.config.max_points - n_pts, 3), dtype=np.float32)
            pts = np.vstack([pts, pad])

        # GT components array — pad to max_components
        max_c = self.config.max_components
        comp_arr = np.zeros((max_c, 5), dtype=np.float32)
        n_c = min(len(components), max_c)
        for i in range(n_c):
            c = components[i]
            comp_arr[i] = [c.A, c.omega, c.kx, c.ky, c.phi]

        return {
            'points': torch.tensor(pts, dtype=torch.float32),
            'components': torch.tensor(comp_arr, dtype=torch.float32),
            'n_components': n_c,
            't': torch.tensor(t, dtype=torch.float32),
            'n_valid': n_pts,
        }
