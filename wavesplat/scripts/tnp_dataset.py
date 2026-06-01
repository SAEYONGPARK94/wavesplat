"""
tnp_dataset.py

WaveSplat HDF5 episode → TNP `Batch` 변환 dataset.

매 task = *short window (default 3 frames)* — WaveSplat 의 multi-frame init 과 같은 paradigm:
  - Context = LiDAR points in window (sparse, off-grid)
  - Target  = GT grid points in window (dense, on-grid)

Batch layout (TNP base.Batch):
  - x  = concat(xc, xt)         (B, Nc+Nt, 3)   # (x, y, t)
  - y  = concat(yc, yt)         (B, Nc+Nt, 1)   # η
  - xc = LiDAR coords           (B, Nc, 3)
  - yc = LiDAR heights          (B, Nc, 1)
  - xt = GT grid coords         (B, Nt, 3)
  - yt = GT grid heights        (B, Nt, 1)
"""
import os
import sys
from glob import glob
from typing import List, Optional, Tuple

import h5py
import numpy as np
import torch

# Make tnp importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'external', 'gridded-tnp'))

from tnp.data.base import Batch, DataGenerator  # noqa: E402


def _list_episode_files(data_dir: str, pattern: str = 'episode_BF*.h5') -> List[str]:
    return sorted(glob(os.path.join(data_dir, pattern)))


def _load_episode(path: str):
    """Load a single episode HDF5 into Python dict of tensors (numpy)."""
    with h5py.File(path, 'r') as f:
        gt_grid_x = f['frames/gt_grid_x'][:]                  # (Nx,)
        gt_grid_y = f['frames/gt_grid_y'][:]                  # (Ny,)
        gt_wave   = f['frames/gt_wave_field'][:]              # (T, Ny, Nx)
        lidar_pts = f['frames/lidar_points'][:]               # (T, Nmax, 3)
        lidar_mask = f['frames/lidar_valid_mask'][:]          # (T, Nmax)
        times = f['frames/timestamps'][:]                     # (T,)
    return dict(
        gt_grid_x=gt_grid_x, gt_grid_y=gt_grid_y,
        gt_wave=gt_wave, lidar_pts=lidar_pts, lidar_mask=lidar_mask,
        times=times,
    )


class WaveSplatTNPGenerator(DataGenerator):
    """Generates TNP `Batch` from one or more WaveSplat episodes.

    Each task = short window of `window_size` consecutive frames.
    Context = LiDAR points (sparse, x_obs, y_obs, t_obs, η_obs).
    Target  = GT grid points (dense, x_gt, y_gt, t_gt, η_gt).

    Normalization:
      - Coordinates (x, y) scaled to [-1, 1]  (using domain extents)
      - Time t scaled to [0, 1]               (within window)
      - η normalized per-batch (std-scaled)   (optional)
    """

    def __init__(
        self,
        *,
        episode_paths: List[str],
        samples_per_epoch: int,
        batch_size: int,
        window_size: int = 3,
        max_context: int = 256,
        max_target: int = 1024,
        normalise_eta: bool = True,
        deterministic: bool = False,
        deterministic_seed: int = 0,
        **kwargs,
    ):
        super().__init__(
            samples_per_epoch=samples_per_epoch,
            batch_size=batch_size,
            deterministic=deterministic,
            deterministic_seed=deterministic_seed,
            **kwargs,
        )
        assert len(episode_paths) > 0, "episode_paths must be non-empty"

        self.episode_paths = episode_paths
        self.window_size = window_size
        self.max_context = max_context
        self.max_target = max_target
        self.normalise_eta = normalise_eta

        # Pre-load all episodes into memory (small dataset: ~110 ep × ~12 MB = 1.3 GB)
        self.episodes = [_load_episode(p) for p in episode_paths]

        # Domain extents (for coordinate normalization)
        ep0 = self.episodes[0]
        self.x_min, self.x_max = float(ep0['gt_grid_x'][0]), float(ep0['gt_grid_x'][-1])
        self.y_min, self.y_max = float(ep0['gt_grid_y'][0]), float(ep0['gt_grid_y'][-1])

    def _normalise_xy(self, x: np.ndarray, y: np.ndarray):
        xn = 2 * (x - self.x_min) / (self.x_max - self.x_min) - 1
        yn = 2 * (y - self.y_min) / (self.y_max - self.y_min) - 1
        return xn, yn

    def _sample_one_task(self, rng: np.random.Generator):
        """Sample (xc, yc, xt, yt) for one task (one window of one episode)."""
        ep_idx = rng.integers(0, len(self.episodes))
        ep = self.episodes[ep_idx]
        T = ep['gt_wave'].shape[0]
        t0 = int(rng.integers(0, T - self.window_size))
        ts = ep['times'][t0:t0 + self.window_size]
        t_min = float(ts[0])
        t_scale = max(float(ts[-1] - ts[0]), 1e-6)

        # ── Context: LiDAR points in window ──
        ctx_x, ctx_y, ctx_t, ctx_eta = [], [], [], []
        for k in range(self.window_size):
            mask = ep['lidar_mask'][t0 + k]            # (Nmax,)
            pts = ep['lidar_pts'][t0 + k][mask]         # (n_valid, 3)
            if len(pts) == 0:
                continue
            ctx_x.append(pts[:, 0])
            ctx_y.append(pts[:, 1])
            ctx_t.append(np.full(len(pts), (ts[k] - t_min) / t_scale))
            ctx_eta.append(pts[:, 2])
        ctx_x = np.concatenate(ctx_x) if ctx_x else np.zeros(0)
        ctx_y = np.concatenate(ctx_y) if ctx_y else np.zeros(0)
        ctx_t = np.concatenate(ctx_t) if ctx_t else np.zeros(0)
        ctx_eta = np.concatenate(ctx_eta) if ctx_eta else np.zeros(0)

        # Normalize x, y
        ctx_x, ctx_y = self._normalise_xy(ctx_x, ctx_y)

        # Cap context size
        if len(ctx_x) > self.max_context:
            idx = rng.choice(len(ctx_x), self.max_context, replace=False)
            ctx_x = ctx_x[idx]; ctx_y = ctx_y[idx]
            ctx_t = ctx_t[idx]; ctx_eta = ctx_eta[idx]

        xc = np.stack([ctx_x, ctx_y, ctx_t], axis=-1).astype(np.float32)  # (Nc, 3)
        yc = ctx_eta.astype(np.float32)[:, None]                          # (Nc, 1)

        # ── Target: GT grid points (subsample) ──
        gx, gy = ep['gt_grid_x'], ep['gt_grid_y']
        gxn, gyn = self._normalise_xy(gx, gy)
        Ny, Nx = ep['gt_wave'].shape[1], ep['gt_wave'].shape[2]
        # Random subsample of (frame_in_window, y_idx, x_idx)
        n_target = min(self.max_target, self.window_size * Ny * Nx)
        ks = rng.integers(0, self.window_size, n_target)
        yi = rng.integers(0, Ny, n_target)
        xi = rng.integers(0, Nx, n_target)
        tgt_x = gxn[xi].astype(np.float32)
        tgt_y = gyn[yi].astype(np.float32)
        tgt_t = ((ts[ks] - t_min) / t_scale).astype(np.float32)
        tgt_eta = ep['gt_wave'][t0 + ks, yi, xi].astype(np.float32)

        xt = np.stack([tgt_x, tgt_y, tgt_t], axis=-1)  # (Nt, 3)
        yt = tgt_eta[:, None]                          # (Nt, 1)

        return xc, yc, xt, yt

    def generate_batch(self, batch_shape: Optional[torch.Size] = None) -> Batch:
        batch_size = self.batch_size if batch_shape is None else batch_shape[0]
        rng = np.random.default_rng()

        xcs, ycs, xts, yts = [], [], [], []
        for _ in range(batch_size):
            xc, yc, xt, yt = self._sample_one_task(rng)
            xcs.append(xc); ycs.append(yc); xts.append(xt); yts.append(yt)

        # Pad context (variable Nc per task) to common Nc_max
        Nc_max = max(len(x) for x in xcs)
        Nt_max = max(len(x) for x in xts)

        def _pad(arr_list, N_max, dim):
            padded = np.zeros((batch_size, N_max, dim), dtype=np.float32)
            for i, a in enumerate(arr_list):
                padded[i, :len(a)] = a
            return padded

        xc = _pad(xcs, Nc_max, 3)
        yc = _pad(ycs, Nc_max, 1)
        xt = _pad(xts, Nt_max, 3)
        yt = _pad(yts, Nt_max, 1)

        # Optionally normalize eta (batch-wise) for stable training
        if self.normalise_eta:
            eta_std = max(float(np.std(yc) + np.std(yt)), 1e-4)
            yc = yc / eta_std
            yt = yt / eta_std

        x = np.concatenate([xc, xt], axis=1)
        y = np.concatenate([yc, yt], axis=1)

        return Batch(
            x=torch.from_numpy(x), y=torch.from_numpy(y),
            xc=torch.from_numpy(xc), yc=torch.from_numpy(yc),
            xt=torch.from_numpy(xt), yt=torch.from_numpy(yt),
        )


def build_episode_split(
    data_dir: str,
    train_pattern: str = 'episode_BF[2345]*_uniform.h5',
    val_pattern: str = 'episode_BF[2345]*_slope.h5',
    test_pattern: str = 'episode_BF[67]*.h5',
) -> Tuple[List[str], List[str], List[str]]:
    """Split episodes by pattern (no leakage)."""
    train = sorted(glob(os.path.join(data_dir, train_pattern)))
    val = sorted(glob(os.path.join(data_dir, val_pattern)))
    test = sorted(glob(os.path.join(data_dir, test_pattern)))
    return train, val, test


if __name__ == '__main__':
    # Smoke test
    DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
    eps = _list_episode_files(DATA_DIR, 'episode_BF3_2comp_seed0_uniform.h5')
    print(f'Test episodes: {eps}')

    gen = WaveSplatTNPGenerator(
        episode_paths=eps,
        samples_per_epoch=8,
        batch_size=2,
        window_size=3,
        max_context=256,
        max_target=512,
    )
    batch = next(iter(gen))
    print(f'  x:  {batch.x.shape}  ({batch.x.dtype})')
    print(f'  y:  {batch.y.shape}  ({batch.y.dtype})')
    print(f'  xc: {batch.xc.shape}')
    print(f'  yc: {batch.yc.shape}')
    print(f'  xt: {batch.xt.shape}')
    print(f'  yt: {batch.yt.shape}')
    print(f'  context η range: [{batch.yc.min():.3f}, {batch.yc.max():.3f}]')
    print(f'  target η range:  [{batch.yt.min():.3f}, {batch.yt.max():.3f}]')
