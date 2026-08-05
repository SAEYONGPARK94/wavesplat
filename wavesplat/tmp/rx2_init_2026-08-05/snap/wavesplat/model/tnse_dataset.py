"""
tnse_dataset.py — PyTorch Dataset for T-NSE training.

Each sample:
  - Input:  W consecutive frames of sparse LiDAR (x, y, t, z, conf)
  - Target: GT wave elevation at next frame on the prediction grid

Normalisation:
  - x, y: divided by x_range (250m), centred
  - t:    divided by Tp (peak period)
  - z:    divided by Hs estimate
  - conf: raw [0, 1]
"""

import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset

from wavesplat.data_loader import EpisodeDataLoader
from wavesplat.model.tnse import MAX_PTS


class TNSEDataset(Dataset):
    """
    Sliding-window dataset over HDF5 episodes.

    Args:
        data_dir:  path to directory containing .h5 files
        h5_files:  list of filenames (None = use all *.h5 in data_dir)
        W:         temporal window size (number of conditioning frames)
        tp_factor: skip init window (first tp_factor * Tp seconds)
        augment:   random horizontal flip augmentation
    """

    def __init__(self, data_dir, h5_files=None, W=10, tp_factor=2.0, augment=False):
        self.W = W
        self.augment = augment
        self.samples = []   # list of (loader_idx, frame_idx_end)
        self.loaders = []

        if h5_files is None:
            h5_files = [os.path.basename(p)
                        for p in sorted(glob.glob(os.path.join(data_dir, '*.h5')))
                        if os.path.getsize(p) > 1000]

        for fname in h5_files:
            fpath = os.path.join(data_dir, fname)
            if not os.path.exists(fpath) or os.path.getsize(fpath) < 1000:
                continue
            try:
                loader = EpisodeDataLoader(fpath, device='cpu')
            except Exception as e:
                print(f"  Warning: failed to load {fname}: {e}")
                continue

            # Collect sliding windows in the tracking region
            t_init = tp_factor * loader.Tp_true
            start_frame = None
            for i in range(loader.n_frames):
                if loader.timestamps[i] > t_init:
                    start_frame = i
                    break
            if start_frame is None:
                continue

            # Window ends at frame j, starts at j-W+1
            # Need at least W frames + 1 target frame after init
            for j in range(start_frame + W - 1, loader.n_frames - 1):
                self.samples.append((len(self.loaders), j))

            self.loaders.append(loader)

        print(f"TNSEDataset: {len(self.loaders)} episodes, {len(self.samples)} samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        loader_idx, frame_end = self.samples[idx]
        loader = self.loaders[loader_idx]
        W = self.W

        Hs   = max(loader.Hs_true, 0.1)
        Tp   = max(loader.Tp_true, 1.0)
        depth = loader.depth

        # Ground-truth wave components for direction supervision.
        # wave_components columns: [A, omega, kx, ky, phi, spread].
        wc = loader.wave_components
        if wc is not None and len(wc) > 0:
            wc = np.asarray(wc, dtype=np.float32)
            true_k = wc[:, 2:4].copy()      # (Nt, 2) = (kx, ky)
            true_w = wc[:, 0].copy() ** 2   # (Nt,) energy weight = A^2
        else:
            true_k = np.zeros((0, 2), dtype=np.float32)
            true_w = np.zeros((0,),   dtype=np.float32)

        gx = loader.gt_grid_x.numpy()   # (Nx,)
        gy = loader.gt_grid_y.numpy()   # (Ny,)
        x_scale = float(gx.max() - gx.min()) or 250.0

        # ── Build W-frame window ──────────────────────────────────────────────
        frames_tokens = np.zeros((W, MAX_PTS, 5), dtype=np.float32)
        frames_mask   = np.ones( (W, MAX_PTS),    dtype=bool)   # True = padding
        frame_ids     = np.arange(W, dtype=np.int64)

        window_obs_xyz = []  # collect all (x,y,t,z) for LS target

        # Window start time → relative-time normalisation. Using absolute t_raw/Tp
        # grows unbounded across an episode (t=200s → t_n=22), so windows at
        # different episode times look incomparable to the network. Relative time
        # keeps the input bounded and generalisable. NOTE: physics rendering (LS
        # fit / dispersion) still uses absolute t via window_obs — this only
        # affects the model's input feature.
        t0 = float(loader[frame_end - W + 1].t)

        # NOTE: horizontal-flip augmentation was removed — the previous version
        # flipped x independently per frame AND left the GT field unflipped,
        # corrupting every augmented sample. Proper reflection augmentation
        # (consistent per-window flip + GT column reversal) is future work.
        for fi, j in enumerate(range(frame_end - W + 1, frame_end + 1)):
            frame = loader[j]
            x_raw = frame.x_obs.numpy()
            y_raw = frame.y_obs.numpy()
            z_raw = frame.z_obs.numpy()
            conf  = frame.confidence.numpy()
            t_raw = float(frame.t)

            M = min(len(x_raw), MAX_PTS)
            if M == 0:
                continue

            # Random subset if too many points
            if len(x_raw) > MAX_PTS:
                idx_sub = np.random.choice(len(x_raw), MAX_PTS, replace=False)
                x_raw = x_raw[idx_sub]; y_raw = y_raw[idx_sub]
                z_raw = z_raw[idx_sub]; conf  = conf[idx_sub]
                M = MAX_PTS

            x_n = x_raw / x_scale
            y_n = y_raw / x_scale
            t_n = (t_raw - t0) / Tp        # relative time within window
            z_n = z_raw / Hs

            frames_tokens[fi, :M, 0] = x_n
            frames_tokens[fi, :M, 1] = y_n
            frames_tokens[fi, :M, 2] = t_n
            frames_tokens[fi, :M, 3] = z_n
            frames_tokens[fi, :M, 4] = conf
            frames_mask[fi, :M] = False  # valid

            window_obs_xyz.extend(zip(x_raw, y_raw, [t_raw]*M, z_raw))

        window_obs_xyz = np.array(window_obs_xyz, dtype=np.float32)  # (M_total, 4)

        # ── Target: GT field at the frame AFTER the window ───────────────────
        target_frame = loader[frame_end + 1]
        gt_field = target_frame.gt_field.numpy()     # (Ny, Nx)
        t_target = float(target_frame.t)

        # Also get current frame observations for LS fitting context
        cur_frame = loader[frame_end]
        cur_x = cur_frame.x_obs.numpy()
        cur_y = cur_frame.y_obs.numpy()
        cur_z = cur_frame.z_obs.numpy()
        cur_t = np.full_like(cur_x, float(cur_frame.t))

        return {
            'frames_tokens': torch.from_numpy(frames_tokens),    # (W, MAX_PTS, 5)
            'frames_mask':   torch.from_numpy(frames_mask),      # (W, MAX_PTS)
            'frame_ids':     torch.from_numpy(frame_ids),        # (W,)
            'gt_field':      torch.from_numpy(gt_field),         # (Ny, Nx)
            't_target':      t_target,
            'window_obs':    torch.from_numpy(window_obs_xyz),   # (M_total, 4)
            'cur_obs_x':     torch.from_numpy(cur_x.astype(np.float32)),
            'cur_obs_y':     torch.from_numpy(cur_y.astype(np.float32)),
            'cur_obs_z':     torch.from_numpy(cur_z.astype(np.float32)),
            'cur_obs_t':     torch.from_numpy(cur_t.astype(np.float32)),
            'gt_grid_x':     torch.from_numpy(gx.astype(np.float32)),
            'gt_grid_y':     torch.from_numpy(gy.astype(np.float32)),
            'true_k':        torch.from_numpy(true_k),            # (Nt, 2) GT (kx, ky)
            'true_w':        torch.from_numpy(true_w),            # (Nt,) energy weight A^2
            'Hs':            Hs,
            'depth':         depth,
            'x_scale':       x_scale,
        }


def collate_fn(batch):
    """Custom collate: variable-length obs are kept as lists."""
    keys_tensor = ['frames_tokens', 'frames_mask', 'frame_ids', 'gt_field']
    keys_scalar = ['t_target', 'Hs', 'depth', 'x_scale']
    keys_varlen = ['window_obs', 'cur_obs_x', 'cur_obs_y', 'cur_obs_z',
                   'cur_obs_t', 'gt_grid_x', 'gt_grid_y', 'true_k', 'true_w']

    out = {}
    for k in keys_tensor:
        out[k] = torch.stack([b[k] for b in batch])
    for k in keys_scalar:
        out[k] = torch.tensor([b[k] for b in batch], dtype=torch.float32)
    for k in keys_varlen:
        out[k] = [b[k] for b in batch]   # list of tensors (variable length)
    return out
