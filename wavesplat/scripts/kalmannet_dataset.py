"""
kalmannet_dataset.py

WaveSplat HDF5 episode → tracking sequence batch (KalmanNet 용).

Pipeline:
  1. Load episode HDF5
  2. Run WaveSplat init (existing v2_konly) on first ~Tp*tp_factor seconds
  3. Extract subsequent tracking frames (LiDAR obs + GT)
  4. Pack as PyTorch sequence batch

Output sequence dict (per task):
  init_state:    {kx, ky, w0, phi0}  — fixed from init
  frames:        list of dict per frame:
                   {dt, x_obs, y_obs, eta_obs, eta_gt_at_obs}
  meta:          episode tag, BF, N, etc.

Stage 1 sanity-only: single episode, ~20 tracking frames.
"""
import os
import sys
from typing import Dict, List, Optional

import h5py
import numpy as np
import torch

# Make wavesplat importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from wavesplat.data_loader import EpisodeDataLoader, FrameData  # noqa: E402
from wavesplat.optimizer.sequential_update import SequentialUpdater  # noqa: E402
from wavesplat.run import set_seed  # noqa: E402


def build_sequence_from_episode(
    h5_path: str,
    tp_factor: float = 2.0,
    n_tracking_frames: int = 20,
    init_config: Optional[Dict] = None,
    seed: int = 42,
) -> Dict:
    """Run WaveSplat init then extract a tracking sequence.

    Returns dict with:
      init_state: (kx, ky, w_init, phi_init) tensors — k fixed during tracking
      frames:     list of dicts with (dt, x_obs, y_obs, eta_obs, t)
      gt_frames:  list of GT FrameData (for evaluation)
      meta:       episode tag, depth, n_components
    """
    set_seed(seed)
    loader = EpisodeDataLoader(h5_path)

    # ── 1) Init ──
    cfg = dict(init_config or {})
    cfg.setdefault('init_method', 'v2_konly')
    cfg.setdefault('use_dispersion', True)
    cfg.setdefault('use_confidence', True)
    cfg.setdefault('adaptive_n', True)
    cfg.setdefault('tracking_window', 3)
    cfg.setdefault('seed', seed)
    cfg['depth'] = loader.depth

    updater = SequentialUpdater(cfg)
    init_frames = loader.get_init_window(tp_factor=tp_factor)
    updater.initialize(
        init_frames,
        x_range=(float(loader.gt_grid_x[0]), float(loader.gt_grid_x[-1])),
        y_range=(float(loader.gt_grid_y[0]), float(loader.gt_grid_y[-1])),
    )

    # Snapshot init state
    with torch.no_grad():
        init_state = {
            'kx':  updater.model.kx.detach().clone(),
            'ky':  updater.model.ky.detach().clone(),
            'w':   updater.model.w.detach().clone(),
            'phi': updater.model.phi.detach().clone(),
        }

    # ── 2) Tracking frames ──
    tracking_frames = loader.get_tracking_frames(tp_factor=tp_factor)
    sequence = []
    gt_frames = []
    t_prev = float(init_frames[-1].t) if len(init_frames) > 0 else 0.0

    for i, f in enumerate(tracking_frames[:n_tracking_frames]):
        dt = float(f.t) - t_prev
        t_prev = float(f.t)
        if len(f.x_obs) == 0:
            continue
        sequence.append({
            't':       float(f.t),
            'dt':      dt,
            'x_obs':   f.x_obs.detach().clone(),
            'y_obs':   f.y_obs.detach().clone(),
            'eta_obs': f.z_obs.detach().clone(),
        })
        gt_frames.append(f)

    return {
        'init_state': init_state,
        'frames':     sequence,
        'gt_frames':  gt_frames,
        'meta': {
            'episode': os.path.basename(h5_path),
            'depth':   float(loader.depth),
            'n_init':  init_state['kx'].shape[0],
            'Tp':      float(loader.Tp_true),
            'Hs':      float(loader.Hs_true),
        },
    }


def forward_lidar(
    kx: torch.Tensor, ky: torch.Tensor, w: torch.Tensor, phi: torch.Tensor,
    omega: torch.Tensor,
    x_obs: torch.Tensor, y_obs: torch.Tensor, t: float,
) -> torch.Tensor:
    """Linear sinusoidal forward model at LiDAR points.

    η(x, y, t) = Σ_n w_n · cos(kx_n·x + ky_n·y - ω_n·t + φ_n)

    Args:
        kx, ky, w, phi, omega: (N,) tensors
        x_obs, y_obs:          (M,) tensors
        t:                     scalar
    Returns:
        eta_pred: (M,) tensor
    """
    # (M, 1) broadcast against (1, N)
    x = x_obs[:, None]; y = y_obs[:, None]
    arg = kx[None, :] * x + ky[None, :] * y - omega[None, :] * t + phi[None, :]
    return (w[None, :] * torch.cos(arg)).sum(dim=-1)


if __name__ == '__main__':
    # Smoke test
    DATA = os.path.join(os.path.dirname(__file__), '..', 'data',
                        'episode_BF3_2comp_seed0_uniform.h5')
    seq = build_sequence_from_episode(
        DATA, tp_factor=2.0, n_tracking_frames=10
    )
    print(f"Episode: {seq['meta']['episode']}")
    print(f"  Hs={seq['meta']['Hs']:.2f} m, Tp={seq['meta']['Tp']:.2f} s")
    print(f"  Init N components: {seq['meta']['n_init']}")
    print(f"  Tracking frames extracted: {len(seq['frames'])}")
    print()
    print(f"Init state shapes:")
    for k, v in seq['init_state'].items():
        print(f"  {k}: {tuple(v.shape)}  ({v.dtype})")
    print()
    print(f"First frame:")
    f0 = seq['frames'][0]
    print(f"  t={f0['t']:.3f}, dt={f0['dt']:.3f}")
    print(f"  n_obs: {f0['x_obs'].shape[0]}")
    print(f"  eta_obs range: [{f0['eta_obs'].min():.3f}, {f0['eta_obs'].max():.3f}]")

    # Verify forward_lidar
    from wavesplat.model.dispersion import dispersion_omega
    k_mag = torch.sqrt(seq['init_state']['kx']**2 + seq['init_state']['ky']**2 + 1e-12)
    omega = dispersion_omega(k_mag, seq['meta']['depth'])
    eta_pred = forward_lidar(
        seq['init_state']['kx'], seq['init_state']['ky'],
        seq['init_state']['w'], seq['init_state']['phi'],
        omega, f0['x_obs'], f0['y_obs'], f0['t'],
    )
    residual = f0['eta_obs'] - eta_pred
    print(f"\nForward LiDAR test (init state):")
    print(f"  eta_pred range: [{eta_pred.min():.3f}, {eta_pred.max():.3f}]")
    print(f"  residual RMS: {(residual**2).mean().sqrt().item():.4f}")
