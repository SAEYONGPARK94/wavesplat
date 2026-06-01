#!/usr/bin/env python3
"""
Generate training data for USGO (Unrolled Spectral Gaussian Optimizer).

For each episode, extracts:
  - Sparse LiDAR observations (input)
  - GT spectral parameters (label)
  - Beamforming init output (warm-start)

Usage:
    python3 -m scripts.generate_usgo_data --output data/usgo_train.npz
    python3 -m scripts.generate_usgo_data --output data/usgo_train.npz --max_episodes 20
"""

import numpy as np
import torch
import sys
import os
import argparse
import glob
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from wavesim.data_writer import load_episode
from wavesplat.model.dispersion import dispersion_omega
from wavesplat.optimizer.initializer import initialize_from_frames
from wavesplat.data_loader import EpisodeDataLoader, FrameData


def extract_episode_samples(h5_path: str, n_init: int = 15,
                            frames_per_episode: int = 10,
                            depth_default: float = 50.0) -> list:
    """Extract training samples from one episode.

    Each sample: {
        'obs_x', 'obs_y', 'obs_z', 'obs_conf': sparse observations (padded),
        'n_obs': number of valid observations,
        'gt_kx', 'gt_ky', 'gt_w', 'gt_phi': GT spectral params,
        'gt_omega': GT angular frequencies,
        'n_comp': number of GT components,
        'init_kx', 'init_ky', 'init_w', 'init_phi': beamforming init,
        't': frame time,
        'depth': water depth,
        'Hs': significant wave height,
        'sea_state': Beaufort scale,
    }
    """
    try:
        loader = EpisodeDataLoader(h5_path, device='cpu')
    except Exception as e:
        print(f"  Skip {h5_path}: {e}")
        return []

    depth = loader.depth if loader.depth else depth_default
    wave_comp = loader.wave_components
    if wave_comp is None:
        print(f"  Skip {h5_path}: no wave_components")
        return []

    # GT params (constant for the episode — Gerstner wave components)
    gt_A = wave_comp[:, 0]      # amplitude
    gt_omega = wave_comp[:, 1]  # angular frequency
    gt_kx = wave_comp[:, 2]     # wavenumber x
    gt_ky = wave_comp[:, 3]     # wavenumber y
    gt_phi0 = wave_comp[:, 4]   # initial phase
    n_comp = len(gt_A)

    # Get init window frames
    init_frames = loader.get_init_window(tp_factor=2.0)
    if len(init_frames) < 5:
        print(f"  Skip {h5_path}: too few init frames ({len(init_frames)})")
        return []

    x_range = (float(loader.gt_grid_x[0]), float(loader.gt_grid_x[-1]))
    y_range = (float(loader.gt_grid_y[0]), float(loader.gt_grid_y[-1]))

    # Beamforming init (v2_konly)
    try:
        init_kx, init_ky, init_w, init_phi = initialize_from_frames(
            init_frames, n_init, depth, x_range, y_range,
            init_method='v2_konly', seed=42
        )
    except Exception as e:
        print(f"  Skip {h5_path}: init failed: {e}")
        return []

    # Sample frames evenly across tracking phase
    tracking_frames = loader.get_tracking_frames(tp_factor=2.0)
    if len(tracking_frames) < frames_per_episode:
        indices = list(range(len(tracking_frames)))
    else:
        indices = np.linspace(0, len(tracking_frames) - 1,
                              frames_per_episode, dtype=int).tolist()

    samples = []
    for idx in indices:
        frame = tracking_frames[idx]
        if len(frame.x_obs) < 10:
            continue

        # GT phase at this time: phi(t) = phi0 - omega * t (dispersion)
        gt_phi_t = gt_phi0 - gt_omega * frame.t

        samples.append({
            'obs_x': frame.x_obs.numpy(),
            'obs_y': frame.y_obs.numpy(),
            'obs_z': frame.z_obs.numpy(),
            'obs_conf': frame.confidence.numpy(),
            'n_obs': len(frame.x_obs),
            'gt_kx': gt_kx.copy(),
            'gt_ky': gt_ky.copy(),
            'gt_w': gt_A.copy(),  # amplitude = weight
            'gt_phi': gt_phi_t.copy(),
            'gt_omega': gt_omega.copy(),
            'n_comp': n_comp,
            'init_kx': init_kx.numpy().copy(),
            'init_ky': init_ky.numpy().copy(),
            'init_w': init_w.numpy().copy(),
            'init_phi': init_phi.numpy().copy(),
            'n_init': n_init,
            't': frame.t,
            'depth': depth,
            'Hs': loader.Hs_true,
            'sea_state': loader.sea_state,
        })

    return samples


def pad_to_max(arrays: list, max_len: int) -> np.ndarray:
    """Pad variable-length arrays to max_len, stack into (N, max_len)."""
    result = np.zeros((len(arrays), max_len), dtype=np.float32)
    for i, arr in enumerate(arrays):
        n = min(len(arr), max_len)
        result[i, :n] = arr[:n]
    return result


def main():
    parser = argparse.ArgumentParser(description='Generate USGO training data')
    parser.add_argument('--output', type=str, default='data/usgo_train.npz')
    parser.add_argument('--data_dir', type=str, default='data')
    parser.add_argument('--n_init', type=int, default=15,
                        help='Number of init components for beamforming')
    parser.add_argument('--frames_per_episode', type=int, default=10)
    parser.add_argument('--max_episodes', type=int, default=None)
    parser.add_argument('--max_obs', type=int, default=200,
                        help='Max observations per frame (pad/truncate)')
    parser.add_argument('--max_comp', type=int, default=30,
                        help='Max GT components (pad)')
    args = parser.parse_args()

    # Find all episodes
    pattern = os.path.join(args.data_dir, 'episode_BF*.h5')
    episodes = sorted(glob.glob(pattern))
    if args.max_episodes:
        episodes = episodes[:args.max_episodes]
    print(f"Found {len(episodes)} episodes")

    all_samples = []
    t0 = time.time()

    for i, ep_path in enumerate(episodes):
        ep_name = os.path.basename(ep_path)
        samples = extract_episode_samples(
            ep_path, n_init=args.n_init,
            frames_per_episode=args.frames_per_episode
        )
        all_samples.extend(samples)
        if (i + 1) % 10 == 0 or i == 0:
            print(f"  [{i+1}/{len(episodes)}] {ep_name}: "
                  f"{len(samples)} samples (total: {len(all_samples)})")

    elapsed = time.time() - t0
    print(f"\nTotal: {len(all_samples)} samples from {len(episodes)} episodes "
          f"({elapsed:.1f}s)")

    if not all_samples:
        print("No samples generated!")
        return

    # Pack into arrays
    N = len(all_samples)
    max_obs = args.max_obs
    max_comp = args.max_comp
    n_init = args.n_init

    obs_x = pad_to_max([s['obs_x'] for s in all_samples], max_obs)
    obs_y = pad_to_max([s['obs_y'] for s in all_samples], max_obs)
    obs_z = pad_to_max([s['obs_z'] for s in all_samples], max_obs)
    obs_conf = pad_to_max([s['obs_conf'] for s in all_samples], max_obs)
    n_obs = np.array([s['n_obs'] for s in all_samples], dtype=np.int32)

    gt_kx = pad_to_max([s['gt_kx'] for s in all_samples], max_comp)
    gt_ky = pad_to_max([s['gt_ky'] for s in all_samples], max_comp)
    gt_w = pad_to_max([s['gt_w'] for s in all_samples], max_comp)
    gt_phi = pad_to_max([s['gt_phi'] for s in all_samples], max_comp)
    gt_omega = pad_to_max([s['gt_omega'] for s in all_samples], max_comp)
    n_comp = np.array([s['n_comp'] for s in all_samples], dtype=np.int32)

    init_kx = np.stack([s['init_kx'] for s in all_samples])
    init_ky = np.stack([s['init_ky'] for s in all_samples])
    init_w = np.stack([s['init_w'] for s in all_samples])
    init_phi = np.stack([s['init_phi'] for s in all_samples])

    t_arr = np.array([s['t'] for s in all_samples], dtype=np.float32)
    depth_arr = np.array([s['depth'] for s in all_samples], dtype=np.float32)
    Hs_arr = np.array([s['Hs'] for s in all_samples], dtype=np.float32)

    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    np.savez_compressed(args.output,
                        obs_x=obs_x, obs_y=obs_y, obs_z=obs_z,
                        obs_conf=obs_conf, n_obs=n_obs,
                        gt_kx=gt_kx, gt_ky=gt_ky, gt_w=gt_w,
                        gt_phi=gt_phi, gt_omega=gt_omega, n_comp=n_comp,
                        init_kx=init_kx, init_ky=init_ky,
                        init_w=init_w, init_phi=init_phi,
                        t=t_arr, depth=depth_arr, Hs=Hs_arr,
                        n_init=n_init, max_obs=max_obs, max_comp=max_comp)

    size_mb = os.path.getsize(args.output) / 1e6
    print(f"\nSaved {args.output} ({size_mb:.1f} MB)")
    print(f"  Samples: {N}")
    print(f"  Obs range: {n_obs.min()}-{n_obs.max()}")
    print(f"  Comp range: {n_comp.min()}-{n_comp.max()}")
    print(f"  Hs range: {Hs_arr.min():.2f}-{Hs_arr.max():.2f}")


if __name__ == '__main__':
    main()
