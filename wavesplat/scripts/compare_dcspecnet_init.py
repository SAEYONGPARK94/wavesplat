"""
Compare DCSpecNet init vs beamforming init for WaveSplat.

Evaluates on multiple episodes:
  1. Beamforming init (v2_konly) — current default
  2. DCSpecNet spectral prediction — grid-based model
"""
import sys
import os
import time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dcspecnet.model import DCSpecNet, DCSpecNetConfig
from wavesplat.data_loader import EpisodeDataLoader
from wavesplat.run import run_estimation, DEFAULT_CONFIG
from wavesplat.evaluation.metrics import nrmse as compute_nrmse


def load_dcspecnet(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location='cpu')
    cfg = ckpt['config']
    model = DCSpecNet(cfg)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    return model


def evaluate_dcspecnet_init(model, ep_path):
    """Evaluate DCSpecNet's spectral prediction on an episode's init window."""
    loader = EpisodeDataLoader(ep_path)
    init_frames = loader.get_init_window(tp_factor=2.0)

    # Gather LiDAR points from init frames
    all_x, all_y, all_z = [], [], []
    for f in init_frames:
        if len(f.x_obs) > 0:
            all_x.append(f.x_obs.numpy())
            all_y.append(f.y_obs.numpy())
            all_z.append(f.z_obs.numpy())

    if not all_x:
        return float('nan'), {}

    x = np.concatenate(all_x)
    y = np.concatenate(all_y)
    z = np.concatenate(all_z)

    # Subsample to 100 points
    rng = np.random.default_rng(42)
    if len(x) > 100:
        idx = rng.choice(len(x), 100, replace=False)
        x, y, z = x[idx], y[idx], z[idx]

    pts = torch.tensor(np.stack([x, y, z], axis=-1),
                       dtype=torch.float32).unsqueeze(0)

    with torch.no_grad():
        params = model(pts)
        top = model.top_components(params, n_top=8)

        # Reconstruct on GT grid
        grid_x = loader.gt_grid_x.numpy()
        grid_y = loader.gt_grid_y.numpy()
        gx, gy = np.meshgrid(grid_x, grid_y)
        X_flat = torch.tensor(gx.flatten(), dtype=torch.float32).unsqueeze(0)
        Y_flat = torch.tensor(gy.flatten(), dtype=torch.float32).unsqueeze(0)

        eta = model.reconstruct(params, X_flat, Y_flat, t=0.0)
        pred = eta[0].numpy().reshape(gx.shape)

    gt = init_frames[-1].gt_field.numpy()
    Hs = loader.Hs_true

    nrmse_val = compute_nrmse(pred, gt, Hs)

    # Top component info
    top_info = {
        'kx': top['kx'][0].numpy().tolist(),
        'ky': top['ky'][0].numpy().tolist(),
        'A': top['A'][0].numpy().tolist(),
    }

    return nrmse_val, top_info


def main():
    ckpt_path = 'checkpoints/dcspecnet/best_grid.pt'
    if not os.path.exists(ckpt_path):
        print(f"ERROR: No checkpoint at {ckpt_path}")
        return

    model = load_dcspecnet(ckpt_path)
    print(f"Loaded DCSpecNet: k_grid_n={model.cfg.k_grid_n}, "
          f"k_max={model.cfg.k_max}")

    episodes = [
        'data/episode_BF3_5comp_seed0_slope.h5',
        'data/episode_BF4_5comp_seed0_slope.h5',
        'data/episode_BF4_15comp_seed1_uniform.h5',
        'data/episode_BF5_30comp_seed0_uniform.h5',
    ]

    header = "{:<40} {:>14} {:>14}".format(
        "Episode", "Beamform nRMSE", "DCSpec nRMSE")
    print("\n" + "=" * 70)
    print(header)
    print("=" * 70)

    beam_scores = []
    dc_scores = []

    for ep in episodes:
        ep_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               '..', ep)
        ep_path = os.path.normpath(ep_path)
        name = os.path.basename(ep_path).replace('episode_', '').replace('.h5', '')

        # 1. Beamforming init
        cfg_beam = DEFAULT_CONFIG.copy()
        result_beam = run_estimation(ep_path, cfg_beam, verbose=False)
        nrmses = [f['nrmse'] for f in result_beam['frame_results']
                  if 'nrmse' in f]
        nrmse_beam = np.mean(nrmses) if nrmses else float('nan')
        beam_scores.append(nrmse_beam)

        # 2. DCSpecNet init (spectral reconstruction at t=0)
        nrmse_dc, top_info = evaluate_dcspecnet_init(model, ep_path)
        dc_scores.append(nrmse_dc)

        print("{:<40} {:>14.4f} {:>14.4f}".format(name, nrmse_beam, nrmse_dc))

    print("=" * 70)
    print("{:<40} {:>14.4f} {:>14.4f}".format(
        "AVERAGE", np.nanmean(beam_scores), np.nanmean(dc_scores)))
    print("=" * 70)

    print("\nNote: DCSpec nRMSE is init-only (no tracking/optimization).")
    print("Beamform nRMSE includes full tracking pipeline.")


if __name__ == '__main__':
    main()
