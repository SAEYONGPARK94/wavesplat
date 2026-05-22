#!/usr/bin/env python3
"""
Train and evaluate WaveNP (Neural Process) for ocean wave field estimation.

Usage:
    cd wavesplat && python3 -u scripts/train_wave_np.py
"""

import sys
import os
import time
import json
import glob
import numpy as np
import torch
import torch.optim as optim

sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', buffering=1)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from wavesplat.data_loader import EpisodeDataLoader
from wavesplat.model.wave_np import WaveNP
from wavesplat.evaluation.metrics import nrmse


# ─── Configuration ───────────────────────────────────────────────────────────

TRAIN_EPISODES = [
    # BF3
    'episode_BF3_1comp_seed0_uniform.h5', 'episode_BF3_1comp_seed1_uniform.h5',
    'episode_BF3_2comp_seed0_uniform.h5', 'episode_BF3_2comp_seed1_uniform.h5',
    'episode_BF3_5comp_seed0_uniform.h5', 'episode_BF3_5comp_seed1_uniform.h5',
    'episode_BF3_1comp_seed0_slope.h5', 'episode_BF3_1comp_seed1_slope.h5',
    'episode_BF3_2comp_seed0_slope.h5', 'episode_BF3_2comp_seed1_slope.h5',
    'episode_BF3_5comp_seed0_slope.h5', 'episode_BF3_5comp_seed1_slope.h5',
    # BF4
    'episode_BF4_1comp_seed0_uniform.h5', 'episode_BF4_1comp_seed1_uniform.h5',
    'episode_BF4_5comp_seed0_uniform.h5', 'episode_BF4_5comp_seed1_uniform.h5',
    'episode_BF4_15comp_seed0_uniform.h5', 'episode_BF4_15comp_seed1_uniform.h5',
    'episode_BF4_30comp_seed0_uniform.h5',
    'episode_BF4_1comp_seed0_slope.h5', 'episode_BF4_1comp_seed1_slope.h5',
    'episode_BF4_5comp_seed0_slope.h5', 'episode_BF4_5comp_seed1_slope.h5',
    'episode_BF4_15comp_seed0_slope.h5', 'episode_BF4_15comp_seed1_slope.h5',
    # BF5
    'episode_BF5_15comp_seed0_uniform.h5', 'episode_BF5_15comp_seed1_uniform.h5',
    'episode_BF5_30comp_seed0_uniform.h5', 'episode_BF5_30comp_seed1_uniform.h5',
    'episode_BF5_15comp_seed0_slope.h5', 'episode_BF5_15comp_seed1_slope.h5',
    'episode_BF5_30comp_seed0_slope.h5', 'episode_BF5_30comp_seed1_slope.h5',
]

VAL_EPISODES = [
    'episode_BF3_5comp_seed2_uniform.h5',
    'episode_BF4_5comp_seed2_uniform.h5',
    'episode_BF4_15comp_seed2_uniform.h5',
    'episode_BF5_15comp_seed2_uniform.h5',
]

TEST_EPISODES = [
    'episode_BF3_5comp_seed0_slope.h5',
    'episode_BF4_5comp_seed0_uniform.h5',
    'episode_BF4_15comp_seed0_uniform.h5',
    'episode_BF4_30comp_seed0_uniform.h5',
    'episode_BF5_15comp_seed0_uniform.h5',
    'episode_BF5_30comp_seed0_uniform.h5',
]

EPOCHS = 50
BATCH_SIZE = 4          # frames per optimization step
LR = 3e-4
EMBED_DIM = 128
N_HEADS = 4
N_LAYERS = 4
GRID_NX = 16
GRID_NY = 12
N_FREQ = 8
MAX_TARGET_PTS = 1024   # subsample targets per step (memory)


# ─── Data Loading ────────────────────────────────────────────────────────────

def load_episodes(episode_names, data_dir):
    """Load episodes and extract all frames as (xc, yc, xt, yt) tensors."""
    episodes = []
    for name in episode_names:
        path = os.path.join(data_dir, name)
        if not os.path.exists(path):
            continue
        loader = EpisodeDataLoader(path)
        episodes.append(loader)
    return episodes


def sample_batch(episodes, batch_size, max_target_pts, gt_grid_x, gt_grid_y):
    """Sample a random batch of (context, target) pairs from episodes."""
    xc_list, yc_list, xt_list, yt_list = [], [], [], []

    for _ in range(batch_size):
        ep = episodes[np.random.randint(len(episodes))]
        frame_idx = np.random.randint(len(ep))
        frame = ep[frame_idx]

        # Context: LiDAR observations
        xc = torch.stack([frame.x_obs, frame.y_obs], dim=-1)  # [Nc, 2]
        yc = frame.z_obs.unsqueeze(-1)  # [Nc, 1]

        # Target: GT grid (subsample for memory)
        gx = gt_grid_x
        gy = gt_grid_y
        xx, yy = torch.meshgrid(gx, gy, indexing='ij')
        xt_full = torch.stack([xx.flatten(), yy.flatten()], dim=-1)  # [Nt, 2]
        yt_full = frame.gt_field.T.flatten().unsqueeze(-1)  # [Nt, 1] (transpose: [Ny,Nx] -> [Nx,Ny])

        if max_target_pts < xt_full.shape[0]:
            idx = torch.randperm(xt_full.shape[0])[:max_target_pts]
            xt = xt_full[idx]
            yt = yt_full[idx]
        else:
            xt = xt_full
            yt = yt_full

        xc_list.append(xc)
        yc_list.append(yc)
        xt_list.append(xt)
        yt_list.append(yt)

    # Pad to same size within batch
    max_nc = max(x.shape[0] for x in xc_list)
    max_nt = max(x.shape[0] for x in xt_list)

    xc_batch = torch.zeros(batch_size, max_nc, 2)
    yc_batch = torch.zeros(batch_size, max_nc, 1)
    xt_batch = torch.zeros(batch_size, max_nt, 2)
    yt_batch = torch.zeros(batch_size, max_nt, 1)

    for i in range(batch_size):
        nc = xc_list[i].shape[0]
        nt = xt_list[i].shape[0]
        xc_batch[i, :nc] = xc_list[i]
        yc_batch[i, :nc] = yc_list[i]
        xt_batch[i, :nt] = xt_list[i]
        yt_batch[i, :nt] = yt_list[i]

    return xc_batch, yc_batch, xt_batch, yt_batch


# ─── Evaluation ──────────────────────────────────────────────────────────────

def evaluate(model, episodes, gt_grid_x, gt_grid_y, n_frames=10):
    """Evaluate model on episodes, return avg nRMSE."""
    model.eval()
    all_nrmse = []

    gx = gt_grid_x
    gy = gt_grid_y
    xx, yy = torch.meshgrid(gx, gy, indexing='ij')
    xt = torch.stack([xx.flatten(), yy.flatten()], dim=-1).unsqueeze(0)  # [1, Nt, 2]

    for ep in episodes:
        Hs = ep.Hs_true
        frame_indices = np.linspace(10, len(ep) - 1, n_frames, dtype=int)
        for fi in frame_indices:
            frame = ep[fi]
            xc = torch.stack([frame.x_obs, frame.y_obs], dim=-1).unsqueeze(0)
            yc = frame.z_obs.unsqueeze(-1).unsqueeze(0)

            with torch.no_grad():
                pred = model.predict(xc, yc, xt)  # [1, Nt, 1]

            pred_field = pred[0, :, 0].reshape(len(gx), len(gy)).T.numpy()
            gt_field = frame.gt_field.numpy()

            nrmse_val = nrmse(pred_field, gt_field, Hs)
            if not np.isnan(nrmse_val):
                all_nrmse.append(nrmse_val)

    model.train()
    return float(np.mean(all_nrmse)) if all_nrmse else 1.0


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            'data')

    print("=" * 80)
    print("WaveNP: Neural Process for Ocean Wave Field Estimation")
    print("=" * 80)

    # Load data
    print("\nLoading episodes...")
    train_eps = load_episodes(TRAIN_EPISODES, data_dir)
    val_eps = load_episodes(VAL_EPISODES, data_dir)
    test_eps = load_episodes(TEST_EPISODES, data_dir)
    print(f"  Train: {len(train_eps)} episodes")
    print(f"  Val:   {len(val_eps)} episodes")
    print(f"  Test:  {len(test_eps)} episodes")

    if not train_eps:
        print("ERROR: No training episodes found!")
        return

    # Get grid info from first episode
    gt_grid_x = train_eps[0].gt_grid_x
    gt_grid_y = train_eps[0].gt_grid_y
    x_range = (float(gt_grid_x[0]), float(gt_grid_x[-1]))
    y_range = (float(gt_grid_y[0]), float(gt_grid_y[-1]))

    # Build model
    model = WaveNP(
        embed_dim=EMBED_DIM,
        n_heads=N_HEADS,
        n_layers=N_LAYERS,
        grid_nx=GRID_NX,
        grid_ny=GRID_NY,
        x_range=x_range,
        y_range=y_range,
        n_freq=N_FREQ,
    )
    print(f"\nModel: {model.n_params:,} parameters")
    print(f"  Grid: {GRID_NX}x{GRID_NY} = {GRID_NX*GRID_NY} pseudo-tokens")
    print(f"  Embed dim: {EMBED_DIM}, Heads: {N_HEADS}, Layers: {N_LAYERS}")

    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    # Training
    print(f"\nTraining for {EPOCHS} epochs...")
    steps_per_epoch = 100
    best_val = 1.0
    best_epoch = 0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        epoch_loss = 0.0

        for step in range(steps_per_epoch):
            xc, yc, xt, yt = sample_batch(
                train_eps, BATCH_SIZE, MAX_TARGET_PTS, gt_grid_x, gt_grid_y)

            dist = model(xc, yc, xt)
            nll = -dist.log_prob(yt).mean()

            optimizer.zero_grad()
            nll.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += nll.item()

        scheduler.step()
        avg_loss = epoch_loss / steps_per_epoch

        # Validate every 5 epochs
        if epoch % 5 == 0 or epoch == 1:
            val_nrmse = evaluate(model, val_eps, gt_grid_x, gt_grid_y)
            improved = ""
            if val_nrmse < best_val:
                best_val = val_nrmse
                best_epoch = epoch
                torch.save(model.state_dict(), os.path.join(data_dir, '..', 'checkpoints', 'wave_np_best.pt'))
                improved = " *"
            print(f"  Epoch {epoch:3d}/{EPOCHS}: loss={avg_loss:.4f}, "
                  f"val_nRMSE={val_nrmse:.4f}, lr={scheduler.get_last_lr()[0]:.6f}{improved}")
        else:
            print(f"  Epoch {epoch:3d}/{EPOCHS}: loss={avg_loss:.4f}, "
                  f"lr={scheduler.get_last_lr()[0]:.6f}")

    print(f"\nBest val nRMSE: {best_val:.4f} at epoch {best_epoch}")

    # Test evaluation
    print("\n" + "=" * 80)
    print("Test Evaluation")
    print("=" * 80)

    # Load best model
    ckpt_path = os.path.join(data_dir, '..', 'checkpoints', 'wave_np_best.pt')
    if os.path.exists(ckpt_path):
        model.load_state_dict(torch.load(ckpt_path, map_location='cpu'))

    model.eval()

    gx = gt_grid_x
    gy = gt_grid_y
    xx, yy = torch.meshgrid(gx, gy, indexing='ij')
    xt = torch.stack([xx.flatten(), yy.flatten()], dim=-1).unsqueeze(0)

    results = []
    for ep in test_eps:
        ep_nrmses = []
        n_eval = min(20, len(ep))
        frame_indices = np.linspace(10, len(ep) - 1, n_eval, dtype=int)

        for fi in frame_indices:
            frame = ep[fi]
            xc = torch.stack([frame.x_obs, frame.y_obs], dim=-1).unsqueeze(0)
            yc = frame.z_obs.unsqueeze(-1).unsqueeze(0)

            with torch.no_grad():
                pred = model.predict(xc, yc, xt)
            pred_field = pred[0, :, 0].reshape(len(gx), len(gy)).T.numpy()
            gt_field = frame.gt_field.numpy()
            ep_nrmses.append(nrmse(pred_field, gt_field, ep.Hs_true))

        avg = float(np.mean(ep_nrmses))
        ep_name = os.path.basename(getattr(ep, 'data', {}).get('h5_path', 'unknown'))
        results.append(avg)
        print(f"  {avg:.4f}")

    print(f"\n  Average test nRMSE: {np.mean(results):.4f}")
    print(f"  (WaveSplat reference: ~0.22)")

    # Save results
    results_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                'results')
    os.makedirs(results_dir, exist_ok=True)
    with open(os.path.join(results_dir, 'wave_np_eval.json'), 'w') as f:
        json.dump({
            'best_val_nRMSE': best_val,
            'best_epoch': best_epoch,
            'test_nRMSEs': results,
            'test_avg_nRMSE': float(np.mean(results)),
            'n_params': model.n_params,
            'config': {
                'embed_dim': EMBED_DIM, 'n_heads': N_HEADS,
                'n_layers': N_LAYERS, 'grid_nx': GRID_NX, 'grid_ny': GRID_NY,
            },
        }, f, indent=2)


if __name__ == '__main__':
    os.makedirs(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                              'checkpoints'), exist_ok=True)
    main()
