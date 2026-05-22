#!/usr/bin/env python3
"""
Train WaveNP v2: dispersion-constrained Neural Process for ocean wave estimation.

v2 improvements over v1:
  1. Dispersion spectral consistency loss (physics constraint)
  2. Multi-frame temporal context (3 frames)
  3. Longer training (200 epochs) with warmup
  4. Data augmentation (y-flip, noise injection)

Usage:
    cd wavesplat && python3 -u scripts/train_wave_np_v2.py
"""

import sys
import os
import time
import json
import math
import numpy as np
import torch
import torch.optim as optim

sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', buffering=1)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from wavesplat.data_loader import EpisodeDataLoader
from wavesplat.model.wave_np import (
    WaveNP, WaveNPMultiFrame,
    dispersion_spectral_loss, dispersion_gradient_loss,
)
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

# Training config
EPOCHS = 200
BATCH_SIZE = 4
LR = 3e-4
WARMUP_EPOCHS = 10

# Model config
EMBED_DIM = 128
N_HEADS = 4
N_LAYERS = 6          # v2: deeper (was 4)
GRID_NX = 16
GRID_NY = 12
N_FREQ = 8
N_CONTEXT_FRAMES = 3  # v2: multi-frame

# Physics loss config
LAMBDA_DISP = 0.1     # dispersion spectral loss weight
LAMBDA_GRAD = 0.01    # gradient smoothness loss weight
DEPTH = 50.0          # water depth [m]
K_MIN = 0.02          # min plausible wavenumber [rad/m]
K_MAX = 0.5           # max plausible wavenumber [rad/m]

# Data config
MAX_TARGET_PTS = 1024


# ─── Data Loading ────────────────────────────────────────────────────────────

def load_episodes(episode_names, data_dir):
    episodes = []
    for name in episode_names:
        path = os.path.join(data_dir, name)
        if not os.path.exists(path):
            continue
        loader = EpisodeDataLoader(path)
        episodes.append(loader)
    return episodes


def sample_batch_multiframe(episodes, batch_size, max_target_pts,
                            gt_grid_x, gt_grid_y, n_context_frames=3):
    """Sample batch with multi-frame context for temporal coherence.

    For each sample, picks n_context_frames consecutive frames.
    The last frame is the "current" frame whose GT is the target.
    """
    gx = gt_grid_x
    gy = gt_grid_y
    Nx, Ny = len(gx), len(gy)
    xx, yy = torch.meshgrid(gx, gy, indexing='ij')
    xt_full = torch.stack([xx.flatten(), yy.flatten()], dim=-1)  # [Nt, 2]

    # Per-frame lists (will be padded)
    xc_frames_list = [[] for _ in range(n_context_frames)]
    yc_frames_list = [[] for _ in range(n_context_frames)]
    dt_list = [[] for _ in range(n_context_frames)]
    xt_list, yt_list = [], []
    gt_field_list = []  # for physics loss (full grid)

    for _ in range(batch_size):
        ep = episodes[np.random.randint(len(episodes))]
        # Ensure enough frames for context window
        min_idx = n_context_frames - 1
        frame_idx = np.random.randint(min_idx, len(ep))

        # Get consecutive frames
        for f in range(n_context_frames):
            ctx_idx = frame_idx - (n_context_frames - 1 - f)
            frame = ep[ctx_idx]
            xc = torch.stack([frame.x_obs, frame.y_obs], dim=-1)
            yc = frame.z_obs.unsqueeze(-1)

            # Data augmentation: y-flip with 50% probability
            if np.random.random() < 0.5:
                xc = xc.clone()
                xc[:, 1] = -xc[:, 1]

            xc_frames_list[f].append(xc)
            yc_frames_list[f].append(yc)

            # Relative time offset (0 = current frame)
            dt = float(ctx_idx - frame_idx)
            dt_list[f].append(dt)

        # Target: current (last) frame's GT
        current_frame = ep[frame_idx]
        gt_field = current_frame.gt_field  # [Ny, Nx]
        yt_full = gt_field.T.flatten().unsqueeze(-1)  # [Nx*Ny, 1]
        gt_field_list.append(gt_field.T)  # [Nx, Ny] for physics loss

        # Subsample targets
        if max_target_pts < xt_full.shape[0]:
            idx = torch.randperm(xt_full.shape[0])[:max_target_pts]
            xt_list.append(xt_full[idx])
            yt_list.append(yt_full[idx])
        else:
            xt_list.append(xt_full)
            yt_list.append(yt_full)

    # Pad context frames
    xc_batches, yc_batches, dt_batches = [], [], []
    for f in range(n_context_frames):
        max_nc = max(x.shape[0] for x in xc_frames_list[f])
        xc_batch = torch.zeros(batch_size, max_nc, 2)
        yc_batch = torch.zeros(batch_size, max_nc, 1)
        dt_batch = torch.zeros(batch_size, 1)
        for i in range(batch_size):
            nc = xc_frames_list[f][i].shape[0]
            xc_batch[i, :nc] = xc_frames_list[f][i]
            yc_batch[i, :nc] = yc_frames_list[f][i]
            dt_batch[i, 0] = dt_list[f][i]
        xc_batches.append(xc_batch)
        yc_batches.append(yc_batch)
        dt_batches.append(dt_batch)

    # Pad targets
    max_nt = max(x.shape[0] for x in xt_list)
    xt_batch = torch.zeros(batch_size, max_nt, 2)
    yt_batch = torch.zeros(batch_size, max_nt, 1)
    for i in range(batch_size):
        nt = xt_list[i].shape[0]
        xt_batch[i, :nt] = xt_list[i]
        yt_batch[i, :nt] = yt_list[i]

    # GT fields for physics loss [B, Nx, Ny]
    gt_fields = torch.stack(gt_field_list, dim=0)

    return xc_batches, yc_batches, dt_batches, xt_batch, yt_batch, gt_fields


def sample_batch_single(episodes, batch_size, max_target_pts, gt_grid_x, gt_grid_y):
    """Single-frame batch (for base WaveNP without multi-frame)."""
    xc_list, yc_list, xt_list, yt_list = [], [], [], []
    gt_field_list = []

    gx = gt_grid_x
    gy = gt_grid_y
    xx, yy = torch.meshgrid(gx, gy, indexing='ij')
    xt_full = torch.stack([xx.flatten(), yy.flatten()], dim=-1)

    for _ in range(batch_size):
        ep = episodes[np.random.randint(len(episodes))]
        frame_idx = np.random.randint(len(ep))
        frame = ep[frame_idx]

        xc = torch.stack([frame.x_obs, frame.y_obs], dim=-1)
        yc = frame.z_obs.unsqueeze(-1)

        # Data augmentation: y-flip
        if np.random.random() < 0.5:
            xc = xc.clone()
            xc[:, 1] = -xc[:, 1]

        gt_field = frame.gt_field  # [Ny, Nx]
        yt_full = gt_field.T.flatten().unsqueeze(-1)
        gt_field_list.append(gt_field.T)  # [Nx, Ny]

        if max_target_pts < xt_full.shape[0]:
            idx = torch.randperm(xt_full.shape[0])[:max_target_pts]
            xt_list.append(xt_full[idx])
            yt_list.append(yt_full[idx])
        else:
            xt_list.append(xt_full)
            yt_list.append(yt_full)

        xc_list.append(xc)
        yc_list.append(yc)

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

    gt_fields = torch.stack(gt_field_list, dim=0)
    return xc_batch, yc_batch, xt_batch, yt_batch, gt_fields


# ─── Evaluation ──────────────────────────────────────────────────────────────

def evaluate(model, episodes, gt_grid_x, gt_grid_y, n_frames=10,
             multi_frame=False, n_context_frames=3):
    """Evaluate model on episodes, return avg nRMSE."""
    model.eval()
    all_nrmse = []

    gx = gt_grid_x
    gy = gt_grid_y
    xx, yy = torch.meshgrid(gx, gy, indexing='ij')
    xt = torch.stack([xx.flatten(), yy.flatten()], dim=-1).unsqueeze(0)

    for ep in episodes:
        Hs = ep.Hs_true
        frame_indices = np.linspace(
            max(10, n_context_frames), len(ep) - 1, n_frames, dtype=int)

        for fi in frame_indices:
            if multi_frame:
                # Build multi-frame context
                xc_frames, yc_frames, dt_frames = [], [], []
                for f_offset in range(n_context_frames):
                    ctx_idx = fi - (n_context_frames - 1 - f_offset)
                    ctx_idx = max(0, ctx_idx)
                    frame = ep[ctx_idx]
                    xc = torch.stack([frame.x_obs, frame.y_obs], dim=-1).unsqueeze(0)
                    yc = frame.z_obs.unsqueeze(-1).unsqueeze(0)
                    dt = torch.tensor([[float(ctx_idx - fi)]])
                    xc_frames.append(xc)
                    yc_frames.append(yc)
                    dt_frames.append(dt)

                with torch.no_grad():
                    pred = model.predict(xc_frames, yc_frames, dt_frames, xt)
            else:
                frame = ep[fi]
                xc = torch.stack([frame.x_obs, frame.y_obs], dim=-1).unsqueeze(0)
                yc = frame.z_obs.unsqueeze(-1).unsqueeze(0)

                with torch.no_grad():
                    pred = model.predict(xc, yc, xt)

            pred_field = pred[0, :, 0].reshape(len(gx), len(gy)).T.numpy()
            gt_field = ep[fi].gt_field.numpy()

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
    print("WaveNP v2: Dispersion-Constrained Neural Process")
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

    # Get grid info
    gt_grid_x = train_eps[0].gt_grid_x
    gt_grid_y = train_eps[0].gt_grid_y
    x_range = (float(gt_grid_x[0]), float(gt_grid_x[-1]))
    y_range = (float(gt_grid_y[0]), float(gt_grid_y[-1]))
    Nx, Ny = len(gt_grid_x), len(gt_grid_y)
    dx = float(gt_grid_x[1] - gt_grid_x[0])
    dy = float(gt_grid_y[1] - gt_grid_y[0])

    print(f"\n  Grid: {Nx}x{Ny}, dx={dx:.1f}m, dy={dy:.1f}m")
    print(f"  Depth: {DEPTH}m")
    print(f"  k-band: [{K_MIN}, {K_MAX}] rad/m")

    # Build model
    use_multi_frame = N_CONTEXT_FRAMES > 1

    base_model = WaveNP(
        embed_dim=EMBED_DIM,
        n_heads=N_HEADS,
        n_layers=N_LAYERS,
        grid_nx=GRID_NX,
        grid_ny=GRID_NY,
        x_range=x_range,
        y_range=y_range,
        n_freq=N_FREQ,
    )

    if use_multi_frame:
        model = WaveNPMultiFrame(base_model, n_context_frames=N_CONTEXT_FRAMES)
        print(f"\n  Model: WaveNP v2 (multi-frame, {N_CONTEXT_FRAMES} frames)")
    else:
        model = base_model
        print(f"\n  Model: WaveNP v2 (single-frame)")

    print(f"  Parameters: {model.n_params:,}")
    print(f"  Grid: {GRID_NX}x{GRID_NY} = {GRID_NX*GRID_NY} pseudo-tokens")
    print(f"  Embed dim: {EMBED_DIM}, Heads: {N_HEADS}, Layers: {N_LAYERS}")
    print(f"  Physics loss: λ_disp={LAMBDA_DISP}, λ_grad={LAMBDA_GRAD}")

    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    # Training
    print(f"\nTraining for {EPOCHS} epochs (warmup: {WARMUP_EPOCHS})...")
    steps_per_epoch = 100
    best_val = 1.0
    best_epoch = 0
    t_start = time.time()

    # For full-grid prediction (physics loss)
    xx, yy = torch.meshgrid(gt_grid_x, gt_grid_y, indexing='ij')
    xt_full = torch.stack([xx.flatten(), yy.flatten()], dim=-1)  # [Nx*Ny, 2]

    for epoch in range(1, EPOCHS + 1):
        model.train()
        epoch_nll = 0.0
        epoch_disp = 0.0
        epoch_grad = 0.0

        # Warmup: ramp up physics loss weight
        warmup_factor = min(1.0, epoch / WARMUP_EPOCHS)
        lambda_disp = LAMBDA_DISP * warmup_factor
        lambda_grad = LAMBDA_GRAD * warmup_factor

        for step in range(steps_per_epoch):
            if use_multi_frame:
                xc_batches, yc_batches, dt_batches, xt, yt, gt_fields = \
                    sample_batch_multiframe(
                        train_eps, BATCH_SIZE, MAX_TARGET_PTS,
                        gt_grid_x, gt_grid_y, N_CONTEXT_FRAMES)

                dist = model(xc_batches, yc_batches, dt_batches, xt)
            else:
                xc, yc, xt, yt, gt_fields = sample_batch_single(
                    train_eps, BATCH_SIZE, MAX_TARGET_PTS,
                    gt_grid_x, gt_grid_y)

                dist = model(xc, yc, xt)

            # NLL loss (primary)
            nll = -dist.log_prob(yt).mean()

            # Physics losses on full-grid prediction
            # Use model's mean prediction on full grid for physics loss
            B = gt_fields.shape[0]
            xt_full_batch = xt_full.unsqueeze(0).expand(B, -1, -1)

            if use_multi_frame:
                dist_full = model(xc_batches, yc_batches, dt_batches, xt_full_batch)
            else:
                dist_full = model(xc, yc, xt_full_batch)

            pred_full = dist_full.mean  # [B, Nx*Ny, 1]
            pred_field_2d = pred_full[:, :, 0].reshape(B, Nx, Ny)

            # Dispersion spectral consistency loss
            disp_loss = dispersion_spectral_loss(
                pred_field_2d, dx, dy, depth=DEPTH,
                k_min=K_MIN, k_max=K_MAX)

            # Gradient smoothness loss
            grad_loss = dispersion_gradient_loss(pred_field_2d, dx, dy, depth=DEPTH)

            # Total loss
            loss = nll + lambda_disp * disp_loss + lambda_grad * grad_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_nll += nll.item()
            epoch_disp += disp_loss.item()
            epoch_grad += grad_loss.item()

        scheduler.step()
        avg_nll = epoch_nll / steps_per_epoch
        avg_disp = epoch_disp / steps_per_epoch
        avg_grad = epoch_grad / steps_per_epoch

        # Validate every 10 epochs (more frequent early on)
        val_interval = 5 if epoch <= 50 else 10
        if epoch % val_interval == 0 or epoch == 1:
            val_nrmse = evaluate(model, val_eps, gt_grid_x, gt_grid_y,
                                 multi_frame=use_multi_frame,
                                 n_context_frames=N_CONTEXT_FRAMES)
            improved = ""
            if val_nrmse < best_val:
                best_val = val_nrmse
                best_epoch = epoch
                ckpt_dir = os.path.join(data_dir, '..', 'checkpoints')
                os.makedirs(ckpt_dir, exist_ok=True)
                torch.save(model.state_dict(),
                           os.path.join(ckpt_dir, 'wave_np_v2_best.pt'))
                improved = " *"

            elapsed = time.time() - t_start
            print(f"  Epoch {epoch:3d}/{EPOCHS}: nll={avg_nll:.4f}, "
                  f"disp={avg_disp:.4f}, grad={avg_grad:.4f}, "
                  f"val_nRMSE={val_nrmse:.4f}, "
                  f"lr={scheduler.get_last_lr()[0]:.6f} "
                  f"[{elapsed:.0f}s]{improved}")
        elif epoch % 20 == 0:
            elapsed = time.time() - t_start
            print(f"  Epoch {epoch:3d}/{EPOCHS}: nll={avg_nll:.4f}, "
                  f"disp={avg_disp:.4f}, grad={avg_grad:.4f}, "
                  f"lr={scheduler.get_last_lr()[0]:.6f} [{elapsed:.0f}s]")

    print(f"\nBest val nRMSE: {best_val:.4f} at epoch {best_epoch}")
    print(f"Total training time: {time.time() - t_start:.0f}s")

    # ─── Test Evaluation ─────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("Test Evaluation (WaveNP v2)")
    print("=" * 80)

    # Load best model
    ckpt_path = os.path.join(data_dir, '..', 'checkpoints', 'wave_np_v2_best.pt')
    if os.path.exists(ckpt_path):
        model.load_state_dict(torch.load(ckpt_path, map_location='cpu'))
        print("  Loaded best checkpoint")

    model.eval()

    gx = gt_grid_x
    gy = gt_grid_y
    xx, yy = torch.meshgrid(gx, gy, indexing='ij')
    xt = torch.stack([xx.flatten(), yy.flatten()], dim=-1).unsqueeze(0)

    results = {}
    all_nrmses = []

    for ep in test_eps:
        ep_nrmses = []
        n_eval = min(20, len(ep))
        frame_indices = np.linspace(
            max(10, N_CONTEXT_FRAMES), len(ep) - 1, n_eval, dtype=int)

        for fi in frame_indices:
            if use_multi_frame:
                xc_frames, yc_frames, dt_frames = [], [], []
                for f_offset in range(N_CONTEXT_FRAMES):
                    ctx_idx = fi - (N_CONTEXT_FRAMES - 1 - f_offset)
                    ctx_idx = max(0, ctx_idx)
                    frame = ep[ctx_idx]
                    xc = torch.stack([frame.x_obs, frame.y_obs], dim=-1).unsqueeze(0)
                    yc = frame.z_obs.unsqueeze(-1).unsqueeze(0)
                    dt = torch.tensor([[float(ctx_idx - fi)]])
                    xc_frames.append(xc)
                    yc_frames.append(yc)
                    dt_frames.append(dt)

                with torch.no_grad():
                    pred = model.predict(xc_frames, yc_frames, dt_frames, xt)
            else:
                frame = ep[fi]
                xc = torch.stack([frame.x_obs, frame.y_obs], dim=-1).unsqueeze(0)
                yc = frame.z_obs.unsqueeze(-1).unsqueeze(0)

                with torch.no_grad():
                    pred = model.predict(xc, yc, xt)

            pred_field = pred[0, :, 0].reshape(len(gx), len(gy)).T.numpy()
            gt_field = ep[fi].gt_field.numpy()
            ep_nrmses.append(nrmse(pred_field, gt_field, ep.Hs_true))

        avg = float(np.mean(ep_nrmses))
        # Try to extract episode name
        ep_name = 'unknown'
        if hasattr(ep, 'data') and isinstance(ep.data, dict):
            ep_name = os.path.basename(ep.data.get('h5_path', 'unknown'))
        all_nrmses.append(avg)
        results[ep_name] = avg
        print(f"  {ep_name}: {avg:.4f}")

    overall_avg = float(np.mean(all_nrmses))
    print(f"\n  Average test nRMSE: {overall_avg:.4f}")
    print(f"  (WaveSplat reference: ~0.22)")
    print(f"  (WaveNP v1 reference: ~0.259)")

    # Save results
    results_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                'results')
    os.makedirs(results_dir, exist_ok=True)

    out = {
        'version': 'v2',
        'best_val_nRMSE': best_val,
        'best_epoch': best_epoch,
        'test_avg_nRMSE': overall_avg,
        'test_per_episode': results,
        'n_params': model.n_params,
        'training_time_s': time.time() - t_start,
        'config': {
            'embed_dim': EMBED_DIM, 'n_heads': N_HEADS,
            'n_layers': N_LAYERS, 'grid_nx': GRID_NX, 'grid_ny': GRID_NY,
            'n_context_frames': N_CONTEXT_FRAMES,
            'lambda_disp': LAMBDA_DISP, 'lambda_grad': LAMBDA_GRAD,
            'epochs': EPOCHS, 'lr': LR,
        },
    }

    with open(os.path.join(results_dir, 'wave_np_v2_eval.json'), 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\n  Results saved to results/wave_np_v2_eval.json")


if __name__ == '__main__':
    os.makedirs(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                              'checkpoints'), exist_ok=True)
    main()
