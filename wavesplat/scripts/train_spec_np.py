#!/usr/bin/env python3
"""
Train SpecNP: Spectral Neural Process (FFT-based) for ocean wave estimation.

Predicts FFT coefficients at low-k bins, with both spatial and spectral supervision.

Usage:
    cd wavesplat && python3 -u scripts/train_spec_np.py
"""

import sys
import os
import time
import json
import math
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', buffering=1)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from wavesplat.data_loader import EpisodeDataLoader
from wavesplat.model.spec_np import SpecNP, spec_np_loss, build_fft_k_grid
from wavesplat.evaluation.metrics import nrmse


# --- Configuration ---

TRAIN_EPISODES = [
    'episode_BF3_1comp_seed0_uniform.h5', 'episode_BF3_1comp_seed1_uniform.h5',
    'episode_BF3_2comp_seed0_uniform.h5', 'episode_BF3_2comp_seed1_uniform.h5',
    'episode_BF3_5comp_seed0_uniform.h5', 'episode_BF3_5comp_seed1_uniform.h5',
    'episode_BF3_1comp_seed0_slope.h5', 'episode_BF3_1comp_seed1_slope.h5',
    'episode_BF3_2comp_seed0_slope.h5', 'episode_BF3_2comp_seed1_slope.h5',
    'episode_BF3_5comp_seed0_slope.h5', 'episode_BF3_5comp_seed1_slope.h5',
    'episode_BF4_1comp_seed0_uniform.h5', 'episode_BF4_1comp_seed1_uniform.h5',
    'episode_BF4_5comp_seed0_uniform.h5', 'episode_BF4_5comp_seed1_uniform.h5',
    'episode_BF4_15comp_seed0_uniform.h5', 'episode_BF4_15comp_seed1_uniform.h5',
    'episode_BF4_30comp_seed0_uniform.h5',
    'episode_BF4_1comp_seed0_slope.h5', 'episode_BF4_1comp_seed1_slope.h5',
    'episode_BF4_5comp_seed0_slope.h5', 'episode_BF4_5comp_seed1_slope.h5',
    'episode_BF4_15comp_seed0_slope.h5', 'episode_BF4_15comp_seed1_slope.h5',
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

# Model
EMBED_DIM = 128
N_HEADS = 4
N_LAYERS = 4
GRID_NX = 16
GRID_NY = 12
N_FREQ = 8
DEPTH = 50.0
K_MAX = 0.15

# Training
EPOCHS = 200
BATCH_SIZE = 8
LR = 1e-3
LAMBDA_SPEC = 0.1
LAMBDA_L1 = 0.0001
WARMUP_EPOCHS = 10


# --- Data ---

def load_episodes(episode_names, data_dir):
    episodes = []
    for name in episode_names:
        path = os.path.join(data_dir, name)
        if not os.path.exists(path):
            continue
        episodes.append(EpisodeDataLoader(path))
    return episodes


def compute_gt_spectrum(gt_field, bin_iy, bin_ix):
    """Compute GT FFT coefficients at selected bins.

    Args:
        gt_field: [Ny, Nx] numpy array
        bin_iy, bin_ix: index arrays for selected FFT bins

    Returns:
        spec_real, spec_imag: [N_bins] tensors
    """
    spectrum = np.fft.fft2(gt_field, norm='ortho')
    selected = spectrum[bin_iy, bin_ix]
    return torch.tensor(selected.real, dtype=torch.float32), \
           torch.tensor(selected.imag, dtype=torch.float32)


def sample_batch(episodes, batch_size, bin_iy, bin_ix):
    """Sample a training batch with GT fields and GT spectra, normalized by Hs."""
    xc_list, yc_list, field_list = [], [], []
    spec_real_list, spec_imag_list = [], []

    for _ in range(batch_size):
        ep = episodes[np.random.randint(len(episodes))]
        frame = ep[np.random.randint(len(ep))]
        Hs = max(ep.Hs_true, 0.1)  # avoid division by zero

        xc = torch.stack([frame.x_obs, frame.y_obs], dim=-1)
        yc = frame.z_obs.unsqueeze(-1) / Hs  # normalize by Hs

        gt_field = frame.gt_field.numpy() / Hs  # normalize

        # y-flip augmentation
        if np.random.random() < 0.5:
            xc = xc.clone()
            xc[:, 1] = -xc[:, 1]
            gt_field = gt_field[::-1, :].copy()

        field_list.append(torch.tensor(gt_field, dtype=torch.float32))
        xc_list.append(xc)
        yc_list.append(yc)

        # GT spectrum
        sr, si = compute_gt_spectrum(gt_field, bin_iy, bin_ix)
        spec_real_list.append(sr)
        spec_imag_list.append(si)

    # Pad context
    max_nc = max(x.shape[0] for x in xc_list)
    xc_batch = torch.zeros(batch_size, max_nc, 2)
    yc_batch = torch.zeros(batch_size, max_nc, 1)
    for i in range(batch_size):
        nc = xc_list[i].shape[0]
        xc_batch[i, :nc] = xc_list[i]
        yc_batch[i, :nc] = yc_list[i]

    field_batch = torch.stack(field_list)  # [B, Ny, Nx]
    spec_real_batch = torch.stack(spec_real_list)  # [B, N_bins]
    spec_imag_batch = torch.stack(spec_imag_list)  # [B, N_bins]

    return xc_batch, yc_batch, field_batch, spec_real_batch, spec_imag_batch


# --- Evaluation ---

def evaluate(model, episodes, n_frames=10):
    model.eval()
    all_nrmse = []

    for ep in episodes:
        Hs = max(ep.Hs_true, 0.1)
        frame_indices = np.linspace(10, len(ep) - 1, n_frames, dtype=int)

        for fi in frame_indices:
            frame = ep[fi]
            xc = torch.stack([frame.x_obs, frame.y_obs], dim=-1).unsqueeze(0)
            yc = frame.z_obs.unsqueeze(-1).unsqueeze(0) / Hs

            with torch.no_grad():
                field = model.predict(xc, yc)

            pred_field = field[0].numpy() * Hs  # denormalize
            gt_field = frame.gt_field.numpy()

            val = nrmse(pred_field, gt_field, ep.Hs_true)
            if not np.isnan(val):
                all_nrmse.append(val)

    model.train()
    return float(np.mean(all_nrmse)) if all_nrmse else 1.0


# --- Main ---

def main():
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            'data')

    print("=" * 80)
    print("SpecNP: Spectral Neural Process (FFT-based)")
    print("=" * 80)

    # Load
    train_eps = load_episodes(TRAIN_EPISODES, data_dir)
    val_eps = load_episodes(VAL_EPISODES, data_dir)
    test_eps = load_episodes(TEST_EPISODES, data_dir)
    print(f"\n  Train: {len(train_eps)}, Val: {len(val_eps)}, Test: {len(test_eps)}")

    if not train_eps:
        print("ERROR: No training episodes!")
        return

    # Get grid info
    gt_grid_x = train_eps[0].gt_grid_x
    gt_grid_y = train_eps[0].gt_grid_y
    gt_nx = len(gt_grid_x)
    gt_ny = len(gt_grid_y)
    dx = (gt_grid_x[-1] - gt_grid_x[0]).item() / (gt_nx - 1)
    dy = (gt_grid_y[-1] - gt_grid_y[0]).item() / (gt_ny - 1)
    x_range = (float(gt_grid_x[0]), float(gt_grid_x[-1]))
    y_range = (float(gt_grid_y[0]), float(gt_grid_y[-1]))

    # Build FFT k-grid for GT spectrum computation
    _, _, fft_mask, bin_indices = build_fft_k_grid(gt_nx, gt_ny, dx, dy, K_MAX)
    bin_iy = bin_indices[:, 0]
    bin_ix = bin_indices[:, 1]
    n_bins = len(bin_indices)

    model = SpecNP(
        embed_dim=EMBED_DIM, n_heads=N_HEADS, n_layers=N_LAYERS,
        grid_nx=GRID_NX, grid_ny=GRID_NY,
        x_range=x_range, y_range=y_range, n_freq=N_FREQ, depth=DEPTH,
        gt_nx=gt_nx, gt_ny=gt_ny, dx=dx, dy=dy, k_max=K_MAX,
    )
    print(f"  Model: SpecNP (FFT-based), {model.n_params:,} params")
    print(f"  FFT bins: {n_bins} (|k| < {K_MAX}), GT grid: {gt_nx}x{gt_ny}")
    print(f"  Grid: {GRID_NX}x{GRID_NY}, dx={dx:.2f}m, dy={dy:.2f}m")

    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    # Warmup + cosine annealing
    warmup_sched = optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.01, total_iters=WARMUP_EPOCHS)
    cosine_sched = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS - WARMUP_EPOCHS)
    scheduler = optim.lr_scheduler.SequentialLR(
        optimizer, [warmup_sched, cosine_sched], milestones=[WARMUP_EPOCHS])

    # Train
    print(f"\n  Training for {EPOCHS} epochs (spec_loss={LAMBDA_SPEC})...")
    steps_per_epoch = 100
    best_val = 1.0
    best_epoch = 0
    t_start = time.time()

    for epoch in range(1, EPOCHS + 1):
        model.train()
        epoch_spatial = 0.0
        epoch_spectral = 0.0
        epoch_n_active = 0.0

        for step in range(steps_per_epoch):
            xc, yc, field_gt, gt_sr, gt_si = sample_batch(
                train_eps, BATCH_SIZE, bin_iy, bin_ix)

            field_pred, params = model(xc, yc)
            loss, loss_dict = spec_np_loss(
                field_pred, field_gt, params,
                gt_spec_real=gt_sr, gt_spec_imag=gt_si,
                lambda_spec=LAMBDA_SPEC, lambda_l1=LAMBDA_L1)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_spatial += loss_dict['spatial']
            epoch_spectral += loss_dict.get('spectral', 0.0)
            epoch_n_active += loss_dict['n_active']

        scheduler.step()
        avg_spatial = epoch_spatial / steps_per_epoch
        avg_spectral = epoch_spectral / steps_per_epoch
        avg_n = epoch_n_active / steps_per_epoch

        val_interval = 5 if epoch <= 50 else 10
        if epoch % val_interval == 0 or epoch == 1:
            val_nrmse = evaluate(model, val_eps)
            improved = ""
            if val_nrmse < best_val:
                best_val = val_nrmse
                best_epoch = epoch
                ckpt_dir = os.path.join(data_dir, '..', 'checkpoints')
                os.makedirs(ckpt_dir, exist_ok=True)
                torch.save(model.state_dict(),
                           os.path.join(ckpt_dir, 'spec_np_best.pt'))
                improved = " *"

            elapsed = time.time() - t_start
            print(f"  Epoch {epoch:3d}/{EPOCHS}: spatial={avg_spatial:.6f}, "
                  f"spectral={avg_spectral:.6f}, N_active={avg_n:.1f}, "
                  f"val_nRMSE={val_nrmse:.4f}, lr={scheduler.get_last_lr()[0]:.6f} "
                  f"[{elapsed:.0f}s]{improved}")
        elif epoch % 10 == 0:
            elapsed = time.time() - t_start
            print(f"  Epoch {epoch:3d}/{EPOCHS}: spatial={avg_spatial:.6f}, "
                  f"spectral={avg_spectral:.6f}, N_active={avg_n:.1f}, "
                  f"lr={scheduler.get_last_lr()[0]:.6f} [{elapsed:.0f}s]")

    print(f"\n  Best val nRMSE: {best_val:.4f} at epoch {best_epoch}")

    # --- Test ---
    print("\n" + "=" * 80)
    print("Test Evaluation (SpecNP FFT-based)")
    print("=" * 80)

    ckpt_path = os.path.join(data_dir, '..', 'checkpoints', 'spec_np_best.pt')
    if os.path.exists(ckpt_path):
        model.load_state_dict(torch.load(ckpt_path, map_location='cpu',
                                          weights_only=True))
        print("  Loaded best checkpoint")

    model.eval()
    results = {}
    all_nrmses = []

    for ep_idx, ep in enumerate(test_eps):
        ep_nrmses = []
        n_eval = min(20, len(ep))
        frame_indices = np.linspace(10, len(ep) - 1, n_eval, dtype=int)

        Hs = max(ep.Hs_true, 0.1)
        for fi in frame_indices:
            frame = ep[fi]
            xc = torch.stack([frame.x_obs, frame.y_obs], dim=-1).unsqueeze(0)
            yc = frame.z_obs.unsqueeze(-1).unsqueeze(0) / Hs

            with torch.no_grad():
                field, params = model(xc, yc)

            pred_field = field[0].numpy() * Hs  # denormalize
            gt_field = frame.gt_field.numpy()
            ep_nrmses.append(nrmse(pred_field, gt_field, ep.Hs_true))

        avg = float(np.mean(ep_nrmses))
        ep_name = TEST_EPISODES[ep_idx].replace('episode_', '').replace('.h5', '')
        results[ep_name] = {'nRMSE': avg}
        all_nrmses.append(avg)
        print(f"  {ep_name}: nRMSE={avg:.4f}")

    overall = float(np.mean(all_nrmses))
    print(f"\n  Average test nRMSE: {overall:.4f}")
    print(f"  (WaveSplat ref: ~0.223, WaveNP ref: ~0.259)")

    # Save
    results_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                'results')
    os.makedirs(results_dir, exist_ok=True)
    out = {
        'model': 'SpecNP-FFT',
        'best_val_nRMSE': best_val,
        'best_epoch': best_epoch,
        'test_avg_nRMSE': overall,
        'test_per_episode': results,
        'n_params': model.n_params,
        'n_fft_bins': n_bins,
        'training_time_s': time.time() - t_start,
        'config': {
            'embed_dim': EMBED_DIM, 'n_heads': N_HEADS,
            'n_layers': N_LAYERS, 'n_fft_bins': n_bins,
            'grid_nx': GRID_NX, 'grid_ny': GRID_NY,
            'k_max': K_MAX,
            'lambda_spec': LAMBDA_SPEC, 'lambda_l1': LAMBDA_L1,
            'epochs': EPOCHS, 'lr': LR,
        },
    }
    with open(os.path.join(results_dir, 'spec_np_eval.json'), 'w') as f:
        json.dump(out, f, indent=2)
    print(f"  Results saved to results/spec_np_eval.json")


if __name__ == '__main__':
    os.makedirs(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                              'checkpoints'), exist_ok=True)
    main()
