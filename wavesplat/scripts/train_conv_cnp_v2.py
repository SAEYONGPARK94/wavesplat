#!/usr/bin/env python3
"""
Train ConvCNP v2: correlation-aware loss + full-grid supervision.

Key changes from v1 (test 0.248):
  - Full-grid MSE (resample GT to model grid, no target subsampling)
  - Correlation loss: L_corr = 1 - corr(pred, gt) per batch
  - Combined: L = MSE + lambda_corr * L_corr
  - Encourages spatial pattern matching, not just minimizing MSE

Usage:
    cd wavesplat && python3 -u scripts/train_conv_cnp_v2.py
"""

import sys
import os
import time
import json
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', buffering=1)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from wavesplat.data_loader import EpisodeDataLoader
from wavesplat.model.conv_cnp import ConvCNP
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
GRID_NX = 64
GRID_NY = 48
BASE_CH = 32
INIT_LENGTHSCALE = 10.0

# Training
EPOCHS = 100
BATCH_SIZE = 8
LR = 1e-3
WARMUP_EPOCHS = 5
LAMBDA_CORR = 1.0  # correlation loss weight


def load_episodes(episode_names, data_dir):
    episodes = []
    for name in episode_names:
        path = os.path.join(data_dir, name)
        if not os.path.exists(path):
            continue
        episodes.append(EpisodeDataLoader(path))
    return episodes


def correlation_loss(pred, target):
    """Batch-wise negative Pearson correlation loss.

    Args:
        pred: [B, N] predicted values
        target: [B, N] target values

    Returns:
        scalar loss: 1 - mean_correlation (0 = perfect, 2 = anti-correlated)
    """
    pred_c = pred - pred.mean(dim=1, keepdim=True)
    target_c = target - target.mean(dim=1, keepdim=True)

    pred_std = pred_c.norm(dim=1, keepdim=True).clamp(min=1e-8)
    target_std = target_c.norm(dim=1, keepdim=True).clamp(min=1e-8)

    corr = (pred_c * target_c).sum(dim=1) / (pred_std.squeeze() * target_std.squeeze())
    return 1.0 - corr.mean()


def sample_batch_fullgrid(episodes, batch_size, model_grid_x, model_grid_y,
                          gt_grid_x, gt_grid_y):
    """Sample batch with full-grid GT resampled to model grid."""
    xc_list, yc_list = [], []
    gt_list = []

    for _ in range(batch_size):
        ep = episodes[np.random.randint(len(episodes))]
        frame = ep[np.random.randint(len(ep))]
        Hs = max(ep.Hs_true, 0.1)

        xc = torch.stack([frame.x_obs, frame.y_obs], dim=-1)
        yc = frame.z_obs.unsqueeze(-1) / Hs

        # GT field: [Ny, Nx] - resample to model grid using bilinear interpolation
        gt = frame.gt_field / Hs  # [Ny_data, Nx_data]

        # Resample GT to model grid
        # gt is [Ny, Nx], treat as image [1, 1, Ny, Nx]
        gt_img = gt.unsqueeze(0).unsqueeze(0).float()

        # Create sample grid for model grid points in data grid normalized coords
        # Model grid is [Gx, Gy] in physical space, need to map to data grid [-1,1]
        mgx = model_grid_x  # [Gx]
        mgy = model_grid_y  # [Gy]

        # Data grid bounds
        dx0, dx1 = float(gt_grid_x[0]), float(gt_grid_x[-1])
        dy0, dy1 = float(gt_grid_y[0]), float(gt_grid_y[-1])

        # Normalize model grid coords to [-1, 1] relative to data grid
        x_norm = 2.0 * (mgx - dx0) / (dx1 - dx0) - 1.0  # [Gx]
        y_norm = 2.0 * (mgy - dy0) / (dy1 - dy0) - 1.0  # [Gy]

        # gt_img is [1, 1, Ny, Nx] — grid_sample expects [1, H_out, W_out, 2]
        # with (x_norm, y_norm) where x->width(Nx), y->height(Ny)
        yy_grid, xx_grid = torch.meshgrid(y_norm, x_norm, indexing='ij')
        sample_grid = torch.stack([xx_grid, yy_grid], dim=-1).unsqueeze(0)  # [1, Gy, Gx, 2]

        gt_resampled = F.grid_sample(
            gt_img, sample_grid, mode='bilinear', align_corners=True,
            padding_mode='border'
        )  # [1, 1, Gy, Gx]
        gt_on_model = gt_resampled[0, 0]  # [Gy, Gx]

        # y-flip augmentation
        if np.random.random() < 0.5:
            xc = xc.clone()
            xc[:, 1] = -xc[:, 1]
            gt_on_model = gt_on_model.flip(0)  # flip y-axis

        xc_list.append(xc)
        yc_list.append(yc)
        gt_list.append(gt_on_model)

    # Pad context
    max_nc = max(x.shape[0] for x in xc_list)
    xc_batch = torch.zeros(batch_size, max_nc, 2)
    yc_batch = torch.zeros(batch_size, max_nc, 1)

    for i in range(batch_size):
        nc = xc_list[i].shape[0]
        xc_batch[i, :nc] = xc_list[i]
        yc_batch[i, :nc] = yc_list[i]

    gt_batch = torch.stack(gt_list)  # [B, Gy, Gx]

    return xc_batch, yc_batch, gt_batch


def evaluate(model, episodes, gt_grid_x, gt_grid_y, n_frames=10):
    model.eval()
    all_nrmse = []

    gx, gy = gt_grid_x, gt_grid_y
    xx, yy = torch.meshgrid(gx, gy, indexing='ij')
    xt = torch.stack([xx.flatten(), yy.flatten()], dim=-1).unsqueeze(0)

    for ep in episodes:
        Hs = max(ep.Hs_true, 0.1)
        frame_indices = np.linspace(10, len(ep) - 1, n_frames, dtype=int)

        for fi in frame_indices:
            frame = ep[fi]
            xc = torch.stack([frame.x_obs, frame.y_obs], dim=-1).unsqueeze(0)
            yc = frame.z_obs.unsqueeze(-1).unsqueeze(0) / Hs

            with torch.no_grad():
                pred = model.predict(xc, yc, xt)

            pred_field = (pred[0, :, 0] * Hs).reshape(len(gx), len(gy)).T.numpy()
            gt_field = frame.gt_field.numpy()

            val = nrmse(pred_field, gt_field, ep.Hs_true)
            if not np.isnan(val):
                all_nrmse.append(val)

    model.train()
    return float(np.mean(all_nrmse)) if all_nrmse else 1.0


def main():
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            'data')

    print("=" * 80)
    print("ConvCNP v2: Correlation-Aware Convolutional Neural Process")
    print("=" * 80)

    train_eps = load_episodes(TRAIN_EPISODES, data_dir)
    val_eps = load_episodes(VAL_EPISODES, data_dir)
    test_eps = load_episodes(TEST_EPISODES, data_dir)
    print("  Train: %d, Val: %d, Test: %d" % (len(train_eps), len(val_eps), len(test_eps)))

    if not train_eps:
        print("ERROR: No training episodes!")
        return

    gt_grid_x = train_eps[0].gt_grid_x
    gt_grid_y = train_eps[0].gt_grid_y
    x_range = (float(gt_grid_x[0]), float(gt_grid_x[-1]))
    y_range = (float(gt_grid_y[0]), float(gt_grid_y[-1]))

    # Model grid (same physical range as data)
    model_grid_x = torch.linspace(x_range[0], x_range[1], GRID_NX)
    model_grid_y = torch.linspace(y_range[0], y_range[1], GRID_NY)

    model = ConvCNP(
        grid_nx=GRID_NX, grid_ny=GRID_NY,
        x_range=x_range, y_range=y_range,
        base_ch=BASE_CH,
        init_lengthscale=INIT_LENGTHSCALE,
    )
    print("  Model: ConvCNP v2, %s params" % format(model.n_params, ','))
    print("  Grid: %dx%d, lambda_corr=%.1f" % (GRID_NX, GRID_NY, LAMBDA_CORR))

    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    warmup_sched = optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.01, total_iters=WARMUP_EPOCHS)
    cosine_sched = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS - WARMUP_EPOCHS)
    scheduler = optim.lr_scheduler.SequentialLR(
        optimizer, [warmup_sched, cosine_sched], milestones=[WARMUP_EPOCHS])

    print("  Training for %d epochs..." % EPOCHS)
    steps_per_epoch = 100
    best_val = 1.0
    best_epoch = 0
    t_start = time.time()

    for epoch in range(1, EPOCHS + 1):
        model.train()
        epoch_mse = 0.0
        epoch_corr = 0.0

        for step in range(steps_per_epoch):
            xc, yc, gt_grid = sample_batch_fullgrid(
                train_eps, BATCH_SIZE, model_grid_x, model_grid_y,
                gt_grid_x, gt_grid_y)

            # Forward: get UNet grid output directly
            B = xc.shape[0]
            grid_feat = model.setconv(xc, yc, model.grid_xy)
            grid_img = grid_feat.permute(0, 2, 1).reshape(B, 2, GRID_NX, GRID_NY)
            out_img = model.unet(grid_img)  # [B, 1, Gx, Gy]

            # pred is [B, 1, Gx, Gy], gt is [B, Gy, Gx]
            # Need to match: out_img[b, 0, :, :] = [Gx, Gy], gt = [Gy, Gx]
            pred_flat = out_img[:, 0].reshape(B, -1)  # [B, Gx*Gy]
            gt_flat = gt_grid.permute(0, 2, 1).reshape(B, -1)  # [B, Gx*Gy]

            # MSE loss
            mse_loss = F.mse_loss(pred_flat, gt_flat)

            # Correlation loss
            corr_loss = correlation_loss(pred_flat, gt_flat)

            loss = mse_loss + LAMBDA_CORR * corr_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_mse += mse_loss.item()
            epoch_corr += corr_loss.item()

        scheduler.step()
        avg_mse = epoch_mse / steps_per_epoch
        avg_corr = epoch_corr / steps_per_epoch

        val_interval = 5 if epoch <= 50 else 10
        if epoch % val_interval == 0 or epoch == 1:
            val_nrmse = evaluate(model, val_eps, gt_grid_x, gt_grid_y)
            improved = ""
            if val_nrmse < best_val:
                best_val = val_nrmse
                best_epoch = epoch
                ckpt_dir = os.path.join(data_dir, '..', 'checkpoints')
                os.makedirs(ckpt_dir, exist_ok=True)
                torch.save(model.state_dict(),
                           os.path.join(ckpt_dir, 'conv_cnp_v2_best.pt'))
                improved = " *"

            elapsed = time.time() - t_start
            ls = model.setconv.log_lengthscale.exp().item()
            print("  Epoch %3d/%d: mse=%.6f corr=%.4f val=%.4f ls=%.1fm "
                  "lr=%.6f [%.0fs]%s" % (
                      epoch, EPOCHS, avg_mse, avg_corr, val_nrmse, ls,
                      scheduler.get_last_lr()[0], elapsed, improved))
        elif epoch % 10 == 0:
            elapsed = time.time() - t_start
            print("  Epoch %3d/%d: mse=%.6f corr=%.4f lr=%.6f [%.0fs]" % (
                epoch, EPOCHS, avg_mse, avg_corr,
                scheduler.get_last_lr()[0], elapsed))

    print("  Best val nRMSE: %.4f at epoch %d" % (best_val, best_epoch))

    # --- Test ---
    print("\n" + "=" * 80)
    print("Test Evaluation (ConvCNP v2)")
    print("=" * 80)

    ckpt_path = os.path.join(data_dir, '..', 'checkpoints', 'conv_cnp_v2_best.pt')
    if os.path.exists(ckpt_path):
        model.load_state_dict(torch.load(ckpt_path, map_location='cpu',
                                          weights_only=True))
        print("  Loaded best checkpoint")

    model.eval()
    gx, gy = gt_grid_x, gt_grid_y
    xx, yy = torch.meshgrid(gx, gy, indexing='ij')
    xt = torch.stack([xx.flatten(), yy.flatten()], dim=-1).unsqueeze(0)

    results = {}
    all_nrmses = []

    for ep_idx, ep in enumerate(test_eps):
        ep_nrmses = []
        Hs = max(ep.Hs_true, 0.1)
        n_eval = min(20, len(ep))
        frame_indices = np.linspace(10, len(ep) - 1, n_eval, dtype=int)

        for fi in frame_indices:
            frame = ep[fi]
            xc = torch.stack([frame.x_obs, frame.y_obs], dim=-1).unsqueeze(0)
            yc = frame.z_obs.unsqueeze(-1).unsqueeze(0) / Hs

            with torch.no_grad():
                pred = model.predict(xc, yc, xt)

            pred_field = (pred[0, :, 0] * Hs).reshape(len(gx), len(gy)).T.numpy()
            gt_field = frame.gt_field.numpy()
            ep_nrmses.append(nrmse(pred_field, gt_field, ep.Hs_true))

        avg = float(np.mean(ep_nrmses))
        ep_name = TEST_EPISODES[ep_idx].replace('episode_', '').replace('.h5', '')
        results[ep_name] = avg
        all_nrmses.append(avg)
        print("  %s: nRMSE=%.4f" % (ep_name, avg))

    overall = float(np.mean(all_nrmses))
    print("  Average test nRMSE: %.4f" % overall)
    print("  (ConvCNP v1: 0.248, WaveNP v3: 0.249, WaveSplat: 0.223)")

    results_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                'results')
    os.makedirs(results_dir, exist_ok=True)
    out = {
        'version': 'conv_cnp_v2',
        'best_val_nRMSE': best_val,
        'best_epoch': best_epoch,
        'test_avg_nRMSE': overall,
        'test_per_episode': results,
        'n_params': model.n_params,
        'training_time_s': time.time() - t_start,
        'learned_lengthscale': model.setconv.log_lengthscale.exp().item(),
        'config': {
            'grid_nx': GRID_NX, 'grid_ny': GRID_NY,
            'base_ch': BASE_CH, 'lambda_corr': LAMBDA_CORR,
            'epochs': EPOCHS, 'lr': LR, 'batch_size': BATCH_SIZE,
        },
    }
    with open(os.path.join(results_dir, 'conv_cnp_v2_eval.json'), 'w') as f:
        json.dump(out, f, indent=2)
    print("  Results saved to results/conv_cnp_v2_eval.json")


if __name__ == '__main__':
    os.makedirs(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                              'checkpoints'), exist_ok=True)
    main()
