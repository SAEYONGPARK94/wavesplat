#!/usr/bin/env python3
"""
Train ConvCNP v3: full-grid MSE supervision (no correlation loss).

Based on lambda sweep showing MSE-only outperforms MSE+corr.
Changes from v1: full-grid GT (not subsampled 1024 targets).

Usage:
    cd wavesplat && python3 -u scripts/train_conv_cnp_v3.py
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

GRID_NX = 64
GRID_NY = 48
BASE_CH = 32
INIT_LENGTHSCALE = 10.0

EPOCHS = 100
BATCH_SIZE = 8
LR = 1e-3
WARMUP_EPOCHS = 5


def load_episodes(episode_names, data_dir):
    episodes = []
    for name in episode_names:
        path = os.path.join(data_dir, name)
        if os.path.exists(path):
            episodes.append(EpisodeDataLoader(path))
    return episodes


def sample_batch_fullgrid(episodes, batch_size, grid_nx, grid_ny,
                          gt_grid_x, gt_grid_y, x_range, y_range):
    """Sample batch with full grid target (resampled GT)."""
    xc_list, yc_list = [], []
    xt_list, yt_list = [], []

    gx, gy = gt_grid_x, gt_grid_y
    xx, yy = torch.meshgrid(gx, gy, indexing='ij')
    xt_full = torch.stack([xx.flatten(), yy.flatten()], dim=-1)  # data grid targets

    for _ in range(batch_size):
        ep = episodes[np.random.randint(len(episodes))]
        frame = ep[np.random.randint(len(ep))]
        Hs = max(ep.Hs_true, 0.1)

        xc = torch.stack([frame.x_obs, frame.y_obs], dim=-1)
        yc = frame.z_obs.unsqueeze(-1) / Hs

        yt_full = frame.gt_field.T.flatten().unsqueeze(-1) / Hs

        # y-flip augmentation
        if np.random.random() < 0.5:
            xc = xc.clone()
            xc[:, 1] = -xc[:, 1]
            yt_full = frame.gt_field.flip(0).T.flatten().unsqueeze(-1) / Hs

        xc_list.append(xc)
        yc_list.append(yc)
        xt_list.append(xt_full)
        yt_list.append(yt_full)

    max_nc = max(x.shape[0] for x in xc_list)
    Nt = xt_full.shape[0]

    xc_batch = torch.zeros(batch_size, max_nc, 2)
    yc_batch = torch.zeros(batch_size, max_nc, 1)
    xt_batch = xt_full.unsqueeze(0).expand(batch_size, -1, -1)
    yt_batch = torch.stack([y.squeeze(-1) for y in yt_list])  # [B, Nt]

    for i in range(batch_size):
        nc = xc_list[i].shape[0]
        xc_batch[i, :nc] = xc_list[i]
        yc_batch[i, :nc] = yc_list[i]

    return xc_batch, yc_batch, xt_batch, yt_batch


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
    print("ConvCNP v3: Full-Grid MSE Convolutional Neural Process")
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

    model = ConvCNP(
        grid_nx=GRID_NX, grid_ny=GRID_NY,
        x_range=x_range, y_range=y_range,
        base_ch=BASE_CH,
        init_lengthscale=INIT_LENGTHSCALE,
    )
    print("  Model: ConvCNP v3, %s params" % format(model.n_params, ','))
    print("  Grid: %dx%d, pure MSE, full-grid targets" % (GRID_NX, GRID_NY))

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
        epoch_loss = 0.0

        for step in range(steps_per_epoch):
            xc, yc, xt, yt = sample_batch_fullgrid(
                train_eps, BATCH_SIZE, GRID_NX, GRID_NY,
                gt_grid_x, gt_grid_y, x_range, y_range)

            pred = model(xc, yc, xt)  # [B, Nt, 1]
            loss = F.mse_loss(pred.squeeze(-1), yt)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()

        scheduler.step()
        avg_loss = epoch_loss / steps_per_epoch

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
                           os.path.join(ckpt_dir, 'conv_cnp_v3_best.pt'))
                improved = " *"

            elapsed = time.time() - t_start
            ls = model.setconv.log_lengthscale.exp().item()
            print("  Epoch %3d/%d: loss=%.6f val=%.4f ls=%.1fm lr=%.6f [%.0fs]%s" % (
                epoch, EPOCHS, avg_loss, val_nrmse, ls,
                scheduler.get_last_lr()[0], elapsed, improved))
        elif epoch % 10 == 0:
            elapsed = time.time() - t_start
            print("  Epoch %3d/%d: loss=%.6f lr=%.6f [%.0fs]" % (
                epoch, EPOCHS, avg_loss,
                scheduler.get_last_lr()[0], elapsed))

    print("  Best val nRMSE: %.4f at epoch %d" % (best_val, best_epoch))

    # --- Test ---
    print("\n" + "=" * 80)
    print("Test Evaluation (ConvCNP v3)")
    print("=" * 80)

    ckpt_path = os.path.join(data_dir, '..', 'checkpoints', 'conv_cnp_v3_best.pt')
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
    print("  (v1: 0.248, v2: 0.244, WaveNP v3: 0.249, WaveSplat: 0.223)")

    results_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                'results')
    os.makedirs(results_dir, exist_ok=True)
    out = {
        'version': 'conv_cnp_v3',
        'best_val_nRMSE': best_val,
        'best_epoch': best_epoch,
        'test_avg_nRMSE': overall,
        'test_per_episode': results,
        'n_params': model.n_params,
        'training_time_s': time.time() - t_start,
        'learned_lengthscale': model.setconv.log_lengthscale.exp().item(),
        'config': {
            'grid_nx': GRID_NX, 'grid_ny': GRID_NY,
            'base_ch': BASE_CH,
            'epochs': EPOCHS, 'lr': LR, 'batch_size': BATCH_SIZE,
        },
    }
    with open(os.path.join(results_dir, 'conv_cnp_v3_eval.json'), 'w') as f:
        json.dump(out, f, indent=2)
    print("  Results saved to results/conv_cnp_v3_eval.json")


if __name__ == '__main__':
    os.makedirs(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                              'checkpoints'), exist_ok=True)
    main()
