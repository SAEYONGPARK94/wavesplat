#!/usr/bin/env python3
"""
pino_stage1_sanity.py

WavePINO Stage 1: Single-episode overfit sanity check.

Purpose:
  Verify that the FNO architecture can memorize one episode — i.e.,
  gradient flows correctly through SpectralConv2d and the model can
  learn wave spatial structure from sparse LiDAR.

Pass criteria:
  - nRMSE < 0.30 on the training episode (overfit)
  - MSE loss drops by factor > 10x

Architecture: small model (fno_width=8, layers=3) for fast iteration.
No physics loss at Stage 1.

Usage:
    cd wavesplat && python3 -u scripts/pino_stage1_sanity.py
"""

import os
import sys
import time
import json

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', buffering=1)

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(REPO_ROOT, 'wavesplat'))

from wavesplat.data_loader import EpisodeDataLoader
from wavesplat.model.wave_pino import WavePINO
from wavesplat.evaluation.metrics import nrmse

# --- Config ---
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DATA_DIR = os.path.join(REPO_ROOT, 'wavesplat', 'data')
EPISODE  = 'episode_BF4_5comp_seed0_uniform.h5'

# Small model for fast sanity
FNO_WIDTH   = 8
FNO_LAYERS  = 3
FNO_MODES_H = 12
FNO_MODES_W = 10
GRID_NX = 64
GRID_NY = 48

# Training
N_STEPS    = 5000
LR         = 1e-3
BATCH_SIZE = 8
LOG_EVERY  = 500


def sample_batch(ep, batch_size, model_grid_x, model_grid_y,
                 gt_grid_x, gt_grid_y, device):
    xc_list, yc_list, gt_list = [], [], []
    Hs = max(ep.Hs_true, 0.1)

    for _ in range(batch_size):
        fi = np.random.randint(len(ep))
        frame = ep[fi]

        xc = torch.stack([frame.x_obs, frame.y_obs], dim=-1)
        yc = frame.z_obs.unsqueeze(-1) / Hs

        gt = frame.gt_field / Hs  # [Ny, Nx]

        # Resample GT to model grid via bilinear interpolation
        dx0, dx1 = float(gt_grid_x[0]), float(gt_grid_x[-1])
        dy0, dy1 = float(gt_grid_y[0]), float(gt_grid_y[-1])
        x_norm = 2.0 * (model_grid_x - dx0) / (dx1 - dx0) - 1.0
        y_norm = 2.0 * (model_grid_y - dy0) / (dy1 - dy0) - 1.0
        yy, xx = torch.meshgrid(y_norm, x_norm, indexing='ij')
        sg = torch.stack([xx, yy], dim=-1).unsqueeze(0)  # [1, Gy, Gx, 2]
        gt_r = F.grid_sample(gt.unsqueeze(0).unsqueeze(0).float(), sg,
                              mode='bilinear', align_corners=True,
                              padding_mode='border')[0, 0]  # [Gy, Gx]
        # gt_r is [Gy, Gx]; we need [Gx, Gy] to match model output [B,1,Gx,Gy]
        gt_r = gt_r.T  # [Gx, Gy]

        xc_list.append(xc)
        yc_list.append(yc)
        gt_list.append(gt_r)

    max_nc = max(x.shape[0] for x in xc_list)
    xc_b = torch.zeros(batch_size, max_nc, 2)
    yc_b = torch.zeros(batch_size, max_nc, 1)
    for i in range(batch_size):
        nc = xc_list[i].shape[0]
        xc_b[i, :nc] = xc_list[i]
        yc_b[i, :nc] = yc_list[i]
    gt_b = torch.stack(gt_list)  # [B, Gx, Gy]

    return xc_b.to(device), yc_b.to(device), gt_b.to(device)


def evaluate(model, ep, gt_grid_x, gt_grid_y, device, n_frames=10):
    model.eval()
    Hs = max(ep.Hs_true, 0.1)
    gx, gy = gt_grid_x, gt_grid_y
    xx, yy = torch.meshgrid(gx, gy, indexing='ij')
    xt = torch.stack([xx.flatten(), yy.flatten()], dim=-1).unsqueeze(0).to(device)

    nrmse_list = []
    idx_list = np.linspace(5, len(ep) - 1, n_frames, dtype=int)
    with torch.no_grad():
        for fi in idx_list:
            frame = ep[fi]
            xc = torch.stack([frame.x_obs, frame.y_obs], dim=-1).unsqueeze(0).to(device)
            yc = frame.z_obs.unsqueeze(-1).unsqueeze(0).to(device) / Hs
            pred = model(xc, yc, xt)
            pred_field = (pred[0, :, 0] * Hs).cpu().numpy()
            pred_field = pred_field.reshape(len(gx), len(gy)).T  # [Ny, Nx]
            gt_field = frame.gt_field.numpy()
            v = nrmse(pred_field, gt_field, ep.Hs_true)
            if not np.isnan(v):
                nrmse_list.append(v)
    model.train()
    return float(np.mean(nrmse_list)) if nrmse_list else 1.0


def main():
    print(f"Device: {DEVICE}")
    ep_path = os.path.join(DATA_DIR, EPISODE)
    if not os.path.exists(ep_path):
        print(f"ERROR: episode not found: {ep_path}")
        return

    ep = EpisodeDataLoader(ep_path)
    gt_grid_x = ep.gt_grid_x
    gt_grid_y = ep.gt_grid_y
    x_range = (float(gt_grid_x[0]), float(gt_grid_x[-1]))
    y_range = (float(gt_grid_y[0]), float(gt_grid_y[-1]))

    model_grid_x = torch.linspace(x_range[0], x_range[1], GRID_NX)
    model_grid_y = torch.linspace(y_range[0], y_range[1], GRID_NY)

    model = WavePINO(
        grid_nx=GRID_NX, grid_ny=GRID_NY,
        x_range=x_range, y_range=y_range,
        fno_width=FNO_WIDTH, fno_layers=FNO_LAYERS,
        fno_modes_h=FNO_MODES_H, fno_modes_w=FNO_MODES_W,
        init_lengthscale=10.0,
    ).to(DEVICE)

    print(f"WavePINO (Stage 1 / small): {model.n_params:,} params")
    print(f"  fno_width={FNO_WIDTH}, layers={FNO_LAYERS}, "
          f"modes=({FNO_MODES_H}, {FNO_MODES_W})")
    print(f"  Episode: {EPISODE}, frames: {len(ep)}")
    print(f"  Training: {N_STEPS} steps, BS={BATCH_SIZE}, LR={LR}")

    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=N_STEPS)

    print(f"\n{'Step':>6}  {'MSE':>10}  {'Time':>6}")
    print("-" * 32)

    t0 = time.time()
    mse_history = []

    for step in range(1, N_STEPS + 1):
        model.train()
        xc, yc, gt = sample_batch(ep, BATCH_SIZE, model_grid_x, model_grid_y,
                                   gt_grid_x, gt_grid_y, DEVICE)

        # Predict full grid [B, 1, Gx, Gy]
        pred_grid = model.predict_grid(xc, yc)
        # gt is [B, Gx, Gy]
        mse_loss = F.mse_loss(pred_grid[:, 0], gt)

        optimizer.zero_grad()
        mse_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        mse_history.append(mse_loss.item())

        if step % LOG_EVERY == 0 or step == 1:
            elapsed = time.time() - t0
            print(f"{step:>6}  {mse_loss.item():>10.6f}  {elapsed:>5.1f}s")

    # Final evaluation
    nrmse_val = evaluate(model, ep, gt_grid_x, gt_grid_y, DEVICE, n_frames=15)
    total = time.time() - t0
    mse_drop = mse_history[0] / (min(mse_history) + 1e-12)

    print(f"\n{'='*40}")
    print("WavePINO Stage 1 Results:")
    print(f"  First MSE:   {mse_history[0]:.6f}")
    print(f"  Best MSE:    {min(mse_history):.6f}")
    print(f"  MSE drop ×:  {mse_drop:.1f}x")
    print(f"  nRMSE:       {nrmse_val:.4f}")
    print(f"  Time:        {total:.1f}s  ({total/N_STEPS*1000:.1f}ms/step)")

    passed = mse_drop >= 10.0 and nrmse_val < 0.30
    verdict = 'PASS' if passed else 'FAIL'
    print(f"\nSTAGE 1: {verdict}")
    print(f"  MSE drop {mse_drop:.1f}x (>=10x?) | nRMSE {nrmse_val:.3f} (<0.30?)")

    result = {
        'stage': 1,
        'description': 'WavePINO single-episode overfit sanity check',
        'episode': EPISODE,
        'model': {
            'fno_width': FNO_WIDTH, 'fno_layers': FNO_LAYERS,
            'fno_modes_h': FNO_MODES_H, 'fno_modes_w': FNO_MODES_W,
            'params': model.n_params,
        },
        'training': {
            'steps': N_STEPS, 'lr': LR, 'batch_size': BATCH_SIZE,
        },
        'results': {
            'first_mse': mse_history[0],
            'best_mse': min(mse_history),
            'mse_drop_x': mse_drop,
            'nrmse': nrmse_val,
            'time_s': total,
            'ms_per_step': total / N_STEPS * 1000,
        },
        'verdict': verdict,
    }

    out_dir = os.path.join(REPO_ROOT, 'wavesplat', 'results')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'pino_stage1_sanity.json')
    with open(out_path, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"\nResults saved: {out_path}")


if __name__ == '__main__':
    main()
