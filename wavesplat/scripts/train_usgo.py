#!/usr/bin/env python3
"""
Train USGO (Unrolled Spectral Gaussian Optimizer).

Two training modes:
  - Deep supervision: sum of observation losses at all layers (default)
  - Grid reconstruction: MSE on dense grid (for evaluation)

Two model variants:
  - simple: learned step sizes + momentum only (~80 params)
  - mlp: per-component MLP (~5K-50K params)

Usage:
    python3 -m scripts.train_usgo --data data/usgo_train.npz --model simple
    python3 -m scripts.train_usgo --data data/usgo_train.npz --model mlp --n_layers 10
"""

import argparse
import os
import sys
import time
import json
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from wavesplat.usgo.model import USGOSimple, USGOModel
from wavesplat.usgo.dataset import USGODataset
from wavesplat.model.dispersion import dispersion_omega


def reconstruction_loss(kx, ky, w, phi, depth,
                        gt_kx, gt_ky, gt_w, gt_phi, gt_omega,
                        grid_x, grid_y, t):
    """MSE between predicted and GT fields on evaluation grid."""
    k_mag = torch.sqrt(kx ** 2 + ky ** 2 + 1e-12)
    omega = dispersion_omega(k_mag, depth)

    Xg, Yg = torch.meshgrid(grid_x, grid_y, indexing='xy')
    x_flat = Xg.reshape(-1)
    y_flat = Yg.reshape(-1)

    phase_pred = (kx[None, :] * x_flat[:, None]
                  + ky[None, :] * y_flat[:, None]
                  - omega[None, :] * t + phi[None, :])
    z_pred = torch.sum(w[None, :] * torch.cos(phase_pred), dim=1)

    phase_gt = (gt_kx[None, :] * x_flat[:, None]
                + gt_ky[None, :] * y_flat[:, None]
                - gt_omega[None, :] * t + gt_phi[None, :])
    z_gt = torch.sum(gt_w[None, :] * torch.cos(phase_gt), dim=1)

    return nn.functional.mse_loss(z_pred, z_gt)


def train_one_epoch(model, dataset, optimizer, grid_x, grid_y,
                    device, indices, deep_supervision_weight=1.0):
    """Train one epoch with deep supervision + grid reconstruction loss."""
    model.train()
    total_loss = 0.0
    total_recon = 0.0
    total_init_recon = 0.0
    n_samples = 0

    np.random.shuffle(indices)

    for idx in indices:
        sample = dataset[idx]

        obs_x = sample['obs_x'].to(device)
        obs_y = sample['obs_y'].to(device)
        obs_z = sample['obs_z'].to(device)
        obs_conf = sample['obs_conf'].to(device)
        t = float(sample['t'])
        depth = float(sample['depth'])

        init_kx = sample['init_kx'].to(device)
        init_ky = sample['init_ky'].to(device)
        init_w = sample['init_w'].to(device)
        init_phi = sample['init_phi'].to(device)

        gt_kx = sample['gt_kx'].to(device)
        gt_ky = sample['gt_ky'].to(device)
        gt_w = sample['gt_w'].to(device)
        gt_phi = sample['gt_phi'].to(device)
        gt_omega = sample['gt_omega'].to(device)

        gx = grid_x.to(device)
        gy = grid_y.to(device)

        # Init reconstruction loss
        with torch.no_grad():
            init_recon = reconstruction_loss(
                init_kx, init_ky, init_w, init_phi, depth,
                gt_kx, gt_ky, gt_w, gt_phi, gt_omega, gx, gy, t)
            total_init_recon += init_recon.item()

        # Forward
        out_kx, out_ky, out_w, out_phi, layer_losses = model(
            init_kx, init_ky, init_w, init_phi,
            obs_x, obs_y, obs_z, t, depth, obs_conf)

        # Deep supervision: weighted sum of all layer obs losses
        # Later layers weighted more (progressive schedule)
        K = len(layer_losses)
        deep_loss = sum(
            (k + 1) / K * layer_losses[k] for k in range(K)) / K

        # Grid reconstruction loss (final output only)
        recon_loss = reconstruction_loss(
            out_kx, out_ky, out_w, out_phi, depth,
            gt_kx, gt_ky, gt_w, gt_phi, gt_omega, gx, gy, t)

        # Combined loss
        loss = deep_supervision_weight * deep_loss + recon_loss

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        total_recon += recon_loss.item()
        n_samples += 1

    return (total_loss / max(n_samples, 1),
            total_recon / max(n_samples, 1),
            total_init_recon / max(n_samples, 1))


def evaluate(model, dataset, grid_x, grid_y, device, indices):
    """Evaluate: grid reconstruction loss + observation loss."""
    model.eval()
    total_recon = 0.0
    total_init_recon = 0.0
    total_obs = 0.0
    total_init_obs = 0.0
    n_samples = 0

    for idx in indices:
        sample = dataset[idx]

        obs_x = sample['obs_x'].to(device)
        obs_y = sample['obs_y'].to(device)
        obs_z = sample['obs_z'].to(device)
        obs_conf = sample['obs_conf'].to(device)
        t = float(sample['t'])
        depth = float(sample['depth'])

        init_kx = sample['init_kx'].to(device)
        init_ky = sample['init_ky'].to(device)
        init_w = sample['init_w'].to(device)
        init_phi = sample['init_phi'].to(device)

        gt_kx = sample['gt_kx'].to(device)
        gt_ky = sample['gt_ky'].to(device)
        gt_w = sample['gt_w'].to(device)
        gt_phi = sample['gt_phi'].to(device)
        gt_omega = sample['gt_omega'].to(device)

        gx = grid_x.to(device)
        gy = grid_y.to(device)

        # Init losses
        init_recon = reconstruction_loss(
            init_kx, init_ky, init_w, init_phi, depth,
            gt_kx, gt_ky, gt_w, gt_phi, gt_omega, gx, gy, t)
        total_init_recon += init_recon.item()

        # USGO forward
        out_kx, out_ky, out_w, out_phi, layer_losses = model(
            init_kx, init_ky, init_w, init_phi,
            obs_x, obs_y, obs_z, t, depth, obs_conf)

        # Init observation loss = first layer loss
        total_init_obs += layer_losses[0].item()
        # Final observation loss
        total_obs += layer_losses[-1].item()

        recon = reconstruction_loss(
            out_kx, out_ky, out_w, out_phi, depth,
            gt_kx, gt_ky, gt_w, gt_phi, gt_omega, gx, gy, t)
        total_recon += recon.item()
        n_samples += 1

    n = max(n_samples, 1)
    return {
        'recon': total_recon / n,
        'init_recon': total_init_recon / n,
        'obs': total_obs / n,
        'init_obs': total_init_obs / n,
    }


def main():
    parser = argparse.ArgumentParser(description='Train USGO')
    parser.add_argument('--data', type=str, default='data/usgo_train.npz')
    parser.add_argument('--model', type=str, default='simple',
                        choices=['simple', 'mlp'])
    parser.add_argument('--n_layers', type=int, default=10)
    parser.add_argument('--hidden_dim', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-2)
    parser.add_argument('--deep_sup_weight', type=float, default=1.0)
    parser.add_argument('--val_split', type=float, default=0.2)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--save_dir', type=str, default='checkpoints/usgo')
    parser.add_argument('--n_grid', type=int, default=32)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    dataset = USGODataset(args.data, device=args.device)
    N = len(dataset)
    n_val = int(N * args.val_split)
    n_train = N - n_val

    perm = torch.randperm(N).tolist()
    train_idx = perm[:n_train]
    val_idx = perm[n_train:]

    print(f"Dataset: {N} samples (train={n_train}, val={n_val})")

    # Model
    if args.model == 'simple':
        model = USGOSimple(n_layers=args.n_layers, depth=50.0)
    else:
        model = USGOModel(n_layers=args.n_layers, hidden_dim=args.hidden_dim,
                          depth=50.0)
    model = model.to(args.device)

    n_params = model.count_parameters()
    print(f"USGO-{args.model}: {args.n_layers} layers, {n_params} params")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-5)

    grid_x, grid_y = torch.linspace(0, 250, args.n_grid), \
                      torch.linspace(-100, 100, args.n_grid)

    best_val_recon = float('inf')
    history = []

    print(f"\n{'Ep':>4} {'TrLoss':>9} {'TrRecon':>9} {'ValRecon':>9} "
          f"{'InitRec':>9} {'Improv%':>8} {'ValObs':>9} {'InitObs':>9} "
          f"{'OImpv%':>7} {'Time':>5}")
    print("-" * 95)

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        train_loss, train_recon, train_init = train_one_epoch(
            model, dataset, optimizer, grid_x, grid_y,
            args.device, train_idx, args.deep_sup_weight)

        val = evaluate(model, dataset, grid_x, grid_y,
                       args.device, val_idx)

        scheduler.step()
        elapsed = time.time() - t0

        recon_improv = (1 - val['recon'] / (val['init_recon'] + 1e-12)) * 100
        obs_improv = (1 - val['obs'] / (val['init_obs'] + 1e-12)) * 100

        lr = scheduler.get_last_lr()[0]
        history.append({
            'epoch': epoch,
            'train_loss': train_loss,
            'train_recon': train_recon,
            'val_recon': val['recon'],
            'val_init_recon': val['init_recon'],
            'recon_improv': recon_improv,
            'val_obs': val['obs'],
            'val_init_obs': val['init_obs'],
            'obs_improv': obs_improv,
        })

        if epoch <= 5 or epoch % 10 == 0 or epoch == args.epochs:
            print(f"{epoch:4d} {train_loss:9.5f} {train_recon:9.5f} "
                  f"{val['recon']:9.5f} {val['init_recon']:9.5f} "
                  f"{recon_improv:7.1f}% {val['obs']:9.5f} "
                  f"{val['init_obs']:9.5f} {obs_improv:6.1f}% "
                  f"{elapsed:4.1f}s")

        if val['recon'] < best_val_recon:
            best_val_recon = val['recon']
            os.makedirs(args.save_dir, exist_ok=True)
            torch.save({
                'model_state_dict': model.state_dict(),
                'model_type': args.model,
                'n_layers': args.n_layers,
                'hidden_dim': args.hidden_dim,
                'depth': 50.0,
                'epoch': epoch,
                'val_recon': val['recon'],
                'val_init_recon': val['init_recon'],
                'n_params': n_params,
            }, os.path.join(args.save_dir, 'usgo_best.pt'))

    best = min(history, key=lambda h: h['val_recon'])
    print(f"\nBest: epoch {best['epoch']}, "
          f"val_recon={best['val_recon']:.5f} ({best['recon_improv']:.1f}%), "
          f"val_obs={best['val_obs']:.5f} ({best['obs_improv']:.1f}%)")

    with open(os.path.join(args.save_dir, 'train_history.json'), 'w') as f:
        json.dump(history, f, indent=2)


if __name__ == '__main__':
    main()
