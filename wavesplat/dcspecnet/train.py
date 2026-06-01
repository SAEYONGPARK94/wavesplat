"""
Training script for DCSpecNet.

Usage:
    python -m dcspecnet.train --epochs 100 --batch_size 32
    python -m dcspecnet.train --from_episodes data/episode_*.h5
"""

import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import numpy as np
import json
from pathlib import Path

from dcspecnet.model import DCSpecNet, DCSpecNetConfig, DCSpecNetLoss
from dcspecnet.dataset import OnTheFlyDataset, DataConfig


def train_epoch(model, loader, loss_fn, optimizer, device, epoch=0):
    model.train()
    total_loss = 0
    n_batches = 0
    metrics = {'recon': 0, 'sparse': 0, 'n_active': 0}

    for batch in loader:
        points = batch['points'].to(device)
        t = batch['t'].to(device)

        optimizer.zero_grad()
        params = model(points)
        nv = batch['n_valid'].to(device) if 'n_valid' in batch else None
        loss, info = loss_fn(params, points, t=0.0, epoch=epoch, n_valid=nv)
        loss.backward()

        # Gradient clipping
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += info['total']
        metrics['recon'] += info['recon']
        metrics['sparse'] += info['sparse']
        metrics['n_active'] += info['n_active']
        n_batches += 1

    return {k: v / n_batches for k, v in metrics.items()}, total_loss / n_batches


@torch.no_grad()
def eval_epoch(model, loader, loss_fn, device, epoch=0):
    model.eval()
    total_loss = 0
    n_batches = 0
    metrics = {'recon': 0, 'sparse': 0, 'n_active': 0}

    for batch in loader:
        points = batch['points'].to(device)
        params = model(points)
        nv = batch['n_valid'].to(device) if 'n_valid' in batch else None
        loss, info = loss_fn(params, points, t=0.0, epoch=epoch, n_valid=nv)

        total_loss += info['total']
        metrics['recon'] += info['recon']
        metrics['sparse'] += info['sparse']
        metrics['n_active'] += info['n_active']
        n_batches += 1

    return {k: v / n_batches for k, v in metrics.items()}, total_loss / n_batches


def main():
    parser = argparse.ArgumentParser(description='Train DCSpecNet')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--n_samples', type=int, default=10000)
    parser.add_argument('--n_slots', type=int, default=16)
    parser.add_argument('--embed_dim', type=int, default=128)
    parser.add_argument('--depth', type=float, default=50.0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--save_dir', type=str, default='checkpoints/dcspecnet')
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    # Dataset
    data_cfg = DataConfig(max_points=200, depth=args.depth)
    dataset = OnTheFlyDataset(n_samples=args.n_samples, config=data_cfg,
                              seed=args.seed)

    # Train/val split
    n_val = max(1, int(0.1 * len(dataset)))
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(dataset, [n_train, n_val])

    train_loader = DataLoader(train_set, batch_size=args.batch_size,
                              shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=args.batch_size,
                            shuffle=False, num_workers=0)

    # Model
    model_cfg = DCSpecNetConfig(
        embed_dim=args.embed_dim,
        k_grid_n=args.n_slots,  # reuse n_slots as grid size per axis
        depth=args.depth,
    )
    model = DCSpecNet(model_cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    K = args.n_slots ** 2
    print(f'DCSpecNet: {n_params:,} parameters, {K} grid points ({args.n_slots}x{args.n_slots})')

    # Loss and optimizer
    loss_fn = DCSpecNetLoss(lambda_sparse=0.001)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-5)

    # Training loop
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    best_val_loss = float('inf')
    history = []

    for epoch in range(1, args.epochs + 1):
        train_metrics, train_loss = train_epoch(
            model, train_loader, loss_fn, optimizer, device, epoch=epoch)
        val_metrics, val_loss = eval_epoch(
            model, val_loader, loss_fn, device, epoch=epoch)
        scheduler.step()

        lr = optimizer.param_groups[0]['lr']
        print(f'Epoch {epoch:3d}/{args.epochs} | '
              f'Train: recon={train_metrics["recon"]:.6f} '
              f'N_act={train_metrics["n_active"]:.1f} | '
              f'Val: recon={val_metrics["recon"]:.6f} '
              f'N_act={val_metrics["n_active"]:.1f} | '
              f'lr={lr:.2e}')

        history.append({
            'epoch': epoch, 'train_loss': train_loss, 'val_loss': val_loss,
            **{f'train_{k}': v for k, v in train_metrics.items()},
            **{f'val_{k}': v for k, v in val_metrics.items()},
        })

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'config': model_cfg,
                'val_loss': val_loss,
            }, save_dir / 'best.pt')

    # Save final
    torch.save({
        'epoch': args.epochs,
        'model_state_dict': model.state_dict(),
        'config': model_cfg,
    }, save_dir / 'final.pt')

    with open(save_dir / 'history.json', 'w') as f:
        json.dump(history, f, indent=2)

    print(f'\nTraining complete. Best val loss: {best_val_loss:.6f}')
    print(f'Saved to {save_dir}/')


if __name__ == '__main__':
    main()
