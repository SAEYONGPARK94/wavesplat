#!/usr/bin/env python3
"""
Train the Neural Field Corrector on HOS training data.

Supports two architectures and three residual targets:
  - Architecture: V1 (DeepSets) and V2 (Set Transformer with self-attention)
  - Targets: 'full' (M3-M1), '3rd' (M3-M2), 'nn' (M3-M1-Stokes2)

Usage:
    python3 -m scripts.train_neural_corrector --data data/nn_train_1k.npz --arch v2 --target nn
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import sys
import os
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from wavesplat.model.neural_corrector import NeuralFieldCorrector, NeuralFieldCorrectorV2


class NNTrainingDataset(torch.utils.data.Dataset):
    """Dataset for neural field corrector training."""

    def __init__(self, data_path: str, horizon_idx: int = 1, target: str = 'full'):
        data = np.load(data_path)
        self.kx = torch.tensor(data['kx'], dtype=torch.float32)
        self.ky = torch.tensor(data['ky'], dtype=torch.float32)
        self.w = torch.tensor(data['w'], dtype=torch.float32)
        self.phi = torch.tensor(data['phi'], dtype=torch.float32)
        self.omega = torch.tensor(data['omega'], dtype=torch.float32)
        self.Hs = torch.tensor(data['Hs'], dtype=torch.float32)

        # Select target residual
        if target == 'full':
            key = 'residuals_full' if 'residuals_full' in data else 'residuals'
        elif target == '3rd':
            key = 'residuals_3rd'
        elif target == 'nn':
            key = 'residuals_nn'
        else:
            raise ValueError(f"Unknown target: {target}")

        self.residuals = torch.tensor(data[key][:, horizon_idx],
                                       dtype=torch.float32)
        self.eta_hos = torch.tensor(data['eta_hos'][:, horizon_idx],
                                     dtype=torch.float32)

        self.x_grid = torch.tensor(data['x_grid'], dtype=torch.float32)
        self.y_grid = torch.tensor(data['y_grid'], dtype=torch.float32)
        self.horizon = float(data['horizons'][horizon_idx])
        self.depth = float(data['depth'])

        # Filter NaN samples
        valid = ~torch.any(torch.isnan(self.residuals), dim=1)
        if not valid.all():
            n_removed = (~valid).sum().item()
            print(f"  Removed {n_removed} NaN samples")
            self.kx = self.kx[valid]
            self.ky = self.ky[valid]
            self.w = self.w[valid]
            self.phi = self.phi[valid]
            self.omega = self.omega[valid]
            self.Hs = self.Hs[valid]
            self.residuals = self.residuals[valid]
            self.eta_hos = self.eta_hos[valid]

    def __len__(self):
        return len(self.kx)

    def __getitem__(self, idx):
        return {
            'kx': self.kx[idx],
            'ky': self.ky[idx],
            'w': self.w[idx],
            'phi': self.phi[idx],
            'omega': self.omega[idx],
            'residual': self.residuals[idx],
            'eta_hos': self.eta_hos[idx],
            'Hs': self.Hs[idx],
        }


def create_model(arch, hidden_dim, n_layers, n_heads):
    if arch == 'v1':
        return NeuralFieldCorrector(hidden_dim=hidden_dim, n_layers=n_layers)
    elif arch == 'v2':
        return NeuralFieldCorrectorV2(
            hidden_dim=hidden_dim, n_heads=n_heads, n_attn_layers=n_layers)
    else:
        raise ValueError(f"Unknown arch: {arch}")


def train(args):
    print(f"Loading data from {args.data}...")
    data_check = np.load(args.data)
    horizons = data_check['horizons']
    n_samples = len(data_check['kx'])
    print(f"  {n_samples} samples, horizons={horizons}")

    available_targets = []
    for key in ['residuals_full', 'residuals_3rd', 'residuals_nn', 'residuals']:
        if key in data_check:
            available_targets.append(key)
    print(f"  Available: {available_targets}")

    dataset = NNTrainingDataset(args.data, horizon_idx=args.horizon_idx,
                                 target=args.target)
    horizon_t = dataset.horizon
    depth = dataset.depth
    x_grid = dataset.x_grid
    y_grid = dataset.y_grid

    # Report target statistics
    res_std = dataset.residuals.std().item()
    hos_std = dataset.eta_hos.std().item()
    print(f"  Target '{args.target}': horizon={horizon_t}s, {len(dataset)} samples")
    print(f"  Residual std={res_std:.5f}m, HOS std={hos_std:.4f}m, ratio={res_std/hos_std:.4f}")

    # Train/val split
    n_train = int(0.8 * len(dataset))
    n_val = len(dataset) - n_train
    train_set, val_set = torch.utils.data.random_split(
        dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(42))

    # Spatial subsampling: use subset of grid points per forward pass (speed)
    n_grid_total = len(x_grid)
    n_subsample = min(args.n_subsample, n_grid_total)

    # Batch size > 1: accumulate gradients over mini-batch
    batch_size = min(args.batch_size, n_train)
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=1, shuffle=True)
    val_loader = torch.utils.data.DataLoader(
        val_set, batch_size=1, shuffle=False)

    # Model
    model = create_model(args.arch, args.hidden_dim, args.n_layers, args.n_heads)
    print(f"  Model: {args.arch}, {model.count_parameters()} params")

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)

    best_val_loss = float('inf')
    best_state = None
    patience = 0
    max_patience = 50

    for epoch in range(args.epochs):
        model.train()
        train_losses = []
        optimizer.zero_grad()

        for i, batch in enumerate(train_loader):
            kx = batch['kx'].squeeze(0)
            ky = batch['ky'].squeeze(0)
            w = batch['w'].squeeze(0)
            phi_b = batch['phi'].squeeze(0)
            omega = batch['omega'].squeeze(0)
            target = batch['residual'].squeeze(0)

            mask = torch.abs(w) > 1e-10
            if mask.sum() < 3:
                continue

            # Spatial subsampling for training speed
            if n_subsample < n_grid_total:
                idx = torch.randperm(n_grid_total)[:n_subsample]
                x_sub = x_grid[idx]
                y_sub = y_grid[idx]
                target_sub = target[idx]
            else:
                x_sub = x_grid
                y_sub = y_grid
                target_sub = target

            delta_eta = model(kx[mask], ky[mask], w[mask], phi_b[mask],
                              omega[mask], depth, x_sub, y_sub, horizon_t)

            loss_mse = nn.functional.mse_loss(delta_eta, target_sub)
            loss_reg = 0.001 * (delta_eta**2).mean()
            loss = (loss_mse + loss_reg) / batch_size

            loss.backward()
            train_losses.append(loss_mse.item())

            # Gradient accumulation
            if (i + 1) % batch_size == 0 or (i + 1) == len(train_loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()

        scheduler.step()

        # Validation
        model.eval()
        val_losses = []
        val_nrmses = []

        with torch.no_grad():
            for batch in val_loader:
                kx = batch['kx'].squeeze(0)
                ky = batch['ky'].squeeze(0)
                w = batch['w'].squeeze(0)
                phi_b = batch['phi'].squeeze(0)
                omega = batch['omega'].squeeze(0)
                target = batch['residual'].squeeze(0)

                mask = torch.abs(w) > 1e-10
                if mask.sum() < 3:
                    continue

                delta_eta = model(kx[mask], ky[mask], w[mask], phi_b[mask],
                                  omega[mask], depth, x_grid, y_grid, horizon_t)

                loss_mse = nn.functional.mse_loss(delta_eta, target)
                val_losses.append(loss_mse.item())

                residual_std = target.std()
                if residual_std > 1e-6:
                    nrmse = torch.sqrt(loss_mse) / residual_std
                    val_nrmses.append(nrmse.item())

        mean_train = np.mean(train_losses) if train_losses else 0
        mean_val = np.mean(val_losses) if val_losses else 0
        mean_nrmse = np.mean(val_nrmses) if val_nrmses else 0

        if mean_val < best_val_loss:
            best_val_loss = mean_val
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:3d}: train_mse={mean_train:.6f}, "
                  f"val_mse={mean_val:.6f}, val_nrmse={mean_nrmse:.3f}, "
                  f"lr={scheduler.get_last_lr()[0]:.6f}")

        if patience >= max_patience:
            print(f"  Early stopping at epoch {epoch+1}")
            break

    # Save best model
    if best_state is not None:
        model.load_state_dict(best_state)

    os.makedirs('checkpoints', exist_ok=True)
    save_path = f'checkpoints/field_corrector_{args.arch}_{args.target}_h{int(horizon_t)}s.pt'
    torch.save({
        'model_state_dict': model.state_dict(),
        'arch': args.arch,
        'hidden_dim': args.hidden_dim,
        'n_layers': args.n_layers,
        'n_heads': args.n_heads,
        'target': args.target,
        'horizon': horizon_t,
        'depth': depth,
        'best_val_loss': best_val_loss,
    }, save_path)
    print(f"\nBest model saved to {save_path} (val_mse={best_val_loss:.6f})")

    # Final evaluation
    print("\nFinal evaluation on validation set:")
    model.eval()
    with torch.no_grad():
        total_residual_var = 0
        total_pred_mse = 0
        n_eval = 0
        for batch in val_loader:
            kx = batch['kx'].squeeze(0)
            ky = batch['ky'].squeeze(0)
            w = batch['w'].squeeze(0)
            phi_b = batch['phi'].squeeze(0)
            omega = batch['omega'].squeeze(0)
            target = batch['residual'].squeeze(0)

            mask = torch.abs(w) > 1e-10
            if mask.sum() < 3:
                continue

            delta_eta = model(kx[mask], ky[mask], w[mask], phi_b[mask],
                              omega[mask], depth, x_grid, y_grid, horizon_t)

            total_residual_var += target.var().item()
            total_pred_mse += nn.functional.mse_loss(delta_eta, target).item()
            n_eval += 1

    if n_eval > 0:
        r2 = 1 - total_pred_mse / total_residual_var
        mean_mse = total_pred_mse / n_eval
        mean_var = total_residual_var / n_eval
        nrmse = np.sqrt(mean_mse / mean_var) if mean_var > 0 else float('inf')
        print(f"  R² = {r2:.4f}")
        print(f"  Mean MSE = {mean_mse:.6f}")
        print(f"  Mean nRMSE = {nrmse:.4f}")

    return best_val_loss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='data/nn_train_1k.npz')
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--hidden_dim', type=int, default=64)
    parser.add_argument('--n_layers', type=int, default=2)
    parser.add_argument('--n_heads', type=int, default=4)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--horizon_idx', type=int, default=1,
                        help='Horizon index (0=5s, 1=10s)')
    parser.add_argument('--arch', type=str, default='v2',
                        choices=['v1', 'v2'],
                        help='Architecture: v1=DeepSets, v2=SetTransformer')
    parser.add_argument('--target', type=str, default='full',
                        choices=['full', '3rd', 'nn'],
                        help='Residual target: full=M3-M1, 3rd=M3-M2, nn=M3-M1-S2')
    parser.add_argument('--n_subsample', type=int, default=256,
                        help='Spatial grid points per forward pass (train speed)')
    args = parser.parse_args()
    train(args)


if __name__ == '__main__':
    main()
