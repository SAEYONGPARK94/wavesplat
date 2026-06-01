#!/usr/bin/env python3
"""
Evaluate trained Neural Field Corrector on held-out HOS test data.

Computes:
  - nRMSE of NN prediction vs true residual
  - R² score
  - Comparison: (Linear only) vs (Linear + Stokes2) vs (Linear + Stokes2 + NN)
  - Per-sea-state breakdown

Usage:
    python3 -m scripts.evaluate_nn_corrector --data data/nn_train_1k.npz --checkpoint checkpoints/field_corrector_v2_nn_h10s.pt
"""

import numpy as np
import torch
import sys
import os
import argparse
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from wavesplat.model.neural_corrector import NeuralFieldCorrector, NeuralFieldCorrectorV2


def load_model(checkpoint_path):
    ckpt = torch.load(checkpoint_path, map_location='cpu')
    arch = ckpt.get('arch', 'v1')
    if arch == 'v2':
        model = NeuralFieldCorrectorV2(
            hidden_dim=ckpt['hidden_dim'],
            n_heads=ckpt.get('n_heads', 4),
            n_attn_layers=ckpt['n_layers'],
        )
    else:
        model = NeuralFieldCorrector(
            hidden_dim=ckpt['hidden_dim'],
            n_layers=ckpt['n_layers'],
        )
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    return model, ckpt


def evaluate(args):
    print(f"Loading data from {args.data}...")
    data = np.load(args.data)

    # Load model
    print(f"Loading model from {args.checkpoint}...")
    model, ckpt = load_model(args.checkpoint)
    arch = ckpt.get('arch', 'v1')
    target = ckpt.get('target', 'full')
    print(f"  Arch: {arch}, target: {target}, horizon: {ckpt['horizon']}s")
    print(f"  Params: {model.count_parameters()}")
    print(f"  Best val_mse: {ckpt['best_val_loss']:.6f}")

    horizon_idx = args.horizon_idx
    horizons = data['horizons']
    depth = float(data['depth'])
    x_grid = torch.tensor(data['x_grid'], dtype=torch.float32)
    y_grid = torch.tensor(data['y_grid'], dtype=torch.float32)
    horizon_t = float(horizons[horizon_idx])

    # Use last 20% as test set (same split as training)
    n_total = len(data['kx'])
    n_train = int(0.8 * n_total)
    test_indices = list(range(n_train, n_total))

    print(f"\n  Evaluating on {len(test_indices)} test samples, horizon={horizon_t}s")

    # Select target residual for evaluation
    if 'residuals_full' in data:
        residual_full = data['residuals_full'][:, horizon_idx]
        residual_nn = data['residuals_nn'][:, horizon_idx]
        residual_3rd = data['residuals_3rd'][:, horizon_idx]
        stokes2_corr = data['stokes2_corr'][:, horizon_idx]
        has_detailed = True
    else:
        residual_full = data['residuals'][:, horizon_idx]
        has_detailed = False

    eta_m3 = data['eta_hos'][:, horizon_idx]
    eta_m1 = data['eta_pred'][:, horizon_idx]

    results = {
        'arch': arch, 'target': target, 'horizon': horizon_t,
        'n_test': len(test_indices),
        'samples': [],
    }

    # Aggregate metrics
    total_nn_mse = 0.0
    total_target_var = 0.0
    total_m1_mse = 0.0  # Linear vs M3
    total_m1s2_mse = 0.0  # Linear+Stokes2 vs M3
    total_m1s2nn_mse = 0.0  # Linear+Stokes2+NN vs M3
    total_m3_var = 0.0

    with torch.no_grad():
        for idx in test_indices:
            kx = torch.tensor(data['kx'][idx], dtype=torch.float32)
            ky = torch.tensor(data['ky'][idx], dtype=torch.float32)
            w = torch.tensor(data['w'][idx], dtype=torch.float32)
            phi = torch.tensor(data['phi'][idx], dtype=torch.float32)
            omega = torch.tensor(data['omega'][idx], dtype=torch.float32)

            mask = torch.abs(w) > 1e-10
            if mask.sum() < 3:
                continue

            # NN prediction
            nn_pred = model(kx[mask], ky[mask], w[mask], phi[mask],
                            omega[mask], depth, x_grid, y_grid, horizon_t)
            nn_pred_np = nn_pred.numpy()

            # Target residual (what NN should predict)
            if target == 'nn' and has_detailed:
                target_res = residual_nn[idx]
            elif target == '3rd' and has_detailed:
                target_res = residual_3rd[idx]
            else:
                target_res = residual_full[idx]

            # Metrics for NN prediction vs target
            nn_mse = np.mean((nn_pred_np - target_res)**2)
            target_var = np.var(target_res)
            total_nn_mse += nn_mse
            total_target_var += target_var

            # Full pipeline comparison (all relative to HOS M3)
            m3 = eta_m3[idx]
            m1 = eta_m1[idx]
            m3_var = np.var(m3)
            total_m3_var += m3_var

            # Linear only vs M3
            total_m1_mse += np.mean((m1 - m3)**2)

            # Linear + Stokes2 vs M3
            if has_detailed:
                m1_s2 = m1 + stokes2_corr[idx]
                total_m1s2_mse += np.mean((m1_s2 - m3)**2)

                # Linear + Stokes2 + NN vs M3
                m1_s2_nn = m1_s2 + nn_pred_np
                total_m1s2nn_mse += np.mean((m1_s2_nn - m3)**2)

    n_eval = len(test_indices)
    if n_eval == 0:
        print("  No valid test samples!")
        return

    # NN correction quality
    r2 = 1 - total_nn_mse / max(total_target_var, 1e-12)
    mean_nrmse = np.sqrt(total_nn_mse / n_eval / max(total_target_var / n_eval, 1e-12))

    print(f"\n=== NN Correction Quality (target: {target}) ===")
    print(f"  R² = {r2:.4f}")
    print(f"  nRMSE = {mean_nrmse:.4f}")

    # Full pipeline comparison
    linear_nrmse = np.sqrt(total_m1_mse / n_eval / max(total_m3_var / n_eval, 1e-12))
    print(f"\n=== Full Pipeline Comparison (vs HOS M3) ===")
    print(f"  Linear only:           nRMSE = {linear_nrmse:.4f}")

    if has_detailed:
        s2_nrmse = np.sqrt(total_m1s2_mse / n_eval / max(total_m3_var / n_eval, 1e-12))
        s2nn_nrmse = np.sqrt(total_m1s2nn_mse / n_eval / max(total_m3_var / n_eval, 1e-12))
        print(f"  Linear + Stokes2:      nRMSE = {s2_nrmse:.4f}")
        print(f"  Linear + Stokes2 + NN: nRMSE = {s2nn_nrmse:.4f}")

        improvement_s2 = (linear_nrmse - s2_nrmse) / linear_nrmse * 100
        improvement_nn = (s2_nrmse - s2nn_nrmse) / s2_nrmse * 100
        print(f"\n  Stokes2 improvement: {improvement_s2:.1f}%")
        print(f"  NN improvement (over S2): {improvement_nn:.1f}%")

        results['linear_nrmse'] = float(linear_nrmse)
        results['s2_nrmse'] = float(s2_nrmse)
        results['s2nn_nrmse'] = float(s2nn_nrmse)

    results['r2'] = float(r2)
    results['nrmse'] = float(mean_nrmse)

    # Save results
    os.makedirs('results', exist_ok=True)
    out_path = f'results/nn_corrector_eval_{arch}_{target}_h{int(horizon_t)}s.json'
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='data/nn_train_1k.npz')
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--horizon_idx', type=int, default=1)
    args = parser.parse_args()
    evaluate(args)


if __name__ == '__main__':
    main()
