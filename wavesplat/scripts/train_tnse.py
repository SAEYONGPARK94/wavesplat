"""
train_tnse.py — T-NSE training script.

Stage 1 (sanity):  overfit on a single episode → verify model can learn k-vectors
Stage 2 (general): train on 24 episodes, validate on 6 → check generalisation

Loss: reconstruction nRMSE on the target frame GT field,
      computed via differentiable LS amplitude fitting.

Usage:
  python scripts/train_tnse.py --stage 1              # sanity overfit
  python scripts/train_tnse.py --stage 2              # full training
  python scripts/train_tnse.py --stage 2 --resume results/tnse_best.pt
"""

import os, sys, json, time, argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
os.chdir(os.path.join(os.path.dirname(__file__), '..'))

from wavesplat.model.tnse import TNSE, tnse_forward_with_ls, render_wave_field_torch, differentiable_ls_fit
from wavesplat.model.tnse_dataset import TNSEDataset, collate_fn

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ── Episode splits ────────────────────────────────────────────────────────────
SEA_STATES = [('BF3', 5), ('BF4', 5), ('BF4', 15), ('BF4', 30), ('BF5', 15), ('BF5', 30)]

TRAIN_FILES = [f'episode_{ss}_{nc}comp_seed{s}_uniform.h5'
               for ss, nc in SEA_STATES for s in [0, 1, 2, 3]]  # 24 episodes

VAL_FILES   = [f'episode_{ss}_{nc}comp_seed{s}_uniform.h5'
               for ss, nc in SEA_STATES for s in [4]]           # 6 episodes

STAGE1_FILE = ['episode_BF4_5comp_seed0_uniform.h5']  # single easy episode


# ── Training configuration ────────────────────────────────────────────────────
CFG = {
    'n_components': 15,
    'W': 10,
    'batch_size': 16,
    'lr': 3e-4,
    'weight_decay': 1e-4,
    'lambda_reg': 0.5,   # LS regularisation
    'lambda_l1': 0.0,    # amplitude sparsity (off: it biased amplitudes low → low VR)
    'lambda_dir': 5.0,   # Hungarian matched-direction MSE weight
    'lambda_conf': 0.5,  # objectness BCE weight (no-object class)
    'n_epochs_stage1': 200,
    'n_epochs_stage2': 100,
    'd_model': 128,
    'n_heads': 4,
    'n_layers': 4,
    'd_ff': 256,
    'dropout': 0.1,
}


def hungarian_direction_loss(kx_pred, ky_pred, conf_logit, true_k, true_w, eps=1e-8):
    """
    DETR-style bipartite-matching direction loss.

    The earlier energy-weighted Chamfer (true → nearest pred) was satisfiable by a
    single fixed dictionary of directions (multiple true components can share one
    prediction), so the model collapsed to an input-invariant constant predictor.
    Hungarian 1:1 matching forbids that: each true component is assigned to a
    distinct prediction, so a fixed dictionary cannot simultaneously fit diverse
    episodes — the model is forced to produce input-specific directions.

    Args:
        kx_pred, ky_pred: (N,) predicted wavenumber components [rad/m]
        conf_logit:       (N,) objectness logits
        true_k: (Nt, 2) ground-truth (kx, ky)
        true_w: (Nt,) energy weights (A^2)

    Returns:
        (d_match, l_conf): energy-weighted matched direction MSE, and objectness BCE.
    """
    N = kx_pred.shape[0]
    device = kx_pred.device

    if true_k.shape[0] == 0:
        zero = torch.zeros((), device=device)
        l_conf = F.binary_cross_entropy_with_logits(conf_logit, torch.zeros(N, device=device))
        return zero, l_conf

    pred = torch.stack([kx_pred, ky_pred], dim=1)   # (N, 2)
    Nt = true_k.shape[0]
    # More true components than queries → keep the top-N by energy.
    if Nt > N:
        idx = torch.topk(true_w, N).indices
        true_k = true_k[idx]; true_w = true_w[idx]; Nt = N

    C = ((pred[:, None, :] - true_k[None, :, :]) ** 2).sum(-1)   # (N, Nt) squared-L2 cost
    with torch.no_grad():
        row, col = linear_sum_assignment(C.detach().cpu().numpy())
    row = torch.as_tensor(row, device=device, dtype=torch.long)
    col = torch.as_tensor(col, device=device, dtype=torch.long)

    w = true_w[col] / (true_w[col].sum() + eps)                  # energy-weight matched pairs
    d_match = (w * C[row, col]).sum()

    conf_t = torch.zeros(N, device=device)
    conf_t[row] = 1.0                                           # matched → object, else no-object
    l_conf = F.binary_cross_entropy_with_logits(conf_logit, conf_t)
    return d_match, l_conf


def compute_batch_loss(model, batch, lambda_reg=0.5, lambda_l1=0.0,
                       lambda_dir=5.0, lambda_conf=0.5):
    """
    Forward pass + differentiable LS + reconstruction loss + direction supervision.

    Returns (loss, recon_nRMSE). The returned nRMSE is reconstruction-only
    (the val metric); the optimised loss additionally includes the Hungarian
    direction-matching and objectness terms.
    """
    frames_tokens = batch['frames_tokens'].to(DEVICE)   # (B, W, P, 5)
    frames_mask   = batch['frames_mask'].to(DEVICE)     # (B, W, P)
    frame_ids     = batch['frame_ids'].to(DEVICE)       # (B, W)
    gt_fields     = batch['gt_field'].to(DEVICE)        # (B, Ny, Nx)
    Hs_batch      = batch['Hs'].to(DEVICE)              # (B,)

    B = frames_tokens.shape[0]
    depths = batch['depth']  # list or tensor

    # Predict kx, ky, objectness
    kx_pred, ky_pred, conf_pred = model(frames_tokens, frames_mask, frame_ids)  # (B, N)

    total_loss = torch.tensor(0.0, device=DEVICE)
    nrmse_list = []

    for i in range(B):
        depth = float(depths[i])
        Hs = float(Hs_batch[i])

        # Fit A, phi over the FULL window (all W frames) — not just the current
        # frame. This matches eval_tnse.py (which also uses the full window) and
        # is far better conditioned: ~500 obs for 2N params vs ~50, and it
        # exploits the dispersion-driven temporal structure across frames.
        wobs = batch['window_obs'][i].to(DEVICE)   # (M, 4): x, y, t, z
        if wobs.shape[0] < 2 * model.n_components:
            continue
        wx, wy, wt, wz = wobs[:, 0], wobs[:, 1], wobs[:, 2], wobs[:, 3]

        kx_i = kx_pred[i:i+1]  # (1, N)
        ky_i = ky_pred[i:i+1]  # (1, N)

        Phi = render_wave_field_torch(kx_i, ky_i, wx, wy, wt, depth)  # (1, M, 2N)
        coef = differentiable_ls_fit(Phi, wz.unsqueeze(0), lambda_reg)  # (1, 2N)

        N = kx_i.shape[1]
        a = coef[0, :N]; b = coef[0, N:]
        A = torch.sqrt(a**2 + b**2 + 1e-8)  # (N,)

        # Render GT field on grid for loss
        gx = batch['gt_grid_x'][i].to(DEVICE)  # (Nx,)
        gy = batch['gt_grid_y'][i].to(DEVICE)  # (Ny,)
        Nx, Ny = len(gx), len(gy)
        Xg, Yg = torch.meshgrid(gx, gy, indexing='xy')  # (Ny, Nx)
        x_flat = Xg.reshape(-1); y_flat = Yg.reshape(-1)

        t_target = batch['t_target'][i].item()
        t_flat = torch.full_like(x_flat, t_target)

        phi_i = torch.atan2(-b, a)  # (N,)
        A_i = A.unsqueeze(0)        # (1, N)
        phi_t = phi_i.unsqueeze(0)  # (1, N)

        # Render predicted field
        pred_flat = render_wave_field_torch(
            kx_i, ky_i, x_flat, y_flat, t_flat, depth, A=A_i, phi=phi_t
        ).squeeze(0)  # (Ny*Nx,)
        pred_field = pred_flat.reshape(Ny, Nx)

        # nRMSE loss
        gt = gt_fields[i]  # (Ny, Nx)
        mse = torch.mean((pred_field - gt)**2)
        nrmse = torch.sqrt(mse) / max(Hs, 0.1)

        # L1 sparsity on amplitudes
        l1 = A.mean()

        # Supervised direction term — Hungarian 1:1 matching forces input-specific
        # directions (a fixed dictionary can't satisfy the assignment across episodes).
        tk = batch['true_k'][i].to(DEVICE)   # (Nt, 2)
        tw = batch['true_w'][i].to(DEVICE)   # (Nt,)
        d_match, l_conf = hungarian_direction_loss(
            kx_pred[i], ky_pred[i], conf_pred[i], tk, tw)

        loss_i = nrmse + lambda_l1 * l1 + lambda_dir * d_match + lambda_conf * l_conf
        total_loss = total_loss + loss_i
        nrmse_list.append(nrmse.item())

    return total_loss / max(B, 1), float(np.mean(nrmse_list)) if nrmse_list else 1.0


@torch.no_grad()
def evaluate(model, loader_val, lambda_reg=0.5):
    model.eval()
    nrmse_all = []
    for batch in loader_val:
        _, nrmse = compute_batch_loss(model, batch, lambda_reg=lambda_reg, lambda_l1=0.0)
        nrmse_all.append(nrmse)
    return float(np.mean(nrmse_all))


def train(args):
    stage = args.stage
    data_dir = 'data'

    if stage == 1:
        h5_train = STAGE1_FILE; h5_val = STAGE1_FILE
        n_epochs = CFG['n_epochs_stage1']
        print("Stage 1: Sanity overfit on single episode")
    else:
        h5_train = TRAIN_FILES; h5_val = VAL_FILES
        n_epochs = CFG['n_epochs_stage2']
        print("Stage 2: Full training (24 train, 6 val episodes)")

    print(f"Device: {DEVICE}")

    # augment=False: horizontal-flip augmentation was removed (see tnse_dataset.py).
    # Re-enable here once proper per-window reflection + GT flip is implemented.
    ds_train = TNSEDataset(data_dir, h5_train, W=CFG['W'], augment=False)
    ds_val   = TNSEDataset(data_dir, h5_val,   W=CFG['W'], augment=False)

    if len(ds_train) == 0:
        print("ERROR: no training data. Generate episodes first.")
        return

    dl_train = DataLoader(ds_train, batch_size=CFG['batch_size'], shuffle=True,
                          num_workers=2, pin_memory=True, collate_fn=collate_fn,
                          drop_last=True)
    dl_val   = DataLoader(ds_val,   batch_size=CFG['batch_size'], shuffle=False,
                          num_workers=2, collate_fn=collate_fn)

    model = TNSE(
        n_components=CFG['n_components'],
        depth=50.0,
        d_model=CFG['d_model'],
        n_heads=CFG['n_heads'],
        n_layers=CFG['n_layers'],
        d_ff=CFG['d_ff'],
        dropout=CFG['dropout'],
    ).to(DEVICE)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {n_params/1e6:.2f}M")

    if args.resume and os.path.exists(args.resume):
        ckpt = torch.load(args.resume, map_location=DEVICE)
        model.load_state_dict(ckpt['model'])
        print(f"Resumed from {args.resume}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=CFG['lr'],
                                  weight_decay=CFG['weight_decay'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)

    best_val_nrmse = float('inf')
    best_ckpt = f'results/tnse_stage{stage}_best.pt'
    log_path  = f'results/tnse_stage{stage}_log.json'
    log = []

    print(f"\n{'Epoch':>6}  {'Train nRMSE':>12}  {'Val nRMSE':>12}  {'LR':>10}  {'Time':>6}")
    print("-" * 56)

    for epoch in range(1, n_epochs + 1):
        model.train()
        t0 = time.time()
        train_nrmse = []

        for batch in dl_train:
            optimizer.zero_grad()
            loss, nrmse = compute_batch_loss(
                model, batch, lambda_reg=CFG['lambda_reg'],
                lambda_l1=CFG['lambda_l1'], lambda_dir=CFG['lambda_dir'],
                lambda_conf=CFG['lambda_conf'])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_nrmse.append(nrmse)

        scheduler.step()
        tr_nrmse = float(np.mean(train_nrmse))
        val_nrmse = evaluate(model, dl_val, lambda_reg=CFG['lambda_reg'])
        elapsed = time.time() - t0
        lr = optimizer.param_groups[0]['lr']

        print(f"{epoch:>6}  {tr_nrmse:>12.4f}  {val_nrmse:>12.4f}  {lr:>10.2e}  {elapsed:>5.1f}s")
        log.append({'epoch': epoch, 'train_nRMSE': tr_nrmse, 'val_nRMSE': val_nrmse, 'lr': lr})

        if val_nrmse < best_val_nrmse:
            best_val_nrmse = val_nrmse
            torch.save({'model': model.state_dict(), 'cfg': CFG,
                        'epoch': epoch, 'val_nRMSE': val_nrmse}, best_ckpt)

        # Early stopping for Stage 1 (overfit check)
        if stage == 1 and tr_nrmse < 0.05:
            print(f"Stage 1: overfit achieved at epoch {epoch}. nRMSE={tr_nrmse:.4f}")
            break

    print(f"\nBest val nRMSE: {best_val_nrmse:.4f}  → {best_ckpt}")
    with open(log_path, 'w') as f:
        json.dump({'config': CFG, 'best_val_nRMSE': best_val_nrmse,
                   'stage': stage, 'log': log}, f, indent=2)
    print(f"Log saved: {log_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage',  type=int, default=1, choices=[1, 2])
    parser.add_argument('--resume', type=str, default=None)
    args = parser.parse_args()
    train(args)
