#!/usr/bin/env python3
"""
pino_stage2_train.py

WavePINO Stage 2: Cross-episode generalization with physics loss.

Two experiments run sequentially:
  (A) FNO baseline  — same loss as ConvCNP v2 (MSE + correlation)
  (B) PINO          — adds spectral physics loss (λ_spec=0.5)

Physics loss:
  L_spec = fraction of 2D FFT power outside the physical wave band
           k_min ≤ |k| ≤ k_max,  where k derived from dispersion ω²=g|k|tanh(|k|d)
  T_min = 4s  → k_max = 0.252 rad/m (wavelength ~25m)
  T_max = 25s → k_min = 0.006 rad/m

Same train/val/test split as ConvCNP v2 for fair comparison.

Usage:
    cd wavesplat && python3 -u scripts/pino_stage2_train.py
"""

import os
import sys
import time
import json
import copy

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', buffering=1)

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(REPO_ROOT, 'wavesplat'))

from wavesplat.data_loader import EpisodeDataLoader
from wavesplat.model.wave_pino import WavePINO, spectral_physics_loss
from wavesplat.evaluation.metrics import nrmse

# ---------------------------------------------------------------------------
# Episode lists (same as ConvCNP v2 for fair comparison)
# ---------------------------------------------------------------------------
TRAIN_EPISODES = [
    'episode_BF3_1comp_seed0_uniform.h5', 'episode_BF3_1comp_seed1_uniform.h5',
    'episode_BF3_2comp_seed0_uniform.h5', 'episode_BF3_2comp_seed1_uniform.h5',
    'episode_BF3_5comp_seed0_uniform.h5', 'episode_BF3_5comp_seed1_uniform.h5',
    'episode_BF3_1comp_seed0_slope.h5',  'episode_BF3_1comp_seed1_slope.h5',
    'episode_BF3_2comp_seed0_slope.h5',  'episode_BF3_2comp_seed1_slope.h5',
    'episode_BF3_5comp_seed0_slope.h5',  'episode_BF3_5comp_seed1_slope.h5',
    'episode_BF4_1comp_seed0_uniform.h5','episode_BF4_1comp_seed1_uniform.h5',
    'episode_BF4_5comp_seed0_uniform.h5','episode_BF4_5comp_seed1_uniform.h5',
    'episode_BF4_15comp_seed0_uniform.h5','episode_BF4_15comp_seed1_uniform.h5',
    'episode_BF4_30comp_seed0_uniform.h5',
    'episode_BF4_1comp_seed0_slope.h5',  'episode_BF4_1comp_seed1_slope.h5',
    'episode_BF4_5comp_seed0_slope.h5',  'episode_BF4_5comp_seed1_slope.h5',
    'episode_BF4_15comp_seed0_slope.h5', 'episode_BF4_15comp_seed1_slope.h5',
    'episode_BF5_15comp_seed0_uniform.h5','episode_BF5_15comp_seed1_uniform.h5',
    'episode_BF5_30comp_seed0_uniform.h5','episode_BF5_30comp_seed1_uniform.h5',
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

# ---------------------------------------------------------------------------
# Model hyperparameters (full model)
# ---------------------------------------------------------------------------
GRID_NX     = 64
GRID_NY     = 48
FNO_WIDTH   = 16
FNO_LAYERS  = 4
FNO_MODES_H = 16   # covers λ_min ≈ 16m in x (240m domain)
FNO_MODES_W = 12   # covers λ_min ≈ 17m in y (200m domain)

# Training
EPOCHS          = 100
STEPS_PER_EPOCH = 100
BATCH_SIZE      = 8
LR              = 1e-3
WARMUP_EPOCHS   = 5
LAMBDA_CORR     = 1.0   # correlation loss weight (same as ConvCNP v2)
LAMBDA_SPEC_B   = 0.5   # spectral physics loss weight for PINO run

# Physics loss parameters
DEPTH = 50.0    # water depth [m]
T_MIN = 4.0     # minimum wave period [s]
T_MAX = 25.0    # maximum wave period [s]

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ---------------------------------------------------------------------------
# Data utilities
# ---------------------------------------------------------------------------

def load_episodes(names, data_dir):
    eps = []
    for n in names:
        p = os.path.join(data_dir, n)
        if os.path.exists(p):
            eps.append(EpisodeDataLoader(p))
    return eps


def correlation_loss(pred, target):
    """Batch-wise 1 - Pearson correlation. Returns in [0, 2]."""
    pc = pred   - pred.mean(dim=1, keepdim=True)
    tc = target - target.mean(dim=1, keepdim=True)
    ps = pc.norm(dim=1, keepdim=True).clamp(min=1e-8)
    ts = tc.norm(dim=1, keepdim=True).clamp(min=1e-8)
    corr = (pc * tc).sum(dim=1) / (ps.squeeze() * ts.squeeze())
    return 1.0 - corr.mean()


def sample_batch_fullgrid(episodes, batch_size, model_grid_x, model_grid_y,
                           gt_grid_x, gt_grid_y, device):
    """Sample batch; resample GT to model grid for full-grid supervision."""
    xc_list, yc_list, gt_list = [], [], []

    for _ in range(batch_size):
        ep = episodes[np.random.randint(len(episodes))]
        fi = np.random.randint(len(ep))
        frame = ep[fi]
        Hs = max(ep.Hs_true, 0.1)

        xc = torch.stack([frame.x_obs, frame.y_obs], dim=-1)
        yc = frame.z_obs.unsqueeze(-1) / Hs
        gt = frame.gt_field / Hs  # [Ny, Nx]

        # Resample GT to model grid [Gy, Gx] then transpose to [Gx, Gy]
        dx0, dx1 = float(gt_grid_x[0]), float(gt_grid_x[-1])
        dy0, dy1 = float(gt_grid_y[0]), float(gt_grid_y[-1])
        x_n = 2.0 * (model_grid_x - dx0) / (dx1 - dx0) - 1.0
        y_n = 2.0 * (model_grid_y - dy0) / (dy1 - dy0) - 1.0
        yy, xx = torch.meshgrid(y_n, x_n, indexing='ij')
        sg = torch.stack([xx, yy], dim=-1).unsqueeze(0)
        gt_r = F.grid_sample(gt.unsqueeze(0).unsqueeze(0).float(), sg,
                              mode='bilinear', align_corners=True,
                              padding_mode='border')[0, 0]  # [Gy, Gx]
        gt_r = gt_r.T  # [Gx, Gy]

        # Y-flip augmentation
        if np.random.random() < 0.5:
            xc = xc.clone()
            xc[:, 1] = -xc[:, 1]
            gt_r = gt_r.flip(1)  # flip Gy dimension

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


def evaluate(model, episodes, gt_grid_x, gt_grid_y, device, n_frames=10):
    model.eval()
    all_nrmse = []
    gx, gy = gt_grid_x, gt_grid_y
    xx, yy = torch.meshgrid(gx, gy, indexing='ij')
    xt = torch.stack([xx.flatten(), yy.flatten()], dim=-1).unsqueeze(0).to(device)

    for ep in episodes:
        Hs = max(ep.Hs_true, 0.1)
        idx_list = np.linspace(10, len(ep) - 1, n_frames, dtype=int)
        for fi in idx_list:
            frame = ep[fi]
            xc = torch.stack([frame.x_obs, frame.y_obs], dim=-1).unsqueeze(0).to(device)
            yc = frame.z_obs.unsqueeze(-1).unsqueeze(0).to(device) / Hs
            with torch.no_grad():
                pred = model(xc, yc, xt)
            pred_field = (pred[0, :, 0] * Hs).cpu().numpy()
            pred_field = pred_field.reshape(len(gx), len(gy)).T  # [Ny, Nx]
            v = nrmse(pred_field, frame.gt_field.numpy(), ep.Hs_true)
            if not np.isnan(v):
                all_nrmse.append(v)

    model.train()
    return float(np.mean(all_nrmse)) if all_nrmse else 1.0


def test_evaluation(model, test_eps, gt_grid_x, gt_grid_y, device, n_frames=20):
    """Detailed per-episode test evaluation."""
    model.eval()
    gx, gy = gt_grid_x, gt_grid_y
    xx, yy = torch.meshgrid(gx, gy, indexing='ij')
    xt = torch.stack([xx.flatten(), yy.flatten()], dim=-1).unsqueeze(0).to(device)
    results = {}

    for ep_idx, ep in enumerate(test_eps):
        Hs = max(ep.Hs_true, 0.1)
        idx_list = np.linspace(10, len(ep) - 1, n_frames, dtype=int)
        ep_nrmse = []
        for fi in idx_list:
            frame = ep[fi]
            xc = torch.stack([frame.x_obs, frame.y_obs], dim=-1).unsqueeze(0).to(device)
            yc = frame.z_obs.unsqueeze(-1).unsqueeze(0).to(device) / Hs
            with torch.no_grad():
                pred = model(xc, yc, xt)
            pred_field = (pred[0, :, 0] * Hs).cpu().numpy()
            pred_field = pred_field.reshape(len(gx), len(gy)).T
            v = nrmse(pred_field, frame.gt_field.numpy(), ep.Hs_true)
            if not np.isnan(v):
                ep_nrmse.append(v)
        ep_name = TEST_EPISODES[ep_idx].replace('episode_', '').replace('.h5', '')
        avg = float(np.mean(ep_nrmse)) if ep_nrmse else 1.0
        results[ep_name] = avg
        print(f"  {ep_name}: {avg:.4f}")

    return results


# ---------------------------------------------------------------------------
# Training loop (shared, parametrised)
# ---------------------------------------------------------------------------

def build_model(x_range, y_range):
    return WavePINO(
        grid_nx=GRID_NX, grid_ny=GRID_NY,
        x_range=x_range, y_range=y_range,
        fno_width=FNO_WIDTH, fno_layers=FNO_LAYERS,
        fno_modes_h=FNO_MODES_H, fno_modes_w=FNO_MODES_W,
        init_lengthscale=10.0,
    ).to(DEVICE)


def train_one_run(tag, lambda_spec, train_eps, val_eps,
                  model_grid_x, model_grid_y, gt_grid_x, gt_grid_y,
                  x_range, y_range, ckpt_dir):
    """Run one full training: FNO-only (lambda_spec=0) or PINO (lambda_spec>0)."""
    print(f"\n{'='*70}")
    print(f"  WavePINO — {tag}  (λ_spec={lambda_spec})")
    print(f"{'='*70}")

    x_extent = x_range[1] - x_range[0]
    y_extent = y_range[1] - y_range[0]

    model = build_model(x_range, y_range)
    print(f"  Model: {model.n_params:,} params")
    print(f"  FNO: width={FNO_WIDTH}, layers={FNO_LAYERS}, "
          f"modes=({FNO_MODES_H},{FNO_MODES_W})")

    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    warmup = optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.01, total_iters=WARMUP_EPOCHS)
    cosine = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS - WARMUP_EPOCHS)
    scheduler = optim.lr_scheduler.SequentialLR(
        optimizer, [warmup, cosine], milestones=[WARMUP_EPOCHS])

    best_val  = 1.0
    best_ckpt = os.path.join(ckpt_dir, f'wave_pino_{tag}_best.pt')
    t_start   = time.time()

    header = f"{'Ep':>4}  {'MSE':>8}  {'Corr':>7}  {'Spec':>7}  {'ValNRMSE':>9}  {'Time':>6}"
    print(f"\n{header}")
    print("-" * len(header))

    for epoch in range(1, EPOCHS + 1):
        model.train()
        ep_mse = ep_corr = ep_spec = 0.0

        for _ in range(STEPS_PER_EPOCH):
            xc, yc, gt = sample_batch_fullgrid(
                train_eps, BATCH_SIZE, model_grid_x, model_grid_y,
                gt_grid_x, gt_grid_y, DEVICE)

            # Full-grid prediction [B, 1, Gx, Gy]
            pred_grid = model.predict_grid(xc, yc)

            # Losses
            pred_flat = pred_grid[:, 0].reshape(BATCH_SIZE, -1)  # [B, Gx*Gy]
            gt_flat   = gt.reshape(BATCH_SIZE, -1)               # [B, Gx*Gy]

            mse_loss  = F.mse_loss(pred_flat, gt_flat)
            corr_loss = correlation_loss(pred_flat, gt_flat)

            loss = mse_loss + LAMBDA_CORR * corr_loss

            if lambda_spec > 0.0:
                spec_loss = spectral_physics_loss(
                    pred_grid, x_extent, y_extent, DEPTH, T_MIN, T_MAX)
                loss = loss + lambda_spec * spec_loss
                ep_spec += spec_loss.item()

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            ep_mse  += mse_loss.item()
            ep_corr += corr_loss.item()

        scheduler.step()
        ep_mse  /= STEPS_PER_EPOCH
        ep_corr /= STEPS_PER_EPOCH
        ep_spec /= STEPS_PER_EPOCH

        val_interval = 5 if epoch <= 50 else 10
        if epoch % val_interval == 0 or epoch == 1:
            val_nrmse = evaluate(model, val_eps, gt_grid_x, gt_grid_y, DEVICE)
            flag = ''
            if val_nrmse < best_val:
                best_val = val_nrmse
                torch.save(model.state_dict(), best_ckpt)
                flag = ' *'
            elapsed = time.time() - t_start
            ls = model.log_lengthscale.exp().item()
            print(f"{epoch:>4}  {ep_mse:>8.5f}  {ep_corr:>7.4f}  "
                  f"{ep_spec:>7.4f}  {val_nrmse:>9.4f}  "
                  f"{elapsed:>5.0f}s  ls={ls:.1f}m{flag}")
        elif epoch % 10 == 0:
            elapsed = time.time() - t_start
            print(f"{epoch:>4}  {ep_mse:>8.5f}  {ep_corr:>7.4f}  "
                  f"{ep_spec:>7.4f}  {'---':>9}  {elapsed:>5.0f}s")

    print(f"\n  Best val nRMSE: {best_val:.4f}")
    return model, best_val, best_ckpt, time.time() - t_start


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    data_dir = os.path.join(REPO_ROOT, 'wavesplat', 'data')
    results_dir = os.path.join(REPO_ROOT, 'wavesplat', 'results')
    ckpt_dir    = os.path.join(REPO_ROOT, 'wavesplat', 'checkpoints')
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)

    print("=" * 70)
    print("  WavePINO Stage 2: Cross-Episode Generalization")
    print("  Device:", DEVICE)
    print("=" * 70)

    train_eps = load_episodes(TRAIN_EPISODES, data_dir)
    val_eps   = load_episodes(VAL_EPISODES,   data_dir)
    test_eps  = load_episodes(TEST_EPISODES,  data_dir)
    print(f"  Train: {len(train_eps)}, Val: {len(val_eps)}, Test: {len(test_eps)}")

    if not train_eps:
        print("ERROR: No training episodes found in", data_dir)
        return

    gt_grid_x = train_eps[0].gt_grid_x
    gt_grid_y = train_eps[0].gt_grid_y
    x_range   = (float(gt_grid_x[0]), float(gt_grid_x[-1]))
    y_range   = (float(gt_grid_y[0]), float(gt_grid_y[-1]))

    model_grid_x = torch.linspace(x_range[0], x_range[1], GRID_NX)
    model_grid_y = torch.linspace(y_range[0], y_range[1], GRID_NY)

    all_results = {}

    # -----------------------------------------------------------------------
    # Run A: FNO baseline (no physics loss)
    # -----------------------------------------------------------------------
    model_a, best_val_a, ckpt_a, time_a = train_one_run(
        tag='fno', lambda_spec=0.0,
        train_eps=train_eps, val_eps=val_eps,
        model_grid_x=model_grid_x, model_grid_y=model_grid_y,
        gt_grid_x=gt_grid_x, gt_grid_y=gt_grid_y,
        x_range=x_range, y_range=y_range,
        ckpt_dir=ckpt_dir)

    # Load best checkpoint and evaluate on test
    model_a.load_state_dict(torch.load(ckpt_a, map_location=DEVICE,
                                        weights_only=True))
    print("\nFNO Test Evaluation:")
    test_a = test_evaluation(model_a, test_eps, gt_grid_x, gt_grid_y, DEVICE)
    avg_a  = float(np.mean(list(test_a.values())))
    print(f"  FNO avg test nRMSE: {avg_a:.4f}")

    all_results['fno'] = {
        'label': 'FNO (no physics loss)',
        'best_val_nrmse': best_val_a,
        'test_avg_nrmse': avg_a,
        'test_per_episode': test_a,
        'total_time_s': time_a,
    }

    # -----------------------------------------------------------------------
    # Run B: PINO (with spectral physics loss)
    # -----------------------------------------------------------------------
    model_b, best_val_b, ckpt_b, time_b = train_one_run(
        tag='pino', lambda_spec=LAMBDA_SPEC_B,
        train_eps=train_eps, val_eps=val_eps,
        model_grid_x=model_grid_x, model_grid_y=model_grid_y,
        gt_grid_x=gt_grid_x, gt_grid_y=gt_grid_y,
        x_range=x_range, y_range=y_range,
        ckpt_dir=ckpt_dir)

    model_b.load_state_dict(torch.load(ckpt_b, map_location=DEVICE,
                                        weights_only=True))
    print("\nPINO Test Evaluation:")
    test_b = test_evaluation(model_b, test_eps, gt_grid_x, gt_grid_y, DEVICE)
    avg_b  = float(np.mean(list(test_b.values())))
    print(f"  PINO avg test nRMSE: {avg_b:.4f}")

    all_results['pino'] = {
        'label': f'PINO (lambda_spec={LAMBDA_SPEC_B})',
        'best_val_nrmse': best_val_b,
        'test_avg_nrmse': avg_b,
        'test_per_episode': test_b,
        'total_time_s': time_b,
    }

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  PINO Stage 2 Summary")
    print("=" * 70)
    print(f"  {'Method':<22}  {'Val':>7}  {'Test':>7}")
    print(f"  {'-'*22}  {'-'*7}  {'-'*7}")
    print(f"  {'WaveSplat (baseline)':<22}  {'---':>7}  {'0.223':>7}")
    print(f"  {'ConvCNP v2':<22}  {'---':>7}  {'0.244':>7}")
    print(f"  {'FNO (no physics)':<22}  {best_val_a:>7.4f}  {avg_a:>7.4f}")
    print(f"  {'PINO (λ_spec=0.5)':<22}  {best_val_b:>7.4f}  {avg_b:>7.4f}")

    # Physics loss verdict
    physics_delta = avg_a - avg_b
    if physics_delta > 0.005:
        phys_verdict = f"HELPS (delta={physics_delta:.4f})"
    elif physics_delta < -0.005:
        phys_verdict = f"HURTS (delta={physics_delta:.4f})"
    else:
        phys_verdict = f"NEUTRAL (delta={physics_delta:.4f})"
    print(f"\n  Physics loss effect: {phys_verdict}")

    best_test = min(avg_a, avg_b)
    if best_test < 0.223:
        final_verdict = f"SUCCESS — beats WaveSplat (best={best_test:.4f} < 0.223)"
    elif best_test < 0.244:
        final_verdict = f"PARTIAL — beats ConvCNP v2 (best={best_test:.4f} < 0.244)"
    else:
        final_verdict = f"FAIL — below ConvCNP v2 (best={best_test:.4f} >= 0.244)"
    print(f"  Final verdict: {final_verdict}")

    all_results['training_config'] = {
        'grid': f'{GRID_NX}x{GRID_NY}',
        'fno_width': FNO_WIDTH, 'fno_layers': FNO_LAYERS,
        'fno_modes': f'{FNO_MODES_H}x{FNO_MODES_W}',
        'epochs': EPOCHS, 'steps_per_epoch': STEPS_PER_EPOCH,
        'batch_size': BATCH_SIZE, 'lr': LR,
        'lambda_corr': LAMBDA_CORR, 'lambda_spec_b': LAMBDA_SPEC_B,
        'T_min': T_MIN, 'T_max': T_MAX, 'depth': DEPTH,
    }
    all_results['physics_loss_verdict'] = phys_verdict
    all_results['final_verdict'] = final_verdict
    all_results['best_test_nrmse'] = best_test

    out_path = os.path.join(results_dir, 'pino_stage2_train.json')
    with open(out_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  Results saved: {out_path}")


if __name__ == '__main__':
    main()
