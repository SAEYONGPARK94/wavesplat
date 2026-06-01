#!/usr/bin/env python3
"""
Train Conditional DDPM for ocean wave field estimation.

Key design:
  - Train on existing HDF5 episodes (gt_field as x0, sparse LiDAR as condition)
  - Hs-normalized fields (like ConvCNP v2)
  - SetConv condition on the same grid as GT (80x60)
  - DDIM sampling for fast inference (50 steps)
  - Evaluate nRMSE: single sample and 5-sample ensemble mean

Goal: Verify that diffusion solves amplitude hedging (35-47% of GT amplitude
under MSE loss) by sampling from p(field | sparse_obs) instead of E[field | obs].

Usage:
    cd wavesplat && python3 -u scripts/train_wave_diffusion.py
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
from wavesplat.model.wave_diffusion import WaveDiffusion, GaussianDiffusion
from wavesplat.evaluation.metrics import nrmse


# --- Episode lists (same as ConvCNP v2) ---

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

# --- Config ---
GRID_NX = 80  # Match GT grid exactly
GRID_NY = 60
BASE_CH = 32  # Smaller than 64 for speed
N_TIMESTEPS = 1000
DDIM_STEPS = 50

EPOCHS = 100
BATCH_SIZE = 8
LR = 2e-4
WARMUP_EPOCHS = 5
STEPS_PER_EPOCH = 100


def load_episodes(names, data_dir):
    eps = []
    for name in names:
        path = os.path.join(data_dir, name)
        if os.path.exists(path):
            eps.append(EpisodeDataLoader(path))
    return eps


def sample_batch(episodes, batch_size):
    """Sample batch: (obs_xy, obs_z, gt_field) with Hs normalization."""
    xc_list, yc_list, gt_list = [], [], []

    for _ in range(batch_size):
        ep = episodes[np.random.randint(len(episodes))]
        frame = ep[np.random.randint(len(ep))]
        Hs = max(ep.Hs_true, 0.1)

        xc = torch.stack([frame.x_obs, frame.y_obs], dim=-1)
        yc = frame.z_obs.unsqueeze(-1) / Hs
        gt = frame.gt_field / Hs  # [Ny, Nx]

        # y-flip augmentation
        if np.random.random() < 0.5:
            xc = xc.clone()
            xc[:, 1] = -xc[:, 1]
            gt = gt.flip(0)

        xc_list.append(xc)
        yc_list.append(yc)
        gt_list.append(gt)

    # Pad context points
    max_nc = max(x.shape[0] for x in xc_list)
    xc_batch = torch.zeros(batch_size, max_nc, 2)
    yc_batch = torch.zeros(batch_size, max_nc, 1)
    for i in range(batch_size):
        nc = xc_list[i].shape[0]
        xc_batch[i, :nc] = xc_list[i]
        yc_batch[i, :nc] = yc_list[i]

    # GT: [B, Ny, Nx] → [B, 1, Nx, Ny] to match grid_xy ordering (ij meshgrid)
    # gt_field is [Ny, Nx], our grid is meshgrid(gx, gy, indexing='ij') → [Gx, Gy]
    # So we need gt as [Gx, Gy] = [Nx, Ny] → transpose
    gt_batch = torch.stack(gt_list)  # [B, Ny, Nx]
    gt_batch = gt_batch.permute(0, 2, 1).unsqueeze(1)  # [B, 1, Nx, Ny] = [B, 1, Gx, Gy]

    return xc_batch, yc_batch, gt_batch


def evaluate_diffusion(model, diffusion, episodes, n_frames=10, ddim_steps=50,
                       n_samples=1):
    """Evaluate: generate wave field and compute nRMSE."""
    model.eval()
    all_nrmse = []

    for ep in episodes:
        Hs = max(ep.Hs_true, 0.1)
        frame_indices = np.linspace(10, len(ep) - 1, n_frames, dtype=int)

        for fi in frame_indices:
            frame = ep[fi]
            xc = torch.stack([frame.x_obs, frame.y_obs], dim=-1).unsqueeze(0)
            yc = frame.z_obs.unsqueeze(-1).unsqueeze(0) / Hs

            # Get condition
            with torch.no_grad():
                cond = model.get_condition(xc, yc)  # [1, 2, Gx, Gy]

            shape = (1, 1, model.grid_nx, model.grid_ny)

            # Generate sample(s)
            preds = []
            for _ in range(n_samples):
                with torch.no_grad():
                    sample = diffusion.sample_ddim(model, cond, shape,
                                                   n_steps=ddim_steps)
                preds.append(sample)

            # Ensemble mean
            pred = torch.stack(preds).mean(0)  # [1, 1, Gx, Gy]

            # Convert to [Ny, Nx] format
            pred_field = (pred[0, 0].permute(1, 0) * Hs).numpy()  # [Gy, Gx] = [Ny, Nx]
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
    print("WaveDiffusion: Conditional DDPM for Ocean Wave Field Estimation")
    print("=" * 80)

    train_eps = load_episodes(TRAIN_EPISODES, data_dir)
    val_eps = load_episodes(VAL_EPISODES, data_dir)
    test_eps = load_episodes(TEST_EPISODES, data_dir)
    print("  Train: %d, Val: %d, Test: %d" % (len(train_eps), len(val_eps), len(test_eps)))

    if not train_eps:
        print("ERROR: No training episodes!")
        return

    ep0 = train_eps[0]
    x_range = (float(ep0.gt_grid_x[0]), float(ep0.gt_grid_x[-1]))
    y_range = (float(ep0.gt_grid_y[0]), float(ep0.gt_grid_y[-1]))

    model = WaveDiffusion(
        grid_nx=GRID_NX, grid_ny=GRID_NY,
        x_range=x_range, y_range=y_range,
        base_ch=BASE_CH,
        init_lengthscale=10.0,
        n_timesteps=N_TIMESTEPS,
    )
    print("  Model: WaveDiffusion, %s params" % format(model.n_params, ','))
    print("  Grid: %dx%d, T=%d, DDIM=%d steps" % (
        GRID_NX, GRID_NY, N_TIMESTEPS, DDIM_STEPS))

    diffusion = GaussianDiffusion(
        n_timesteps=N_TIMESTEPS, beta_start=1e-4, beta_end=0.02)

    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    warmup_sched = optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.01, total_iters=WARMUP_EPOCHS)
    cosine_sched = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS - WARMUP_EPOCHS)
    scheduler = optim.lr_scheduler.SequentialLR(
        optimizer, [warmup_sched, cosine_sched], milestones=[WARMUP_EPOCHS])

    ckpt_dir = os.path.join(data_dir, '..', 'checkpoints')
    os.makedirs(ckpt_dir, exist_ok=True)

    print("  Training for %d epochs (%d steps/epoch)..." % (EPOCHS, STEPS_PER_EPOCH))
    best_val = 1.0
    best_epoch = 0
    t_start = time.time()

    for epoch in range(1, EPOCHS + 1):
        model.train()
        epoch_loss = 0.0

        for step in range(STEPS_PER_EPOCH):
            xc, yc, gt = sample_batch(train_eps, BATCH_SIZE)

            # Compute condition from sparse obs
            cond = model.get_condition(xc, yc)  # [B, 2, Gx, Gy]

            # Sample random timesteps
            t = torch.randint(0, N_TIMESTEPS, (BATCH_SIZE,))

            # Forward diffusion: add noise to GT
            x_noisy, noise = diffusion.q_sample(gt, t)

            # Predict noise
            pred_noise = model(x_noisy, t, cond)

            # Simple MSE loss on noise (standard DDPM)
            loss = F.mse_loss(pred_noise, noise)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()

        scheduler.step()
        avg_loss = epoch_loss / STEPS_PER_EPOCH

        # Validation (less frequent — sampling is slow)
        val_interval = 10 if epoch <= 50 else 20
        if epoch % val_interval == 0 or epoch == 1:
            val_nrmse = evaluate_diffusion(
                model, diffusion, val_eps, n_frames=5, ddim_steps=DDIM_STEPS,
                n_samples=1)
            improved = ""
            if val_nrmse < best_val:
                best_val = val_nrmse
                best_epoch = epoch
                torch.save(model.state_dict(),
                           os.path.join(ckpt_dir, 'wave_diffusion_best.pt'))
                improved = " *"

            elapsed = time.time() - t_start
            ls = model.setconv.log_lengthscale.exp().item()
            print("  Epoch %3d/%d: loss=%.6f val=%.4f ls=%.1fm lr=%.6f [%.0fs]%s" % (
                epoch, EPOCHS, avg_loss, val_nrmse, ls,
                scheduler.get_last_lr()[0], elapsed, improved))
        elif epoch % 5 == 0:
            elapsed = time.time() - t_start
            print("  Epoch %3d/%d: loss=%.6f lr=%.6f [%.0fs]" % (
                epoch, EPOCHS, avg_loss, scheduler.get_last_lr()[0], elapsed))

    print("  Best val nRMSE: %.4f at epoch %d" % (best_val, best_epoch))

    # --- Test ---
    print("\n" + "=" * 80)
    print("Test Evaluation (WaveDiffusion)")
    print("=" * 80)

    ckpt_path = os.path.join(ckpt_dir, 'wave_diffusion_best.pt')
    if os.path.exists(ckpt_path):
        model.load_state_dict(torch.load(ckpt_path, map_location='cpu',
                                          weights_only=True))
        print("  Loaded best checkpoint")

    model.eval()

    # Test with different sample counts
    for n_samples in [1, 5]:
        print("\n  --- %d-sample evaluation ---" % n_samples)
        results = {}
        all_nrmses = []
        amp_ratios = []

        for ep_idx, ep in enumerate(test_eps):
            ep_nrmses = []
            ep_amp = []
            Hs = max(ep.Hs_true, 0.1)
            n_eval = min(20, len(ep))
            frame_indices = np.linspace(10, len(ep) - 1, n_eval, dtype=int)

            for fi in frame_indices:
                frame = ep[fi]
                xc = torch.stack([frame.x_obs, frame.y_obs], dim=-1).unsqueeze(0)
                yc = frame.z_obs.unsqueeze(-1).unsqueeze(0) / Hs

                with torch.no_grad():
                    cond = model.get_condition(xc, yc)

                shape = (1, 1, model.grid_nx, model.grid_ny)
                preds = []
                for _ in range(n_samples):
                    with torch.no_grad():
                        s = diffusion.sample_ddim(model, cond, shape,
                                                   n_steps=DDIM_STEPS)
                    preds.append(s)

                pred = torch.stack(preds).mean(0)
                pred_field = (pred[0, 0].permute(1, 0) * Hs).numpy()
                gt_field = frame.gt_field.numpy()

                val = nrmse(pred_field, gt_field, ep.Hs_true)
                if not np.isnan(val):
                    ep_nrmses.append(val)

                # Amplitude ratio (key metric for hedging)
                pred_std = float(np.std(pred_field))
                gt_std = float(np.std(gt_field))
                if gt_std > 0.01:
                    ep_amp.append(pred_std / gt_std)

            avg = float(np.mean(ep_nrmses))
            amp = float(np.mean(ep_amp)) if ep_amp else 0.0
            ep_name = TEST_EPISODES[ep_idx].replace('episode_', '').replace('.h5', '')
            results[ep_name] = avg
            all_nrmses.append(avg)
            amp_ratios.append(amp)
            print("    %s: nRMSE=%.4f amp_ratio=%.2f" % (ep_name, avg, amp))

        overall = float(np.mean(all_nrmses))
        avg_amp = float(np.mean(amp_ratios))
        print("    Average: nRMSE=%.4f, amp_ratio=%.2f" % (overall, avg_amp))
        print("    (ConvCNP v2: 0.244 amp~0.4, WaveSplat: 0.223)")

        if n_samples == 1:
            results_1 = results.copy()
            overall_1 = overall
            amp_1 = avg_amp
        else:
            results_5 = results.copy()

    # Save results
    results_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                'results')
    os.makedirs(results_dir, exist_ok=True)
    out = {
        'version': 'wave_diffusion_v1',
        'best_val_nRMSE': best_val,
        'best_epoch': best_epoch,
        'test_1sample_avg': overall_1,
        'test_1sample_amp_ratio': amp_1,
        'test_1sample_per_episode': results_1,
        'test_5sample_avg': float(np.mean(list(results_5.values()))),
        'test_5sample_per_episode': results_5,
        'n_params': model.n_params,
        'training_time_s': time.time() - t_start,
        'config': {
            'grid_nx': GRID_NX, 'grid_ny': GRID_NY,
            'base_ch': BASE_CH, 'n_timesteps': N_TIMESTEPS,
            'ddim_steps': DDIM_STEPS,
            'epochs': EPOCHS, 'lr': LR, 'batch_size': BATCH_SIZE,
        },
    }
    with open(os.path.join(results_dir, 'wave_diffusion_eval.json'), 'w') as f:
        json.dump(out, f, indent=2)
    print("\n  Results saved to results/wave_diffusion_eval.json")


if __name__ == '__main__':
    main()
