#!/usr/bin/env python3
"""
Train WaveDiffusion v3: RePaint + Dispersion Projection + Expanded Data.

Key changes from v2 (test nRMSE 0.261, amp_ratio 0.84):
  1. Observation-guided sampling (RePaint): replace grid cells near obs during denoising
  2. Dispersion-constrained denoising: wavenumber bandpass filter on x0_pred
  3. Expanded training data: 33 -> ~96 episodes (3x more data)
  4. base_ch=48, 300 epochs (longer training for more data)

Usage:
    cd wavesplat && python3 -u scripts/train_wave_diffusion_v3.py
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


# --- Episode splits ---
# Test: same as v2 for fair comparison
TEST_EPISODES = [
    'episode_BF3_5comp_seed0_slope.h5',
    'episode_BF4_5comp_seed0_uniform.h5',
    'episode_BF4_15comp_seed0_uniform.h5',
    'episode_BF4_30comp_seed0_uniform.h5',
    'episode_BF5_15comp_seed0_uniform.h5',
    'episode_BF5_30comp_seed0_uniform.h5',
]

# Val: same as v2
VAL_EPISODES = [
    'episode_BF3_5comp_seed2_uniform.h5',
    'episode_BF4_5comp_seed2_uniform.h5',
    'episode_BF4_15comp_seed2_uniform.h5',
    'episode_BF5_15comp_seed2_uniform.h5',
]

# Train: everything else (auto-discovered)
EXCLUDE = set(TEST_EPISODES + VAL_EPISODES)

# --- Config ---
GRID_NX = 80
GRID_NY = 60
BASE_CH = 48
N_TIMESTEPS = 1000
DDIM_STEPS = 50
GUIDANCE_SCALE = 3.0
COND_DROP_PROB = 0.1

EPOCHS = 300
BATCH_SIZE = 8
LR = 2e-4
WARMUP_EPOCHS = 10
STEPS_PER_EPOCH = 100

# RePaint config
REPAINT_RADIUS = 2  # grid cells

# Dispersion config
DISP_K_MIN = 0.02   # rad/m (Tp ~ 14s)
DISP_K_MAX = 0.5    # rad/m (Tp ~ 3s)
DISP_STRENGTH = 0.3  # soft blending


def discover_train_episodes(data_dir):
    """Auto-discover all episodes not in test/val sets."""
    import glob
    all_eps = glob.glob(os.path.join(data_dir, 'episode_*.h5'))
    train = []
    for p in sorted(all_eps):
        name = os.path.basename(p)
        if name not in EXCLUDE:
            train.append(name)
    return train


def load_episodes(names, data_dir):
    eps = []
    for name in names:
        path = os.path.join(data_dir, name)
        if os.path.exists(path):
            eps.append(EpisodeDataLoader(path))
    return eps


def sample_batch(episodes, batch_size):
    """Sample batch with Hs normalization and y-flip augmentation."""
    xc_list, yc_list, gt_list = [], [], []

    for _ in range(batch_size):
        ep = episodes[np.random.randint(len(episodes))]
        frame = ep[np.random.randint(len(ep))]
        Hs = max(ep.Hs_true, 0.1)

        xc = torch.stack([frame.x_obs, frame.y_obs], dim=-1)
        yc = frame.z_obs.unsqueeze(-1) / Hs
        gt = frame.gt_field / Hs

        if np.random.random() < 0.5:
            xc = xc.clone()
            xc[:, 1] = -xc[:, 1]
            gt = gt.flip(0)

        xc_list.append(xc)
        yc_list.append(yc)
        gt_list.append(gt)

    max_nc = max(x.shape[0] for x in xc_list)
    xc_batch = torch.zeros(batch_size, max_nc, 2)
    yc_batch = torch.zeros(batch_size, max_nc, 1)
    for i in range(batch_size):
        nc = xc_list[i].shape[0]
        xc_batch[i, :nc] = xc_list[i]
        yc_batch[i, :nc] = yc_list[i]

    gt_batch = torch.stack(gt_list)
    gt_batch = gt_batch.permute(0, 2, 1).unsqueeze(1)  # [B, 1, Nx, Ny]

    return xc_batch, yc_batch, gt_batch


def evaluate_diffusion(model, diffusion, episodes, n_frames=5,
                       ddim_steps=50, guidance_scale=1.0, n_samples=1,
                       use_repaint=False, use_dispersion=False):
    """Evaluate with optional RePaint and dispersion projection."""
    model.eval()
    all_nrmse = []

    for ep in episodes:
        Hs = max(ep.Hs_true, 0.1)
        frame_indices = np.linspace(10, len(ep) - 1, n_frames, dtype=int)

        for fi in frame_indices:
            frame = ep[fi]
            xc = torch.stack([frame.x_obs, frame.y_obs], dim=-1).unsqueeze(0)
            yc = frame.z_obs.unsqueeze(-1).unsqueeze(0) / Hs

            with torch.no_grad():
                cond = model.get_condition(xc, yc)

            shape = (1, 1, model.grid_nx, model.grid_ny)

            # Prepare RePaint inputs
            obs_mask, obs_values = None, None
            if use_repaint:
                obs_mask, obs_values = model.make_obs_grid(xc, yc, radius=REPAINT_RADIUS)

            preds = []
            for _ in range(n_samples):
                with torch.no_grad():
                    if use_repaint or use_dispersion:
                        s = diffusion.sample_ddim_guided(
                            model, cond, shape,
                            n_steps=ddim_steps,
                            guidance_scale=guidance_scale,
                            obs_mask=obs_mask if use_repaint else None,
                            obs_values=obs_values if use_repaint else None,
                            k_min=DISP_K_MIN, k_max=DISP_K_MAX,
                            disp_strength=DISP_STRENGTH if use_dispersion else 0.0)
                    else:
                        s = diffusion.sample_ddim(
                            model, cond, shape,
                            n_steps=ddim_steps,
                            guidance_scale=guidance_scale)
                preds.append(s)

            pred = torch.stack(preds).mean(0)
            pred_field = (pred[0, 0].permute(1, 0) * Hs).numpy()
            gt_field = frame.gt_field.numpy()

            val = nrmse(pred_field, gt_field, ep.Hs_true)
            if not np.isnan(val):
                all_nrmse.append(val)

    model.train()
    return float(np.mean(all_nrmse)) if all_nrmse else 1.0


def test_eval_full(model, diffusion, test_eps, ddim_steps, guidance_configs):
    """Full test evaluation with multiple configs."""
    model.eval()
    results = {}

    for gs, n_samples, use_rp, use_disp, label in guidance_configs:
        print("\n  --- %s ---" % label)
        ep_results = []
        amp_ratios = []

        for ep_idx, ep in enumerate(test_eps):
            Hs = max(ep.Hs_true, 0.1)
            n_eval = min(20, len(ep))
            frame_indices = np.linspace(10, len(ep) - 1, n_eval, dtype=int)
            ep_nrmses = []
            ep_amp = []

            for fi in frame_indices:
                frame = ep[fi]
                xc = torch.stack([frame.x_obs, frame.y_obs], dim=-1).unsqueeze(0)
                yc = frame.z_obs.unsqueeze(-1).unsqueeze(0) / Hs

                with torch.no_grad():
                    cond = model.get_condition(xc, yc)
                shape = (1, 1, model.grid_nx, model.grid_ny)

                obs_mask, obs_values = None, None
                if use_rp:
                    obs_mask, obs_values = model.make_obs_grid(xc, yc, radius=REPAINT_RADIUS)

                preds = []
                for _ in range(n_samples):
                    with torch.no_grad():
                        if use_rp or use_disp:
                            s = diffusion.sample_ddim_guided(
                                model, cond, shape,
                                n_steps=ddim_steps,
                                guidance_scale=gs,
                                obs_mask=obs_mask if use_rp else None,
                                obs_values=obs_values if use_rp else None,
                                k_min=DISP_K_MIN, k_max=DISP_K_MAX,
                                disp_strength=DISP_STRENGTH if use_disp else 0.0)
                        else:
                            s = diffusion.sample_ddim(model, cond, shape,
                                                       n_steps=ddim_steps,
                                                       guidance_scale=gs)
                    preds.append(s)

                pred = torch.stack(preds).mean(0)
                pred_field = (pred[0, 0].permute(1, 0) * Hs).numpy()
                gt_field = frame.gt_field.numpy()

                val = nrmse(pred_field, gt_field, ep.Hs_true)
                if not np.isnan(val):
                    ep_nrmses.append(val)
                pred_std = float(np.std(pred_field))
                gt_std = float(np.std(gt_field))
                if gt_std > 0.01:
                    ep_amp.append(pred_std / gt_std)

            avg = float(np.mean(ep_nrmses))
            amp = float(np.mean(ep_amp)) if ep_amp else 0.0
            ep_name = TEST_EPISODES[ep_idx].replace('episode_', '').replace('.h5', '')
            ep_results.append(avg)
            amp_ratios.append(amp)
            print("    %s: nRMSE=%.4f amp=%.2f" % (ep_name, avg, amp))

        overall = float(np.mean(ep_results))
        avg_amp = float(np.mean(amp_ratios))
        print("    Avg: nRMSE=%.4f amp=%.2f" % (overall, avg_amp))
        results[label] = {'nrmse': overall, 'amp': avg_amp,
                          'per_episode': dict(zip(
                              [n.replace('episode_', '').replace('.h5', '') for n in TEST_EPISODES],
                              ep_results))}

    return results


def main():
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            'data')

    print("=" * 80)
    print("WaveDiffusion v3: RePaint + Dispersion + Expanded Data")
    print("=" * 80)

    # Discover training episodes
    train_names = discover_train_episodes(data_dir)
    train_eps = load_episodes(train_names, data_dir)
    val_eps = load_episodes(VAL_EPISODES, data_dir)
    test_eps = load_episodes(TEST_EPISODES, data_dir)
    print("  Train: %d, Val: %d, Test: %d" % (len(train_eps), len(val_eps), len(test_eps)))

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
    print("  Model: %s params, base_ch=%d" % (format(model.n_params, ','), BASE_CH))
    print("  CFG: drop_prob=%.2f, guidance_scale=%.1f" % (COND_DROP_PROB, GUIDANCE_SCALE))
    print("  RePaint: radius=%d, Dispersion: k=[%.2f, %.2f] strength=%.1f" % (
        REPAINT_RADIUS, DISP_K_MIN, DISP_K_MAX, DISP_STRENGTH))

    diffusion = GaussianDiffusion(n_timesteps=N_TIMESTEPS)

    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    warmup_sched = optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.01, total_iters=WARMUP_EPOCHS)
    cosine_sched = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS - WARMUP_EPOCHS)
    scheduler = optim.lr_scheduler.SequentialLR(
        optimizer, [warmup_sched, cosine_sched], milestones=[WARMUP_EPOCHS])

    ckpt_dir = os.path.join(data_dir, '..', 'checkpoints')
    os.makedirs(ckpt_dir, exist_ok=True)

    print("  Training for %d epochs..." % EPOCHS)
    best_val = 1.0
    best_epoch = 0
    t_start = time.time()

    for epoch in range(1, EPOCHS + 1):
        model.train()
        epoch_loss = 0.0

        for step in range(STEPS_PER_EPOCH):
            xc, yc, gt = sample_batch(train_eps, BATCH_SIZE)

            cond = model.get_condition(xc, yc)

            # CFG: randomly drop condition
            if COND_DROP_PROB > 0:
                drop_mask = torch.rand(BATCH_SIZE) < COND_DROP_PROB
                if drop_mask.any():
                    cond[drop_mask] = 0.0

            t = torch.randint(0, N_TIMESTEPS, (BATCH_SIZE,))
            x_noisy, noise = diffusion.q_sample(gt, t)

            pred_noise = model(x_noisy, t, cond)
            loss = F.mse_loss(pred_noise, noise)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()

        scheduler.step()
        avg_loss = epoch_loss / STEPS_PER_EPOCH

        # Validation (with RePaint + dispersion for true v3 evaluation)
        val_interval = 10 if epoch <= 50 else 20
        if epoch % val_interval == 0 or epoch == 1:
            val_nrmse = evaluate_diffusion(
                model, diffusion, val_eps, n_frames=5, ddim_steps=DDIM_STEPS,
                guidance_scale=GUIDANCE_SCALE, n_samples=1,
                use_repaint=True, use_dispersion=True)
            improved = ""
            if val_nrmse < best_val:
                best_val = val_nrmse
                best_epoch = epoch
                torch.save(model.state_dict(),
                           os.path.join(ckpt_dir, 'wave_diffusion_v3_best.pt'))
                improved = " *"

            elapsed = time.time() - t_start
            ls = model.setconv.log_lengthscale.exp().item()
            print("  Epoch %3d/%d: loss=%.6f val=%.4f ls=%.1fm lr=%.6f [%.0fs]%s" % (
                epoch, EPOCHS, avg_loss, val_nrmse, ls,
                scheduler.get_last_lr()[0], elapsed, improved))
        elif epoch % 10 == 0:
            elapsed = time.time() - t_start
            print("  Epoch %3d/%d: loss=%.6f lr=%.6f [%.0fs]" % (
                epoch, EPOCHS, avg_loss, scheduler.get_last_lr()[0], elapsed))

    print("  Best val nRMSE: %.4f at epoch %d" % (best_val, best_epoch))

    # --- Test ---
    print("\n" + "=" * 80)
    print("Test Evaluation (WaveDiffusion v3)")
    print("=" * 80)

    ckpt_path = os.path.join(ckpt_dir, 'wave_diffusion_v3_best.pt')
    if os.path.exists(ckpt_path):
        model.load_state_dict(torch.load(ckpt_path, map_location='cpu',
                                          weights_only=True))
        print("  Loaded best checkpoint (epoch %d)" % best_epoch)

    # Test configs: (guidance_scale, n_samples, use_repaint, use_dispersion, label)
    configs = [
        # Baseline (same as v2)
        (1.0, 1, False, False, "baseline gs=1 1s"),
        (1.0, 5, False, False, "baseline gs=1 5s"),
        (3.0, 1, False, False, "baseline gs=3 1s"),
        # RePaint only
        (3.0, 1, True, False, "repaint gs=3 1s"),
        (3.0, 5, True, False, "repaint gs=3 5s"),
        # Dispersion only
        (3.0, 1, False, True, "disp gs=3 1s"),
        # RePaint + Dispersion (full v3)
        (3.0, 1, True, True, "full_v3 gs=3 1s"),
        (3.0, 5, True, True, "full_v3 gs=3 5s"),
        (5.0, 1, True, True, "full_v3 gs=5 1s"),
    ]

    all_results = test_eval_full(model, diffusion, test_eps, DDIM_STEPS, configs)

    # Save results
    results_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                'results')
    os.makedirs(results_dir, exist_ok=True)
    out = {
        'version': 'wave_diffusion_v3',
        'best_val_nRMSE': best_val,
        'best_epoch': best_epoch,
        'n_params': model.n_params,
        'n_train_episodes': len(train_eps),
        'training_time_s': time.time() - t_start,
        'config': {
            'grid_nx': GRID_NX, 'grid_ny': GRID_NY,
            'base_ch': BASE_CH, 'n_timesteps': N_TIMESTEPS,
            'ddim_steps': DDIM_STEPS, 'guidance_scale': GUIDANCE_SCALE,
            'cond_drop_prob': COND_DROP_PROB,
            'epochs': EPOCHS, 'lr': LR, 'batch_size': BATCH_SIZE,
            'repaint_radius': REPAINT_RADIUS,
            'disp_k_min': DISP_K_MIN, 'disp_k_max': DISP_K_MAX,
            'disp_strength': DISP_STRENGTH,
        },
        'test_results': all_results,
    }
    with open(os.path.join(results_dir, 'wave_diffusion_v3_eval.json'), 'w') as f:
        json.dump(out, f, indent=2)
    print("\n  Results saved to results/wave_diffusion_v3_eval.json")


if __name__ == '__main__':
    main()
