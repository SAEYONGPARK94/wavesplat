#!/usr/bin/env python3
"""
Train WaveDiffusion v7: ConvCNP-conditioned diffusion.

Key idea: Use frozen ConvCNP v2 output as additional condition channel.
Diffusion learns to refine ConvCNP's deterministic prediction → add realistic
amplitude/detail while preserving spatial accuracy.

Architecture:
  in_ch = 1 (noisy) + 2 (SetConv) + 1 (ConvCNP output) = 4
  Everything else same as v3 (epsilon-pred, linear, base_ch=48, 300 epochs)

Usage:
    cd wavesplat && python3 -u scripts/train_wave_diffusion_v7.py
"""

import sys, os, time, json, glob
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', buffering=1)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from wavesplat.data_loader import EpisodeDataLoader
from wavesplat.model.wave_diffusion import (
    WaveDiffusion, GaussianDiffusion, _make_dispersion_mask, _apply_dispersion_filter
)
from wavesplat.model.conv_cnp import ConvCNP
from wavesplat.evaluation.metrics import nrmse

# --- Episode splits ---
TEST_EPISODES = [
    'episode_BF3_5comp_seed0_slope.h5',
    'episode_BF4_5comp_seed0_uniform.h5',
    'episode_BF4_15comp_seed0_uniform.h5',
    'episode_BF4_30comp_seed0_uniform.h5',
    'episode_BF5_15comp_seed0_uniform.h5',
    'episode_BF5_30comp_seed0_uniform.h5',
]
VAL_EPISODES = [
    'episode_BF3_5comp_seed2_uniform.h5',
    'episode_BF4_5comp_seed2_uniform.h5',
    'episode_BF4_15comp_seed2_uniform.h5',
    'episode_BF5_15comp_seed2_uniform.h5',
]
EXCLUDE = set(TEST_EPISODES + VAL_EPISODES)

# --- Config ---
GRID_NX, GRID_NY = 80, 60
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
DISP_K_MIN, DISP_K_MAX, DISP_STRENGTH = 0.02, 0.5, 0.3

# ConvCNP config (must match v2 checkpoint)
CNP_GRID_NX, CNP_GRID_NY = 64, 48
CNP_BASE_CH = 32


def discover_train_episodes(data_dir):
    all_eps = glob.glob(os.path.join(data_dir, 'episode_*.h5'))
    return [os.path.basename(p) for p in sorted(all_eps) if os.path.basename(p) not in EXCLUDE]


def load_episodes(names, data_dir):
    return [EpisodeDataLoader(os.path.join(data_dir, n))
            for n in names if os.path.exists(os.path.join(data_dir, n))]


def get_cnp_grid_output(cnp_model, xc, yc):
    """Get ConvCNP internal grid prediction [B, 1, 64, 48] and resize to [B, 1, 80, 60]."""
    B = xc.shape[0]
    with torch.no_grad():
        grid_feat = cnp_model.setconv(xc, yc, cnp_model.grid_xy)
        grid_img = grid_feat.permute(0, 2, 1).reshape(B, 2, CNP_GRID_NX, CNP_GRID_NY)
        out_img = cnp_model.unet(grid_img)  # [B, 1, 64, 48]
    # Resize to diffusion grid
    out_resized = F.interpolate(out_img, size=(GRID_NX, GRID_NY), mode='bilinear', align_corners=True)
    return out_resized  # [B, 1, 80, 60]


def sample_batch(episodes, batch_size, cnp_model):
    """Sample training batch with ConvCNP condition."""
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

    gt_batch = torch.stack(gt_list).permute(0, 2, 1).unsqueeze(1)

    # Get ConvCNP output
    cnp_out = get_cnp_grid_output(cnp_model, xc_batch, yc_batch)

    return xc_batch, yc_batch, gt_batch, cnp_out


@torch.no_grad()
def sample_ddim_v7(model, diffusion, cond, cnp_cond, shape, n_steps=50,
                    guidance_scale=1.0, disp_mask=None, disp_strength=0.3):
    """DDIM sampling with ConvCNP condition channel."""
    device = cond.device
    B = shape[0]

    step_size = diffusion.n_timesteps // n_steps
    timesteps = list(range(0, diffusion.n_timesteps, step_size))
    timesteps = list(reversed(timesteps))

    x = torch.randn(shape, device=device)

    # Full condition: SetConv [B,2,H,W] + ConvCNP [B,1,H,W]
    full_cond = torch.cat([cond, cnp_cond], dim=1)  # [B, 3, H, W]
    uncond = torch.zeros_like(full_cond) if guidance_scale > 1.0 else None

    for i, t_cur in enumerate(timesteps):
        t = torch.full((B,), t_cur, device=device, dtype=torch.long)

        if guidance_scale > 1.0 and uncond is not None:
            noise_cond = model(x, t, full_cond)
            noise_uncond = model(x, t, uncond)
            pred_noise = noise_uncond + guidance_scale * (noise_cond - noise_uncond)
        else:
            pred_noise = model(x, t, full_cond)

        alpha_bar = diffusion.alphas_cumprod[t_cur]
        x0_pred = (x - torch.sqrt(1 - alpha_bar) * pred_noise) / torch.sqrt(alpha_bar)

        if disp_mask is not None and disp_strength > 0:
            x0_pred = _apply_dispersion_filter(x0_pred, disp_mask, disp_strength)

        if i < len(timesteps) - 1:
            t_next = timesteps[i + 1]
            alpha_bar_next = diffusion.alphas_cumprod[t_next]
            x = (torch.sqrt(alpha_bar_next) * x0_pred +
                 torch.sqrt(1 - alpha_bar_next) * pred_noise)
        else:
            x = x0_pred

    return x


def evaluate(model, diffusion, cnp_model, episodes, n_frames=5,
             ddim_steps=50, guidance_scale=1.0, n_samples=1,
             use_dispersion=False, disp_mask=None):
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
                cnp_cond = get_cnp_grid_output(cnp_model, xc, yc)
            shape = (1, 1, model.grid_nx, model.grid_ny)
            preds = []
            for _ in range(n_samples):
                s = sample_ddim_v7(
                    model, diffusion, cond, cnp_cond, shape,
                    n_steps=ddim_steps, guidance_scale=guidance_scale,
                    disp_mask=disp_mask if use_dispersion else None,
                    disp_strength=DISP_STRENGTH if use_dispersion else 0.0)
                preds.append(s)
            pred = torch.stack(preds).mean(0)
            pred_field = (pred[0, 0].permute(1, 0) * Hs).numpy()
            gt_field = frame.gt_field.numpy()
            val = nrmse(pred_field, gt_field, ep.Hs_true)
            if not np.isnan(val):
                all_nrmse.append(val)
    model.train()
    return float(np.mean(all_nrmse)) if all_nrmse else 1.0


def test_eval_full(model, diffusion, cnp_model, test_eps, ddim_steps, configs, disp_mask):
    model.eval()
    results = {}
    for gs, n_samples, use_disp, label in configs:
        print("\n  --- %s ---" % label)
        ep_results, amp_ratios = [], []
        for ep_idx, ep in enumerate(test_eps):
            Hs = max(ep.Hs_true, 0.1)
            n_eval = min(20, len(ep))
            frame_indices = np.linspace(10, len(ep) - 1, n_eval, dtype=int)
            ep_nrmses, ep_amp = [], []
            for fi in frame_indices:
                frame = ep[fi]
                xc = torch.stack([frame.x_obs, frame.y_obs], dim=-1).unsqueeze(0)
                yc = frame.z_obs.unsqueeze(-1).unsqueeze(0) / Hs
                with torch.no_grad():
                    cond = model.get_condition(xc, yc)
                    cnp_cond = get_cnp_grid_output(cnp_model, xc, yc)
                shape = (1, 1, model.grid_nx, model.grid_ny)
                preds = []
                for _ in range(n_samples):
                    s = sample_ddim_v7(
                        model, diffusion, cond, cnp_cond, shape,
                        n_steps=ddim_steps, guidance_scale=gs,
                        disp_mask=disp_mask if use_disp else None,
                        disp_strength=DISP_STRENGTH if use_disp else 0.0)
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
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
    ckpt_dir = os.path.join(data_dir, '..', 'checkpoints')

    print("=" * 80)
    print("WaveDiffusion v7: ConvCNP-conditioned diffusion")
    print("=" * 80)

    # --- Load ConvCNP v2 (frozen) ---
    train_names = discover_train_episodes(data_dir)
    train_eps = load_episodes(train_names, data_dir)
    val_eps = load_episodes(VAL_EPISODES, data_dir)
    test_eps = load_episodes(TEST_EPISODES, data_dir)
    print("  Train: %d, Val: %d, Test: %d" % (len(train_eps), len(val_eps), len(test_eps)))

    ep0 = train_eps[0]
    x_range = (float(ep0.gt_grid_x[0]), float(ep0.gt_grid_x[-1]))
    y_range = (float(ep0.gt_grid_y[0]), float(ep0.gt_grid_y[-1]))

    cnp_model = ConvCNP(
        grid_nx=CNP_GRID_NX, grid_ny=CNP_GRID_NY,
        x_range=x_range, y_range=y_range,
        base_ch=CNP_BASE_CH, init_lengthscale=10.0)
    cnp_ckpt = os.path.join(ckpt_dir, 'conv_cnp_v2_best.pt')
    cnp_model.load_state_dict(torch.load(cnp_ckpt, map_location='cpu', weights_only=True))
    cnp_model.eval()
    for p in cnp_model.parameters():
        p.requires_grad = False
    print("  ConvCNP v2 loaded (frozen): %s params" % format(cnp_model.n_params, ','))

    # --- Build WaveDiffusion with 4 input channels ---
    # in_ch = 1 (noisy) + 2 (SetConv) + 1 (ConvCNP) = 4
    model = WaveDiffusion(
        grid_nx=GRID_NX, grid_ny=GRID_NY,
        x_range=x_range, y_range=y_range,
        base_ch=BASE_CH, init_lengthscale=10.0,
        n_timesteps=N_TIMESTEPS, self_cond=True,  # hack: self_cond adds 1 channel (total 4)
    )
    # Override: we use the self_cond channel for ConvCNP output instead
    model.self_cond = False  # disable actual self-conditioning logic
    print("  WaveDiffusion v7: %s params (in_ch=4, base_ch=%d)" % (
        format(model.n_params, ','), BASE_CH))

    diffusion = GaussianDiffusion(n_timesteps=N_TIMESTEPS)
    disp_mask = _make_dispersion_mask(GRID_NX, GRID_NY, DISP_K_MIN, DISP_K_MAX, 'cpu')

    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    warmup_sched = optim.lr_scheduler.LinearLR(optimizer, start_factor=0.01, total_iters=WARMUP_EPOCHS)
    cosine_sched = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS - WARMUP_EPOCHS)
    scheduler = optim.lr_scheduler.SequentialLR(
        optimizer, [warmup_sched, cosine_sched], milestones=[WARMUP_EPOCHS])

    os.makedirs(ckpt_dir, exist_ok=True)
    print("  Training for %d epochs..." % EPOCHS)
    best_val = 1.0
    best_epoch = 0
    t_start = time.time()

    for epoch in range(1, EPOCHS + 1):
        model.train()
        epoch_loss = 0.0

        for step in range(STEPS_PER_EPOCH):
            xc, yc, gt, cnp_out = sample_batch(train_eps, BATCH_SIZE, cnp_model)

            # Get SetConv condition
            cond = model.get_condition(xc, yc)  # [B, 2, H, W]

            # Full condition: SetConv + ConvCNP
            full_cond = torch.cat([cond, cnp_out], dim=1)  # [B, 3, H, W]

            # CFG: randomly drop condition
            if COND_DROP_PROB > 0:
                drop_mask = torch.rand(BATCH_SIZE) < COND_DROP_PROB
                if drop_mask.any():
                    full_cond[drop_mask] = 0.0

            t = torch.randint(0, N_TIMESTEPS, (BATCH_SIZE,))
            x_noisy, noise = diffusion.q_sample(gt, t)

            # Predict noise (model sees: noisy [1ch] + full_cond [3ch] = 4ch)
            pred_noise = model.unet(x_noisy, t, full_cond)
            loss = F.mse_loss(pred_noise, noise)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()

        scheduler.step()
        avg_loss = epoch_loss / STEPS_PER_EPOCH

        # Validation
        val_interval = 10 if epoch <= 50 else 20
        if epoch % val_interval == 0 or epoch == 1:
            val_nrmse = evaluate(
                model, diffusion, cnp_model, val_eps, n_frames=5,
                ddim_steps=DDIM_STEPS, guidance_scale=GUIDANCE_SCALE,
                n_samples=1, use_dispersion=True, disp_mask=disp_mask)
            improved = ""
            if val_nrmse < best_val:
                best_val = val_nrmse
                best_epoch = epoch
                torch.save(model.state_dict(), os.path.join(ckpt_dir, 'wave_diffusion_v7_best.pt'))
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
    print("Test Evaluation (WaveDiffusion v7)")
    print("=" * 80)

    ckpt_path = os.path.join(ckpt_dir, 'wave_diffusion_v7_best.pt')
    if os.path.exists(ckpt_path):
        model.load_state_dict(torch.load(ckpt_path, map_location='cpu', weights_only=True))
        print("  Loaded best checkpoint (epoch %d)" % best_epoch)

    configs = [
        (1.0, 1, False, "baseline gs=1 1s"),
        (3.0, 1, False, "baseline gs=3 1s"),
        (3.0, 1, True, "disp gs=3 1s"),
        (3.0, 5, True, "disp gs=3 5s"),
        (3.0, 3, True, "disp gs=3 3s"),
    ]

    all_results = test_eval_full(model, diffusion, cnp_model, test_eps, DDIM_STEPS, configs, disp_mask)

    results_dir = os.path.join(data_dir, '..', 'results')
    os.makedirs(results_dir, exist_ok=True)
    out = {
        'version': 'wave_diffusion_v7_cnp_cond',
        'best_val_nRMSE': best_val,
        'best_epoch': best_epoch,
        'n_params': model.n_params,
        'cnp_params': cnp_model.n_params,
        'training_time_s': time.time() - t_start,
        'changes': [
            'ConvCNP v2 output as additional condition channel (frozen)',
            'in_ch=4: noisy + SetConv(2) + ConvCNP(1)',
            'epsilon-prediction, linear schedule, base_ch=48',
        ],
        'config': {
            'grid_nx': GRID_NX, 'grid_ny': GRID_NY,
            'base_ch': BASE_CH, 'n_timesteps': N_TIMESTEPS,
            'ddim_steps': DDIM_STEPS, 'guidance_scale': GUIDANCE_SCALE,
            'cond_drop_prob': COND_DROP_PROB,
            'epochs': EPOCHS, 'lr': LR, 'batch_size': BATCH_SIZE,
            'disp_k_min': DISP_K_MIN, 'disp_k_max': DISP_K_MAX,
            'disp_strength': DISP_STRENGTH,
        },
        'test_results': all_results,
    }
    with open(os.path.join(results_dir, 'wave_diffusion_v7_eval.json'), 'w') as f:
        json.dump(out, f, indent=2)
    print("\n  Results saved to results/wave_diffusion_v7_eval.json")


if __name__ == '__main__':
    main()
