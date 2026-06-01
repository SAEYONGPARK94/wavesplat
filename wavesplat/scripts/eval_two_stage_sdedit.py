#!/usr/bin/env python3
"""
Two-stage SDEdit: ConvCNP mean prediction → Diffusion amplitude refinement.

Approach:
  1. ConvCNP v2 produces mean prediction (nRMSE 0.244, amp 0.40)
  2. Resize to diffusion grid (64x48 → 80x60)
  3. Add noise at level t_start (< T) to get x_{t_start}
  4. Denoise from t_start using v3 diffusion model
  5. Expected: spatial pattern from ConvCNP + amplitude from diffusion

Key parameter: t_start — controls noise level
  - Low t_start (e.g. 100): preserves ConvCNP spatial accuracy, less amplitude recovery
  - High t_start (e.g. 500): more freedom for amplitude, risks losing spatial accuracy
  - Sweet spot: where spatial accuracy ≈ ConvCNP but amp_ratio > 0.40

Usage:
    cd wavesplat && python3 -u scripts/eval_two_stage_sdedit.py
"""

import sys, os, time, json
import numpy as np
import torch
import torch.nn.functional as F

sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', buffering=1)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from wavesplat.data_loader import EpisodeDataLoader
from wavesplat.model.conv_cnp import ConvCNP
from wavesplat.model.wave_diffusion import (
    WaveDiffusion, GaussianDiffusion, _make_dispersion_mask, _apply_dispersion_filter
)
from wavesplat.evaluation.metrics import nrmse

TEST_EPISODES = [
    'episode_BF3_5comp_seed0_slope.h5',
    'episode_BF4_5comp_seed0_uniform.h5',
    'episode_BF4_15comp_seed0_uniform.h5',
    'episode_BF4_30comp_seed0_uniform.h5',
    'episode_BF5_15comp_seed0_uniform.h5',
    'episode_BF5_30comp_seed0_uniform.h5',
]

# ConvCNP config
CNP_GRID_NX, CNP_GRID_NY = 64, 48
CNP_BASE_CH = 32

# Diffusion config (v3)
DIFF_GRID_NX, DIFF_GRID_NY = 80, 60
DIFF_BASE_CH = 48
N_TIMESTEPS = 1000
DDIM_STEPS = 50
GUIDANCE_SCALE = 3.0
DISP_K_MIN, DISP_K_MAX, DISP_STRENGTH = 0.02, 0.5, 0.3

# SDEdit t_start values to sweep
T_START_VALUES = [50, 100, 200, 300, 400, 500]


def load_episodes(names, data_dir):
    return [EpisodeDataLoader(os.path.join(data_dir, n)) for n in names
            if os.path.exists(os.path.join(data_dir, n))]


def get_cnp_grid_prediction(cnp_model, frame, Hs, diff_grid_nx, diff_grid_ny,
                             x_range, y_range):
    """Get ConvCNP prediction on diffusion grid (resized).

    Returns: [1, 1, diff_grid_nx, diff_grid_ny] Hs-normalized prediction
    """
    xc = torch.stack([frame.x_obs, frame.y_obs], dim=-1).unsqueeze(0)
    yc = frame.z_obs.unsqueeze(-1).unsqueeze(0) / Hs

    with torch.no_grad():
        # Get ConvCNP's internal grid prediction
        B = 1
        grid_feat = cnp_model.setconv(xc, yc, cnp_model.grid_xy)
        grid_img = grid_feat.permute(0, 2, 1).reshape(
            B, 2, cnp_model.grid_nx, cnp_model.grid_ny)
        out_img = cnp_model.unet(grid_img)  # [1, 1, 64, 48]

    # Resize to diffusion grid
    out_resized = F.interpolate(
        out_img, size=(diff_grid_nx, diff_grid_ny),
        mode='bilinear', align_corners=True)

    return out_resized  # [1, 1, 80, 60]


def sdedit_sample(diffusion_model, diffusion, cnp_pred, cond, t_start,
                  n_steps=50, guidance_scale=1.0,
                  disp_mask=None, disp_strength=0.0):
    """SDEdit: add noise to cnp_pred at t_start, then denoise.

    Args:
        diffusion_model: WaveDiffusion (v3)
        diffusion: GaussianDiffusion
        cnp_pred: [B, 1, H, W] ConvCNP prediction (Hs-normalized)
        cond: [B, 2, H, W] diffusion condition
        t_start: noise level to start denoising from
        n_steps: total DDIM steps
        guidance_scale: CFG scale
        disp_mask: dispersion bandpass mask
        disp_strength: dispersion strength

    Returns:
        [B, 1, H, W] refined prediction
    """
    device = cond.device
    B = cnp_pred.shape[0]

    # Add noise at t_start
    t = torch.full((B,), t_start, device=device, dtype=torch.long)
    noise = torch.randn_like(cnp_pred)
    x_t = diffusion.q_sample(cnp_pred, t, noise)[0]

    # DDIM timesteps: only use steps <= t_start
    step_size = diffusion.n_timesteps // n_steps
    all_timesteps = list(range(0, diffusion.n_timesteps, step_size))
    all_timesteps = list(reversed(all_timesteps))

    # Filter to only timesteps <= t_start
    timesteps = [ts for ts in all_timesteps if ts <= t_start]

    uncond = torch.zeros_like(cond) if guidance_scale > 1.0 else None
    x = x_t

    for i, t_cur in enumerate(timesteps):
        t_batch = torch.full((B,), t_cur, device=device, dtype=torch.long)

        if guidance_scale > 1.0 and uncond is not None:
            noise_cond = diffusion_model(x, t_batch, cond)
            noise_uncond = diffusion_model(x, t_batch, uncond)
            pred_noise = noise_uncond + guidance_scale * (noise_cond - noise_uncond)
        else:
            pred_noise = diffusion_model(x, t_batch, cond)

        alpha_bar = diffusion.alphas_cumprod[t_cur]
        x0_pred = (x - torch.sqrt(1 - alpha_bar) * pred_noise) / torch.sqrt(alpha_bar)

        # Dispersion filter
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


def evaluate_config(cnp_model, diff_model, diffusion, test_eps,
                    t_start, n_samples, use_disp, disp_mask, guidance_scale):
    """Evaluate one SDEdit configuration."""
    diff_model.eval()
    cnp_model.eval()

    ep_results, amp_ratios = [], []
    for ep_idx, ep in enumerate(test_eps):
        Hs = max(ep.Hs_true, 0.1)
        x_range = (float(ep.gt_grid_x[0]), float(ep.gt_grid_x[-1]))
        y_range = (float(ep.gt_grid_y[0]), float(ep.gt_grid_y[-1]))

        n_eval = min(20, len(ep))
        frame_indices = np.linspace(10, len(ep) - 1, n_eval, dtype=int)
        ep_nrmses, ep_amp = [], []

        for fi in frame_indices:
            frame = ep[fi]

            # Stage 1: ConvCNP prediction
            cnp_pred = get_cnp_grid_prediction(
                cnp_model, frame, Hs,
                DIFF_GRID_NX, DIFF_GRID_NY, x_range, y_range)

            # Diffusion condition
            xc = torch.stack([frame.x_obs, frame.y_obs], dim=-1).unsqueeze(0)
            yc = frame.z_obs.unsqueeze(-1).unsqueeze(0) / Hs
            with torch.no_grad():
                cond = diff_model.get_condition(xc, yc)

            # Stage 2: SDEdit
            preds = []
            for _ in range(n_samples):
                with torch.no_grad():
                    s = sdedit_sample(
                        diff_model, diffusion, cnp_pred, cond,
                        t_start=t_start, n_steps=DDIM_STEPS,
                        guidance_scale=guidance_scale,
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
    return overall, avg_amp, ep_results


def main():
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
    ckpt_dir = os.path.join(data_dir, '..', 'checkpoints')
    test_eps = load_episodes(TEST_EPISODES, data_dir)
    print("Test episodes: %d" % len(test_eps))

    ep0 = test_eps[0]
    x_range = (float(ep0.gt_grid_x[0]), float(ep0.gt_grid_x[-1]))
    y_range = (float(ep0.gt_grid_y[0]), float(ep0.gt_grid_y[-1]))

    # Load ConvCNP v2
    cnp_model = ConvCNP(
        grid_nx=CNP_GRID_NX, grid_ny=CNP_GRID_NY,
        x_range=x_range, y_range=y_range,
        base_ch=CNP_BASE_CH, init_lengthscale=10.0)
    cnp_model.load_state_dict(torch.load(
        os.path.join(ckpt_dir, 'conv_cnp_v2_best.pt'),
        map_location='cpu', weights_only=True))
    cnp_model.eval()
    print("ConvCNP v2: %s params" % format(cnp_model.n_params, ','))

    # Load Diffusion v3
    diff_model = WaveDiffusion(
        grid_nx=DIFF_GRID_NX, grid_ny=DIFF_GRID_NY,
        x_range=x_range, y_range=y_range,
        base_ch=DIFF_BASE_CH, init_lengthscale=10.0,
        n_timesteps=N_TIMESTEPS)
    diff_model.load_state_dict(torch.load(
        os.path.join(ckpt_dir, 'wave_diffusion_v3_best.pt'),
        map_location='cpu', weights_only=True))
    diff_model.eval()
    print("Diffusion v3: %s params" % format(diff_model.n_params, ','))

    diffusion = GaussianDiffusion(n_timesteps=N_TIMESTEPS)
    disp_mask = _make_dispersion_mask(DIFF_GRID_NX, DIFF_GRID_NY, DISP_K_MIN, DISP_K_MAX, 'cpu')

    print("\n" + "=" * 80)
    print("Two-Stage SDEdit: ConvCNP v2 → Diffusion v3")
    print("=" * 80)

    all_results = {}

    # ConvCNP-only baseline (already known but re-evaluate on diffusion grid for fair comparison)
    print("\n--- ConvCNP v2 baseline (resized to 80x60) ---")
    ep_results_cnp, amp_ratios_cnp = [], []
    for ep_idx, ep in enumerate(test_eps):
        Hs = max(ep.Hs_true, 0.1)
        n_eval = min(20, len(ep))
        frame_indices = np.linspace(10, len(ep) - 1, n_eval, dtype=int)
        ep_nrmses, ep_amp = [], []
        for fi in frame_indices:
            frame = ep[fi]
            cnp_pred = get_cnp_grid_prediction(
                cnp_model, frame, Hs,
                DIFF_GRID_NX, DIFF_GRID_NY, x_range, y_range)
            pred_field = (cnp_pred[0, 0].permute(1, 0) * Hs).numpy()
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
        ep_results_cnp.append(avg)
        amp_ratios_cnp.append(amp)
        print("    %s: nRMSE=%.4f amp=%.2f" % (ep_name, avg, amp))
    cnp_overall = float(np.mean(ep_results_cnp))
    cnp_amp = float(np.mean(amp_ratios_cnp))
    print("    Avg: nRMSE=%.4f amp=%.2f" % (cnp_overall, cnp_amp))
    all_results['cnp_baseline'] = {'nrmse': cnp_overall, 'amp': cnp_amp}

    # SDEdit sweep over t_start values
    t0 = time.time()
    for t_start in T_START_VALUES:
        for gs in [1.0, 3.0]:
            for n_s, use_disp in [(1, False), (1, True)]:
                label = "sdedit t=%d gs=%.0f %s%s" % (
                    t_start, gs, "disp" if use_disp else "base", " %ds" % n_s)
                print("\n--- %s ---" % label)
                nrmse_avg, amp_avg, per_ep = evaluate_config(
                    cnp_model, diff_model, diffusion, test_eps,
                    t_start, n_s, use_disp, disp_mask, gs)
                print("    Avg: nRMSE=%.4f amp=%.2f" % (nrmse_avg, amp_avg))
                all_results[label] = {
                    'nrmse': nrmse_avg, 'amp': amp_avg,
                    't_start': t_start, 'guidance_scale': gs,
                    'use_disp': use_disp, 'n_samples': n_s,
                    'per_episode': dict(zip(
                        [n.replace('episode_', '').replace('.h5', '') for n in TEST_EPISODES],
                        per_ep))
                }

    elapsed = time.time() - t0

    # Save results
    out = {
        'version': 'two_stage_sdedit_v1',
        'method': 'ConvCNP v2 → SDEdit → Diffusion v3',
        'eval_time_s': elapsed,
        'results': all_results,
    }
    results_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results')
    os.makedirs(results_dir, exist_ok=True)
    with open(os.path.join(results_dir, 'two_stage_sdedit_eval.json'), 'w') as f:
        json.dump(out, f, indent=2)
    print("\nSaved to results/two_stage_sdedit_eval.json (%.0fs)" % elapsed)

    # Summary
    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)
    print("  ConvCNP v2 (resized): nRMSE=%.4f amp=%.2f" % (cnp_overall, cnp_amp))
    print("  Diffusion v3 (disp gs=3 1s): nRMSE=0.282 amp=0.46")
    print("  Diffusion v3 (baseline gs=3 1s): nRMSE=0.325 amp=0.89")
    for key, val in sorted(all_results.items()):
        if key != 'cnp_baseline':
            print("  %s: nRMSE=%.4f amp=%.2f" % (key, val['nrmse'], val['amp']))


if __name__ == '__main__':
    main()
