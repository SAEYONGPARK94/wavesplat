#!/usr/bin/env python3
"""Evaluate effect of dispersion filter strength on v3 model.

Tests multiple strengths (0.0 to 1.0) and also a "hard+noise" mode
that filters both x0_pred and the noise residual in x_t.

Uses existing v3 checkpoint — no new training needed.
"""

import sys, os, json, time
import numpy as np
import torch
import math
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from wavesplat.data_loader import EpisodeDataLoader
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

GRID_NX, GRID_NY = 80, 60
N_TIMESTEPS = 1000
DDIM_STEPS = 50
DISP_K_MIN, DISP_K_MAX = 0.02, 0.5


@torch.no_grad()
def sample_ddim_disp_v7(model, diffusion, cond, shape, n_steps=50,
                         guidance_scale=3.0, disp_mask=None,
                         disp_strength=0.3, filter_xt=False):
    """DDIM sampling with configurable dispersion projection.

    New option: filter_xt — also apply dispersion filter to x_t after each step,
    preventing out-of-band energy from persisting through the noise residual.
    """
    device = cond.device
    B = shape[0]

    step_size = diffusion.n_timesteps // n_steps
    timesteps = list(range(0, diffusion.n_timesteps, step_size))
    timesteps = list(reversed(timesteps))

    x = torch.randn(shape, device=device)
    uncond = torch.zeros_like(cond) if guidance_scale > 1.0 else None

    for i, t_cur in enumerate(timesteps):
        t = torch.full((B,), t_cur, device=device, dtype=torch.long)

        if guidance_scale > 1.0 and uncond is not None:
            noise_cond = model(x, t, cond)
            noise_uncond = model(x, t, uncond)
            pred_noise = noise_uncond + guidance_scale * (noise_cond - noise_uncond)
        else:
            pred_noise = model(x, t, cond)

        alpha_bar = diffusion.alphas_cumprod[t_cur]
        x0_pred = (x - torch.sqrt(1 - alpha_bar) * pred_noise) / torch.sqrt(alpha_bar)

        # Dispersion projection on x0
        if disp_mask is not None and disp_strength > 0:
            x0_pred = _apply_dispersion_filter(x0_pred, disp_mask, disp_strength)

        if i < len(timesteps) - 1:
            t_next = timesteps[i + 1]
            alpha_bar_next = diffusion.alphas_cumprod[t_next]
            x = (torch.sqrt(alpha_bar_next) * x0_pred +
                 torch.sqrt(1 - alpha_bar_next) * pred_noise)

            # Also filter x_t to prevent noise from re-introducing out-of-band energy
            if filter_xt and disp_mask is not None and disp_strength > 0:
                x = _apply_dispersion_filter(x, disp_mask, disp_strength)
        else:
            x = x0_pred

    return x


def evaluate_config(model, diffusion, test_eps, disp_mask, gs, n_samples,
                    disp_strength, filter_xt, label):
    """Evaluate a single configuration."""
    print("\n--- %s ---" % label)
    model.eval()
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

            shape = (1, 1, model.grid_nx, model.grid_ny)
            preds = []
            for _ in range(n_samples):
                s = sample_ddim_disp_v7(
                    model, diffusion, cond, shape,
                    n_steps=DDIM_STEPS, guidance_scale=gs,
                    disp_mask=disp_mask if disp_strength > 0 else None,
                    disp_strength=disp_strength,
                    filter_xt=filter_xt)
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
        print("  %s: nRMSE=%.4f amp=%.2f" % (ep_name, avg, amp))

    overall = float(np.mean(ep_results))
    avg_amp = float(np.mean(amp_ratios))
    print("  Avg: nRMSE=%.4f amp=%.2f" % (overall, avg_amp))
    return {
        'nrmse': overall, 'amp': avg_amp,
        'per_episode': dict(zip(
            [n.replace('episode_', '').replace('.h5', '') for n in TEST_EPISODES],
            ep_results))
    }


def main():
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')

    # Load test episodes
    test_eps = [EpisodeDataLoader(os.path.join(data_dir, n))
                for n in TEST_EPISODES if os.path.exists(os.path.join(data_dir, n))]
    print("Test episodes: %d" % len(test_eps))

    ep0 = test_eps[0]
    x_range = (float(ep0.gt_grid_x[0]), float(ep0.gt_grid_x[-1]))
    y_range = (float(ep0.gt_grid_y[0]), float(ep0.gt_grid_y[-1]))

    # Load v3 model (epsilon-prediction, base_ch=48)
    model = WaveDiffusion(
        grid_nx=GRID_NX, grid_ny=GRID_NY,
        x_range=x_range, y_range=y_range,
        base_ch=48, init_lengthscale=10.0,
        n_timesteps=N_TIMESTEPS, self_cond=False)

    ckpt_path = os.path.join(data_dir, '..', 'checkpoints', 'wave_diffusion_v3_best.pt')
    model.load_state_dict(torch.load(ckpt_path, map_location='cpu', weights_only=True))
    model.eval()
    print("Loaded v3 checkpoint: %s" % ckpt_path)

    diffusion = GaussianDiffusion(n_timesteps=N_TIMESTEPS)
    disp_mask = _make_dispersion_mask(GRID_NX, GRID_NY, DISP_K_MIN, DISP_K_MAX, 'cpu')

    # Configs: (gs, n_samples, disp_strength, filter_xt, label)
    configs = [
        # Baseline: no dispersion
        (3.0, 1, 0.0, False, "no_disp gs=3 1s"),
        # Current: strength=0.3
        (3.0, 1, 0.3, False, "disp_0.3 gs=3 1s"),
        # Stronger
        (3.0, 1, 0.5, False, "disp_0.5 gs=3 1s"),
        (3.0, 1, 0.7, False, "disp_0.7 gs=3 1s"),
        # Hard projection
        (3.0, 1, 1.0, False, "disp_1.0 gs=3 1s"),
        # Hard + filter x_t (prevent noise leaking)
        (3.0, 1, 1.0, True, "disp_1.0+xt gs=3 1s"),
        # Best configs with multi-sample
        (3.0, 5, 0.3, False, "disp_0.3 gs=3 5s"),
        (3.0, 5, 1.0, False, "disp_1.0 gs=3 5s"),
        (3.0, 5, 1.0, True, "disp_1.0+xt gs=3 5s"),
        # Hard + filter_xt with 3 samples (practical)
        (3.0, 3, 1.0, True, "disp_1.0+xt gs=3 3s"),
    ]

    t0 = time.time()
    all_results = {}
    for gs, n_samples, ds, fxt, label in configs:
        result = evaluate_config(model, diffusion, test_eps, disp_mask,
                                 gs, n_samples, ds, fxt, label)
        all_results[label] = result

    elapsed = time.time() - t0

    # Save
    out = {
        'version': 'dispersion_strength_sweep_v3',
        'model': 'wave_diffusion_v3 (epsilon-pred, base_ch=48)',
        'eval_time_s': elapsed,
        'test_results': all_results,
    }
    results_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results')
    os.makedirs(results_dir, exist_ok=True)
    with open(os.path.join(results_dir, 'disp_strength_eval.json'), 'w') as f:
        json.dump(out, f, indent=2)
    print("\nSaved to results/disp_strength_eval.json (%.0fs)" % elapsed)


if __name__ == '__main__':
    main()
