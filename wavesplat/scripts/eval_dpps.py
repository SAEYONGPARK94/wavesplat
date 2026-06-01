#!/usr/bin/env python3
"""Evaluate Dispersion-Projected Posterior Sampling (DPPS) on v3 model.

DPPS combines:
  1. DPS (Diffusion Posterior Sampling) — observation guidance gradient
  2. Dispersion spectral projection — FFT bandpass constraint

No retraining needed — uses existing v3 checkpoint at inference time.

Sweeps obs_guidance_scale (ζ) × disp_strength × n_samples.
"""

import sys, os, json, time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from wavesplat.data_loader import EpisodeDataLoader
from wavesplat.model.wave_diffusion import (
    WaveDiffusion, GaussianDiffusion, _make_dispersion_mask
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


def evaluate_dpps(model, diffusion, test_eps, gs, n_samples,
                  disp_strength, obs_guidance_scale, label):
    """Evaluate a single DPPS configuration."""
    print("\n--- %s ---" % label)
    model.eval()
    ep_results, amp_ratios = [], []

    disp_mask = _make_dispersion_mask(GRID_NX, GRID_NY, DISP_K_MIN, DISP_K_MAX, 'cpu')

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

            # Build observation grid for DPS guidance
            obs_mask, obs_values = model.make_obs_grid(xc, yc, radius=2)

            shape = (1, 1, model.grid_nx, model.grid_ny)
            preds = []
            for _ in range(n_samples):
                s = diffusion.sample_ddim_dpps(
                    model, cond, shape,
                    n_steps=DDIM_STEPS, guidance_scale=gs,
                    obs_mask=obs_mask, obs_values=obs_values,
                    k_min=DISP_K_MIN, k_max=DISP_K_MAX,
                    disp_strength=disp_strength,
                    obs_guidance_scale=obs_guidance_scale)
                preds.append(s.detach())

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

    test_eps = [EpisodeDataLoader(os.path.join(data_dir, n))
                for n in TEST_EPISODES if os.path.exists(os.path.join(data_dir, n))]
    print("Test episodes: %d" % len(test_eps))

    ep0 = test_eps[0]
    x_range = (float(ep0.gt_grid_x[0]), float(ep0.gt_grid_x[-1]))
    y_range = (float(ep0.gt_grid_y[0]), float(ep0.gt_grid_y[-1]))

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

    # DPPS configs: (gs, n_samples, disp_strength, obs_guidance_scale, label)
    configs = [
        # Baselines (no DPS, for comparison)
        (3.0, 1, 0.0,  0.0,  "baseline gs=3 1s"),
        (3.0, 1, 0.5,  0.0,  "disp_only gs=3 1s"),

        # DPS only (no dispersion) — sweep ζ
        (3.0, 1, 0.0,  0.5,  "dps_0.5 gs=3 1s"),
        (3.0, 1, 0.0,  1.0,  "dps_1.0 gs=3 1s"),
        (3.0, 1, 0.0,  2.0,  "dps_2.0 gs=3 1s"),
        (3.0, 1, 0.0,  5.0,  "dps_5.0 gs=3 1s"),

        # DPPS: DPS + Dispersion combined — sweep ζ with best disp_strength=0.5
        (3.0, 1, 0.5,  0.5,  "dpps_z0.5_d0.5 gs=3 1s"),
        (3.0, 1, 0.5,  1.0,  "dpps_z1.0_d0.5 gs=3 1s"),
        (3.0, 1, 0.5,  2.0,  "dpps_z2.0_d0.5 gs=3 1s"),
        (3.0, 1, 0.5,  5.0,  "dpps_z5.0_d0.5 gs=3 1s"),

        # DPPS with hard dispersion
        (3.0, 1, 1.0,  1.0,  "dpps_z1.0_d1.0 gs=3 1s"),
        (3.0, 1, 1.0,  2.0,  "dpps_z2.0_d1.0 gs=3 1s"),

        # Best DPPS configs with multi-sample
        (3.0, 5, 0.5,  1.0,  "dpps_z1.0_d0.5 gs=3 5s"),
        (3.0, 5, 0.5,  2.0,  "dpps_z2.0_d0.5 gs=3 5s"),
    ]

    t0 = time.time()
    all_results = {}
    for gs, n_samples, ds, zeta, label in configs:
        result = evaluate_dpps(model, diffusion, test_eps,
                               gs, n_samples, ds, zeta, label)
        all_results[label] = result

    elapsed = time.time() - t0

    out = {
        'version': 'dpps_eval_v1',
        'model': 'wave_diffusion_v3 (epsilon-pred, base_ch=48)',
        'eval_time_s': elapsed,
        'test_results': all_results,
    }
    results_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results')
    os.makedirs(results_dir, exist_ok=True)
    out_path = os.path.join(results_dir, 'dpps_eval.json')
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2)
    print("\nSaved to %s (%.0fs)" % (out_path, elapsed))


if __name__ == '__main__':
    main()
