#!/usr/bin/env python3
"""
Evaluate UQ calibration of WaveDiffusion via ensemble sampling.

For each test frame:
  1. Generate N samples from the diffusion model
  2. Compute per-pixel mean, std (uncertainty)
  3. Compute per-pixel absolute error (|mean - GT|)
  4. Measure calibration: does high uncertainty → high error?

Metrics:
  - PICP (Prediction Interval Coverage Probability): fraction of GT within ±2σ
  - MPIW (Mean Prediction Interval Width): average interval width / Hs
  - Spearman correlation(uncertainty, error): should be positive
  - Calibration curve: binned uncertainty vs actual error

Compares: baseline (no dispersion) vs dispersion filter

Usage:
    cd wavesplat && python3 -u scripts/evaluate_uq_calibration.py
"""

import sys
import os
import json
import numpy as np
import torch
from scipy import stats

sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', buffering=1)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from wavesplat.data_loader import EpisodeDataLoader
from wavesplat.model.wave_diffusion import (
    WaveDiffusion, GaussianDiffusion, GaussianDiffusionV4, _make_dispersion_mask
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

N_SAMPLES = 20
N_FRAMES_PER_EP = 10
GUIDANCE_SCALE = 3.0
DDIM_STEPS = 50
DISP_STRENGTH = 0.3


def load_model(ckpt_name, base_ch, use_attn, data_dir):
    """Load a pretrained WaveDiffusion model."""
    ep0 = EpisodeDataLoader(os.path.join(data_dir, TEST_EPISODES[0]))
    x_range = (float(ep0.gt_grid_x[0]), float(ep0.gt_grid_x[-1]))
    y_range = (float(ep0.gt_grid_y[0]), float(ep0.gt_grid_y[-1]))

    model = WaveDiffusion(
        grid_nx=80, grid_ny=60,
        x_range=x_range, y_range=y_range,
        base_ch=base_ch,
        init_lengthscale=10.0,
        n_timesteps=1000,
        self_cond=False,
        use_attn=use_attn,
    )

    ckpt_path = os.path.join(data_dir, '..', 'checkpoints', ckpt_name)
    model.load_state_dict(torch.load(ckpt_path, map_location='cpu', weights_only=True))
    model.eval()
    return model


def evaluate_uq(model, diffusion, test_eps, use_dispersion, disp_mask):
    """Generate ensemble samples and compute UQ metrics."""
    all_picp = []
    all_mpiw = []
    all_spearman = []
    all_nrmse_mean = []
    all_nrmse_single = []
    all_amp_mean = []
    all_amp_single = []

    # For calibration curve
    n_bins = 10
    bin_errors = [[] for _ in range(n_bins)]
    bin_uncertainties = [[] for _ in range(n_bins)]

    for ep_idx, ep in enumerate(test_eps):
        Hs = max(ep.Hs_true, 0.1)
        frame_indices = np.linspace(10, len(ep) - 1, N_FRAMES_PER_EP, dtype=int)
        ep_name = TEST_EPISODES[ep_idx].replace('episode_', '').replace('.h5', '')

        for fi in frame_indices:
            frame = ep[fi]
            xc = torch.stack([frame.x_obs, frame.y_obs], dim=-1).unsqueeze(0)
            yc = frame.z_obs.unsqueeze(-1).unsqueeze(0) / Hs

            with torch.no_grad():
                cond = model.get_condition(xc, yc)

            shape = (1, 1, model.grid_nx, model.grid_ny)

            # Generate N samples (v3 uses epsilon-pred + sample_ddim_guided)
            samples = []
            for _ in range(N_SAMPLES):
                with torch.no_grad():
                    s = diffusion.sample_ddim_guided(
                        model, cond, shape,
                        n_steps=DDIM_STEPS,
                        guidance_scale=GUIDANCE_SCALE,
                        disp_strength=DISP_STRENGTH if use_dispersion else 0.0)
                pred_field = (s[0, 0].permute(1, 0) * Hs).numpy()
                samples.append(pred_field)

            samples = np.array(samples)  # [N, H, W]
            gt_field = frame.gt_field.numpy()  # [H, W]

            # Ensemble statistics
            mean_field = samples.mean(axis=0)
            std_field = samples.std(axis=0)
            abs_error = np.abs(mean_field - gt_field)

            # PICP: fraction within ±2σ
            lower = mean_field - 2 * std_field
            upper = mean_field + 2 * std_field
            covered = (gt_field >= lower) & (gt_field <= upper)
            picp = float(covered.mean())
            all_picp.append(picp)

            # MPIW: normalized by Hs
            mpiw = float((4 * std_field).mean() / Hs)
            all_mpiw.append(mpiw)

            # Spearman correlation(std, |error|)
            flat_std = std_field.flatten()
            flat_err = abs_error.flatten()
            spearman, _ = stats.spearmanr(flat_std, flat_err)
            if not np.isnan(spearman):
                all_spearman.append(float(spearman))

            # nRMSE of ensemble mean
            val_mean = nrmse(mean_field, gt_field, ep.Hs_true)
            if not np.isnan(val_mean):
                all_nrmse_mean.append(val_mean)

            # nRMSE of single sample (first one)
            val_single = nrmse(samples[0], gt_field, ep.Hs_true)
            if not np.isnan(val_single):
                all_nrmse_single.append(val_single)

            # Amplitude ratios
            gt_std_scalar = float(np.std(gt_field))
            if gt_std_scalar > 0.01:
                all_amp_mean.append(float(np.std(mean_field)) / gt_std_scalar)
                all_amp_single.append(float(np.std(samples[0])) / gt_std_scalar)

            # Calibration curve bins
            percentiles = np.percentile(flat_std, np.linspace(0, 100, n_bins + 1))
            for b in range(n_bins):
                mask = (flat_std >= percentiles[b]) & (flat_std < percentiles[b + 1])
                if b == n_bins - 1:
                    mask = (flat_std >= percentiles[b])
                if mask.sum() > 0:
                    bin_errors[b].append(float(flat_err[mask].mean()))
                    bin_uncertainties[b].append(float(flat_std[mask].mean()))

        print("  %s: PICP=%.3f MPIW=%.3f Spearman=%.3f" % (
            ep_name,
            float(np.mean(all_picp[-N_FRAMES_PER_EP:])),
            float(np.mean(all_mpiw[-N_FRAMES_PER_EP:])),
            float(np.mean(all_spearman[-N_FRAMES_PER_EP:])) if all_spearman else 0.0))

    # Build calibration curve
    cal_curve = []
    for b in range(n_bins):
        if bin_errors[b] and bin_uncertainties[b]:
            cal_curve.append({
                'uncertainty': float(np.mean(bin_uncertainties[b])),
                'error': float(np.mean(bin_errors[b])),
            })

    return {
        'picp': float(np.mean(all_picp)),
        'mpiw': float(np.mean(all_mpiw)),
        'spearman': float(np.mean(all_spearman)) if all_spearman else 0.0,
        'nrmse_mean': float(np.mean(all_nrmse_mean)),
        'nrmse_single': float(np.mean(all_nrmse_single)),
        'amp_mean': float(np.mean(all_amp_mean)),
        'amp_single': float(np.mean(all_amp_single)),
        'n_samples': N_SAMPLES,
        'calibration_curve': cal_curve,
    }


def main():
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            'data')

    print("=" * 80)
    print("UQ Calibration Evaluation — WaveDiffusion v3")
    print("  N_SAMPLES=%d, N_FRAMES=%d, gs=%.1f" % (N_SAMPLES, N_FRAMES_PER_EP, GUIDANCE_SCALE))
    print("=" * 80)

    # Load test episodes
    test_eps = []
    for name in TEST_EPISODES:
        path = os.path.join(data_dir, name)
        if os.path.exists(path):
            test_eps.append(EpisodeDataLoader(path))
    print("  Test episodes: %d" % len(test_eps))

    # Load v3 model (base_ch=48, no attention)
    model = load_model('wave_diffusion_v3_best.pt', base_ch=48, use_attn=False,
                        data_dir=data_dir)
    print("  Model: v3, %s params" % format(model.n_params, ','))

    # v3 uses epsilon-prediction with GaussianDiffusion (not V4)
    diffusion = GaussianDiffusion(n_timesteps=1000)
    disp_mask = _make_dispersion_mask(80, 60, 0.02, 0.5, 'cpu')

    # --- Config 1: Baseline (no dispersion) ---
    print("\n--- Baseline (no dispersion filter) ---")
    baseline_results = evaluate_uq(model, diffusion, test_eps,
                                    use_dispersion=False, disp_mask=disp_mask)
    print("\n  Summary: PICP=%.3f MPIW=%.3f Spearman=%.3f" % (
        baseline_results['picp'], baseline_results['mpiw'], baseline_results['spearman']))
    print("  nRMSE: single=%.4f mean(%d)=%.4f" % (
        baseline_results['nrmse_single'], N_SAMPLES, baseline_results['nrmse_mean']))
    print("  Amp: single=%.3f mean(%d)=%.3f" % (
        baseline_results['amp_single'], N_SAMPLES, baseline_results['amp_mean']))

    # --- Config 2: With dispersion filter ---
    print("\n--- Dispersion filter ---")
    disp_results = evaluate_uq(model, diffusion, test_eps,
                                use_dispersion=True, disp_mask=disp_mask)
    print("\n  Summary: PICP=%.3f MPIW=%.3f Spearman=%.3f" % (
        disp_results['picp'], disp_results['mpiw'], disp_results['spearman']))
    print("  nRMSE: single=%.4f mean(%d)=%.4f" % (
        disp_results['nrmse_single'], N_SAMPLES, disp_results['nrmse_mean']))
    print("  Amp: single=%.3f mean(%d)=%.3f" % (
        disp_results['amp_single'], N_SAMPLES, disp_results['amp_mean']))

    # --- Comparison ---
    print("\n" + "=" * 80)
    print("Comparison: Baseline vs Dispersion Filter")
    print("=" * 80)
    print("  PICP:     %.3f → %.3f (%+.1f%%)" % (
        baseline_results['picp'], disp_results['picp'],
        100 * (disp_results['picp'] - baseline_results['picp'])))
    print("  MPIW:     %.3f → %.3f (%+.1f%%)" % (
        baseline_results['mpiw'], disp_results['mpiw'],
        100 * (disp_results['mpiw'] - baseline_results['mpiw']) / baseline_results['mpiw']))
    print("  Spearman: %.3f → %.3f" % (
        baseline_results['spearman'], disp_results['spearman']))
    print("  nRMSE(%ds): %.4f → %.4f (%+.1f%%)" % (
        N_SAMPLES, baseline_results['nrmse_mean'], disp_results['nrmse_mean'],
        100 * (disp_results['nrmse_mean'] - baseline_results['nrmse_mean']) / baseline_results['nrmse_mean']))

    # Save results
    results_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                'results')
    os.makedirs(results_dir, exist_ok=True)
    out = {
        'version': 'wave_diffusion_v3_uq',
        'model': 'v3 (base_ch=48, epsilon-pred, linear schedule)',
        'n_samples': N_SAMPLES,
        'guidance_scale': GUIDANCE_SCALE,
        'baseline': baseline_results,
        'dispersion': disp_results,
    }
    with open(os.path.join(results_dir, 'uq_calibration_eval.json'), 'w') as f:
        json.dump(out, f, indent=2)
    print("\n  Results saved to results/uq_calibration_eval.json")


if __name__ == '__main__':
    main()
