#!/usr/bin/env python3
"""Evaluate ConvCNP v2 with post-hoc dispersion spectral filtering.

Applies the same FFT bandpass filter used in DPPS to ConvCNP's UNet grid output
before interpolating to the target grid. No retraining — pure inference-time
physics constraint.

Sweeps disp_strength ∈ {0.0, 0.3, 0.5, 0.7, 1.0}.

Usage:
    cd wavesplat && python3 -u scripts/eval_conv_cnp_disp.py
"""

import sys, os, json, time, math
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from wavesplat.data_loader import EpisodeDataLoader
from wavesplat.model.conv_cnp import ConvCNP
from wavesplat.evaluation.metrics import nrmse

TEST_EPISODES = [
    'episode_BF3_5comp_seed0_slope.h5',
    'episode_BF4_5comp_seed0_uniform.h5',
    'episode_BF4_15comp_seed0_uniform.h5',
    'episode_BF4_30comp_seed0_uniform.h5',
    'episode_BF5_15comp_seed0_uniform.h5',
    'episode_BF5_30comp_seed0_uniform.h5',
]

GRID_NX = 64
GRID_NY = 48
DISP_K_MIN = 0.02   # rad/m
DISP_K_MAX = 0.5    # rad/m


def make_dispersion_mask(H, W, dx_h, dx_w, k_min, k_max, device):
    """Wavenumber bandpass mask for a grid with physical spacings dx_h, dx_w [m]."""
    freq_h = torch.fft.fftfreq(H, d=1.0, device=device)   # cycles/pixel
    freq_w = torch.fft.fftfreq(W, d=1.0, device=device)
    fh, fw = torch.meshgrid(freq_h, freq_w, indexing='ij')

    # Convert to physical wavenumber [rad/m]
    kh = 2 * math.pi * fh / dx_h
    kw = 2 * math.pi * fw / dx_w
    k_mag = torch.sqrt(kh ** 2 + kw ** 2)

    sharpness = 30.0
    low_pass  = torch.sigmoid(sharpness * (k_mag - k_min))
    high_pass = torch.sigmoid(sharpness * (k_max - k_mag))
    mask = low_pass * high_pass
    mask[0, 0] = 1.0   # keep DC
    return mask


def apply_dispersion_filter(grid_img, disp_mask, strength):
    """Apply bandpass filter to UNet output [B, 1, H, W].

    Returns filtered [B, 1, H, W].
    """
    x_2d = grid_img[:, 0]                       # [B, H, W]
    X_freq = torch.fft.fft2(x_2d)
    X_filtered = X_freq * disp_mask.unsqueeze(0)
    x_filtered = torch.fft.ifft2(X_filtered).real
    x_out = (1 - strength) * x_2d + strength * x_filtered
    return x_out.unsqueeze(1)


def evaluate_config(model, test_eps, disp_mask, strength, gt_grid_x, gt_grid_y, label):
    """Evaluate one disp_strength setting."""
    print("\n--- %s ---" % label)
    model.eval()

    gx, gy = gt_grid_x, gt_grid_y
    xx, yy = torch.meshgrid(gx, gy, indexing='ij')
    xt = torch.stack([xx.flatten(), yy.flatten()], dim=-1).unsqueeze(0)  # [1, Nt, 2]

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
                # --- Step 1: SetConv → grid features ---
                grid_feat = model.setconv(xc, yc, model.grid_xy)   # [1, G, 2]
                grid_img = grid_feat.permute(0, 2, 1).reshape(
                    1, 2, model.grid_nx, model.grid_ny)

                # --- Step 2: UNet ---
                out_img = model.unet(grid_img)                       # [1, 1, Gx, Gy]

                # --- Step 3: Dispersion filter (if strength > 0) ---
                if strength > 0 and disp_mask is not None:
                    out_img = apply_dispersion_filter(out_img, disp_mask, strength)

                # --- Step 4: Interpolate to GT grid ---
                pred = model._interpolate_to_targets(out_img, xt)   # [1, Nt, 1]

            pred_field = (pred[0, :, 0] * Hs).reshape(
                len(gx), len(gy)).T.numpy()
            gt_field = frame.gt_field.numpy()

            val = nrmse(pred_field, gt_field, ep.Hs_true)
            if not np.isnan(val):
                ep_nrmses.append(val)

            pred_std = float(np.std(pred_field))
            gt_std   = float(np.std(gt_field))
            if gt_std > 0.01:
                ep_amp.append(pred_std / gt_std)

        avg = float(np.mean(ep_nrmses))
        amp = float(np.mean(ep_amp)) if ep_amp else 0.0
        ep_name = TEST_EPISODES[ep_idx].replace('episode_', '').replace('.h5', '')
        ep_results.append(avg)
        amp_ratios.append(amp)
        print("  %s: nRMSE=%.4f amp=%.2f" % (ep_name, avg, amp))

    overall  = float(np.mean(ep_results))
    avg_amp  = float(np.mean(amp_ratios))
    print("  Avg: nRMSE=%.4f amp=%.2f" % (overall, avg_amp))
    return {
        'nrmse': overall,
        'amp':   avg_amp,
        'per_episode': dict(zip(
            [n.replace('episode_', '').replace('.h5', '') for n in TEST_EPISODES],
            ep_results)),
    }


def main():
    data_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')

    test_eps = [EpisodeDataLoader(os.path.join(data_dir, n))
                for n in TEST_EPISODES
                if os.path.exists(os.path.join(data_dir, n))]
    print("Test episodes: %d" % len(test_eps))

    ep0 = test_eps[0]
    gt_grid_x = ep0.gt_grid_x
    gt_grid_y = ep0.gt_grid_y
    x_range = (float(gt_grid_x[0]), float(gt_grid_x[-1]))
    y_range = (float(gt_grid_y[0]), float(gt_grid_y[-1]))

    # Physical grid spacing for ConvCNP 64x48
    dx_h = (x_range[1] - x_range[0]) / GRID_NX   # m/pixel along x
    dx_w = (y_range[1] - y_range[0]) / GRID_NY   # m/pixel along y
    print("Grid spacing: dx_h=%.2fm, dx_w=%.2fm" % (dx_h, dx_w))

    # Build model
    model = ConvCNP(
        grid_nx=GRID_NX, grid_ny=GRID_NY,
        x_range=x_range, y_range=y_range,
        base_ch=32, init_lengthscale=10.0,
    )

    ckpt_path = os.path.join(data_dir, '..', 'checkpoints', 'conv_cnp_v2_best.pt')
    model.load_state_dict(
        torch.load(ckpt_path, map_location='cpu', weights_only=True))
    model.eval()
    print("Loaded conv_cnp_v2 checkpoint: %s" % ckpt_path)
    print("Model params: %s" % format(model.n_params, ','))

    # Pre-compute dispersion mask for ConvCNP grid
    disp_mask = make_dispersion_mask(
        GRID_NX, GRID_NY, dx_h, dx_w, DISP_K_MIN, DISP_K_MAX, 'cpu')

    # Sweep
    configs = [
        (0.0,  "baseline (no filter)"),
        (0.3,  "disp_strength=0.3"),
        (0.5,  "disp_strength=0.5"),
        (0.7,  "disp_strength=0.7"),
        (1.0,  "disp_strength=1.0 (hard)"),
    ]

    t0 = time.time()
    all_results = {}
    for strength, label in configs:
        result = evaluate_config(
            model, test_eps,
            disp_mask if strength > 0 else None,
            strength, gt_grid_x, gt_grid_y, label)
        all_results[label] = result

    elapsed = time.time() - t0

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY — ConvCNP v2 + Dispersion Filter")
    print("=" * 60)
    print("  %-35s  nRMSE   amp" % "Config")
    print("  " + "-" * 50)
    for label, r in all_results.items():
        print("  %-35s  %.4f  %.3f" % (label, r['nrmse'], r['amp']))
    print("\n  Reference: ConvCNP v2 (no filter): 0.2438")
    print("  Reference: Diffusion v3+DPPS best:   0.2722")
    print("  Reference: WaveSplat:                0.223")

    out = {
        'version': 'conv_cnp_disp_eval_v1',
        'model': 'conv_cnp_v2 (base_ch=32, grid=64x48)',
        'disp_k_min': DISP_K_MIN,
        'disp_k_max': DISP_K_MAX,
        'dx_h': dx_h,
        'dx_w': dx_w,
        'eval_time_s': elapsed,
        'test_results': all_results,
    }
    results_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results')
    os.makedirs(results_dir, exist_ok=True)
    out_path = os.path.join(results_dir, 'conv_cnp_disp_eval.json')
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2)
    print("\nSaved to %s (%.0fs)" % (out_path, elapsed))


if __name__ == '__main__':
    main()
