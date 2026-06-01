#!/usr/bin/env python3
"""Run test evaluation only for v6 using saved checkpoint."""
import sys, os, json, time
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from wavesplat.data_loader import EpisodeDataLoader
from wavesplat.model.wave_diffusion import (
    WaveDiffusion, GaussianDiffusionV4, _make_dispersion_mask
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
BASE_CH = 64
ATTN_LEVELS = ['bot', 'enc3', 'dec3']
N_TIMESTEPS = 1000
DDIM_STEPS = 50
DISP_K_MIN, DISP_K_MAX, DISP_STRENGTH = 0.02, 0.5, 0.3
MIN_SNR_GAMMA = 5.0

def load_episodes(names, data_dir):
    return [EpisodeDataLoader(os.path.join(data_dir, n)) for n in names
            if os.path.exists(os.path.join(data_dir, n))]

def test_eval_full(model, diffusion, test_eps, ddim_steps, configs, disp_mask):
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
                shape = (1, 1, model.grid_nx, model.grid_ny)
                preds = []
                for _ in range(n_samples):
                    with torch.no_grad():
                        s = diffusion.sample_ddim_v4(
                            model, cond, shape, n_steps=ddim_steps,
                            guidance_scale=gs, self_cond=False,
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
    test_eps = load_episodes(TEST_EPISODES, data_dir)
    print("Test episodes: %d" % len(test_eps))

    ep0 = test_eps[0]
    x_range = (float(ep0.gt_grid_x[0]), float(ep0.gt_grid_x[-1]))
    y_range = (float(ep0.gt_grid_y[0]), float(ep0.gt_grid_y[-1]))

    model = WaveDiffusion(grid_nx=GRID_NX, grid_ny=GRID_NY,
                          x_range=x_range, y_range=y_range,
                          base_ch=BASE_CH, init_lengthscale=10.0,
                          n_timesteps=N_TIMESTEPS, self_cond=False,
                          attn_levels=ATTN_LEVELS)

    ckpt_path = os.path.join(data_dir, '..', 'checkpoints', 'wave_diffusion_v6_best.pt')
    model.load_state_dict(torch.load(ckpt_path, map_location='cpu', weights_only=True))
    print("Loaded checkpoint: %s" % ckpt_path)
    print("Params: %s" % format(model.n_params, ','))

    diffusion = GaussianDiffusionV4(n_timesteps=N_TIMESTEPS, min_snr_gamma=MIN_SNR_GAMMA)
    disp_mask = _make_dispersion_mask(GRID_NX, GRID_NY, DISP_K_MIN, DISP_K_MAX, 'cpu')

    configs = [
        (1.0, 1, False, "baseline gs=1 1s"),
        (3.0, 1, False, "baseline gs=3 1s"),
        (3.0, 1, True, "disp gs=3 1s"),
        (3.0, 5, True, "disp gs=3 5s"),
        (3.0, 3, True, "disp gs=3 3s"),
    ]

    t0 = time.time()
    all_results = test_eval_full(model, diffusion, test_eps, DDIM_STEPS, configs, disp_mask)
    elapsed = time.time() - t0

    out = {
        'version': 'wave_diffusion_v6',
        'best_val_nRMSE': 0.2705,
        'best_epoch': 280,
        'n_params': model.n_params,
        'eval_time_s': elapsed,
        'test_results': all_results,
    }
    results_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results')
    os.makedirs(results_dir, exist_ok=True)
    with open(os.path.join(results_dir, 'wave_diffusion_v6_eval.json'), 'w') as f:
        json.dump(out, f, indent=2)
    print("\nSaved to results/wave_diffusion_v6_eval.json (%.0fs)" % elapsed)

if __name__ == '__main__':
    main()
