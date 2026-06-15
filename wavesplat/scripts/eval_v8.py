#!/usr/bin/env python3
"""Quick test evaluation for WaveDiffusion v8 checkpoint."""

import sys, os, json, time
import numpy as np
import torch

sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', buffering=1)
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
N_TIMESTEPS = 1000
DDIM_STEPS = 50
DISP_K_MIN, DISP_K_MAX = 0.02, 0.5
DISP_STRENGTH = 0.3


def eval_config(model, diffusion, test_eps, gs, n_samples, disp_s, disp_mask, label):
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
                with torch.no_grad():
                    s = diffusion.sample_ddim_v4(
                        model, cond, shape, n_steps=DDIM_STEPS,
                        guidance_scale=gs, self_cond=False,
                        disp_mask=disp_mask if disp_s > 0 else None,
                        disp_strength=disp_s)
                preds.append(s)
            pred = torch.stack(preds).mean(0)
            pred_field = (pred[0, 0].permute(1, 0) * Hs).numpy()
            gt_field = frame.gt_field.numpy()
            val = nrmse(pred_field, gt_field, ep.Hs_true)
            if not np.isnan(val):
                ep_nrmses.append(val)
            ps, gs_ = float(np.std(pred_field)), float(np.std(gt_field))
            if gs_ > 0.01:
                ep_amp.append(ps / gs_)
        avg = float(np.mean(ep_nrmses))
        amp = float(np.mean(ep_amp)) if ep_amp else 0.0
        ep_name = TEST_EPISODES[ep_idx].replace('episode_', '').replace('.h5', '')
        ep_results.append(avg)
        amp_ratios.append(amp)
        print("    %s: %.4f amp=%.2f" % (ep_name, avg, amp))
    overall = float(np.mean(ep_results))
    avg_amp = float(np.mean(amp_ratios)) if amp_ratios else 0.0
    print("    Avg: nRMSE=%.4f amp=%.2f" % (overall, avg_amp))
    return {'nrmse': overall, 'amp': avg_amp,
            'per_episode': dict(zip(
                [n.replace('episode_', '').replace('.h5', '') for n in TEST_EPISODES],
                ep_results))}


def main():
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
    test_eps = [EpisodeDataLoader(os.path.join(data_dir, n))
                for n in TEST_EPISODES if os.path.exists(os.path.join(data_dir, n))]

    ep0 = test_eps[0]
    x_range = (float(ep0.gt_grid_x[0]), float(ep0.gt_grid_x[-1]))
    y_range = (float(ep0.gt_grid_y[0]), float(ep0.gt_grid_y[-1]))

    model = WaveDiffusion(grid_nx=GRID_NX, grid_ny=GRID_NY,
                          x_range=x_range, y_range=y_range,
                          base_ch=48, init_lengthscale=10.0,
                          n_timesteps=N_TIMESTEPS, self_cond=False, use_attn=True)

    ckpt = os.path.join(data_dir, '..', 'checkpoints', 'wave_diffusion_v8_best.pt')
    state = torch.load(ckpt, map_location='cpu', weights_only=True)
    model.load_state_dict(state)
    model.eval()
    print("Loaded v8 checkpoint | %s params" % format(model.n_params, ','))

    diffusion = GaussianDiffusionV4(n_timesteps=N_TIMESTEPS, min_snr_gamma=5.0)
    disp_mask = _make_dispersion_mask(GRID_NX, GRID_NY, DISP_K_MIN, DISP_K_MAX, 'cpu')

    configs = [
        (1.0, 1, 0.0, "baseline gs=1 1s"),
        (1.0, 5, 0.0, "baseline gs=1 5s"),
        (3.0, 1, 0.0, "baseline gs=3 1s"),
        (3.0, 1, DISP_STRENGTH, "disp gs=3 1s"),
        (3.0, 5, DISP_STRENGTH, "disp gs=3 5s"),
        (3.0, 3, DISP_STRENGTH, "disp gs=3 3s"),
    ]

    print("\n" + "=" * 60)
    print("WaveDiffusion v8 Test Evaluation")
    print("=" * 60)
    print("Reference: v5b baseline 5s=0.2632, WaveSplat=0.2230")

    t0 = time.time()
    results = {}
    for gs, ns, ds, label in configs:
        print("\n--- %s ---" % label)
        results[label] = eval_config(model, diffusion, test_eps, gs, ns, ds, disp_mask, label)

    print("\n" + "=" * 60)
    print("SUMMARY")
    for label, r in results.items():
        print("  %-28s: %.4f amp=%.2f" % (label, r['nrmse'], r['amp']))
    best = min(results, key=lambda k: results[k]['nrmse'])
    print("  Best: %s → %.4f" % (best, results[best]['nrmse']))
    print("  v5b best:   0.2632  |  WaveSplat: 0.2230")

    out = {'version': 'wave_diffusion_v8', 'n_params': model.n_params,
           'eval_time_s': time.time() - t0,
           'config': {'grid_nx': GRID_NX, 'grid_ny': GRID_NY, 'base_ch': 48,
                      'ddim_steps': DDIM_STEPS, 'disp_k_min': DISP_K_MIN,
                      'disp_k_max': DISP_K_MAX, 'disp_strength': DISP_STRENGTH,
                      'lambda_spec': 0.2},
           'references': {'v5b_baseline_5s': 0.2632, 'WaveSplat': 0.2230},
           'test_results': results}

    results_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results')
    out_path = os.path.join(results_dir, 'wave_diffusion_v8_eval.json')
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2)
    print("Saved: %s" % out_path)


if __name__ == '__main__':
    main()
