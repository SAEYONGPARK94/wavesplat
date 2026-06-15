#!/usr/bin/env python3
"""
eval_comprehensive_comparison.py

Comprehensive multi-metric comparison: WavePINO Stage 4 vs WaveSplat.

Metrics (same panel as b1_full_comparison.json):
  - nRMSE            : primary accuracy
  - Variance_Ratio   : amplitude accuracy (VR≈1 = good; VR<<1 = over-smoothed)
  - Pearson_rho      : spatial shape correlation
  - Skill_Score      : vs zero-predictor (>0 = beats trivial baseline)
  - Hs_error         : significant wave height relative error
  - Directional_R2   : 2D spectral match

Usage:
    cd wavesplat && python3 -u scripts/eval_comprehensive_comparison.py
"""

import os, sys, json, math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', buffering=1)
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(REPO_ROOT, 'wavesplat'))

from wavesplat.data_loader import EpisodeDataLoader
from wavesplat.model.wave_pino import SpectralConv2d, FNOBlock, spectral_physics_loss
from wavesplat.evaluation.metrics import (
    nrmse, variance_ratio, pearson_correlation, skill_score,
    hs_error, directional_r2
)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

TEST_EPISODES = [
    'episode_BF3_5comp_seed0_slope.h5',
    'episode_BF4_5comp_seed0_uniform.h5',
    'episode_BF4_15comp_seed0_uniform.h5',
    'episode_BF4_30comp_seed0_uniform.h5',
    'episode_BF5_15comp_seed0_uniform.h5',
    'episode_BF5_30comp_seed0_uniform.h5',
]

N_FRAMES  = 3
FRAME_GAP = 5
N_EVAL    = 30   # frames per episode (more than training eval for accuracy)


# ---------------------------------------------------------------------------
# Stage 4 model (copy from pino_stage4_multiframe.py)
# ---------------------------------------------------------------------------
class WavePINOMultiFrame(nn.Module):
    def __init__(self, n_frames=3, grid_nx=64, grid_ny=48,
                 x_range=(10.0, 250.0), y_range=(-100.0, 100.0),
                 fno_width=16, fno_layers=4, fno_modes_h=16, fno_modes_w=12,
                 init_lengthscale=10.0):
        super().__init__()
        self.n_frames = n_frames
        self.grid_nx  = grid_nx
        self.grid_ny  = grid_ny
        self.x_range  = x_range
        self.y_range  = y_range

        gx = torch.linspace(x_range[0], x_range[1], grid_nx)
        gy = torch.linspace(y_range[0], y_range[1], grid_ny)
        xx, yy = torch.meshgrid(gx, gy, indexing='ij')
        self.register_buffer('grid_xy',
                             torch.stack([xx.flatten(), yy.flatten()], dim=-1))
        self.log_lengthscale = nn.Parameter(torch.tensor(math.log(init_lengthscale)))

        self.lifting = nn.Sequential(
            nn.Conv2d(2 * n_frames, fno_width, 1), nn.GELU())
        self.fno_blocks = nn.ModuleList([
            FNOBlock(fno_width, fno_modes_h, fno_modes_w) for _ in range(fno_layers)])
        self.output_head = nn.Sequential(
            nn.Conv2d(fno_width, fno_width * 2, 3, padding=1), nn.GELU(),
            nn.Conv2d(fno_width * 2, 1, 1))

    def _setconv(self, obs_xy, obs_z):
        ls = self.log_lengthscale.exp()
        diff    = self.grid_xy.unsqueeze(0).unsqueeze(2) - obs_xy.unsqueeze(1)
        sq_dist = (diff ** 2).sum(-1)
        weights = torch.exp(-0.5 * sq_dist / (ls ** 2))
        density = weights.sum(-1, keepdim=True)
        signal  = (weights.unsqueeze(-1) * obs_z.unsqueeze(1)).sum(2) / (density + 1e-8)
        features = torch.cat([signal, density], dim=-1)
        B = obs_xy.shape[0]
        return features.permute(0, 2, 1).reshape(B, 2, self.grid_nx, self.grid_ny)

    def forward(self, xc_list, yc_list, xt):
        frame_feats = [self._setconv(xc, yc) for xc, yc in zip(xc_list, yc_list)]
        z = self.lifting(torch.cat(frame_feats, dim=1))
        for block in self.fno_blocks:
            z = block(z)
        grid = self.output_head(z)
        return self._interp(grid, xt)

    def _interp(self, grid_field, xt):
        x_norm = 2.0 * (xt[..., 0] - self.x_range[0]) / (self.x_range[1] - self.x_range[0]) - 1.0
        y_norm = 2.0 * (xt[..., 1] - self.y_range[0]) / (self.y_range[1] - self.y_range[0]) - 1.0
        sg = torch.stack([y_norm, x_norm], dim=-1).unsqueeze(2)
        return F.grid_sample(grid_field, sg, mode='bilinear',
                             align_corners=True, padding_mode='border'
                             ).squeeze(1).squeeze(-1).unsqueeze(-1)


def eval_episode_pino(model, ep, xt, gx, gy, n_eval=30):
    """Evaluate WavePINO on one episode, return per-metric averages."""
    model.eval()
    Hs = max(ep.Hs_true, 0.1)
    min_fi = (N_FRAMES - 1) * FRAME_GAP
    fis = np.linspace(min_fi, len(ep) - 1, n_eval, dtype=int)

    metrics_accum = {k: [] for k in
                     ['nRMSE', 'Variance_Ratio', 'Pearson_rho',
                      'Skill_Score', 'Hs_error', 'Directional_R2']}

    for fi in fis:
        xc_list, yc_list = [], []
        for k in range(N_FRAMES):
            fj = fi - (N_FRAMES - 1 - k) * FRAME_GAP
            frame = ep[fj]
            xc = torch.stack([frame.x_obs, frame.y_obs], dim=-1).unsqueeze(0).to(DEVICE)
            yc = frame.z_obs.unsqueeze(-1).unsqueeze(0).to(DEVICE) / Hs
            xc_list.append(xc)
            yc_list.append(yc)

        with torch.no_grad():
            pred = model(xc_list, yc_list, xt)

        pred_field = (pred[0, :, 0] * Hs).cpu().reshape(len(gx), len(gy)).numpy().T
        gt_field   = ep[fi].gt_field.numpy()

        metrics_accum['nRMSE'].append(nrmse(pred_field, gt_field, ep.Hs_true))
        metrics_accum['Variance_Ratio'].append(variance_ratio(pred_field, gt_field))
        metrics_accum['Pearson_rho'].append(pearson_correlation(pred_field, gt_field))
        metrics_accum['Skill_Score'].append(skill_score(pred_field, gt_field))
        metrics_accum['Hs_error'].append(hs_error(pred_field, ep.Hs_true))
        metrics_accum['Directional_R2'].append(directional_r2(pred_field, gt_field))

    return {k: float(np.nanmean(v)) for k, v in metrics_accum.items()}


def main():
    data_dir  = os.path.join(REPO_ROOT, 'wavesplat', 'data')
    ckpt_path = os.path.join(REPO_ROOT, 'wavesplat', 'checkpoints', 'wavepino_mf_T3_g5.pt')
    ws_path   = os.path.join(REPO_ROOT, 'wavesplat', 'results', 'b1_full_comparison.json')

    print("=" * 70)
    print("Comprehensive Comparison: WavePINO Stage 4 vs WaveSplat")
    print("Metrics: nRMSE | Var.Ratio | Pearson | Skill | Hs_err | Dir.R2")
    print("=" * 70)

    # Load WaveSplat results
    with open(ws_path) as f:
        ws_data = json.load(f)['results']

    # Load episodes
    test_eps, test_names = [], []
    for name in TEST_EPISODES:
        path = os.path.join(data_dir, name)
        if os.path.exists(path):
            test_eps.append(EpisodeDataLoader(path))
            test_names.append(name.replace('episode_', '').replace('.h5', ''))
        else:
            print(f"  MISSING: {name}")

    if not test_eps:
        print("No episodes found."); return

    # Set up grid from first episode
    ep0 = test_eps[0]
    x_range = (float(ep0.gt_grid_x[0]), float(ep0.gt_grid_x[-1]))
    y_range = (float(ep0.gt_grid_y[0]), float(ep0.gt_grid_y[-1]))
    gx = ep0.gt_grid_x.cpu()
    gy = ep0.gt_grid_y.cpu()
    xx, yy = torch.meshgrid(gx, gy, indexing='ij')
    xt = torch.stack([xx.flatten(), yy.flatten()], dim=-1).unsqueeze(0).to(DEVICE)

    # Load WavePINO model
    model = WavePINOMultiFrame(
        n_frames=N_FRAMES, grid_nx=64, grid_ny=48,
        x_range=x_range, y_range=y_range,
        fno_width=16, fno_layers=4, fno_modes_h=16, fno_modes_w=12,
    ).to(DEVICE)

    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    model.load_state_dict(ckpt)
    model.eval()
    nparams = sum(p.numel() for p in model.parameters())
    print(f"  WavePINO Stage 4 loaded: {nparams:,} params | ckpt: {os.path.basename(ckpt_path)}")
    print(f"  Evaluating {N_EVAL} frames/episode on {len(test_eps)} test episodes\n")

    # WaveSplat key map
    ws_key_map = {
        'BF3_5comp_seed0_slope':    'BF3_5comp_seed0_slope',
        'BF4_5comp_seed0_uniform':  'BF4_5comp_seed0_uniform',
        'BF4_15comp_seed0_uniform': 'BF4_15comp_seed0_uniform',
        'BF4_30comp_seed0_uniform': 'BF4_30comp_seed0_uniform',
        'BF5_15comp_seed0_uniform': 'BF5_15comp_seed0_uniform',
        'BF5_30comp_seed0_uniform': 'BF5_30comp_seed0_uniform',
    }

    metrics_keys = ['nRMSE', 'Variance_Ratio', 'Pearson_rho',
                    'Skill_Score', 'Hs_error', 'Directional_R2']
    ws_field_map = {
        'nRMSE':         'avg_nRMSE',
        'Variance_Ratio':'avg_Variance_Ratio',
        'Pearson_rho':   'avg_Pearson_rho',
        'Skill_Score':   'avg_Skill_Score',
        'Hs_error':      'avg_Hs_error',
        'Directional_R2':'avg_Directional_R2',
    }

    results = {}
    pino_agg = {k: [] for k in metrics_keys}
    ws_agg   = {k: [] for k in metrics_keys}

    for ep, name in zip(test_eps, test_names):
        pino_m = eval_episode_pino(model, ep, xt, gx, gy, n_eval=N_EVAL)
        results[name] = {'WavePINO': pino_m}

        # WaveSplat from stored results
        ws_ep = ws_data.get(name, {}).get('WaveSplat', {})
        ws_m = {}
        for mk, fk in ws_field_map.items():
            ws_m[mk] = ws_ep.get(fk, float('nan'))
        results[name]['WaveSplat'] = ws_m

        # Accumulate
        for k in metrics_keys:
            pino_agg[k].append(pino_m[k])
            ws_agg[k].append(ws_m[k])

        # Print episode results
        print(f"  [{name}]")
        header = f"    {'Metric':<18} {'WaveSplat':>12} {'WavePINO':>12} {'Delta':>10}"
        print(header)
        for k in metrics_keys:
            w = ws_m[k]
            p = pino_m[k]
            if not (np.isnan(w) or np.isnan(p)):
                if k in ('nRMSE', 'Hs_error'):
                    d = (p - w) / abs(w) * 100 if abs(w) > 1e-9 else 0
                    sign = '+' if d > 0 else ''
                    better = '(WS)' if d > 0 else '(PI)'
                    print(f"    {k:<18} {w:>12.4f} {p:>12.4f} {sign}{d:>6.1f}% {better}")
                else:
                    d = p - w
                    sign = '+' if d > 0 else ''
                    better = '(PI)' if d > 0 else '(WS)'
                    print(f"    {k:<18} {w:>12.4f} {p:>12.4f} {sign}{d:>+7.4f} {better}")
            else:
                print(f"    {k:<18} {w:>12} {p:>12}")
        print()

    # Summary
    print("=" * 70)
    print("  SUMMARY (averaged over 6 test episodes)")
    print(f"  {'Metric':<20} {'WaveSplat':>12} {'WavePINO':>12} {'Winner':>10}")
    print("-" * 60)
    for k in metrics_keys:
        w_avg = float(np.nanmean(ws_agg[k]))
        p_avg = float(np.nanmean(pino_agg[k]))
        if k in ('nRMSE', 'Hs_error'):
            winner = 'WaveSplat' if w_avg < p_avg else 'WavePINO'
        else:
            winner = 'WavePINO' if p_avg > w_avg else 'WaveSplat'
        print(f"  {k:<20} {w_avg:>12.4f} {p_avg:>12.4f} {winner:>10}")

    # Breakdown by sea complexity
    print("\n  BY SEA COMPLEXITY:")
    simple_eps = ['BF3_5comp_seed0_slope', 'BF4_5comp_seed0_uniform']
    complex_eps = ['BF4_15comp_seed0_uniform', 'BF4_30comp_seed0_uniform',
                   'BF5_15comp_seed0_uniform', 'BF5_30comp_seed0_uniform']
    for label, eps_list in [('Simple sea (1-5 comp)', simple_eps),
                             ('Complex sea (15-30 comp)', complex_eps)]:
        ws_nrmse = [ws_agg['nRMSE'][i] for i, n in enumerate(test_names) if n in eps_list]
        pi_nrmse = [pino_agg['nRMSE'][i] for i, n in enumerate(test_names) if n in eps_list]
        ws_vr    = [ws_agg['Variance_Ratio'][i] for i, n in enumerate(test_names) if n in eps_list]
        pi_vr    = [pino_agg['Variance_Ratio'][i] for i, n in enumerate(test_names) if n in eps_list]
        ws_rho   = [ws_agg['Pearson_rho'][i] for i, n in enumerate(test_names) if n in eps_list]
        pi_rho   = [pino_agg['Pearson_rho'][i] for i, n in enumerate(test_names) if n in eps_list]
        if ws_nrmse:
            delta_pct = (np.mean(pi_nrmse) - np.mean(ws_nrmse)) / np.mean(ws_nrmse) * 100
            print(f"    {label}:")
            print(f"      nRMSE : WS={np.mean(ws_nrmse):.4f}  PI={np.mean(pi_nrmse):.4f}  delta={delta_pct:+.1f}%")
            print(f"      VR    : WS={np.mean(ws_vr):.3f}   PI={np.mean(pi_vr):.3f}")
            print(f"      Pearson: WS={np.mean(ws_rho):.3f}   PI={np.mean(pi_rho):.3f}")

    # Save
    out = {
        'description': 'Comprehensive WavePINO-S4 vs WaveSplat comparison',
        'n_eval_frames': N_EVAL,
        'pino_checkpoint': os.path.basename(ckpt_path),
        'per_episode': results,
        'averages': {
            'WaveSplat': {k: float(np.nanmean(ws_agg[k])) for k in metrics_keys},
            'WavePINO':  {k: float(np.nanmean(pino_agg[k])) for k in metrics_keys},
        }
    }
    out_path = os.path.join(REPO_ROOT, 'wavesplat', 'results', 'pino_vs_wavesplat_comprehensive.json')
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\n  Saved: {out_path}")


if __name__ == '__main__':
    main()
