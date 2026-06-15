#!/usr/bin/env python3
"""
train_wave_cfm.py

Direction 3: Conditional Flow Matching (CFM) for ocean wave estimation.

CFM (Lipman et al. 2022) learns a vector field that transforms noise into data
via a straight ODE path. Key difference from DDPM:
  - ODE is deterministic → no amplitude hedging from stochasticity
  - 10-20 Euler steps (vs 50 DDIM) → faster inference
  - Physics guidance via dispersion projection at each ODE step

Training (Conditional Flow Matching):
  x_t = (1 - t) * noise + t * x0       (linear interpolation)
  v_target = x0 - noise                  (constant velocity field)
  loss = ||model(x_t, t, cond) - v_target||²

Inference (Euler ODE, 20 steps):
  x_0 ~ N(0, I)
  for i in 0..N-1:
    t = i/N, dt = 1/N
    v = model(x_t, t, cond)
    x_t+dt = x_t + dt * v
    [optional: dispersion projection on predicted x1 estimate]

Distinction from PerFlow (arXiv:2605.03548):
  - PerFlow: physics-embedded rectified flow for forward PDE solving (gridded)
  - Ours: inverse problem from sparse LiDAR (SetConv → FNO/UNet), ocean wave
    dispersion guidance during ODE integration

Architecture: same DiffusionUNet as WaveDiffusion v5b (base_ch=48, attn=True)

Usage:
    cd wavesplat && python3 -u scripts/train_wave_cfm.py
"""

import sys, os, time, json, math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', buffering=1)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from wavesplat.data_loader import EpisodeDataLoader
from wavesplat.model.wave_diffusion import (
    WaveDiffusion, DiffusionUNet, SetConv, _make_dispersion_mask, _apply_dispersion_filter
)
from wavesplat.evaluation.metrics import nrmse

# --- Episode splits (same as v5b / v8) ---
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
GRID_NX = 80
GRID_NY  = 60
BASE_CH  = 48
EPOCHS   = 300
BATCH_SIZE = 8
LR = 2e-4
WARMUP_EPOCHS = 10
STEPS_PER_EPOCH = 100
COND_DROP_PROB = 0.1

ODE_STEPS = 20          # Euler steps at inference
CFG_SCALE = 3.0         # classifier-free guidance scale

DISP_K_MIN = 0.02
DISP_K_MAX = 0.5
DISP_STRENGTH = 0.3     # dispersion projection strength during ODE


# ---------------------------------------------------------------------------
# WaveCFM model
# ---------------------------------------------------------------------------

class WaveCFM(nn.Module):
    """Conditional Flow Matching model for ocean wave estimation.

    Identical conditioning setup to WaveDiffusion:
      SetConv(sparse LiDAR) → [B,2,H,W] condition grid
      DiffusionUNet predicts vector field v(x_t, t, cond)

    The t input uses same sinusoidal embedding as DDPM timestep,
    but now t ∈ [0,1] (continuous) instead of integer index.
    We map t → integer-like via t * 999 to reuse existing time_mlp.
    """

    def __init__(self, grid_nx=80, grid_ny=60,
                 x_range=(10.0, 250.0), y_range=(-100.0, 100.0),
                 base_ch=48, init_lengthscale=10.0):
        super().__init__()
        self.grid_nx = grid_nx
        self.grid_ny = grid_ny
        self.x_range = x_range
        self.y_range = y_range

        gx = torch.linspace(x_range[0], x_range[1], grid_nx)
        gy = torch.linspace(y_range[0], y_range[1], grid_ny)
        xx, yy = torch.meshgrid(gx, gy, indexing='ij')
        self.register_buffer('grid_xy',
                             torch.stack([xx.flatten(), yy.flatten()], dim=-1))

        self.setconv = SetConv(init_lengthscale=init_lengthscale)
        # in_ch=3: 1 (noisy field) + 2 (cond)
        self.unet = DiffusionUNet(in_ch=3, base_ch=base_ch, time_dim=128, use_attn=True)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, nonlinearity='linear')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def get_condition(self, obs_xy, obs_z):
        B = obs_xy.shape[0]
        grid_feat = self.setconv(obs_xy, obs_z, self.grid_xy)
        return grid_feat.permute(0, 2, 1).reshape(B, 2, self.grid_nx, self.grid_ny)

    def forward(self, x_t, t_cont, cond):
        """Predict velocity field v(x_t, t, cond).

        Args:
            x_t:    [B, 1, H, W] interpolated noisy field
            t_cont: [B] continuous time in [0, 1]
            cond:   [B, 2, H, W] SetConv condition

        Returns:
            [B, 1, H, W] predicted velocity (toward data)
        """
        # Map t ∈ [0,1] → integer in [0,999] for time_mlp reuse
        t_int = (t_cont * 999).long().clamp(0, 999)
        return self.unet(x_t, t_int, cond)

    @property
    def n_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# CFM training utilities
# ---------------------------------------------------------------------------

def cfm_loss(model, x0, cond):
    """Compute CFM loss for a batch.

    Samples t ~ U[0,1], constructs x_t = (1-t)*noise + t*x0,
    trains model to predict v = x0 - noise.
    """
    B = x0.shape[0]
    t = torch.rand(B, device=x0.device)  # [B] ∈ [0,1]

    noise = torch.randn_like(x0)
    t_4d = t[:, None, None, None]
    x_t = (1 - t_4d) * noise + t_4d * x0  # linear interpolation
    v_target = x0 - noise                   # constant velocity field

    v_pred = model(x_t, t, cond)
    return F.mse_loss(v_pred, v_target)


@torch.no_grad()
def cfm_sample(model, cond, shape, n_steps=20, cfg_scale=1.0,
               disp_mask=None, disp_strength=0.0):
    """Euler ODE integration for CFM sampling with optional physics guidance.

    Integrates dx/dt = v(x_t, t, cond) from t=0 to t=1.
    Optionally applies dispersion projection on the estimated x0 at each step.
    """
    device = cond.device
    B = shape[0]
    uncond = torch.zeros_like(cond) if cfg_scale > 1.0 else None

    x = torch.randn(shape, device=device)
    dt = 1.0 / n_steps

    for i in range(n_steps):
        t_val = i / n_steps
        t = torch.full((B,), t_val, device=device)

        if cfg_scale > 1.0 and uncond is not None:
            v_cond   = model(x, t, cond)
            v_uncond = model(x, t, uncond)
            v = v_uncond + cfg_scale * (v_cond - v_uncond)
        else:
            v = model(x, t, cond)

        # Estimate x0 from current x_t and velocity
        # x_t = (1-t)*noise + t*x0  →  x0 ≈ (x_t - (1-t)*noise) / t
        # Simpler: x0_est = x + (1-t_val) * v  (from straight-line geometry)
        if disp_mask is not None and disp_strength > 0 and t_val > 0:
            t_rem = 1.0 - t_val
            x0_est = x + t_rem * v
            x0_est = _apply_dispersion_filter(x0_est, disp_mask, disp_strength)
            # Recompute velocity from corrected x0_est
            if t_rem > 1e-6:
                noise_est = (x - t_val * x0_est) / max(1 - t_val, 1e-6)
                v = x0_est - noise_est

        x = x + dt * v

    # Optional: final dispersion cleanup
    if disp_mask is not None and disp_strength > 0:
        x = _apply_dispersion_filter(x, disp_mask, disp_strength)

    return x


# ---------------------------------------------------------------------------
# Data utilities
# ---------------------------------------------------------------------------

def discover_train_episodes(data_dir):
    import glob
    all_eps = glob.glob(os.path.join(data_dir, 'episode_*.h5'))
    return [os.path.basename(p) for p in sorted(all_eps)
            if os.path.basename(p) not in EXCLUDE]


def load_episodes(names, data_dir):
    return [EpisodeDataLoader(os.path.join(data_dir, n))
            for n in names if os.path.exists(os.path.join(data_dir, n))]


def sample_batch(episodes, batch_size):
    xc_list, yc_list, gt_list = [], [], []
    for _ in range(batch_size):
        ep = episodes[np.random.randint(len(episodes))]
        frame = ep[np.random.randint(len(ep))]
        Hs = max(ep.Hs_true, 0.1)
        xc = torch.stack([frame.x_obs, frame.y_obs], dim=-1)
        yc = frame.z_obs.unsqueeze(-1) / Hs
        gt = frame.gt_field / Hs
        if np.random.random() < 0.5:
            xc = xc.clone(); xc[:, 1] = -xc[:, 1]; gt = gt.flip(0)
        xc_list.append(xc)
        yc_list.append(yc)
        gt_list.append(gt)
    max_nc = max(x.shape[0] for x in xc_list)
    xc_b = torch.zeros(batch_size, max_nc, 2)
    yc_b = torch.zeros(batch_size, max_nc, 1)
    for i in range(batch_size):
        nc = xc_list[i].shape[0]
        xc_b[i, :nc] = xc_list[i]
        yc_b[i, :nc] = yc_list[i]
    gt_b = torch.stack(gt_list).permute(0, 2, 1).unsqueeze(1)
    return xc_b, yc_b, gt_b


def evaluate(model, episodes, n_frames=5, n_steps=ODE_STEPS,
             cfg_scale=1.0, disp_mask=None, disp_strength=0.0, n_samples=1):
    model.eval()
    vals = []
    for ep in episodes:
        Hs = max(ep.Hs_true, 0.1)
        fis = np.linspace(10, len(ep) - 1, n_frames, dtype=int)
        for fi in fis:
            frame = ep[fi]
            xc = torch.stack([frame.x_obs, frame.y_obs], dim=-1).unsqueeze(0)
            yc = frame.z_obs.unsqueeze(-1).unsqueeze(0) / Hs
            with torch.no_grad():
                cond = model.get_condition(xc, yc)
            shape = (1, 1, model.grid_nx, model.grid_ny)
            preds = []
            for _ in range(n_samples):
                with torch.no_grad():
                    s = cfm_sample(model, cond, shape, n_steps=n_steps,
                                   cfg_scale=cfg_scale, disp_mask=disp_mask,
                                   disp_strength=disp_strength)
                preds.append(s)
            pred = torch.stack(preds).mean(0)
            pred_field = (pred[0, 0].permute(1, 0) * Hs).numpy()
            v = nrmse(pred_field, frame.gt_field.numpy(), ep.Hs_true)
            if not np.isnan(v):
                vals.append(v)
    model.train()
    return float(np.mean(vals)) if vals else 1.0


def test_eval_full(model, test_eps, disp_mask):
    model.eval()
    configs = [
        (1.0, 1, 0.0, "baseline gs=1 1s"),
        (1.0, 5, 0.0, "baseline gs=1 5s"),
        (3.0, 1, 0.0, "baseline gs=3 1s"),
        (3.0, 1, DISP_STRENGTH, "disp gs=3 1s"),
        (3.0, 5, DISP_STRENGTH, "disp gs=3 5s"),
    ]
    results = {}
    for cfg_s, n_samp, disp_s, label in configs:
        print("\n  --- %s ---" % label)
        ep_results, amp_ratios = [], []
        for ep_idx, ep in enumerate(test_eps):
            Hs = max(ep.Hs_true, 0.1)
            fis = np.linspace(10, len(ep) - 1, 20, dtype=int)
            ep_vals, ep_amp = [], []
            for fi in fis:
                frame = ep[fi]
                xc = torch.stack([frame.x_obs, frame.y_obs], dim=-1).unsqueeze(0)
                yc = frame.z_obs.unsqueeze(-1).unsqueeze(0) / Hs
                with torch.no_grad():
                    cond = model.get_condition(xc, yc)
                shape = (1, 1, model.grid_nx, model.grid_ny)
                preds = []
                for _ in range(n_samp):
                    with torch.no_grad():
                        s = cfm_sample(model, cond, shape, n_steps=ODE_STEPS,
                                       cfg_scale=cfg_s, disp_mask=disp_mask,
                                       disp_strength=disp_s)
                    preds.append(s)
                pred = torch.stack(preds).mean(0)
                pred_field = (pred[0, 0].permute(1, 0) * Hs).numpy()
                gt_field = frame.gt_field.numpy()
                v = nrmse(pred_field, gt_field, ep.Hs_true)
                if not np.isnan(v):
                    ep_vals.append(v)
                ps, gs_ = float(np.std(pred_field)), float(np.std(gt_field))
                if gs_ > 0.01:
                    ep_amp.append(ps / gs_)
            avg = float(np.mean(ep_vals))
            amp = float(np.mean(ep_amp)) if ep_amp else 0.0
            ep_name = TEST_EPISODES[ep_idx].replace('episode_', '').replace('.h5', '')
            ep_results.append(avg)
            amp_ratios.append(amp)
            print("    %s: %.4f amp=%.2f" % (ep_name, avg, amp))
        overall = float(np.mean(ep_results))
        avg_amp = float(np.mean(amp_ratios)) if amp_ratios else 0.0
        print("    Avg: nRMSE=%.4f amp=%.2f" % (overall, avg_amp))
        results[label] = {'nrmse': overall, 'amp': avg_amp,
                          'per_episode': dict(zip(
                              [n.replace('episode_', '').replace('.h5', '')
                               for n in TEST_EPISODES], ep_results))}
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')

    print("=" * 70)
    print("WaveCFM: Conditional Flow Matching for ocean wave estimation")
    print("Distinction from PerFlow: inverse problem (sparse LiDAR), ocean physics")
    print("=" * 70)

    train_names = discover_train_episodes(data_dir)
    train_eps = load_episodes(train_names, data_dir)
    val_eps   = load_episodes(VAL_EPISODES, data_dir)
    test_eps  = load_episodes(TEST_EPISODES, data_dir)
    print("  Train %d | Val %d | Test %d" % (len(train_eps), len(val_eps), len(test_eps)))

    ep0 = train_eps[0]
    x_range = (float(ep0.gt_grid_x[0]), float(ep0.gt_grid_x[-1]))
    y_range = (float(ep0.gt_grid_y[0]), float(ep0.gt_grid_y[-1]))

    model = WaveCFM(grid_nx=GRID_NX, grid_ny=GRID_NY,
                    x_range=x_range, y_range=y_range,
                    base_ch=BASE_CH, init_lengthscale=10.0)
    print("  Params: %s | base_ch=%d | ODE steps=%d | CFG=%.1f" % (
        format(model.n_params, ','), BASE_CH, ODE_STEPS, CFG_SCALE))

    disp_mask = _make_dispersion_mask(GRID_NX, GRID_NY, DISP_K_MIN, DISP_K_MAX, 'cpu')

    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    warmup = optim.lr_scheduler.LinearLR(optimizer, start_factor=0.01, total_iters=WARMUP_EPOCHS)
    cosine = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS - WARMUP_EPOCHS)
    scheduler = optim.lr_scheduler.SequentialLR(optimizer, [warmup, cosine], milestones=[WARMUP_EPOCHS])

    ckpt_dir = os.path.join(data_dir, '..', 'checkpoints')
    os.makedirs(ckpt_dir, exist_ok=True)

    best_val, best_epoch = 1.0, 0
    t0 = time.time()

    print("  Training %d epochs..." % EPOCHS)
    for epoch in range(1, EPOCHS + 1):
        model.train()
        epoch_loss = 0.0

        for _ in range(STEPS_PER_EPOCH):
            xc, yc, gt = sample_batch(train_eps, BATCH_SIZE)
            cond = model.get_condition(xc, yc)

            # CFG: randomly drop condition
            if COND_DROP_PROB > 0:
                drop_mask = torch.rand(BATCH_SIZE) < COND_DROP_PROB
                if drop_mask.any():
                    cond[drop_mask] = 0.0

            loss = cfm_loss(model, gt, cond)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()

        scheduler.step()
        avg_loss = epoch_loss / STEPS_PER_EPOCH

        val_interval = 10 if epoch <= 50 else 20
        if epoch % val_interval == 0 or epoch == 1:
            val_nrmse = evaluate(model, val_eps, n_frames=5, n_steps=ODE_STEPS,
                                  cfg_scale=CFG_SCALE, disp_mask=disp_mask,
                                  disp_strength=DISP_STRENGTH, n_samples=1)
            improved = ""
            if val_nrmse < best_val:
                best_val, best_epoch = val_nrmse, epoch
                torch.save(model.state_dict(), os.path.join(ckpt_dir, 'wave_cfm_best.pt'))
                improved = " *"
            ls = model.setconv.log_lengthscale.exp().item()
            print("  Ep %3d/%d: loss=%.5f val=%.4f ls=%.1fm [%.0fs]%s" % (
                epoch, EPOCHS, avg_loss, val_nrmse, ls, time.time() - t0, improved))
        elif epoch % 10 == 0:
            print("  Ep %3d/%d: loss=%.5f [%.0fs]" % (
                epoch, EPOCHS, avg_loss, time.time() - t0))

    print("  Best val: %.4f at epoch %d" % (best_val, best_epoch))

    # Test evaluation
    ckpt_path = os.path.join(ckpt_dir, 'wave_cfm_best.pt')
    if os.path.exists(ckpt_path):
        model.load_state_dict(torch.load(ckpt_path, map_location='cpu', weights_only=True))

    print("\n" + "=" * 70)
    print("Test Evaluation — WaveCFM")
    print("=" * 70)
    print("Reference: v5b baseline 5s=0.2632, WaveSplat=0.2230")

    all_results = test_eval_full(model, test_eps, disp_mask)

    print("\n  SUMMARY")
    for label, r in all_results.items():
        print("  %-28s: %.4f amp=%.2f" % (label, r['nrmse'], r['amp']))
    best_label = min(all_results, key=lambda k: all_results[k]['nrmse'])
    print("  Best: %s → %.4f" % (best_label, all_results[best_label]['nrmse']))
    print("  v5b best:   0.2632  |  WaveSplat: 0.2230")

    total_time = time.time() - t0
    out = {
        'version': 'wave_cfm_v1',
        'best_val_nRMSE': best_val, 'best_epoch': best_epoch,
        'n_params': model.n_params, 'training_time_s': total_time,
        'config': {'grid_nx': GRID_NX, 'grid_ny': GRID_NY, 'base_ch': BASE_CH,
                   'ode_steps': ODE_STEPS, 'cfg_scale': CFG_SCALE,
                   'cond_drop_prob': COND_DROP_PROB, 'epochs': EPOCHS,
                   'disp_k_min': DISP_K_MIN, 'disp_k_max': DISP_K_MAX,
                   'disp_strength': DISP_STRENGTH},
        'references': {'v5b_baseline_5s': 0.2632, 'WaveSplat': 0.2230,
                       'WavePINO_MF_T3': 0.2201},
        'test_results': all_results,
    }
    results_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results')
    os.makedirs(results_dir, exist_ok=True)
    out_path = os.path.join(results_dir, 'wave_cfm_eval.json')
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2)
    print("  Saved: %s" % out_path)


if __name__ == '__main__':
    main()
