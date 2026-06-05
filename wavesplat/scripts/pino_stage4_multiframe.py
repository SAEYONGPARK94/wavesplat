#!/usr/bin/env python3
"""
pino_stage4_multiframe.py

WavePINO Stage 4: Multi-frame temporal stacking.

Key idea:
  Ship-mounted LiDAR scans different spatial positions each frame as the ship
  moves.  Stacking T consecutive frames gives DENSER spatial coverage of the
  wave field, while also providing temporal phase information.  An FNO that
  sees T frames simultaneously can learn to disambiguate wave components that
  appear aliased in a single snapshot.

Architecture (WavePINO-MF):
  T × (sparse LiDAR frame) → shared SetConv → concat [B, 2T, Gx, Gy]
                                             → Lift(2T→width) → FNO → output

Comparison:
  Stage 2 PINO (T=1): 0.2220
  Stage 4 PINO-MF  (T=3, gap=5 → Δt=0.5s/step, window=1.0s): ?

Physics loss: spectral band penalty (same λ_spec=0.5 as Stage 2).

Usage:
    cd wavesplat && python3 -u scripts/pino_stage4_multiframe.py
"""

import os
import sys
import time
import json
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', buffering=1)

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(REPO_ROOT, 'wavesplat'))

from wavesplat.data_loader import EpisodeDataLoader
from wavesplat.model.wave_pino import (
    SpectralConv2d, FNOBlock, spectral_physics_loss
)
from wavesplat.evaluation.metrics import nrmse

# ---------------------------------------------------------------------------
# Episode lists (identical to Stage 2 / ConvCNP v2)
# ---------------------------------------------------------------------------
TRAIN_EPISODES = [
    'episode_BF3_1comp_seed0_uniform.h5', 'episode_BF3_1comp_seed1_uniform.h5',
    'episode_BF3_2comp_seed0_uniform.h5', 'episode_BF3_2comp_seed1_uniform.h5',
    'episode_BF3_5comp_seed0_uniform.h5', 'episode_BF3_5comp_seed1_uniform.h5',
    'episode_BF3_1comp_seed0_slope.h5',  'episode_BF3_1comp_seed1_slope.h5',
    'episode_BF3_2comp_seed0_slope.h5',  'episode_BF3_2comp_seed1_slope.h5',
    'episode_BF3_5comp_seed0_slope.h5',  'episode_BF3_5comp_seed1_slope.h5',
    'episode_BF4_1comp_seed0_uniform.h5','episode_BF4_1comp_seed1_uniform.h5',
    'episode_BF4_5comp_seed0_uniform.h5','episode_BF4_5comp_seed1_uniform.h5',
    'episode_BF4_15comp_seed0_uniform.h5','episode_BF4_15comp_seed1_uniform.h5',
    'episode_BF4_30comp_seed0_uniform.h5',
    'episode_BF4_1comp_seed0_slope.h5',  'episode_BF4_1comp_seed1_slope.h5',
    'episode_BF4_5comp_seed0_slope.h5',  'episode_BF4_5comp_seed1_slope.h5',
    'episode_BF4_15comp_seed0_slope.h5', 'episode_BF4_15comp_seed1_slope.h5',
    'episode_BF5_15comp_seed0_uniform.h5','episode_BF5_15comp_seed1_uniform.h5',
    'episode_BF5_30comp_seed0_uniform.h5','episode_BF5_30comp_seed1_uniform.h5',
    'episode_BF5_15comp_seed0_slope.h5', 'episode_BF5_15comp_seed1_slope.h5',
    'episode_BF5_30comp_seed0_slope.h5', 'episode_BF5_30comp_seed1_slope.h5',
]

VAL_EPISODES = [
    'episode_BF3_5comp_seed2_uniform.h5',
    'episode_BF4_5comp_seed2_uniform.h5',
    'episode_BF4_15comp_seed2_uniform.h5',
    'episode_BF5_15comp_seed2_uniform.h5',
]

TEST_EPISODES = [
    'episode_BF3_5comp_seed0_slope.h5',
    'episode_BF4_5comp_seed0_uniform.h5',
    'episode_BF4_15comp_seed0_uniform.h5',
    'episode_BF4_30comp_seed0_uniform.h5',
    'episode_BF5_15comp_seed0_uniform.h5',
    'episode_BF5_30comp_seed0_uniform.h5',
]

# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------
GRID_NX     = 64
GRID_NY     = 48
FNO_WIDTH   = 16
FNO_LAYERS  = 4
FNO_MODES_H = 16
FNO_MODES_W = 12

N_FRAMES    = 3        # number of temporal frames stacked
FRAME_GAP   = 5        # frames between consecutive inputs (Δt = 0.5s each step)
# Total temporal window: (N_FRAMES - 1) * FRAME_GAP * 0.1s = 1.0s

EPOCHS          = 100
STEPS_PER_EPOCH = 100
BATCH_SIZE      = 8
LR              = 1e-3
WARMUP_EPOCHS   = 5
LAMBDA_CORR     = 1.0
LAMBDA_SPEC     = 0.5

DEPTH = 50.0
T_MIN = 4.0
T_MAX = 25.0

G = 9.81

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ---------------------------------------------------------------------------
# Multi-frame WavePINO
# ---------------------------------------------------------------------------

class WavePINOMultiFrame(nn.Module):
    """Multi-frame PINO: T temporal LiDAR frames → shared SetConv → FNO.

    Each frame is aggregated onto the same spatial grid via shared SetConv
    weights.  The T feature maps are concatenated channel-wise before lifting,
    so the FNO sees (2T) channels containing:
      - per-frame signal (weighted average elevation)
      - per-frame density (observation coverage mask)

    This allows the model to learn: 'denser coverage from T positions gives
    more confident reconstruction' and 'phase shift between frames encodes
    propagation direction and speed'.
    """

    def __init__(
        self,
        n_frames: int = 3,
        grid_nx: int = 64,
        grid_ny: int = 48,
        x_range=(10.0, 250.0),
        y_range=(-100.0, 100.0),
        fno_width: int = 16,
        fno_layers: int = 4,
        fno_modes_h: int = 16,
        fno_modes_w: int = 12,
        init_lengthscale: float = 10.0,
    ):
        super().__init__()
        self.n_frames = n_frames
        self.grid_nx  = grid_nx
        self.grid_ny  = grid_ny
        self.x_range  = x_range
        self.y_range  = y_range
        self.fno_width = fno_width

        # SetConv grid coordinates
        gx = torch.linspace(x_range[0], x_range[1], grid_nx)
        gy = torch.linspace(y_range[0], y_range[1], grid_ny)
        xx, yy = torch.meshgrid(gx, gy, indexing='ij')
        self.register_buffer('grid_xy',
                             torch.stack([xx.flatten(), yy.flatten()], dim=-1))

        # Shared SetConv lengthscale (one parameter for all frames)
        self.log_lengthscale = nn.Parameter(
            torch.tensor(math.log(init_lengthscale)))

        # Lifting: 2*n_frames → fno_width  (wider input than single-frame)
        self.lifting = nn.Sequential(
            nn.Conv2d(2 * n_frames, fno_width, 1),
            nn.GELU(),
        )

        self.fno_blocks = nn.ModuleList([
            FNOBlock(fno_width, fno_modes_h, fno_modes_w)
            for _ in range(fno_layers)
        ])

        self.output_head = nn.Sequential(
            nn.Conv2d(fno_width, fno_width * 2, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(fno_width * 2, 1, 1),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='linear')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _setconv(self, obs_xy, obs_z):
        """Scatter one frame's observations to grid. [B,N,2],[B,N,1]->[B,2,Gx,Gy]"""
        ls = self.log_lengthscale.exp()
        diff    = (self.grid_xy.unsqueeze(0).unsqueeze(2)
                   - obs_xy.unsqueeze(1))          # [B, G, N, 2]
        sq_dist = (diff ** 2).sum(-1)              # [B, G, N]
        weights = torch.exp(-0.5 * sq_dist / (ls ** 2))  # [B, G, N]
        density = weights.sum(-1, keepdim=True)    # [B, G, 1]
        signal  = (weights.unsqueeze(-1) * obs_z.unsqueeze(1)).sum(2)
        signal  = signal / (density + 1e-8)
        features = torch.cat([signal, density], dim=-1)  # [B, G, 2]
        B = obs_xy.shape[0]
        return features.permute(0, 2, 1).reshape(B, 2, self.grid_nx, self.grid_ny)

    def _grid_predict(self, xc_list, yc_list):
        """xc_list: list of T [B,N,2]; yc_list: list of T [B,N,1]."""
        frame_feats = [self._setconv(xc, yc)
                       for xc, yc in zip(xc_list, yc_list)]  # T × [B,2,Gx,Gy]
        z = torch.cat(frame_feats, dim=1)  # [B, 2T, Gx, Gy]
        z = self.lifting(z)
        for block in self.fno_blocks:
            z = block(z)
        return self.output_head(z)          # [B, 1, Gx, Gy]

    def forward(self, xc_list, yc_list, xt):
        """xt: [B, Nt, 2] query points for the current (last) frame."""
        grid = self._grid_predict(xc_list, yc_list)
        return self._interp_to_targets(grid, xt)

    def predict_grid(self, xc_list, yc_list):
        return self._grid_predict(xc_list, yc_list)

    def _interp_to_targets(self, grid_field, xt):
        x_norm = (2.0 * (xt[..., 0] - self.x_range[0])
                  / (self.x_range[1] - self.x_range[0]) - 1.0)
        y_norm = (2.0 * (xt[..., 1] - self.y_range[0])
                  / (self.y_range[1] - self.y_range[0]) - 1.0)
        sample_grid = torch.stack([y_norm, x_norm], dim=-1).unsqueeze(2)
        sampled = F.grid_sample(
            grid_field, sample_grid,
            mode='bilinear', align_corners=True, padding_mode='border')
        return sampled.squeeze(1).squeeze(-1).unsqueeze(-1)

    @property
    def n_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Data utilities
# ---------------------------------------------------------------------------

def load_episodes(names, data_dir):
    eps = []
    for n in names:
        p = os.path.join(data_dir, n)
        if os.path.exists(p):
            eps.append(EpisodeDataLoader(p))
    return eps


def correlation_loss(pred, target):
    pc = pred   - pred.mean(dim=1, keepdim=True)
    tc = target - target.mean(dim=1, keepdim=True)
    ps = pc.norm(dim=1, keepdim=True).clamp(min=1e-8)
    ts = tc.norm(dim=1, keepdim=True).clamp(min=1e-8)
    corr = (pc * tc).sum(dim=1) / (ps.squeeze() * ts.squeeze())
    return 1.0 - corr.mean()


def sample_multiframe_batch(
    episodes, batch_size, n_frames, frame_gap,
    model_grid_x, model_grid_y, gt_grid_x, gt_grid_y, device
):
    """Sample a batch of T-frame stacks.

    Returns:
        xc_list: list of n_frames tensors [B, max_N, 2]
        yc_list: list of n_frames tensors [B, max_N, 1]
        gt_b:    [B, Gx, Gy] ground truth for the LAST (current) frame
    """
    min_fi = (n_frames - 1) * frame_gap  # need enough history

    xc_lists  = [[] for _ in range(n_frames)]
    yc_lists  = [[] for _ in range(n_frames)]
    gt_list   = []
    flip_mask = []

    for _ in range(batch_size):
        ep = episodes[np.random.randint(len(episodes))]
        fi = np.random.randint(min_fi, len(ep))
        Hs = max(ep.Hs_true, 0.1)

        do_flip = np.random.random() < 0.5
        flip_mask.append(do_flip)

        # Collect T frames: oldest → newest (current = index fi)
        for t_idx in range(n_frames):
            frame_fi = fi - (n_frames - 1 - t_idx) * frame_gap
            fr = ep[frame_fi]
            xc = torch.stack([fr.x_obs, fr.y_obs], dim=-1)
            yc = fr.z_obs.unsqueeze(-1) / Hs
            if do_flip:
                xc = xc.clone()
                xc[:, 1] = -xc[:, 1]
            xc_lists[t_idx].append(xc)
            yc_lists[t_idx].append(yc)

        # GT: current frame
        current_frame = ep[fi]
        gt = current_frame.gt_field / Hs  # [Ny, Nx]

        dx0, dx1 = float(gt_grid_x[0]), float(gt_grid_x[-1])
        dy0, dy1 = float(gt_grid_y[0]), float(gt_grid_y[-1])
        x_n = 2.0 * (model_grid_x - dx0) / (dx1 - dx0) - 1.0
        y_n = 2.0 * (model_grid_y - dy0) / (dy1 - dy0) - 1.0
        yy, xx = torch.meshgrid(y_n, x_n, indexing='ij')
        sg = torch.stack([xx, yy], dim=-1).unsqueeze(0)
        gt_r = F.grid_sample(
            gt.unsqueeze(0).unsqueeze(0).float(), sg,
            mode='bilinear', align_corners=True,
            padding_mode='border')[0, 0]
        gt_r = gt_r.T  # [Gx, Gy]
        if do_flip:
            gt_r = gt_r.flip(1)
        gt_list.append(gt_r)

    # Pad to max N per frame and stack
    xc_out, yc_out = [], []
    for t_idx in range(n_frames):
        max_n = max(x.shape[0] for x in xc_lists[t_idx])
        xc_b = torch.zeros(batch_size, max_n, 2)
        yc_b = torch.zeros(batch_size, max_n, 1)
        for i, (xc, yc) in enumerate(zip(xc_lists[t_idx], yc_lists[t_idx])):
            nc = xc.shape[0]
            xc_b[i, :nc] = xc
            yc_b[i, :nc] = yc
        xc_out.append(xc_b.to(device))
        yc_out.append(yc_b.to(device))

    gt_b = torch.stack(gt_list).to(device)  # [B, Gx, Gy]
    return xc_out, yc_out, gt_b


def evaluate(model, episodes, gt_grid_x, gt_grid_y, device,
             n_frames_model, frame_gap, n_eval=10):
    model.eval()
    all_nrmse = []
    gx, gy = gt_grid_x, gt_grid_y
    xx, yy = torch.meshgrid(gx, gy, indexing='ij')
    xt = torch.stack([xx.flatten(), yy.flatten()], dim=-1).unsqueeze(0).to(device)
    min_fi = (n_frames_model - 1) * frame_gap

    for ep in episodes:
        Hs = max(ep.Hs_true, 0.1)
        idx_list = np.linspace(min_fi, len(ep) - 1, n_eval, dtype=int)
        for fi in idx_list:
            xc_list, yc_list = [], []
            for t_idx in range(n_frames_model):
                fr = ep[fi - (n_frames_model - 1 - t_idx) * frame_gap]
                xc = torch.stack([fr.x_obs, fr.y_obs], dim=-1).unsqueeze(0).to(device)
                yc = fr.z_obs.unsqueeze(-1).unsqueeze(0).to(device) / Hs
                xc_list.append(xc)
                yc_list.append(yc)
            with torch.no_grad():
                pred = model(xc_list, yc_list, xt)
            pred_field = (pred[0, :, 0] * Hs).cpu().numpy()
            pred_field = pred_field.reshape(len(gx), len(gy)).T
            fr_cur = ep[fi]
            v = nrmse(pred_field, fr_cur.gt_field.numpy(), ep.Hs_true)
            if not np.isnan(v):
                all_nrmse.append(v)

    model.train()
    return float(np.mean(all_nrmse)) if all_nrmse else 1.0


def test_evaluation(model, test_eps_names, test_eps, gt_grid_x, gt_grid_y,
                    device, n_frames_model, frame_gap, n_eval=20):
    model.eval()
    gx, gy = gt_grid_x, gt_grid_y
    xx, yy = torch.meshgrid(gx, gy, indexing='ij')
    xt = torch.stack([xx.flatten(), yy.flatten()], dim=-1).unsqueeze(0).to(device)
    min_fi = (n_frames_model - 1) * frame_gap
    results = {}

    for ep_name, ep in zip(test_eps_names, test_eps):
        Hs = max(ep.Hs_true, 0.1)
        idx_list = np.linspace(min_fi, len(ep) - 1, n_eval, dtype=int)
        ep_nrmse = []
        for fi in idx_list:
            xc_list, yc_list = [], []
            for t_idx in range(n_frames_model):
                fr = ep[fi - (n_frames_model - 1 - t_idx) * frame_gap]
                xc = torch.stack([fr.x_obs, fr.y_obs], dim=-1).unsqueeze(0).to(device)
                yc = fr.z_obs.unsqueeze(-1).unsqueeze(0).to(device) / Hs
                xc_list.append(xc)
                yc_list.append(yc)
            with torch.no_grad():
                pred = model(xc_list, yc_list, xt)
            pred_field = (pred[0, :, 0] * Hs).cpu().numpy()
            pred_field = pred_field.reshape(len(gx), len(gy)).T
            fr_cur = ep[fi]
            v = nrmse(pred_field, fr_cur.gt_field.numpy(), ep.Hs_true)
            if not np.isnan(v):
                ep_nrmse.append(v)
        key = ep_name.replace('episode_', '').replace('.h5', '')
        avg = float(np.mean(ep_nrmse)) if ep_nrmse else 1.0
        results[key] = avg
        print(f"  {key}: {avg:.4f}")

    return results


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(model, train_eps, val_eps,
          model_grid_x, model_grid_y, gt_grid_x, gt_grid_y,
          x_range, y_range, ckpt_path):
    x_extent = x_range[1] - x_range[0]
    y_extent = y_range[1] - y_range[0]

    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    warmup = optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.1, end_factor=1.0, total_iters=WARMUP_EPOCHS)
    cosine = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS - WARMUP_EPOCHS, eta_min=LR * 0.01)
    scheduler = optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup, cosine], milestones=[WARMUP_EPOCHS])

    best_val = 1.0
    best_state = None
    t0 = time.time()

    for epoch in range(1, EPOCHS + 1):
        model.train()
        epoch_loss = 0.0

        for _ in range(STEPS_PER_EPOCH):
            xc_list, yc_list, gt_b = sample_multiframe_batch(
                train_eps, BATCH_SIZE, N_FRAMES, FRAME_GAP,
                model_grid_x, model_grid_y, gt_grid_x, gt_grid_y, DEVICE)

            grid_pred = model.predict_grid(xc_list, yc_list)  # [B, 1, Gx, Gy]
            gt_grid   = gt_b.unsqueeze(1)                      # [B, 1, Gx, Gy]

            l_mse  = F.mse_loss(grid_pred, gt_grid)
            pred_f = grid_pred.squeeze(1).flatten(1)
            gt_f   = gt_grid.squeeze(1).flatten(1)
            l_corr = correlation_loss(pred_f, gt_f)
            l_spec = spectral_physics_loss(
                grid_pred, x_extent, y_extent, DEPTH, T_MIN, T_MAX)

            loss = l_mse + LAMBDA_CORR * l_corr + LAMBDA_SPEC * l_spec
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()

        scheduler.step()
        avg_loss = epoch_loss / STEPS_PER_EPOCH

        if epoch % 10 == 0:
            val_nrmse = evaluate(model, val_eps, gt_grid_x, gt_grid_y, DEVICE,
                                 N_FRAMES, FRAME_GAP)
            if val_nrmse < best_val:
                best_val = val_nrmse
                best_state = {k: v.cpu().clone()
                              for k, v in model.state_dict().items()}
            elapsed = time.time() - t0
            print(f"  Epoch {epoch:3d}/{EPOCHS}  loss={avg_loss:.4f}"
                  f"  val={val_nrmse:.4f}  best={best_val:.4f}"
                  f"  [{elapsed:.0f}s]")

    if best_state is not None:
        model.load_state_dict({k: v.to(DEVICE) for k, v in best_state.items()})
        torch.save(best_state, ckpt_path)
        print(f"  Best checkpoint saved: {ckpt_path}")

    return best_val


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print(f"WavePINO Stage 4 — Multi-frame (T={N_FRAMES}, gap={FRAME_GAP}, "
          f"Δt={FRAME_GAP*0.1:.1f}s/step, window={(N_FRAMES-1)*FRAME_GAP*0.1:.1f}s)")
    print(f"Device: {DEVICE}")
    print(f"Temporal window: {(N_FRAMES-1)*FRAME_GAP*0.1:.1f}s  "
          f"(wave phases shift {(N_FRAMES-1)*FRAME_GAP*0.1 * 4:.0f}–"
          f"{(N_FRAMES-1)*FRAME_GAP*0.1 * 14:.0f}m at 4–14 m/s)")

    data_dir = os.path.join(REPO_ROOT, 'wavesplat', 'data')
    ckpt_dir = os.path.join(REPO_ROOT, 'wavesplat', 'checkpoints')
    os.makedirs(ckpt_dir, exist_ok=True)

    print("\nLoading episodes ...")
    train_eps = load_episodes(TRAIN_EPISODES, data_dir)
    val_eps   = load_episodes(VAL_EPISODES,   data_dir)
    test_eps  = load_episodes(TEST_EPISODES,  data_dir)
    print(f"  Train: {len(train_eps)}  Val: {len(val_eps)}  Test: {len(test_eps)}")

    # Grab grid info from first episode
    ep0          = train_eps[0]
    gt_grid_x    = ep0.gt_grid_x.cpu()
    gt_grid_y    = ep0.gt_grid_y.cpu()
    x_range      = (float(gt_grid_x[0]),  float(gt_grid_x[-1]))
    y_range      = (float(gt_grid_y[0]),  float(gt_grid_y[-1]))
    model_grid_x = torch.linspace(x_range[0], x_range[1], GRID_NX)
    model_grid_y = torch.linspace(y_range[0], y_range[1], GRID_NY)

    # Build model
    model = WavePINOMultiFrame(
        n_frames=N_FRAMES,
        grid_nx=GRID_NX, grid_ny=GRID_NY,
        x_range=x_range, y_range=y_range,
        fno_width=FNO_WIDTH, fno_layers=FNO_LAYERS,
        fno_modes_h=FNO_MODES_H, fno_modes_w=FNO_MODES_W,
        init_lengthscale=10.0,
    ).to(DEVICE)
    print(f"\n  Model: {model.n_params:,} params  "
          f"(vs Stage 2 single-frame: 792,370)")
    print(f"  Input channels: 2×{N_FRAMES}={2*N_FRAMES}  "
          f"(signal+density per frame)")

    ckpt_path = os.path.join(ckpt_dir, f'wavepino_mf_T{N_FRAMES}_g{FRAME_GAP}.pt')
    t_start = time.time()

    print(f"\n{'='*70}")
    print(f"  Training WavePINO-MF  T={N_FRAMES}  λ_spec={LAMBDA_SPEC}")
    print(f"{'='*70}")

    best_val = train(model, train_eps, val_eps,
                     model_grid_x, model_grid_y, gt_grid_x, gt_grid_y,
                     x_range, y_range, ckpt_path)

    print(f"\nTest evaluation ...")
    test_results = test_evaluation(
        model, TEST_EPISODES, test_eps,
        gt_grid_x, gt_grid_y, DEVICE,
        N_FRAMES, FRAME_GAP)

    test_avg = float(np.mean(list(test_results.values())))
    total_s  = time.time() - t_start

    result = {
        "method": f"WavePINO-MF T={N_FRAMES} gap={FRAME_GAP}",
        "n_frames": N_FRAMES,
        "frame_gap": FRAME_GAP,
        "dt_s": FRAME_GAP * 0.1,
        "temporal_window_s": (N_FRAMES - 1) * FRAME_GAP * 0.1,
        "best_val_nrmse": best_val,
        "test_avg_nrmse": test_avg,
        "test_per_episode": test_results,
        "config": {
            "fno_width": FNO_WIDTH,
            "fno_layers": FNO_LAYERS,
            "fno_modes": f"{FNO_MODES_H}x{FNO_MODES_W}",
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "lr": LR,
            "lambda_corr": LAMBDA_CORR,
            "lambda_spec": LAMBDA_SPEC,
        },
        "references": {
            "WaveSplat": 0.223,
            "ConvCNP_v2": 0.244,
            "FNO_no_phys": 0.2334,
            "PINO_v1_spec_T1": 0.2220,
        },
        "delta_vs_pino1": test_avg - 0.2220,
        "total_time_s": total_s,
    }

    out_path = os.path.join(REPO_ROOT, 'wavesplat', 'results',
                            'pino_stage4_multiframe.json')
    with open(out_path, 'w') as f:
        json.dump(result, f, indent=2)

    print(f"\n{'='*70}")
    print(f"  RESULT SUMMARY")
    print(f"{'='*70}")
    print(f"  WavePINO-MF T={N_FRAMES}: test nRMSE = {test_avg:.4f}")
    print(f"  PINO v1 (T=1):            test nRMSE = 0.2220")
    print(f"  WaveSplat:                test nRMSE = 0.2230")
    delta = test_avg - 0.2220
    verdict = f"BETTER by {-delta*100:.1f}%" if delta < 0 else f"WORSE by {delta*100:.1f}%"
    print(f"  vs PINO v1: {verdict}")
    print(f"  Total time: {total_s:.1f}s")
    print(f"  Saved: {out_path}")

    return test_avg


if __name__ == '__main__':
    main()
