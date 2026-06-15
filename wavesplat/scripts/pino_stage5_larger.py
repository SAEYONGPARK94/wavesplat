#!/usr/bin/env python3
"""
pino_stage5_larger.py

WavePINO Stage 5: Larger model + longer training.
Same T=3 multi-frame config as Stage 4 (best result 0.2201),
but with larger FNO (width=24, 5 blocks) and 300 epochs.

Goal: 0.2201 → 0.210 (statistically meaningful win over WaveSplat 0.223)

Key differences from Stage 4:
  - FNO_WIDTH:   16 → 24
  - FNO_LAYERS:   4 → 5
  - FNO_MODES_H: 16 → 20
  - FNO_MODES_W: 12 → 16
  - EPOCHS:     100 → 300

Usage:
    cd wavesplat && python3 -u scripts/pino_stage5_larger.py
"""

import os, sys, time, json, math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', buffering=1)
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(REPO_ROOT, 'wavesplat'))

from wavesplat.data_loader import EpisodeDataLoader
from wavesplat.model.wave_pino import SpectralConv2d, FNOBlock, spectral_physics_loss
from wavesplat.evaluation.metrics import nrmse

# ---------------------------------------------------------------------------
# Episode lists (same as Stage 4)
# ---------------------------------------------------------------------------
TRAIN_EPISODES = [
    'episode_BF3_1comp_seed0_uniform.h5', 'episode_BF3_1comp_seed1_uniform.h5',
    'episode_BF3_2comp_seed0_uniform.h5', 'episode_BF3_2comp_seed1_uniform.h5',
    'episode_BF3_5comp_seed0_uniform.h5', 'episode_BF3_5comp_seed1_uniform.h5',
    'episode_BF3_1comp_seed0_slope.h5',   'episode_BF3_1comp_seed1_slope.h5',
    'episode_BF3_2comp_seed0_slope.h5',   'episode_BF3_2comp_seed1_slope.h5',
    'episode_BF3_5comp_seed0_slope.h5',   'episode_BF3_5comp_seed1_slope.h5',
    'episode_BF4_1comp_seed0_uniform.h5', 'episode_BF4_1comp_seed1_uniform.h5',
    'episode_BF4_5comp_seed0_uniform.h5', 'episode_BF4_5comp_seed1_uniform.h5',
    'episode_BF4_15comp_seed0_uniform.h5','episode_BF4_15comp_seed1_uniform.h5',
    'episode_BF4_30comp_seed0_uniform.h5',
    'episode_BF4_1comp_seed0_slope.h5',   'episode_BF4_1comp_seed1_slope.h5',
    'episode_BF4_5comp_seed0_slope.h5',   'episode_BF4_5comp_seed1_slope.h5',
    'episode_BF4_15comp_seed0_slope.h5',  'episode_BF4_15comp_seed1_slope.h5',
    'episode_BF5_15comp_seed0_uniform.h5','episode_BF5_15comp_seed1_uniform.h5',
    'episode_BF5_30comp_seed0_uniform.h5','episode_BF5_30comp_seed1_uniform.h5',
    'episode_BF5_15comp_seed0_slope.h5',  'episode_BF5_15comp_seed1_slope.h5',
    'episode_BF5_30comp_seed0_slope.h5',  'episode_BF5_30comp_seed1_slope.h5',
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
# Hyperparameters (Stage 5 — larger)
# ---------------------------------------------------------------------------
GRID_NX      = 64
GRID_NY      = 48
FNO_WIDTH    = 24       # up from 16
FNO_LAYERS   = 5        # up from 4
FNO_MODES_H  = 20       # up from 16
FNO_MODES_W  = 16       # up from 12

N_FRAMES     = 3
FRAME_GAP    = 5        # Δt = 0.5s → 1.0s window

EPOCHS           = 300  # up from 100
STEPS_PER_EPOCH  = 100
BATCH_SIZE       = 8
LR               = 1e-3
WARMUP_EPOCHS    = 10
LAMBDA_SPEC      = 0.5

DEPTH = 50.0
T_MIN = 4.0
T_MAX = 25.0

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ---------------------------------------------------------------------------
# Model (same WavePINOMultiFrame, larger config)
# ---------------------------------------------------------------------------

class WavePINOMultiFrameV2(nn.Module):
    """Same as Stage 4 WavePINOMultiFrame, instantiated with larger config."""

    def __init__(self, n_frames=3, grid_nx=64, grid_ny=48,
                 x_range=(10.0, 250.0), y_range=(-100.0, 100.0),
                 fno_width=24, fno_layers=5, fno_modes_h=20, fno_modes_w=16,
                 init_lengthscale=10.0):
        super().__init__()
        self.n_frames  = n_frames
        self.grid_nx   = grid_nx
        self.grid_ny   = grid_ny
        self.x_range   = x_range
        self.y_range   = y_range
        self.fno_width = fno_width

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
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='linear')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

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

    def _grid_predict(self, xc_list, yc_list):
        frame_feats = [self._setconv(xc, yc) for xc, yc in zip(xc_list, yc_list)]
        z = self.lifting(torch.cat(frame_feats, dim=1))
        for block in self.fno_blocks:
            z = block(z)
        return self.output_head(z)

    def forward(self, xc_list, yc_list, xt):
        grid = self._grid_predict(xc_list, yc_list)
        return self._interp(grid, xt)

    def predict_grid(self, xc_list, yc_list):
        return self._grid_predict(xc_list, yc_list)

    def _interp(self, grid_field, xt):
        x_norm = 2.0 * (xt[..., 0] - self.x_range[0]) / (self.x_range[1] - self.x_range[0]) - 1.0
        y_norm = 2.0 * (xt[..., 1] - self.y_range[0]) / (self.y_range[1] - self.y_range[0]) - 1.0
        sg = torch.stack([y_norm, x_norm], dim=-1).unsqueeze(2)
        return F.grid_sample(grid_field, sg, mode='bilinear',
                              align_corners=True, padding_mode='border'
                              ).squeeze(1).squeeze(-1).unsqueeze(-1)

    @property
    def n_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Data utilities
# ---------------------------------------------------------------------------

def load_episodes(names, data_dir):
    return [EpisodeDataLoader(os.path.join(data_dir, n))
            for n in names if os.path.exists(os.path.join(data_dir, n))]


def sample_multiframe_batch(episodes, batch_size, n_frames, frame_gap,
                             model_grid_x, model_grid_y, gt_grid_x, gt_grid_y, device):
    min_fi = (n_frames - 1) * frame_gap
    xc_batch = [[] for _ in range(n_frames)]
    yc_batch = [[] for _ in range(n_frames)]
    gt_batch = []
    max_N = [0] * n_frames

    for _ in range(batch_size):
        ep = episodes[np.random.randint(len(episodes))]
        Hs = max(ep.Hs_true, 0.1)
        fi = np.random.randint(min_fi, len(ep))
        flip = np.random.random() < 0.5

        frames_fi = [fi - (n_frames - 1 - k) * frame_gap for k in range(n_frames)]
        for k, fj in enumerate(frames_fi):
            frame = ep[fj]
            xc = torch.stack([frame.x_obs, frame.y_obs], dim=-1)
            if flip:
                xc = xc.clone(); xc[:, 1] = -xc[:, 1]
            yc = frame.z_obs.unsqueeze(-1) / Hs
            xc_batch[k].append(xc)
            yc_batch[k].append(yc)
            max_N[k] = max(max_N[k], xc.shape[0])

        gt = ep[fi].gt_field / Hs
        if flip:
            gt = gt.flip(0)
        gt_batch.append(gt)

    xc_out, yc_out = [], []
    for k in range(n_frames):
        xb = torch.zeros(batch_size, max_N[k], 2)
        yb = torch.zeros(batch_size, max_N[k], 1)
        for i in range(batch_size):
            nc = xc_batch[k][i].shape[0]
            xb[i, :nc] = xc_batch[k][i]
            yb[i, :nc] = yc_batch[k][i]
        xc_out.append(xb.to(device))
        yc_out.append(yb.to(device))

    gt_out = torch.stack(gt_batch).to(device)  # [B, Gx, Gy]
    return xc_out, yc_out, gt_out


# ---------------------------------------------------------------------------
# Evaluate helpers
# ---------------------------------------------------------------------------

def eval_model(model, episodes, x_extent, y_extent, n_frames=4):
    model.eval()
    vals = []
    for ep in episodes:
        Hs = max(ep.Hs_true, 0.1)
        min_fi = (N_FRAMES - 1) * FRAME_GAP
        fis = np.linspace(min_fi, len(ep) - 1, n_frames, dtype=int)
        gx = ep.gt_grid_x.cpu()
        gy = ep.gt_grid_y.cpu()
        xx, yy = torch.meshgrid(gx, gy, indexing='ij')
        xt = torch.stack([xx.flatten(), yy.flatten()], dim=-1).unsqueeze(0).to(DEVICE)
        for fi in fis:
            xc_list, yc_list = [], []
            for k in range(N_FRAMES):
                fj = fi - (N_FRAMES - 1 - k) * FRAME_GAP
                frame = ep[fj]
                xc = torch.stack([frame.x_obs, frame.y_obs], dim=-1).unsqueeze(0).to(DEVICE)
                yc = frame.z_obs.unsqueeze(-1).unsqueeze(0).to(DEVICE) / Hs
                xc_list.append(xc)
                yc_list.append(yc)
            frame = ep[fi]
            with torch.no_grad():
                pred = model(xc_list, yc_list, xt)
            # reshape [Nx, Ny] then transpose to [Ny, Nx] to match gt_field
            pred_field = (pred[0, :, 0] * Hs).cpu().reshape(
                len(gx), len(gy)).numpy().T
            v = nrmse(pred_field, frame.gt_field.numpy(), ep.Hs_true)
            if not np.isnan(v):
                vals.append(v)
    model.train()
    return float(np.mean(vals)) if vals else 1.0


def test_eval(model, episodes, x_extent, y_extent, n_eval=20):
    model.eval()
    ep_results = []
    for ep in episodes:
        Hs = max(ep.Hs_true, 0.1)
        min_fi = (N_FRAMES - 1) * FRAME_GAP
        fis = np.linspace(min_fi, len(ep) - 1, n_eval, dtype=int)
        gx = ep.gt_grid_x.cpu()
        gy = ep.gt_grid_y.cpu()
        xx, yy = torch.meshgrid(gx, gy, indexing='ij')
        xt = torch.stack([xx.flatten(), yy.flatten()], dim=-1).unsqueeze(0).to(DEVICE)
        ep_vals = []
        for fi in fis:
            xc_list, yc_list = [], []
            for k in range(N_FRAMES):
                fj = fi - (N_FRAMES - 1 - k) * FRAME_GAP
                frame = ep[fj]
                xc = torch.stack([frame.x_obs, frame.y_obs], dim=-1).unsqueeze(0).to(DEVICE)
                yc = frame.z_obs.unsqueeze(-1).unsqueeze(0).to(DEVICE) / Hs
                xc_list.append(xc)
                yc_list.append(yc)
            frame = ep[fi]
            with torch.no_grad():
                pred = model(xc_list, yc_list, xt)
            pred_field = (pred[0, :, 0] * Hs).cpu().reshape(
                len(gx), len(gy)).numpy().T
            v = nrmse(pred_field, frame.gt_field.numpy(), ep.Hs_true)
            if not np.isnan(v):
                ep_vals.append(v)
        ep_results.append(float(np.mean(ep_vals)))
    model.train()
    return ep_results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    data_dir = os.path.join(REPO_ROOT, 'wavesplat', 'data')

    print("=" * 65)
    print("WavePINO Stage 5: Larger model (w=24, L=5) + 300 epochs")
    print("Reference: Stage 4 T=3 gap=5 → 0.2201 | WaveSplat 0.223")
    print("=" * 65)

    train_eps = load_episodes(TRAIN_EPISODES, data_dir)
    val_eps   = load_episodes(VAL_EPISODES,   data_dir)
    test_eps  = load_episodes(TEST_EPISODES,  data_dir)
    print("  Train %d | Val %d | Test %d | Device: %s" % (
        len(train_eps), len(val_eps), len(test_eps), DEVICE))

    ep0 = train_eps[0]
    x_range  = (float(ep0.gt_grid_x[0]), float(ep0.gt_grid_x[-1]))
    y_range  = (float(ep0.gt_grid_y[0]), float(ep0.gt_grid_y[-1]))
    x_extent = x_range[1] - x_range[0]
    y_extent = y_range[1] - y_range[0]
    gt_grid_x = ep0.gt_grid_x
    gt_grid_y = ep0.gt_grid_y

    model = WavePINOMultiFrameV2(
        n_frames=N_FRAMES, grid_nx=GRID_NX, grid_ny=GRID_NY,
        x_range=x_range, y_range=y_range,
        fno_width=FNO_WIDTH, fno_layers=FNO_LAYERS,
        fno_modes_h=FNO_MODES_H, fno_modes_w=FNO_MODES_W,
    ).to(DEVICE)
    print("  Params: %s | width=%d layers=%d modes=(%d,%d) T=%d gap=%d" % (
        format(model.n_params, ','), FNO_WIDTH, FNO_LAYERS,
        FNO_MODES_H, FNO_MODES_W, N_FRAMES, FRAME_GAP))

    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    warmup = optim.lr_scheduler.LinearLR(optimizer, start_factor=0.01, total_iters=WARMUP_EPOCHS)
    cosine = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS - WARMUP_EPOCHS)
    scheduler = optim.lr_scheduler.SequentialLR(optimizer, [warmup, cosine], milestones=[WARMUP_EPOCHS])

    ckpt_dir = os.path.join(REPO_ROOT, 'wavesplat', 'checkpoints')
    os.makedirs(ckpt_dir, exist_ok=True)

    best_val, best_epoch = 1.0, 0
    t0 = time.time()

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = recon_loss_sum = spec_loss_sum = 0.0

        for _ in range(STEPS_PER_EPOCH):
            xc_list, yc_list, gt = sample_multiframe_batch(
                train_eps, BATCH_SIZE, N_FRAMES, FRAME_GAP,
                ep0.gt_grid_x, ep0.gt_grid_y,
                ep0.gt_grid_x, ep0.gt_grid_y, DEVICE)

            # Forward: grid prediction for physics loss + target interp for recon
            grid = model.predict_grid(xc_list, yc_list)  # [B,1,Gx,Gy]

            # Build target query coords from gt grid
            B = gt.shape[0]
            gxx = torch.from_numpy(ep0.gt_grid_x.numpy()).to(DEVICE)
            gyy = torch.from_numpy(ep0.gt_grid_y.numpy()).to(DEVICE)
            XX, YY = torch.meshgrid(gxx, gyy, indexing='ij')
            xt = torch.stack([XX.flatten(), YY.flatten()], dim=-1).unsqueeze(0).expand(B, -1, -1)
            pred_pts = model._interp(grid, xt)   # [B, Gx*Gy, 1]
            gt_flat  = gt.reshape(B, -1).unsqueeze(-1)   # [B, Gx*Gy, 1]

            recon_loss = F.mse_loss(pred_pts, gt_flat)
            spec_loss  = spectral_physics_loss(grid, x_extent, y_extent,
                                                depth=DEPTH, T_min=T_MIN, T_max=T_MAX)
            loss = recon_loss + LAMBDA_SPEC * spec_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss    += loss.item()
            recon_loss_sum += recon_loss.item()
            spec_loss_sum  += spec_loss.item()

        scheduler.step()

        if epoch % 20 == 0 or epoch == 1:
            val_nrmse = eval_model(model, val_eps, x_extent, y_extent, n_frames=4)
            improved = ""
            if val_nrmse < best_val:
                best_val, best_epoch = val_nrmse, epoch
                torch.save(model.state_dict(), os.path.join(ckpt_dir, 'pino_mf_v2_best.pt'))
                improved = " *"
            ls = model.log_lengthscale.exp().item()
            print("  Ep %3d/%d: loss=%.4f (r=%.4f s=%.4f) val=%.4f ls=%.1fm [%.0fs]%s" % (
                epoch, EPOCHS, total_loss / STEPS_PER_EPOCH,
                recon_loss_sum / STEPS_PER_EPOCH, spec_loss_sum / STEPS_PER_EPOCH,
                val_nrmse, ls, time.time() - t0, improved))

    print("  Best val: %.4f at epoch %d" % (best_val, best_epoch))

    # Load best and test
    model.load_state_dict(torch.load(os.path.join(ckpt_dir, 'pino_mf_v2_best.pt'),
                                      map_location='cpu', weights_only=True))
    model.to(DEVICE)

    print("\n" + "=" * 65)
    print("Test Evaluation — WavePINO Stage 5")
    print("=" * 65)

    ep_results = test_eval(model, test_eps, x_extent, y_extent)
    for name, v in zip(TEST_EPISODES, ep_results):
        print("  %-45s %.4f" % (name.replace('episode_', '').replace('.h5', ''), v))
    avg = float(np.mean(ep_results))
    print("  Avg: %.4f" % avg)
    print("  Reference: Stage 4 MF = 0.2201 | WaveSplat = 0.223")

    results_dir = os.path.join(REPO_ROOT, 'wavesplat', 'results')
    os.makedirs(results_dir, exist_ok=True)
    out = {
        'version': 'pino_stage5_larger',
        'test_avg': avg, 'best_val': best_val, 'best_epoch': best_epoch,
        'n_params': model.n_params, 'training_time_s': time.time() - t0,
        'config': {'fno_width': FNO_WIDTH, 'fno_layers': FNO_LAYERS,
                   'fno_modes_h': FNO_MODES_H, 'fno_modes_w': FNO_MODES_W,
                   'n_frames': N_FRAMES, 'frame_gap': FRAME_GAP,
                   'epochs': EPOCHS, 'lambda_spec': LAMBDA_SPEC},
        'references': {'stage4_mf_t3': 0.2201, 'WaveSplat': 0.223},
        'per_episode': dict(zip(
            [n.replace('episode_', '').replace('.h5', '') for n in TEST_EPISODES],
            ep_results)),
    }
    out_path = os.path.join(results_dir, 'pino_stage5_larger.json')
    with open(out_path, 'w') as f:
        import json
        json.dump(out, f, indent=2)
    print("  Saved: %s" % out_path)


if __name__ == '__main__':
    main()
