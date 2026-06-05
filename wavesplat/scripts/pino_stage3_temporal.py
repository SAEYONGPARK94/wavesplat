#!/usr/bin/env python3
"""
pino_stage3_temporal.py

WavePINO Stage 3: Temporal Dispersion Physics Loss.

Motivation (from literature):
  Ehlers et al. (2025, arXiv:2508.03315) demonstrated that embedding
  free-surface boundary conditions into FNO training dramatically improves
  ocean wave reconstruction accuracy. Their approach uses Zakharov BCs
  (nonlinear, requires temporal derivatives) for 1D+time problems.

  For our 2D ship-mounted LiDAR problem, we derive the corresponding
  linear temporal consistency constraint:

    η(t + Δt) ≈ IFFT[ FFT(η(t)) × exp(-i ω(|k|) Δt) ]

  where ω(|k|) = √(g |k| tanh(|k| d)) is the full (non-deep-water)
  dispersion relation. This is the linearized Zakharov kinematic condition
  in the frequency domain, exact for linear waves.

Physics loss:
  L_temporal = MSE( linear_propagate(η_pred(t_a), Δt),  η_pred(t_b) )

  Both η_pred(t_a) and η_pred(t_b) are outputs of the SAME model applied
  to LiDAR observations at two different times from the same episode.
  The loss enforces SELF-CONSISTENCY of consecutive predictions under
  linear wave physics — no additional ground truth needed.

Wave parameters (BF3-5):
  Periods: 2-15s, phase speeds: 4-15 m/s, frame interval: 0.1s
  Frame gap = 10 → Δt = 1.0s: waves move 4-14m (1.7-5.8% of domain)
  → Meaningful constraint while linear approx remains accurate (ε << 1)

Training losses (4 terms):
  L = L_mse + λ_corr × L_corr + λ_spec × L_spec + λ_temp × L_temporal

Usage:
    cd wavesplat && python3 -u scripts/pino_stage3_temporal.py
"""

import os
import sys
import time
import json
import math

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', buffering=1)

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(REPO_ROOT, 'wavesplat'))

from wavesplat.data_loader import EpisodeDataLoader
from wavesplat.model.wave_pino import WavePINO, spectral_physics_loss
from wavesplat.evaluation.metrics import nrmse

G = 9.81
PI = math.pi

# ---------------------------------------------------------------------------
# Episode lists (same as Stage 2 for fair comparison)
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

EPOCHS          = 100
STEPS_PER_EPOCH = 100
BATCH_SIZE      = 8
LR              = 1e-3
WARMUP_EPOCHS   = 5

LAMBDA_CORR  = 1.0   # correlation loss
LAMBDA_SPEC  = 0.5   # spectral physics loss (same as PINO v1)
LAMBDA_TEMP  = 1.0   # temporal dispersion loss (sweep below if time permits)

DEPTH        = 50.0   # water depth [m]
T_MIN        = 4.0    # min wave period [s] for spectral loss
T_MAX        = 25.0   # max wave period [s]
FRAME_GAP    = 10     # frames apart for temporal pair (Δt = 10 × 0.1s = 1.0s)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ---------------------------------------------------------------------------
# Core: linear temporal propagation
# ---------------------------------------------------------------------------

def linear_propagate_2d(
    eta: torch.Tensor,   # [B, H, W] real wave field
    dt: float,           # time step [s]
    x_extent: float,     # physical x extent [m]
    y_extent: float,     # physical y extent [m]
    depth: float = 50.0,
) -> torch.Tensor:       # [B, H, W] propagated field
    """Propagate 2D wave field by dt using linear dispersion relation.

    For a plane wave A cos(k·r - ωt + φ):
      FFT at t:     A exp(i(k·r + φ)) with peak at k=(kx,ky)
      FFT at t+Δt:  A exp(i(k·r + φ)) × exp(-i ω Δt)

    The operation η_ft_new = η_ft × exp(-i ω(|k|) Δt) is applied to
    all Fourier modes, where ω = sqrt(g|k|tanh(|k|d)) is the full
    dispersion relation.

    Args:
        eta:     [B, H, W] predicted wave field (normalized)
        dt:      time step in seconds
        x_extent: physical x extent (H direction) [m]
        y_extent: physical y extent (W direction) [m]
        depth:   water depth [m]

    Returns:
        [B, H, W] wave field propagated by dt
    """
    B, H, W = eta.shape

    # 2D rfft: [B, H, W//2+1] complex
    eta_ft = torch.fft.rfft2(eta)

    # Wavenumber grids [rad/m]
    kx = (2.0 * PI * torch.fft.fftfreq(H, d=x_extent / H)
          .to(eta.device))        # [H], includes negative frequencies
    ky = (2.0 * PI * torch.fft.rfftfreq(W, d=y_extent / W)
          .to(eta.device))        # [W//2+1], positive only

    KX, KY = torch.meshgrid(kx, ky, indexing='ij')   # [H, W//2+1]
    k_mag  = torch.sqrt(KX ** 2 + KY ** 2 + 1e-12)

    # Dispersion: ω = sqrt(g |k| tanh(|k| d))
    omega  = torch.sqrt(G * k_mag * torch.tanh(k_mag * depth) + 1e-12)

    # Phase factor: exp(-i ω Δt)  [H, W//2+1] complex
    phase  = torch.polar(torch.ones_like(omega), -omega * dt)

    # Apply phase shift
    eta_ft_propagated = eta_ft * phase.unsqueeze(0)   # [B, H, W//2+1]

    # Inverse rfft2 → real-valued propagated field
    return torch.fft.irfft2(eta_ft_propagated, s=(H, W))


def temporal_dispersion_loss(
    pred_a: torch.Tensor,    # [B, H, W]
    pred_b: torch.Tensor,    # [B, H, W]
    dt: float,
    x_extent: float,
    y_extent: float,
    depth: float = 50.0,
) -> torch.Tensor:
    """Temporal consistency loss using linear dispersion propagation.

    Enforces: linear_propagate(η_pred(t_a), Δt) ≈ η_pred(t_b)

    This is the linearized kinematic free-surface condition in the
    frequency domain, analogous to the Zakharov conditions used in
    Ehlers et al. (2025) but adapted for 2D spatial reconstruction.

    The loss is a self-consistency check on consecutive model outputs
    — no additional ground truth wave field is required.
    """
    pred_a_propagated = linear_propagate_2d(pred_a, dt, x_extent, y_extent, depth)
    return F.mse_loss(pred_a_propagated, pred_b)


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


def resample_gt(gt_field, model_grid_x, model_grid_y, gt_grid_x, gt_grid_y):
    """Resample GT [Ny, Nx] from data grid to model grid [Gx, Gy]."""
    dx0, dx1 = float(gt_grid_x[0]), float(gt_grid_x[-1])
    dy0, dy1 = float(gt_grid_y[0]), float(gt_grid_y[-1])
    x_n = 2.0 * (model_grid_x - dx0) / (dx1 - dx0) - 1.0
    y_n = 2.0 * (model_grid_y - dy0) / (dy1 - dy0) - 1.0
    yy, xx = torch.meshgrid(y_n, x_n, indexing='ij')
    sg = torch.stack([xx, yy], dim=-1).unsqueeze(0)
    gt_r = F.grid_sample(gt_field.unsqueeze(0).unsqueeze(0).float(), sg,
                          mode='bilinear', align_corners=True,
                          padding_mode='border')[0, 0]   # [Gy, Gx]
    return gt_r.T  # [Gx, Gy]


def sample_temporal_batch(episodes, batch_size, model_grid_x, model_grid_y,
                           gt_grid_x, gt_grid_y, device, frame_gap):
    """Sample pairs of frames (t_a, t_b=t_a+frame_gap) from same episodes.

    Returns:
        xc_a, yc_a: context for predicting η at t_a   [B, Nc, 2/1]
        gt_a:       GT on model grid at t_a            [B, Gx, Gy]
        xc_b, yc_b: context for predicting η at t_b
        gt_b:       GT on model grid at t_b
        dt:         mean time difference (scalar, float)
    """
    xc_a_list, yc_a_list, gt_a_list = [], [], []
    xc_b_list, yc_b_list, gt_b_list = [], [], []
    dts = []

    for _ in range(batch_size):
        ep = episodes[np.random.randint(len(episodes))]
        Hs = max(ep.Hs_true, 0.1)

        # Sample frame pair
        max_idx = len(ep) - frame_gap - 1
        if max_idx < 1:
            max_idx = 1
        i_a = np.random.randint(0, max_idx)
        i_b = i_a + frame_gap

        frame_a = ep[i_a]
        frame_b = ep[i_b]
        dt = float(ep.timestamps[i_b] - ep.timestamps[i_a])
        dts.append(dt)

        for frame, xc_list, yc_list, gt_list in [
            (frame_a, xc_a_list, yc_a_list, gt_a_list),
            (frame_b, xc_b_list, yc_b_list, gt_b_list),
        ]:
            xc = torch.stack([frame.x_obs, frame.y_obs], dim=-1)
            yc = frame.z_obs.unsqueeze(-1) / Hs
            gt = resample_gt(frame.gt_field / Hs,
                             model_grid_x, model_grid_y, gt_grid_x, gt_grid_y)

            # Y-flip augmentation (apply same flip to both frames)
            xc_list.append(xc)
            yc_list.append(yc)
            gt_list.append(gt)

    def pad_and_stack(xc_list, yc_list):
        max_nc = max(x.shape[0] for x in xc_list)
        xb = torch.zeros(batch_size, max_nc, 2)
        yb = torch.zeros(batch_size, max_nc, 1)
        for i in range(batch_size):
            nc = xc_list[i].shape[0]
            xb[i, :nc] = xc_list[i]
            yb[i, :nc] = yc_list[i]
        return xb.to(device), yb.to(device)

    xc_a, yc_a = pad_and_stack(xc_a_list, yc_a_list)
    xc_b, yc_b = pad_and_stack(xc_b_list, yc_b_list)
    gt_a = torch.stack(gt_a_list).to(device)   # [B, Gx, Gy]
    gt_b = torch.stack(gt_b_list).to(device)

    mean_dt = float(np.mean(dts))
    return xc_a, yc_a, gt_a, xc_b, yc_b, gt_b, mean_dt


def evaluate(model, episodes, gt_grid_x, gt_grid_y, device, n_frames=10):
    model.eval()
    all_nrmse = []
    gx, gy = gt_grid_x, gt_grid_y
    xx, yy = torch.meshgrid(gx, gy, indexing='ij')
    xt = torch.stack([xx.flatten(), yy.flatten()], dim=-1).unsqueeze(0).to(device)

    for ep in episodes:
        Hs = max(ep.Hs_true, 0.1)
        for fi in np.linspace(10, len(ep) - 1, n_frames, dtype=int):
            frame = ep[fi]
            xc = torch.stack([frame.x_obs, frame.y_obs], dim=-1).unsqueeze(0).to(device)
            yc = frame.z_obs.unsqueeze(-1).unsqueeze(0).to(device) / Hs
            with torch.no_grad():
                pred = model(xc, yc, xt)
            pred_field = (pred[0, :, 0] * Hs).cpu().numpy()
            pred_field = pred_field.reshape(len(gx), len(gy)).T
            v = nrmse(pred_field, frame.gt_field.numpy(), ep.Hs_true)
            if not np.isnan(v):
                all_nrmse.append(v)

    model.train()
    return float(np.mean(all_nrmse)) if all_nrmse else 1.0


def test_evaluation(model, test_eps, gt_grid_x, gt_grid_y, device, n_frames=20):
    model.eval()
    gx, gy = gt_grid_x, gt_grid_y
    xx, yy = torch.meshgrid(gx, gy, indexing='ij')
    xt = torch.stack([xx.flatten(), yy.flatten()], dim=-1).unsqueeze(0).to(device)
    results = {}

    for ep_idx, ep in enumerate(test_eps):
        Hs = max(ep.Hs_true, 0.1)
        ep_nrmse = []
        for fi in np.linspace(10, len(ep) - 1, n_frames, dtype=int):
            frame = ep[fi]
            xc = torch.stack([frame.x_obs, frame.y_obs], dim=-1).unsqueeze(0).to(device)
            yc = frame.z_obs.unsqueeze(-1).unsqueeze(0).to(device) / Hs
            with torch.no_grad():
                pred = model(xc, yc, xt)
            pred_field = (pred[0, :, 0] * Hs).cpu().numpy()
            pred_field = pred_field.reshape(len(gx), len(gy)).T
            v = nrmse(pred_field, frame.gt_field.numpy(), ep.Hs_true)
            if not np.isnan(v):
                ep_nrmse.append(v)

        ep_name = TEST_EPISODES[ep_idx].replace('episode_', '').replace('.h5', '')
        avg = float(np.mean(ep_nrmse)) if ep_nrmse else 1.0
        results[ep_name] = avg
        print(f"  {ep_name}: {avg:.4f}")

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    data_dir    = os.path.join(REPO_ROOT, 'wavesplat', 'data')
    results_dir = os.path.join(REPO_ROOT, 'wavesplat', 'results')
    ckpt_dir    = os.path.join(REPO_ROOT, 'wavesplat', 'checkpoints')
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)

    print("=" * 70)
    print("  WavePINO Stage 3: Temporal Dispersion Physics Loss")
    print(f"  Device: {DEVICE}")
    print(f"  Δt = {FRAME_GAP} × 0.1s = {FRAME_GAP*0.1:.1f}s")
    print("=" * 70)

    train_eps = load_episodes(TRAIN_EPISODES, data_dir)
    val_eps   = load_episodes(VAL_EPISODES,   data_dir)
    test_eps  = load_episodes(TEST_EPISODES,  data_dir)
    print(f"  Train: {len(train_eps)}, Val: {len(val_eps)}, Test: {len(test_eps)}")

    if not train_eps:
        print("ERROR: No training episodes found in", data_dir)
        return

    gt_grid_x = train_eps[0].gt_grid_x
    gt_grid_y = train_eps[0].gt_grid_y
    x_range   = (float(gt_grid_x[0]), float(gt_grid_x[-1]))
    y_range   = (float(gt_grid_y[0]), float(gt_grid_y[-1]))
    x_extent  = x_range[1] - x_range[0]   # 240m
    y_extent  = y_range[1] - y_range[0]   # 200m

    model_grid_x = torch.linspace(x_range[0], x_range[1], GRID_NX)
    model_grid_y = torch.linspace(y_range[0], y_range[1], GRID_NY)

    model = WavePINO(
        grid_nx=GRID_NX, grid_ny=GRID_NY,
        x_range=x_range, y_range=y_range,
        fno_width=FNO_WIDTH, fno_layers=FNO_LAYERS,
        fno_modes_h=FNO_MODES_H, fno_modes_w=FNO_MODES_W,
        init_lengthscale=10.0,
    ).to(DEVICE)

    print(f"  Model: {model.n_params:,} params  (width={FNO_WIDTH}, L={FNO_LAYERS}, "
          f"modes={FNO_MODES_H}×{FNO_MODES_W})")
    print(f"  Losses: MSE + {LAMBDA_CORR}×corr + {LAMBDA_SPEC}×spec + "
          f"{LAMBDA_TEMP}×temporal")

    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    warmup  = optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.01, total_iters=WARMUP_EPOCHS)
    cosine  = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS - WARMUP_EPOCHS)
    sched   = optim.lr_scheduler.SequentialLR(
        optimizer, [warmup, cosine], milestones=[WARMUP_EPOCHS])

    best_val  = 1.0
    best_ckpt = os.path.join(ckpt_dir, 'wave_pino_temporal_best.pt')
    t_start   = time.time()

    hdr = (f"{'Ep':>4}  {'MSE':>8}  {'Corr':>7}  {'Spec':>7}  "
           f"{'Temp':>7}  {'Val':>8}  {'t':>5}")
    print(f"\n{hdr}")
    print("-" * len(hdr))

    for epoch in range(1, EPOCHS + 1):
        model.train()
        ep_mse = ep_corr = ep_spec = ep_temp = 0.0

        for _ in range(STEPS_PER_EPOCH):
            (xc_a, yc_a, gt_a,
             xc_b, yc_b, gt_b, dt) = sample_temporal_batch(
                train_eps, BATCH_SIZE, model_grid_x, model_grid_y,
                gt_grid_x, gt_grid_y, DEVICE, FRAME_GAP)

            # Forward pass for both time points
            pred_a = model.predict_grid(xc_a, yc_a)  # [B, 1, Gx, Gy]
            pred_b = model.predict_grid(xc_b, yc_b)

            # --- Data fidelity losses ---
            pf_a = pred_a[:, 0].reshape(BATCH_SIZE, -1)
            pf_b = pred_b[:, 0].reshape(BATCH_SIZE, -1)
            gt_f_a = gt_a.reshape(BATCH_SIZE, -1)
            gt_f_b = gt_b.reshape(BATCH_SIZE, -1)

            mse_loss  = (F.mse_loss(pf_a, gt_f_a) + F.mse_loss(pf_b, gt_f_b)) * 0.5
            corr_loss = (correlation_loss(pf_a, gt_f_a)
                         + correlation_loss(pf_b, gt_f_b)) * 0.5

            # --- Spectral physics loss (k-range, on frame a) ---
            spec_loss = spectral_physics_loss(
                pred_a, x_extent, y_extent, DEPTH, T_MIN, T_MAX)

            # --- Temporal dispersion physics loss ---
            # linear_propagate(pred_a, dt) should ≈ pred_b
            temp_loss = temporal_dispersion_loss(
                pred_a[:, 0], pred_b[:, 0], dt, x_extent, y_extent, DEPTH)

            # --- Total loss ---
            loss = (mse_loss
                    + LAMBDA_CORR * corr_loss
                    + LAMBDA_SPEC * spec_loss
                    + LAMBDA_TEMP * temp_loss)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            ep_mse  += mse_loss.item()
            ep_corr += corr_loss.item()
            ep_spec += spec_loss.item()
            ep_temp += temp_loss.item()

        sched.step()
        ep_mse  /= STEPS_PER_EPOCH
        ep_corr /= STEPS_PER_EPOCH
        ep_spec /= STEPS_PER_EPOCH
        ep_temp /= STEPS_PER_EPOCH

        val_interval = 5 if epoch <= 50 else 10
        if epoch % val_interval == 0 or epoch == 1:
            val_nrmse = evaluate(model, val_eps, gt_grid_x, gt_grid_y, DEVICE)
            flag = ''
            if val_nrmse < best_val:
                best_val = val_nrmse
                torch.save(model.state_dict(), best_ckpt)
                flag = ' *'
            elapsed = time.time() - t_start
            ls = model.log_lengthscale.exp().item()
            print(f"{epoch:>4}  {ep_mse:>8.5f}  {ep_corr:>7.4f}  {ep_spec:>7.4f}  "
                  f"{ep_temp:>7.4f}  {val_nrmse:>8.4f}  {elapsed:>4.0f}s  "
                  f"ls={ls:.1f}m{flag}")
        elif epoch % 10 == 0:
            elapsed = time.time() - t_start
            print(f"{epoch:>4}  {ep_mse:>8.5f}  {ep_corr:>7.4f}  {ep_spec:>7.4f}  "
                  f"{ep_temp:>7.4f}  {'---':>8}  {elapsed:>4.0f}s")

    print(f"\n  Best val nRMSE: {best_val:.4f}")

    # --- Test evaluation ---
    model.load_state_dict(torch.load(best_ckpt, map_location=DEVICE,
                                      weights_only=True))
    print("\nTest Evaluation:")
    test_results = test_evaluation(model, test_eps, gt_grid_x, gt_grid_y, DEVICE)
    avg_test = float(np.mean(list(test_results.values())))
    print(f"\n  Temporal PINO test avg nRMSE: {avg_test:.4f}")

    # Comparison
    print("\n" + "=" * 70)
    print("  WavePINO Stage 3: Results Summary")
    print("=" * 70)
    refs = {
        'WaveSplat': 0.223, 'ConvCNP v2': 0.244,
        'FNO (no phys)': 0.2334, 'PINO v1 (spec only)': 0.2220,
    }
    for m, v in refs.items():
        tag = '<< beats WaveSplat!' if v < 0.223 else ''
        print(f"  {m:<28}  {v:.4f}  {tag}")
    print(f"  {'Temporal PINO (this run)':<28}  {avg_test:.4f}  "
          f"{'<< beats WaveSplat!' if avg_test < 0.223 else ''}")

    delta_vs_pino1 = avg_test - 0.2220
    print(f"\n  vs PINO v1 (spec-only): {delta_vs_pino1:+.4f} "
          f"({'better' if delta_vs_pino1 < 0 else 'worse'})")

    total = time.time() - t_start
    result = {
        'method': 'WavePINO Stage 3 (temporal dispersion loss)',
        'best_val_nrmse': best_val,
        'test_avg_nrmse': avg_test,
        'test_per_episode': test_results,
        'config': {
            'fno_width': FNO_WIDTH, 'fno_layers': FNO_LAYERS,
            'fno_modes': f'{FNO_MODES_H}x{FNO_MODES_W}',
            'epochs': EPOCHS, 'batch_size': BATCH_SIZE, 'lr': LR,
            'lambda_corr': LAMBDA_CORR, 'lambda_spec': LAMBDA_SPEC,
            'lambda_temporal': LAMBDA_TEMP, 'frame_gap': FRAME_GAP,
            'dt_s': FRAME_GAP * 0.1, 'depth': DEPTH,
        },
        'references': {
            'WaveSplat': 0.223, 'ConvCNP_v2': 0.244,
            'FNO_no_phys': 0.2334, 'PINO_v1_spec': 0.2220,
        },
        'delta_vs_pino1': delta_vs_pino1,
        'total_time_s': total,
    }

    out = os.path.join(results_dir, 'pino_stage3_temporal.json')
    with open(out, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"\n  Results saved: {out}")


if __name__ == '__main__':
    main()
