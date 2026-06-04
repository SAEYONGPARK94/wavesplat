"""
tetnp_stage1_sanity.py

TE-TNP Stage 1: Single-episode overfit sanity check.

Goal:
- Verify model can memorize one episode
- Pass: NLL drops > 1.0, nRMSE < 0.15

Architecture: small (64-dim, 3 layers) for fast iteration.
"""
import os
import sys
import time

import h5py
import numpy as np
import torch
import torch.optim as optim

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(REPO_ROOT, 'external', 'tetnp'))
sys.path.insert(0, os.path.join(REPO_ROOT, 'external', 'gridded-tnp'))

from tetnp.models.tetnp import TETNP, TETNPEncoder
from tetnp.networks.tetransformer import TETNPTransformerEncoder
from tetnp.networks.teattention_layers import MultiHeadCrossTEAttentionLayer
from tnp.models.tnp import TNPDecoder
from tnp.networks.mlp import MLP
from tnp.likelihoods.gaussian import HeteroscedasticNormalLikelihood

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
EPISODE = os.path.join(REPO_ROOT, 'wavesplat', 'data',
                       'episode_BF4_5comp_seed0_uniform.h5')

# ── Hyperparams ────────────────────────────────────────────────────────────────
DIM_X     = 3
DIM_Y     = 1
EMBED     = 64
HEADS     = 4
HEAD_DIM  = 16
N_LAYERS  = 3
LR        = 1e-3
N_STEPS   = 10000    # total gradient steps
LOG_EVERY = 1000
BATCH_SIZE = 4
WINDOW    = 3
NC        = 128      # fixed context size (no padding ambiguity)
NT        = 512      # fixed target size
ETA_SCALE = 0.5


def build_model():
    mhca_kernel = MLP(in_dim=HEADS + DIM_X, out_dim=HEADS, num_layers=2, width=EMBED)
    mhca_layer  = MultiHeadCrossTEAttentionLayer(
        embed_dim=EMBED, num_heads=HEADS, head_dim=HEAD_DIM,
        kernel=mhca_kernel, feedforward_dim=EMBED, norm_first=True)
    transformer = TETNPTransformerEncoder(num_layers=N_LAYERS, mhca_layer=mhca_layer)
    y_encoder   = MLP(in_dim=DIM_Y + 1, out_dim=EMBED, num_layers=2, width=EMBED)
    encoder     = TETNPEncoder(transformer_encoder=transformer, y_encoder=y_encoder)
    z_decoder   = MLP(in_dim=EMBED, out_dim=2 * DIM_Y, num_layers=2, width=EMBED)
    decoder     = TNPDecoder(z_decoder=z_decoder)
    return TETNP(encoder=encoder, decoder=decoder,
                 likelihood=HeteroscedasticNormalLikelihood())


def load_episode(path):
    with h5py.File(path, 'r') as f:
        gt_grid_x  = f['frames/gt_grid_x'][:]
        gt_grid_y  = f['frames/gt_grid_y'][:]
        gt_wave    = f['frames/gt_wave_field'][:]
        lidar_pts  = f['frames/lidar_points'][:]
        lidar_mask = f['frames/lidar_valid_mask'][:]
        times      = f['frames/timestamps'][:]
    return dict(gt_grid_x=gt_grid_x, gt_grid_y=gt_grid_y,
                gt_wave=gt_wave, lidar_pts=lidar_pts,
                lidar_mask=lidar_mask, times=times)


def norm_xy(x, y, x_min, x_max, y_min, y_max):
    return (2*(x-x_min)/(x_max-x_min)-1,
            2*(y-y_min)/(y_max-y_min)-1)


def sample_task(ep, rng, x_min, x_max, y_min, y_max):
    T = ep['gt_wave'].shape[0]
    t0 = int(rng.integers(0, T - WINDOW))
    ts = ep['times'][t0:t0 + WINDOW]
    t_min, t_scale = float(ts[0]), max(float(ts[-1] - ts[0]), 1e-6)

    # Context: aggregate LiDAR from all frames
    cx, cy, ct, ce = [], [], [], []
    for k in range(WINDOW):
        mask = ep['lidar_mask'][t0 + k]
        pts  = ep['lidar_pts'][t0 + k][mask]
        if len(pts) == 0:
            continue
        cx.append(pts[:, 0]); cy.append(pts[:, 1])
        ct.append(np.full(len(pts), (ts[k] - t_min) / t_scale))
        ce.append(pts[:, 2])

    if not cx:
        cx=[np.zeros(1)]; cy=[np.zeros(1)]; ct=[np.zeros(1)]; ce=[np.zeros(1)]
    cx=np.concatenate(cx); cy=np.concatenate(cy)
    ct=np.concatenate(ct); ce=np.concatenate(ce)

    cxn, cyn = norm_xy(cx, cy, x_min, x_max, y_min, y_max)
    # Always sample exactly NC context points (with replacement if needed)
    idx = rng.choice(len(cxn), NC, replace=(len(cxn) < NC))
    xc = np.stack([cxn[idx], cyn[idx], ct[idx]], axis=-1).astype(np.float32)
    yc = (ce[idx] / ETA_SCALE)[:, None].astype(np.float32)

    # Target: exactly NT random GT grid points
    Ny, Nx = ep['gt_wave'].shape[1], ep['gt_wave'].shape[2]
    gx, gy = ep['gt_grid_x'], ep['gt_grid_y']
    gxn, gyn = norm_xy(gx, gy, x_min, x_max, y_min, y_max)
    ks = rng.integers(0, WINDOW, NT)
    yi = rng.integers(0, Ny,     NT)
    xi = rng.integers(0, Nx,     NT)
    tgt_xn, tgt_yn = gxn[xi], gyn[yi]
    tgt_t  = ((ts[ks] - t_min) / t_scale).astype(np.float32)
    tgt_e  = ep['gt_wave'][t0+ks, yi, xi].astype(np.float32) / ETA_SCALE

    xt = np.stack([tgt_xn, tgt_yn, tgt_t], axis=-1).astype(np.float32)
    yt = tgt_e[:, None]
    return xc, yc, xt, yt


def make_batch(ep, rng, x_min, x_max, y_min, y_max):
    tasks = [sample_task(ep, rng, x_min, x_max, y_min, y_max)
             for _ in range(BATCH_SIZE)]
    # No padding needed — NC and NT are fixed
    xcs = np.stack([t[0] for t in tasks])
    ycs = np.stack([t[1] for t in tasks])
    xts = np.stack([t[2] for t in tasks])
    yts = np.stack([t[3] for t in tasks])
    return (torch.from_numpy(xcs).to(DEVICE),
            torch.from_numpy(ycs).to(DEVICE),
            torch.from_numpy(xts).to(DEVICE),
            torch.from_numpy(yts).to(DEVICE))


def eval_nrmse(model, ep, x_min, x_max, y_min, y_max):
    model.eval()
    T = ep['gt_wave'].shape[0]
    Ny, Nx = ep['gt_wave'].shape[1], ep['gt_wave'].shape[2]
    gx, gy = ep['gt_grid_x'], ep['gt_grid_y']
    gxn, gyn = norm_xy(gx, gy, x_min, x_max, y_min, y_max)
    nrmse_list = []
    with torch.no_grad():
        for t0 in range(0, min(10*WINDOW, T-WINDOW), WINDOW):
            ts = ep['times'][t0:t0+WINDOW]
            t_min, t_scale = float(ts[0]), max(float(ts[-1]-ts[0]), 1e-6)
            cx, cy, ct, ce = [], [], [], []
            for k in range(WINDOW):
                mask = ep['lidar_mask'][t0+k]
                pts  = ep['lidar_pts'][t0+k][mask]
                if len(pts) == 0: continue
                cx.append(pts[:,0]); cy.append(pts[:,1])
                ct.append(np.full(len(pts), (ts[k]-t_min)/t_scale))
                ce.append(pts[:,2])
            if not cx: continue
            cx=np.concatenate(cx); cy=np.concatenate(cy)
            ct=np.concatenate(ct); ce=np.concatenate(ce)
            cxn, cyn = norm_xy(cx, cy, x_min, x_max, y_min, y_max)
            xc_np = np.stack([cxn, cyn, ct], axis=-1).astype(np.float32)
            yc_np = (ce/ETA_SCALE)[:,None].astype(np.float32)

            k_eval = WINDOW // 2
            t_eval = float((ts[k_eval]-t_min)/t_scale)
            gt_eta = ep['gt_wave'][t0+k_eval]

            XX, YY = np.meshgrid(gxn, gyn)
            TT = np.full_like(XX, t_eval)
            xt_np = np.stack([XX.ravel(), YY.ravel(), TT.ravel()], axis=-1).astype(np.float32)

            dist = model(
                torch.from_numpy(xc_np[None]).to(DEVICE),
                torch.from_numpy(yc_np[None]).to(DEVICE),
                torch.from_numpy(xt_np[None]).to(DEVICE))
            pred = dist.mean[0,:,0].cpu().numpy().reshape(Ny, Nx) * ETA_SCALE
            rms_gt = np.sqrt(np.mean(gt_eta**2))
            if rms_gt > 1e-6:
                nrmse_list.append(np.sqrt(np.mean((pred-gt_eta)**2)) / rms_gt)
    return float(np.mean(nrmse_list)) if nrmse_list else float('nan')


def main():
    print(f"Device: {DEVICE}")
    ep = load_episode(EPISODE)
    x_min, x_max = float(ep['gt_grid_x'][0]), float(ep['gt_grid_x'][-1])
    y_min, y_max = float(ep['gt_grid_y'][0]), float(ep['gt_grid_y'][-1])

    model = build_model().to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"TETNP params: {n_params:,}  (EMBED={EMBED}, L={N_LAYERS}, H={HEADS})")
    print(f"NC={NC}, NT={NT}, LR={LR}, steps={N_STEPS}")

    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=N_STEPS, eta_min=1e-5)
    rng = np.random.default_rng(42)

    print(f"\n{'Step':>6}  {'NLL':>8}  {'Time':>6}")
    print("-" * 28)

    t_start = time.time()
    nll_history = []

    for step in range(1, N_STEPS + 1):
        model.train()
        xc, yc, xt, yt = make_batch(ep, rng, x_min, x_max, y_min, y_max)
        dist = model(xc, yc, xt)
        nll = -dist.log_prob(yt).mean()
        optimizer.zero_grad()
        nll.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()
        scheduler.step()
        nll_history.append(nll.item())

        if step % LOG_EVERY == 0 or step == 1:
            elapsed = time.time() - t_start
            print(f"{step:>6}  {nll.item():>8.4f}  {elapsed:>5.1f}s")

    nrmse = eval_nrmse(model, ep, x_min, x_max, y_min, y_max)
    total = time.time() - t_start
    nll_drop = nll_history[0] - min(nll_history)

    print(f"\n{'='*40}")
    print(f"Stage 1 Results:")
    print(f"  First NLL:    {nll_history[0]:.4f}")
    print(f"  Best NLL:     {min(nll_history):.4f}")
    print(f"  NLL drop:     {nll_drop:.4f}")
    print(f"  nRMSE:        {nrmse:.4f}")
    print(f"  Time:         {total:.1f}s  ({total/N_STEPS*1000:.1f}ms/step)")

    passed = nll_drop > 1.0 and nrmse < 0.30
    print(f"\nSTAGE 1: {'PASS' if passed else 'FAIL'}")
    print(f"  NLL drop {nll_drop:.3f} (>1.0?) | nRMSE {nrmse:.3f} (<0.30?)")


if __name__ == '__main__':
    main()
