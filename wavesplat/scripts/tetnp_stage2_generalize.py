"""
tetnp_stage2_generalize.py

TE-TNP Stage 2: Cross-episode generalization training.

Architecture: full model (EMBED=128, 5 layers, 8 heads)
Train: 92 episodes (BF2-5, seed0-3, uniform+slope) — same split as KalmanNet/ConvCNP
Val:   seed4 episodes (8 eps)
Test:  6 standard episodes matching WaveSplat baseline

Goal: test avg nRMSE < 0.244 (beat ConvCNP), ideally < 0.223 (beat WaveSplat)
"""
import json
import os
import sys
import time
from glob import glob

import h5py
import numpy as np
import torch
import torch.optim as optim

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(REPO_ROOT, 'external', 'tetnp'))
sys.path.insert(0, os.path.join(REPO_ROOT, 'external', 'gridded-tnp'))

from tetnp.models.tetnp import TETNP, TETNPEncoder
from tetnp.networks.tetransformer import TETNPTransformerEncoder
from tetnp.networks.teattention_layers import (
    MultiHeadCrossTEAttentionLayer, MultiHeadSelfTEAttentionLayer)
from tnp.models.tnp import TNPDecoder
from tnp.networks.mlp import MLP
from tnp.likelihoods.gaussian import HeteroscedasticNormalLikelihood

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DATA_DIR = os.path.join(REPO_ROOT, 'wavesplat', 'data')
CKPT_PATH = os.path.join(REPO_ROOT, 'wavesplat', 'checkpoints', 'tetnp_stage2_best.pt')
os.makedirs(os.path.dirname(CKPT_PATH), exist_ok=True)

# ── Model ──────────────────────────────────────────────────────────────────────
DIM_X    = 3
DIM_Y    = 1
EMBED    = 128
HEADS    = 8
HEAD_DIM = 16
N_LAYERS = 5

# ── Training ───────────────────────────────────────────────────────────────────
LR         = 5e-4
N_STEPS    = 15000
VAL_EVERY  = 1500
LOG_EVERY  = 500
BATCH_SIZE = 4
WINDOW     = 3
NC         = 128     # context points per task (fixed, no padding)
NT         = 512     # target points per task (fixed)
ETA_SCALE  = 0.5

TEST_EPISODES = [
    'episode_BF3_5comp_seed0_slope.h5',
    'episode_BF4_5comp_seed0_uniform.h5',
    'episode_BF4_15comp_seed0_uniform.h5',
    'episode_BF4_30comp_seed0_uniform.h5',
    'episode_BF5_15comp_seed0_uniform.h5',
    'episode_BF5_30comp_seed0_uniform.h5',
]


def build_model():
    mhca_kernel = MLP(in_dim=HEADS+DIM_X, out_dim=HEADS, num_layers=2, width=EMBED)
    mhsa_kernel = MLP(in_dim=HEADS+DIM_X, out_dim=HEADS, num_layers=2, width=EMBED)
    mhca_layer  = MultiHeadCrossTEAttentionLayer(
        embed_dim=EMBED, num_heads=HEADS, head_dim=HEAD_DIM,
        kernel=mhca_kernel, feedforward_dim=EMBED, norm_first=True)
    mhsa_layer  = MultiHeadSelfTEAttentionLayer(
        embed_dim=EMBED, num_heads=HEADS, head_dim=HEAD_DIM,
        kernel=mhsa_kernel, feedforward_dim=EMBED, norm_first=True)
    transformer = TETNPTransformerEncoder(
        num_layers=N_LAYERS, mhca_layer=mhca_layer, mhsa_layer=mhsa_layer)
    y_encoder   = MLP(in_dim=DIM_Y+1, out_dim=EMBED, num_layers=2, width=EMBED)
    encoder     = TETNPEncoder(transformer_encoder=transformer, y_encoder=y_encoder)
    z_decoder   = MLP(in_dim=EMBED, out_dim=2*DIM_Y, num_layers=2, width=EMBED)
    decoder     = TNPDecoder(z_decoder=z_decoder)
    return TETNP(encoder=encoder, decoder=decoder,
                 likelihood=HeteroscedasticNormalLikelihood())


def load_episode(path):
    with h5py.File(path, 'r') as f:
        return dict(
            gt_grid_x  = f['frames/gt_grid_x'][:],
            gt_grid_y  = f['frames/gt_grid_y'][:],
            gt_wave    = f['frames/gt_wave_field'][:],
            lidar_pts  = f['frames/lidar_points'][:],
            lidar_mask = f['frames/lidar_valid_mask'][:],
            times      = f['frames/timestamps'][:],
        )


def norm_xy(x, y, x_min, x_max, y_min, y_max):
    return (2*(x-x_min)/(x_max-x_min)-1,
            2*(y-y_min)/(y_max-y_min)-1)


def sample_task(ep, rng, x_min, x_max, y_min, y_max):
    T = ep['gt_wave'].shape[0]
    t0 = int(rng.integers(0, T - WINDOW))
    ts = ep['times'][t0:t0+WINDOW]
    t_min, t_sc = float(ts[0]), max(float(ts[-1]-ts[0]), 1e-6)

    cx, cy, ct, ce = [], [], [], []
    for k in range(WINDOW):
        mask = ep['lidar_mask'][t0+k]
        pts  = ep['lidar_pts'][t0+k][mask]
        if len(pts) == 0: continue
        cx.append(pts[:,0]); cy.append(pts[:,1])
        ct.append(np.full(len(pts), (ts[k]-t_min)/t_sc))
        ce.append(pts[:,2])
    if not cx:
        cx=[np.zeros(1)]; cy=[np.zeros(1)]; ct=[np.zeros(1)]; ce=[np.zeros(1)]
    cx=np.concatenate(cx); cy=np.concatenate(cy)
    ct=np.concatenate(ct); ce=np.concatenate(ce)

    cxn, cyn = norm_xy(cx, cy, x_min, x_max, y_min, y_max)
    idx = rng.choice(len(cxn), NC, replace=(len(cxn) < NC))
    xc = np.stack([cxn[idx], cyn[idx], ct[idx]], axis=-1).astype(np.float32)
    yc = (ce[idx]/ETA_SCALE)[:,None].astype(np.float32)

    Ny, Nx = ep['gt_wave'].shape[1], ep['gt_wave'].shape[2]
    gxn, gyn = norm_xy(ep['gt_grid_x'], ep['gt_grid_y'], x_min, x_max, y_min, y_max)
    ks = rng.integers(0, WINDOW, NT)
    yi = rng.integers(0, Ny, NT)
    xi = rng.integers(0, Nx, NT)
    xt = np.stack([gxn[xi], gyn[yi], ((ts[ks]-t_min)/t_sc).astype(np.float32)],
                  axis=-1).astype(np.float32)
    yt = ep['gt_wave'][t0+ks, yi, xi].astype(np.float32)[:,None] / ETA_SCALE
    return xc, yc, xt, yt


def make_batch(episodes, rng, x_min, x_max, y_min, y_max):
    tasks = []
    ep_idxs = rng.integers(0, len(episodes), BATCH_SIZE)
    for i in ep_idxs:
        tasks.append(sample_task(episodes[i], rng, x_min, x_max, y_min, y_max))
    xcs = np.stack([t[0] for t in tasks])
    ycs = np.stack([t[1] for t in tasks])
    xts = np.stack([t[2] for t in tasks])
    yts = np.stack([t[3] for t in tasks])
    return (torch.from_numpy(xcs).to(DEVICE),
            torch.from_numpy(ycs).to(DEVICE),
            torch.from_numpy(xts).to(DEVICE),
            torch.from_numpy(yts).to(DEVICE))


def eval_episode(model, ep, x_min, x_max, y_min, y_max, n_windows=20):
    """Evaluate nRMSE on first n_windows non-overlapping windows."""
    model.eval()
    T = ep['gt_wave'].shape[0]
    Ny, Nx = ep['gt_wave'].shape[1], ep['gt_wave'].shape[2]
    gxn, gyn = norm_xy(ep['gt_grid_x'], ep['gt_grid_y'], x_min, x_max, y_min, y_max)
    nrmse_list = []

    with torch.no_grad():
        for t0 in range(0, min(n_windows*WINDOW, T-WINDOW), WINDOW):
            ts = ep['times'][t0:t0+WINDOW]
            t_min, t_sc = float(ts[0]), max(float(ts[-1]-ts[0]), 1e-6)

            cx, cy, ct, ce = [], [], [], []
            for k in range(WINDOW):
                mask = ep['lidar_mask'][t0+k]
                pts  = ep['lidar_pts'][t0+k][mask]
                if len(pts) == 0: continue
                cx.append(pts[:,0]); cy.append(pts[:,1])
                ct.append(np.full(len(pts), (ts[k]-t_min)/t_sc))
                ce.append(pts[:,2])
            if not cx: continue
            cx=np.concatenate(cx); cy=np.concatenate(cy)
            ct=np.concatenate(ct); ce=np.concatenate(ce)
            cxn, cyn = norm_xy(cx, cy, x_min, x_max, y_min, y_max)
            xc_np = np.stack([cxn, cyn, ct], axis=-1).astype(np.float32)
            yc_np = (ce/ETA_SCALE)[:,None].astype(np.float32)

            k_eval = WINDOW // 2
            t_eval = float((ts[k_eval]-t_min)/t_sc)
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

    # Load episodes
    print("Loading episodes...")
    train_paths, val_paths = [], []
    for bf in ['BF2', 'BF3', 'BF4', 'BF5']:
        for seed in range(4):   # seed0-3: train
            for suf in ['uniform', 'slope']:
                p = glob(os.path.join(DATA_DIR, f'episode_{bf}_*comp_seed{seed}_{suf}.h5'))
                train_paths.extend(p)
        for seed in [4]:        # seed4: val
            for suf in ['uniform', 'slope']:
                p = glob(os.path.join(DATA_DIR, f'episode_{bf}_*comp_seed{seed}_{suf}.h5'))
                val_paths.extend(p)

    # Remove test episodes from train
    test_names = set(TEST_EPISODES)
    train_paths = [p for p in train_paths if os.path.basename(p) not in test_names]
    val_paths   = [p for p in val_paths   if os.path.basename(p) not in test_names]
    train_paths = sorted(set(train_paths))
    val_paths   = sorted(set(val_paths))

    print(f"  Train: {len(train_paths)} episodes")
    print(f"  Val:   {len(val_paths)} episodes")

    train_eps = [load_episode(p) for p in train_paths]
    val_eps   = [load_episode(p) for p in val_paths]

    x_min = float(train_eps[0]['gt_grid_x'][0])
    x_max = float(train_eps[0]['gt_grid_x'][-1])
    y_min = float(train_eps[0]['gt_grid_y'][0])
    y_max = float(train_eps[0]['gt_grid_y'][-1])

    # Build model
    model = build_model().to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nTE-TNP params: {n_params:,}  (EMBED={EMBED}, L={N_LAYERS}, H={HEADS})")

    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=N_STEPS, eta_min=1e-5)
    rng = np.random.default_rng(42)

    print(f"\n{'Step':>6}  {'Train NLL':>10}  {'Val NLL':>8}  {'Time':>7}")
    print("-" * 42)

    t_start = time.time()
    best_val_nll = float('inf')
    nll_history = []
    val_nll_history = []

    for step in range(1, N_STEPS + 1):
        model.train()
        xc, yc, xt, yt = make_batch(train_eps, rng, x_min, x_max, y_min, y_max)
        dist = model(xc, yc, xt)
        nll = -dist.log_prob(yt).mean()
        optimizer.zero_grad()
        nll.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()
        scheduler.step()
        nll_history.append(nll.item())

        # Validation
        if step % VAL_EVERY == 0 or step == 1:
            model.eval()
            val_nlls = []
            with torch.no_grad():
                val_rng = np.random.default_rng(0)
                for _ in range(20):  # 20 val batches
                    xc_v, yc_v, xt_v, yt_v = make_batch(val_eps, val_rng, x_min, x_max, y_min, y_max)
                    d_v = model(xc_v, yc_v, xt_v)
                    val_nlls.append(-d_v.log_prob(yt_v).mean().item())
            val_nll = float(np.mean(val_nlls))
            val_nll_history.append((step, val_nll))

            if val_nll < best_val_nll:
                best_val_nll = val_nll
                torch.save({'step': step, 'model': model.state_dict(),
                            'val_nll': val_nll}, CKPT_PATH)

            elapsed = time.time() - t_start
            print(f"{step:>6}  {nll.item():>10.4f}  {val_nll:>8.4f}  {elapsed:>6.1f}s")

        elif step % LOG_EVERY == 0:
            elapsed = time.time() - t_start
            print(f"{step:>6}  {nll.item():>10.4f}  {'':>8}  {elapsed:>6.1f}s")

    # Load best model and evaluate on test set
    print(f"\nLoading best model (val NLL {best_val_nll:.4f})...")
    ckpt = torch.load(CKPT_PATH, map_location=DEVICE)
    model.load_state_dict(ckpt['model'])

    print("\nEvaluating on test episodes...")
    test_results = {}
    for ep_name in TEST_EPISODES:
        ep_path = os.path.join(DATA_DIR, ep_name)
        if not os.path.exists(ep_path):
            print(f"  {ep_name}: NOT FOUND")
            continue
        ep = load_episode(ep_path)
        nrmse = eval_episode(model, ep, x_min, x_max, y_min, y_max, n_windows=30)
        key = ep_name.replace('episode_', '').replace('.h5', '')
        test_results[key] = round(nrmse, 4)
        print(f"  {key}: nRMSE={nrmse:.4f}")

    avg_nrmse = float(np.mean(list(test_results.values())))
    total_time = time.time() - t_start
    print(f"\nTest avg nRMSE: {avg_nrmse:.4f}")
    print(f"WaveSplat:      0.223")
    print(f"ConvCNP v2:     0.244")

    results = {
        'test_avg_nrmse': round(avg_nrmse, 4),
        'best_val_nll': round(best_val_nll, 4),
        'test_per_episode': test_results,
        'training': {
            'steps': N_STEPS, 'lr': LR, 'batch_size': BATCH_SIZE,
            'nc': NC, 'nt': NT, 'embed': EMBED, 'heads': HEADS,
            'layers': N_LAYERS, 'params': n_params,
        },
        'total_time_s': round(total_time, 1),
    }
    out_path = os.path.join(REPO_ROOT, 'wavesplat', 'results', 'tetnp_stage2_generalize.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved: {out_path}")


if __name__ == '__main__':
    main()
