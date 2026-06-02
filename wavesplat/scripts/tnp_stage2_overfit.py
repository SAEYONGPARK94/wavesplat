"""
tnp_stage2_overfit.py

Stage 2: Overfit Test — Go/No-go #2
1 episode, train=val (intentional), vanilla TNP, GPU accelerated.

Goal: can TNP *memorize* WaveSplat tasks? Loss → near 0?
"""
import os
import sys
import time
import json
import math
from typing import List

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ─────────────────────────────────────────────────────────────
# Minimal Vanilla TNP (no external dependency)
# ─────────────────────────────────────────────────────────────

class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim, n_layers=2):
        super().__init__()
        layers = [nn.Linear(in_dim, hidden_dim), nn.ReLU()]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU()]
        layers.append(nn.Linear(hidden_dim, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.Linear(embed_dim * 2, embed_dim),
        )
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, query, key_value, mask=None):
        # Cross-attention (or self-attention if query == key_value)
        attn_out, _ = self.attn(query, key_value, key_value, key_padding_mask=mask)
        x = self.norm1(query + attn_out)
        x = self.norm2(x + self.ff(x))
        return x


class VanillaTNP(nn.Module):
    """Minimal TNP: MLP encoder → (MHSA on context, MHCA target←context) × L → MLP decoder → (mu, sigma)."""

    def __init__(self, dim_x=3, dim_y=1, embed_dim=128, num_heads=4, num_layers=4):
        super().__init__()
        # Encode context: (x, y, indicator=0) → embed
        self.context_encoder = MLP(dim_x + dim_y + 1, embed_dim, embed_dim)
        # Encode target: (x, 0, indicator=1) → embed
        self.target_encoder = MLP(dim_x + dim_y + 1, embed_dim, embed_dim)

        self.self_attn_layers = nn.ModuleList([
            MultiHeadAttention(embed_dim, num_heads) for _ in range(num_layers)
        ])
        self.cross_attn_layers = nn.ModuleList([
            MultiHeadAttention(embed_dim, num_heads) for _ in range(num_layers)
        ])

        # Decoder: embed → (mu, log_sigma)
        self.decoder = MLP(embed_dim, 2 * dim_y, embed_dim)
        self.min_sigma = 1e-3

    def forward(self, xc, yc, xt, context_mask=None):
        """
        xc: (B, Nc, dim_x)   yc: (B, Nc, dim_y)
        xt: (B, Nt, dim_x)
        context_mask: (B, Nc) True = padded/invalid
        Returns: mu (B, Nt, 1), sigma (B, Nt, 1)
        """
        B, Nc, dx = xc.shape
        Nt = xt.shape[1]
        dy = yc.shape[2]

        # Build input tokens with indicator channel
        ctx_input = torch.cat([xc, yc, torch.zeros(B, Nc, 1, device=xc.device)], dim=-1)
        tgt_input = torch.cat([xt, torch.zeros(B, Nt, dy, device=xc.device),
                               torch.ones(B, Nt, 1, device=xc.device)], dim=-1)

        zc = self.context_encoder(ctx_input)   # (B, Nc, E)
        zt = self.target_encoder(tgt_input)    # (B, Nt, E)

        for sa, ca in zip(self.self_attn_layers, self.cross_attn_layers):
            zc = sa(zc, zc, mask=context_mask)
            zt = ca(zt, zc, mask=context_mask)

        out = self.decoder(zt)  # (B, Nt, 2)
        mu = out[..., :dy]
        log_sigma = out[..., dy:]
        sigma = F.softplus(log_sigma) + self.min_sigma

        return mu, sigma

    def nll_loss(self, xc, yc, xt, yt, context_mask=None):
        mu, sigma = self(xc, yc, xt, context_mask)
        dist = torch.distributions.Normal(mu, sigma)
        return -dist.log_prob(yt).mean()


# ─────────────────────────────────────────────────────────────
# Dataset (simplified, no external dep)
# ─────────────────────────────────────────────────────────────

def load_episode(path):
    with h5py.File(path, 'r') as f:
        return dict(
            gt_grid_x=f['frames/gt_grid_x'][:],
            gt_grid_y=f['frames/gt_grid_y'][:],
            gt_wave=f['frames/gt_wave_field'][:],
            lidar_pts=f['frames/lidar_points'][:],
            lidar_mask=f['frames/lidar_valid_mask'][:],
            times=f['frames/timestamps'][:],
        )


class WaveTNPDataset(torch.utils.data.Dataset):
    """Fixed task list for overfit test. Each task = 3-frame window."""

    def __init__(self, episodes: List[dict], window=3, stride=3,
                 max_context=256, max_target=2048, seed=42):
        self.episodes = episodes
        self.window = window
        self.max_context = max_context
        self.max_target = max_target

        # Build fixed task list
        self.tasks = []
        for ep_idx, ep in enumerate(episodes):
            T = ep['gt_wave'].shape[0]
            for t0 in range(0, T - window, stride):
                self.tasks.append((ep_idx, t0))

        # Domain extents for normalization
        ep0 = episodes[0]
        self.x_min, self.x_max = float(ep0['gt_grid_x'][0]), float(ep0['gt_grid_x'][-1])
        self.y_min, self.y_max = float(ep0['gt_grid_y'][0]), float(ep0['gt_grid_y'][-1])
        self.rng = np.random.default_rng(seed)

    def _norm_xy(self, x, y):
        xn = 2 * (x - self.x_min) / (self.x_max - self.x_min) - 1
        yn = 2 * (y - self.y_min) / (self.y_max - self.y_min) - 1
        return xn, yn

    def __len__(self):
        return len(self.tasks)

    def __getitem__(self, idx):
        ep_idx, t0 = self.tasks[idx]
        ep = self.episodes[ep_idx]
        ts = ep['times'][t0:t0 + self.window]
        t_min, t_scale = float(ts[0]), max(float(ts[-1] - ts[0]), 1e-6)

        # Context: LiDAR points
        cx, cy, ct, ceta = [], [], [], []
        for k in range(self.window):
            mask = ep['lidar_mask'][t0 + k].astype(bool)
            pts = ep['lidar_pts'][t0 + k][mask]
            if len(pts) == 0:
                continue
            cx.append(pts[:, 0]); cy.append(pts[:, 1])
            ct.append(np.full(len(pts), (ts[k] - t_min) / t_scale))
            ceta.append(pts[:, 2])

        cx = np.concatenate(cx) if cx else np.zeros(0)
        cy = np.concatenate(cy) if cy else np.zeros(0)
        ct = np.concatenate(ct) if ct else np.zeros(0)
        ceta = np.concatenate(ceta) if ceta else np.zeros(0)
        cx, cy = self._norm_xy(cx, cy)

        # Cap context
        if len(cx) > self.max_context:
            sel = self.rng.choice(len(cx), self.max_context, replace=False)
            cx, cy, ct, ceta = cx[sel], cy[sel], ct[sel], ceta[sel]

        # Target: GT grid (subsample for memory)
        gx, gy = ep['gt_grid_x'], ep['gt_grid_y']
        gxn, gyn = self._norm_xy(gx, gy)
        Ny, Nx = ep['gt_wave'].shape[1], ep['gt_wave'].shape[2]
        total_grid = self.window * Ny * Nx
        n_tgt = min(self.max_target, total_grid)

        ks = self.rng.integers(0, self.window, n_tgt)
        yi = self.rng.integers(0, Ny, n_tgt)
        xi = self.rng.integers(0, Nx, n_tgt)
        tx = gxn[xi].astype(np.float32)
        ty = gyn[yi].astype(np.float32)
        tt = ((ts[ks] - t_min) / t_scale).astype(np.float32)
        teta = ep['gt_wave'][t0 + ks, yi, xi].astype(np.float32)

        # Normalize eta
        eta_std = max(np.std(np.concatenate([ceta, teta])), 1e-4)

        return dict(
            xc=np.stack([cx, cy, ct], -1).astype(np.float32),
            yc=(ceta / eta_std).astype(np.float32)[:, None],
            xt=np.stack([tx, ty, tt], -1).astype(np.float32),
            yt=(teta / eta_std).astype(np.float32)[:, None],
            eta_std=np.float32(eta_std),
            n_context=len(cx),
        )


def collate_fn(batch):
    """Pad context to max Nc in batch."""
    Nc_max = max(b['n_context'] for b in batch)
    Nt = batch[0]['xt'].shape[0]
    B = len(batch)

    xc = torch.zeros(B, Nc_max, 3)
    yc = torch.zeros(B, Nc_max, 1)
    mask = torch.ones(B, Nc_max, dtype=torch.bool)  # True = padded
    xt = torch.zeros(B, Nt, 3)
    yt = torch.zeros(B, Nt, 1)

    for i, b in enumerate(batch):
        nc = b['n_context']
        xc[i, :nc] = torch.from_numpy(b['xc'])
        yc[i, :nc] = torch.from_numpy(b['yc'])
        mask[i, :nc] = False
        xt[i] = torch.from_numpy(b['xt'])
        yt[i] = torch.from_numpy(b['yt'])

    return xc, yc, xt, yt, mask


# ─────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────

def train_overfit(
    episode_paths: List[str],
    epochs: int = 200,
    batch_size: int = 4,
    lr: float = 1e-4,
    embed_dim: int = 128,
    num_heads: int = 4,
    num_layers: int = 4,
    max_target: int = 2048,
    device: str = 'cuda',
):
    print(f'Loading {len(episode_paths)} episodes...')
    episodes = [load_episode(p) for p in episode_paths]

    dataset = WaveTNPDataset(episodes, window=3, stride=3, max_target=max_target)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn,
        num_workers=0, drop_last=False,
    )
    print(f'Tasks: {len(dataset)}, batches/epoch: {len(loader)}')

    model = VanillaTNP(
        dim_x=3, dim_y=1,
        embed_dim=embed_dim, num_heads=num_heads, num_layers=num_layers,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f'Model: VanillaTNP, {n_params:,} params ({n_params/1e6:.2f}M)')

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_loss = float('inf')
    history = []
    t_start = time.time()

    for epoch in range(epochs):
        model.train()
        epoch_losses = []
        for xc, yc, xt, yt, mask in loader:
            xc, yc, xt, yt, mask = [t.to(device) for t in [xc, yc, xt, yt, mask]]
            optimizer.zero_grad()
            loss = model.nll_loss(xc, yc, xt, yt, context_mask=mask)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_losses.append(loss.item())

        scheduler.step()
        avg_loss = np.mean(epoch_losses)
        history.append(avg_loss)

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), '/tmp/tnp_stage2_best.pt')

        if (epoch + 1) % 10 == 0 or epoch == 0:
            elapsed = time.time() - t_start
            print(f'  epoch {epoch+1:4d}/{epochs}: NLL={avg_loss:.4f}  best={best_loss:.4f}  '
                  f'lr={scheduler.get_last_lr()[0]:.2e}  [{elapsed:.0f}s]')

    elapsed = time.time() - t_start
    print(f'\nTraining done in {elapsed:.0f}s. Best NLL: {best_loss:.4f}')

    # ── Evaluation: reconstruction quality ──
    print('\n── Evaluation (train set, full grid) ──')
    model.load_state_dict(torch.load('/tmp/tnp_stage2_best.pt'))
    model.eval()

    eval_results = evaluate_reconstruction(model, episodes, device)
    for k, v in eval_results.items():
        if isinstance(v, list):
            print(f'  {k}: {[round(x, 4) for x in v]}')
        else:
            print(f'  {k}: {v:.4f}')

    # Save results
    results = {
        'stage': 'Stage 2 overfit',
        'episodes': [os.path.basename(p) for p in episode_paths],
        'n_tasks': len(dataset),
        'n_params': n_params,
        'epochs': epochs,
        'best_nll': best_loss,
        'final_nll': history[-1],
        'training_time_s': elapsed,
        'eval': eval_results,
        'loss_history_every10': [history[i] for i in range(0, len(history), 10)],
    }
    out_path = os.path.join(os.path.dirname(__file__), '..', 'results', 'tnp_stage2_overfit.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'\nResults saved: {out_path}')

    return results


def evaluate_reconstruction(model, episodes, device, n_frames=5, window=3):
    """Evaluate on a few frames: predict full grid from LiDAR context."""
    ep = episodes[0]
    gx, gy = ep['gt_grid_x'], ep['gt_grid_y']
    x_min, x_max = float(gx[0]), float(gx[-1])
    y_min, y_max = float(gy[0]), float(gy[-1])
    Ny, Nx = ep['gt_wave'].shape[1], ep['gt_wave'].shape[2]

    nrmses, amp_ratios = [], []

    for frame_idx in range(10, 10 + n_frames * window, window):
        ts = ep['times'][frame_idx:frame_idx + window]
        t_min, t_scale = float(ts[0]), max(float(ts[-1] - ts[0]), 1e-6)

        # Context: LiDAR
        cx, cy, ct, ceta = [], [], [], []
        for k in range(window):
            mask = ep['lidar_mask'][frame_idx + k].astype(bool)
            pts = ep['lidar_pts'][frame_idx + k][mask]
            if len(pts) == 0:
                continue
            cx.append(pts[:, 0]); cy.append(pts[:, 1])
            ct.append(np.full(len(pts), (ts[k] - t_min) / t_scale))
            ceta.append(pts[:, 2])

        cx = np.concatenate(cx); cy = np.concatenate(cy)
        ct = np.concatenate(ct); ceta = np.concatenate(ceta)
        cxn = 2 * (cx - x_min) / (x_max - x_min) - 1
        cyn = 2 * (cy - y_min) / (y_max - y_min) - 1

        eta_std = max(np.std(ceta), 1e-4)

        xc = torch.tensor(np.stack([cxn, cyn, ct], -1)[None], dtype=torch.float32, device=device)
        yc = torch.tensor((ceta / eta_std)[None, :, None], dtype=torch.float32, device=device)

        # Target: full grid for middle frame
        mid_k = window // 2
        t_val = (ts[mid_k] - t_min) / t_scale
        gt_field = ep['gt_wave'][frame_idx + mid_k]  # (Ny, Nx)

        gxn = 2 * (gx - x_min) / (x_max - x_min) - 1
        gyn = 2 * (gy - y_min) / (y_max - y_min) - 1
        xx, yy = np.meshgrid(gxn, gyn)
        tx = xx.ravel().astype(np.float32)
        ty = yy.ravel().astype(np.float32)
        tt = np.full_like(tx, t_val)

        xt = torch.tensor(np.stack([tx, ty, tt], -1)[None], dtype=torch.float32, device=device)

        with torch.no_grad():
            mu, sigma = model(xc, yc, xt)

        pred = mu[0, :, 0].cpu().numpy() * eta_std
        gt = gt_field.ravel()

        gt_std = max(np.std(gt), 1e-6)
        nrmse = np.sqrt(np.mean((pred - gt)**2)) / gt_std
        amp_ratio = np.std(pred) / gt_std

        nrmses.append(nrmse)
        amp_ratios.append(amp_ratio)

    return {
        'avg_nrmse': float(np.mean(nrmses)),
        'avg_amp_ratio': float(np.mean(amp_ratios)),
        'per_frame_nrmse': [float(x) for x in nrmses],
    }


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--episodes', nargs='+', default=None,
                        help='Episode paths. Default: 3 BF4 episodes')
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--embed_dim', type=int, default=128)
    parser.add_argument('--num_layers', type=int, default=4)
    parser.add_argument('--max_target', type=int, default=2048)
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Device: {device}')

    if args.episodes is None:
        data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
        from glob import glob
        # Use 3 episodes for overfit test
        eps = sorted(glob(os.path.join(data_dir, 'episode_BF4_*comp_seed0_uniform.h5')))[:3]
        if not eps:
            eps = sorted(glob(os.path.join(data_dir, 'episode_BF3_*_seed0_uniform.h5')))[:3]
    else:
        eps = args.episodes

    print(f'Episodes: {[os.path.basename(p) for p in eps]}')
    train_overfit(
        episode_paths=eps,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        embed_dim=args.embed_dim,
        num_heads=4,
        num_layers=args.num_layers,
        max_target=args.max_target,
        device=device,
    )
