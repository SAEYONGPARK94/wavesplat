"""Amortized field regressor + UQ, trained on the EPISODE-MATCHED obs distribution.

The existing amortized nets mean-collapse to ~0.25 on the real episodes — obs-distribution
shift (trained on uniform-box obs). Here we train AmortizedFF (Fourier-feature transformer,
heteroscedastic Gaussian head) on obs sampled from the REAL LiDAR footprint (pool_xy),
varying t-span (K_TP·Tp) and nc∈{5,15,30}, so it sees the deployment distribution. Gives a
sharp field point estimate + aleatoric UQ in one forward pass (a single-network ★1).
Whether it stays sharp on simple seas tests if the true posterior mean is sharp there
(vs the over-dispersed DPS posterior).

Run: python3 scripts/d3_amortized_field.py [--steps 5000] [--eval_only ckpt.pt]
"""
import os, sys, json, time, argparse
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
os.chdir(os.path.join(os.path.dirname(__file__), '..'))

from wavesim.wave_generator import jonswap_components
from wavesplat.data_loader import EpisodeDataLoader
from amortized_ff import AmortizedFF, XS, YS, TS
from d6_integrated_eval import tag_to_file
from d15_sequential_fusion import gather_frames
from d23_main_results import K_TP, panel
from d40_learned_acquisition import pool_xy

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
G = 9.81; DEPTH = 50.0
GW, GH = 28, 20
M_OBS = 400


def make_batch(B, pool, sigma=0.02):
    """obs from the REAL footprint (pool_xy), varying t-span; GT field on a grid."""
    x0, x1 = pool[:, 0].min(), pool[:, 0].max(); y0, y1 = pool[:, 1].min(), pool[:, 1].max()
    gx = np.linspace(x0, x1, GW); gy = np.linspace(y0, y1, GH)
    Xg, Yg = np.meshgrid(gx, gy)
    obs = np.zeros((B, M_OBS, 4), np.float32)
    q = np.zeros((B, GH * GW, 3), np.float32)
    gt = np.zeros((B, GH * GW), np.float32)
    Hs_b = np.zeros(B, np.float32)
    for b in range(B):
        Hs = np.random.uniform(0.5, 3.5); Tp = np.random.uniform(5.0, 12.0)
        nc = int(np.random.choice([5, 15, 30]))
        comp = jonswap_components(Hs=Hs, Tp=Tp, n_components=nc, depth=DEPTH, seed=int(np.random.randint(1e9)))
        A = np.array([c.A for c in comp]); om = np.array([c.omega for c in comp])
        kx = np.array([c.kx for c in comp]); ky = np.array([c.ky for c in comp]); ph = np.array([c.phi for c in comp])
        Tobs = K_TP * Tp; tstar = Tobs
        idx = np.random.choice(len(pool), M_OBS, replace=len(pool) < M_OBS)
        ox, oy = pool[idx, 0], pool[idx, 1]; ot = np.random.uniform(0.0, Tobs, M_OBS)
        psi = kx[None] * ox[:, None] + ky[None] * oy[:, None] - om[None] * ot[:, None] + ph[None]
        oz = (A[None] * np.cos(psi)).sum(1) + np.random.normal(0, sigma, M_OBS)
        obs[b] = np.column_stack([ox, oy, ot, oz])
        psg = (kx[None, None] * Xg[..., None] + ky[None, None] * Yg[..., None] - om[None, None] * tstar + ph[None, None])
        gt[b] = (A[None, None] * np.cos(psg)).sum(-1).ravel()
        q[b] = np.column_stack([Xg.ravel(), Yg.ravel(), np.full(GH * GW, tstar)])
        Hs_b[b] = Hs
    t = lambda a: torch.tensor(a, device=DEVICE)
    return t(obs), t(q), t(gt), t(Hs_b)


def train(steps, B=32, lr=3e-4):
    pool = pool_xy().astype(np.float32)
    model = AmortizedFF(uq=True).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    t0 = time.time()
    for s in range(steps):
        obs, q, gt, Hs = make_batch(B, pool)
        mean, logvar = model(obs, q)
        logvar = logvar.clamp(-8, 4)
        # heteroscedastic Gaussian NLL, Hs-normalised
        nll = (0.5 * torch.exp(-logvar) * (mean - gt) ** 2 + 0.5 * logvar)
        loss = (nll / (Hs[:, None] ** 2 + 1e-6)).mean()
        rmse = (((mean - gt) ** 2).mean(1).sqrt() / Hs).mean()          # ~nRMSE monitor
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        if s % 250 == 0 or s == steps - 1:
            print(f"  step {s:4d}  loss {loss.item():.4f}  nRMSE {rmse.item():.4f}  ({time.time()-t0:.0f}s)", flush=True)
    return model


@torch.no_grad()
def evaluate(model, episodes, m_obs=400):
    model.eval(); out = {}
    for tag in episodes:
        L = EpisodeDataLoader('data/' + tag_to_file(tag)); Hs = L.Hs_true
        gx = L.gt_grid_x.numpy(); gy = L.gt_grid_y.numpy(); nb = int(round(K_TP * L.Tp_true / 0.1))
        i0 = nb + 3; i1 = len(L) - 2; sel = np.linspace(i0, i1, 3).astype(int)
        Xg, Yg = np.meshgrid(gx, gy); nr = []
        for i_star in sel:
            fstar = L[int(i_star)]; P, z = gather_frames(L, int(i_star), nb)
            if len(z) > m_obs:
                idx = np.random.default_rng(0).choice(len(z), m_obs, replace=False); P, z = P[idx], z[idx]
            t0 = P[:, 2].min()
            obs = np.column_stack([P[:, 0], P[:, 1], P[:, 2] - t0, z]).astype(np.float32)
            q = np.column_stack([Xg.ravel(), Yg.ravel(), np.full(Xg.size, float(fstar.t) - t0)]).astype(np.float32)
            mean, _ = model(torch.tensor(obs[None], device=DEVICE), torch.tensor(q[None], device=DEVICE))
            fld = mean[0].cpu().numpy().reshape(len(gy), len(gx))
            nr.append(panel(fld, fstar.gt_field.numpy(), Hs)['nrmse'])
        out[tag] = float(np.mean(nr)); print(f"  {tag:12s} amortized-field nRMSE {out[tag]:.3f}", flush=True)
    return out


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--steps', type=int, default=5000)
    ap.add_argument('--ckpt', default='results/direction3/amortized_field_matched.pt')
    ap.add_argument('--eval_only', default=None)
    a = ap.parse_args()
    eps = ['BF3_5c_s0', 'BF4_5c_s0', 'BF4_15c_s0', 'BF4_30c_s0', 'BF5_15c_s0', 'BF5_30c_s1']
    if a.eval_only:
        model = AmortizedFF(uq=True).to(DEVICE); model.load_state_dict(torch.load(a.eval_only, map_location=DEVICE))
    else:
        print(f"training {a.steps} steps on {DEVICE} (episode-matched amortized field+UQ)...")
        model = train(a.steps); torch.save(model.state_dict(), a.ckpt); print(f"saved {a.ckpt}")
    print("\n=== eval (amortized field vs DPS-BMA simple~0.20, RELAX simple 0.05-0.09) ===")
    res = evaluate(model, eps); print(f"mean nRMSE: {np.mean(list(res.values())):.3f}")
    json.dump(res, open('results/direction3/amortized_field_matched_eval.json', 'w'), indent=2)
