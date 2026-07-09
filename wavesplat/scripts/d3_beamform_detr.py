"""A2 — DETR-style set-prediction net (grid-free by construction).

Instead of a heatmap on a k-grid (which quantizes (kx,ky) → smearing), encode the beamform
power map with a CNN, then a transformer DECODER with N learned queries emits a SET of
continuous atoms directly: each query → (kx, ky, amp, presence). Trained with bipartite
(Hungarian) matching between predicted and true component sets — DETR object detection
transplanted to spectral-component estimation. No grid anywhere, so no grid smearing.

Kept components (presence>0.5) → LS fit A,φ → render → nRMSE (same back-end as A1 for a
fair comparison).

Run: python3 scripts/d3_beamform_detr.py [--steps 10000] [--eval_only ckpt.pt]
"""
import os, sys, json, time, argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
os.chdir(os.path.join(os.path.dirname(__file__), '..'))

from wavesim.wave_generator import jonswap_components
from wavesplat.data_loader import EpisodeDataLoader
from wavesplat.model.tnse import fit_amplitudes_phases, render_wave_field
from d6_integrated_eval import tag_to_file
from d15_sequential_fusion import gather_frames
from d23_main_results import K_TP, panel
from d40_learned_acquisition import pool_xy
from d3_beamform_net import beamform, KLIM, NK, M_OBS, DEPTH

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NQ = 48                    # learned queries (max components in train = 30)


class Encoder(nn.Module):
    def __init__(self, d=96):
        super().__init__()
        def blk(i, o, s=1):
            return nn.Sequential(nn.Conv2d(i, o, 3, s, 1), nn.GroupNorm(8, o), nn.GELU())
        self.net = nn.Sequential(blk(1, d), blk(d, d), blk(d, d, 2),      # 64->32
                                 blk(d, d), blk(d, d, 2),                 # 32->16
                                 blk(d, d))
        self.pos = nn.Parameter(torch.randn(1, 16 * 16, d) * 0.02)

    def forward(self, B):
        f = self.net(B[:, None])                 # (B,d,16,16)
        f = f.flatten(2).transpose(1, 2)         # (B,256,d)
        return f + self.pos


class DETRSet(nn.Module):
    def __init__(self, d=96, nq=NQ, layers=4, heads=4):
        super().__init__()
        self.enc = Encoder(d)
        self.query = nn.Parameter(torch.randn(nq, d) * 0.02)
        dl = nn.TransformerDecoderLayer(d, heads, 4 * d, batch_first=True, activation='gelu')
        self.dec = nn.TransformerDecoder(dl, layers)
        self.coord = nn.Linear(d, 2); self.amp = nn.Linear(d, 1); self.pres = nn.Linear(d, 1)

    def forward(self, B):
        mem = self.enc(B)
        q = self.query[None].expand(B.shape[0], -1, -1)
        h = self.dec(q, mem)                     # (B,nq,d)
        kxy = KLIM * torch.tanh(self.coord(h))   # (B,nq,2) in [-KLIM,KLIM]
        amp = F.softplus(self.amp(h))[..., 0]    # (B,nq)
        pres = self.pres(h)[..., 0]              # (B,nq) logit
        return kxy, amp, pres


def make_batch(Bsz, pool):
    """Synthetic JONSWAP obs → beamform map + list of true component sets (K,3)=(kx,ky,A)."""
    P = np.zeros((Bsz, M_OBS, 3), np.float32); Z = np.zeros((Bsz, M_OBS), np.float32)
    comps = []
    for b in range(Bsz):
        Hs = np.random.uniform(0.5, 3.5); Tp = np.random.uniform(5.0, 12.0)
        nc = int(np.random.choice([5, 15, 30]))
        c = jonswap_components(Hs=Hs, Tp=Tp, n_components=nc, depth=DEPTH, seed=int(np.random.randint(1e9)))
        A = np.array([x.A for x in c]); om = np.array([x.omega for x in c])
        kx = np.array([x.kx for x in c]); ky = np.array([x.ky for x in c]); ph = np.array([x.phi for x in c])
        Tobs = K_TP * Tp
        idx = np.random.choice(len(pool), M_OBS, replace=len(pool) < M_OBS)
        ox, oy = pool[idx, 0], pool[idx, 1]; ot = np.random.uniform(0, Tobs, M_OBS)
        psi = kx[None] * ox[:, None] + ky[None] * oy[:, None] - om[None] * ot[:, None] + ph[None]
        Z[b] = (A[None] * np.cos(psi)).sum(1) + np.random.normal(0, 0.02, M_OBS)
        P[b] = np.column_stack([ox, oy, ot])
        comps.append(np.column_stack([kx, ky, A]).astype(np.float32))
    Bmap = beamform(torch.tensor(P, device=DEVICE), torch.tensor(Z, device=DEVICE))
    return Bmap, comps


def set_loss(kxy, amp, pres, comps):
    """DETR bipartite-matching loss. Returns (loss, matched coord RMSE in rad/m)."""
    B, nq, _ = kxy.shape
    total = 0.0; coord_sq = 0.0; ncoord = 0
    for b in range(B):
        tk = torch.tensor(comps[b], device=kxy.device)     # (K,3)
        K = tk.shape[0]
        with torch.no_grad():
            cost = torch.cdist(kxy[b], tk[:, :2])           # (nq,K)
            ri, ci = linear_sum_assignment(cost.cpu().numpy())
        ri = torch.as_tensor(ri, device=kxy.device, dtype=torch.long)
        ci = torch.as_tensor(ci, device=kxy.device, dtype=torch.long)
        lc = ((kxy[b][ri] - tk[ci, :2]) ** 2).sum(-1).mean()
        la = (amp[b][ri] - tk[ci, 2]).abs().mean()
        pres_t = torch.zeros(nq, device=kxy.device); pres_t[ri] = 1.0
        w = torch.ones(nq, device=kxy.device); w[ri] = nq / max(K, 1)   # balance rare positives
        lp = F.binary_cross_entropy_with_logits(pres[b], pres_t, weight=w)
        total = total + lc + 0.2 * la + lp
        coord_sq += ((kxy[b][ri] - tk[ci, :2]) ** 2).sum().item(); ncoord += K
    return total / B, (coord_sq / max(ncoord, 1)) ** 0.5


def train(steps, Bsz=32, lr=3e-4, log=print):
    pool = pool_xy().astype(np.float32)
    model = DETRSet().to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    t0 = time.time()
    for s in range(steps):
        Bmap, comps = make_batch(Bsz, pool)
        kxy, amp, pres = model(Bmap)
        loss, crmse = set_loss(kxy, amp, pres, comps)
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        if s % 200 == 0 or s == steps - 1:
            with torch.no_grad():
                npos = (torch.sigmoid(pres) > 0.5).float().sum(1).mean().item()
            log(f"  step {s:5d}  loss {loss.item():.4f}  coord_rmse {crmse:.4f} rad/m  "
                f"npred {npos:.1f}  ({time.time()-t0:.0f}s)")
    return model


@torch.no_grad()
def extract(model, Bmap, pres_thresh=0.5, min_keep=3, max_keep=35):
    kxy, amp, pres = model(Bmap)
    p = torch.sigmoid(pres[0])
    keep = p > pres_thresh
    if keep.sum() < min_keep:                       # fall back to top-min_keep
        thr = p.topk(min(min_keep, len(p))).values.min(); keep = p >= thr
    idx = torch.where(keep)[0]
    if len(idx) > max_keep:
        idx = idx[torch.argsort(-p[idx])[:max_keep]]
    kx = kxy[0][idx, 0].cpu().numpy(); ky = kxy[0][idx, 1].cpu().numpy()
    return kx.astype(np.float32), ky.astype(np.float32)


@torch.no_grad()
def evaluate(model, episodes, log=print):
    model.eval(); out = {}
    for tag in episodes:
        L = EpisodeDataLoader('data/' + tag_to_file(tag)); depth = L.depth; Hs = L.Hs_true
        gx = L.gt_grid_x.numpy(); gy = L.gt_grid_y.numpy(); nb = int(round(K_TP * L.Tp_true / 0.1))
        i0 = nb + 3; i1 = len(L) - 2; sel = np.linspace(i0, i1, 3).astype(int); nr = []
        for i_star in sel:
            fstar = L[int(i_star)]; P, z = gather_frames(L, int(i_star), nb)
            if len(z) > M_OBS:
                idx = np.random.default_rng(0).choice(len(z), M_OBS, replace=False); P, z = P[idx], z[idx]
            t0 = P[:, 2].min(); Ps = P.copy(); Ps[:, 2] -= t0
            Bmap = beamform(torch.tensor(Ps[None], device=DEVICE, dtype=torch.float32),
                            torch.tensor(z[None], device=DEVICE, dtype=torch.float32))
            kx, ky = extract(model, Bmap)
            if len(kx) == 0:
                nr.append(1.0); continue
            obs = np.column_stack([Ps[:, 0], Ps[:, 1], Ps[:, 2], z]).astype(np.float32)
            A, phi = fit_amplitudes_phases(kx, ky, obs, depth, lambda_reg=0.5)
            fld = render_wave_field(kx, ky, A, phi, depth, gx, gy, float(fstar.t) - t0)
            nr.append(panel(fld, fstar.gt_field.numpy(), Hs)['nrmse'])
        out[tag] = float(np.mean(nr)); log(f"  {tag:12s} DETR nRMSE {out[tag]:.3f}")
    return out


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--steps', type=int, default=10000)
    ap.add_argument('--ckpt', default='results/direction3/beamform_detr.pt')
    ap.add_argument('--eval_only', default=None)
    a = ap.parse_args()
    eps = ['BF3_5c_s0', 'BF4_5c_s0', 'BF4_15c_s0', 'BF4_30c_s0', 'BF5_15c_s0', 'BF5_30c_s1']
    if a.eval_only:
        model = DETRSet().to(DEVICE); model.load_state_dict(torch.load(a.eval_only, map_location=DEVICE))
    else:
        print(f"training {a.steps} steps on {DEVICE} (DETR set-prediction)...", flush=True)
        model = train(a.steps, log=lambda m: print(m, flush=True))
        torch.save(model.state_dict(), a.ckpt); print(f"saved {a.ckpt}", flush=True)
    print("\n=== A2 eval (DETR set-prediction, grid-free) ===", flush=True)
    res = evaluate(model, eps, log=lambda m: print(m, flush=True))
    mean = float(np.mean(list(res.values())))
    print(f"\nMEAN DETR nRMSE {mean:.3f}  (ref: beamform-detect ~0.06-0.08, RELAX 0.05-0.09)", flush=True)
    json.dump(dict(per_ep=res, mean=mean), open('results/direction3/beamform_detr_eval.json', 'w'), indent=2)
