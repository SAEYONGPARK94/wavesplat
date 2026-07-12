"""M3-v1: point-native amortizer (perceiver-lite) with the incumbent's
data, objective and judgment protocol.

Pre-registration: docs/M3_POINT_prereg_2026-07-12.md (frozen before run).
Same bwtask shards (8,000), same val split (first 666), same task_loss_one
(differentiable MMSE), same output interface (Ylog 51x51 gamma map +
log-smult). Only the encoder changes: raw points (P, z) with kp/Hs
normalization instead of beamform 6ch maps. Cold start, 12,000 steps
(disclosed pretrain-compensation; fairness axis = data/objective/judgment).
"""
import argparse
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
os.chdir(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, 'scripts')

from d3_prior_net_train import band_mask, NGRID, KREL
from d3_prior_net_taskft import load_all, task_loss_one, DEPTH, G

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def point_features(P, z, kp, Hs):
    """(B,N,3),(B,N),(B,),(B,) -> (B,N,5) normalized point tokens."""
    xy = P[..., :2]
    c = xy.mean(dim=1, keepdim=True)
    om = torch.sqrt(G * kp * torch.tanh(kp * DEPTH))
    t_rel = (P[..., 2] - P[..., 2].amax(dim=1, keepdim=True))
    return torch.stack([
        (xy[..., 0] - c[..., 0]) * kp[:, None],
        (xy[..., 1] - c[..., 1]) * kp[:, None],
        t_rel * om[:, None] / 10.0,
        z / (Hs[:, None] / 4.0),
        torch.hypot(xy[..., 0] - c[..., 0], xy[..., 1] - c[..., 1])
        * kp[:, None] / 10.0], dim=-1)


class PointPrior(nn.Module):
    """point tokens -> self-attn -> grid cross-attn -> Ylog map + smult."""

    def __init__(self, d=96, heads=4):
        super().__init__()
        self.embed = nn.Sequential(nn.Linear(5, d), nn.SiLU(),
                                   nn.Linear(d, d))
        self.sa = nn.ModuleList([nn.TransformerEncoderLayer(
            d, heads, 2 * d, dropout=0.0, batch_first=True,
            norm_first=True) for _ in range(2)])
        ax = np.linspace(-KREL, KREL, NGRID)
        KX, KY = np.meshgrid(ax, ax)
        gq = np.stack([KX.ravel(), KY.ravel(),
                       np.hypot(KX, KY).ravel()], -1)
        self.register_buffer('grid', torch.tensor(gq, dtype=torch.float32))
        self.gq_embed = nn.Sequential(nn.Linear(3, d), nn.SiLU(),
                                      nn.Linear(d, d))
        self.ca = nn.MultiheadAttention(d, heads, batch_first=True)
        self.ca2 = nn.MultiheadAttention(d, heads, batch_first=True)
        self.ln_q, self.ln_q2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 2 * d),
                                nn.SiLU(), nn.Linear(2 * d, d))
        self.out = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 1))
        self.sm = nn.Sequential(nn.Linear(2 * d, 32), nn.SiLU(),
                                nn.Linear(32, 1))
        with torch.no_grad():
            self.sm[-1].weight.mul_(0.1)
            self.sm[-1].bias.zero_()
        self.sm_b = nn.Parameter(torch.tensor(float(np.log(3.0))))

    def forward(self, P, z, kp, Hs):
        tok = self.embed(point_features(P, z, kp, Hs))
        for blk in self.sa:
            tok = blk(tok)
        q = self.gq_embed(self.grid)[None].expand(tok.shape[0], -1, -1)
        h, _ = self.ca(self.ln_q(q), tok, tok)
        q = q + h
        h, _ = self.ca2(self.ln_q2(q), tok, tok)
        q = q + h + self.ff(q)
        ylog = self.out(q).squeeze(-1).view(-1, NGRID, NGRID)
        pool = torch.cat([tok.mean(1), tok.std(1)], -1)
        log_smult = self.sm_b + self.sm(pool).squeeze(-1)
        return ylog, log_smult


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--steps', type=int, default=12000)
    ap.add_argument('--bs', type=int, default=8)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--data', default='tmp/prior_train/bwtask')
    ap.add_argument('--ckpt', default='results/direction3/m3_point_v1.pt')
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()
    torch.manual_seed(a.seed)

    D = load_all(a.data)
    n_all = len(D['X'])
    n_val = max(200, n_all // 12)
    net = PointPrior().to(DEVICE)
    npar = sum(p.numel() for p in net.parameters())
    print(f"data {a.data}: {n_all} samples ({n_val} val); "
          f"PointPrior {npar/1e6:.2f}M params (budget 1.5M)", flush=True)
    assert npar <= 1.5e6
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.steps)
    mask_t = torch.tensor(band_mask(), device=DEVICE)
    axn = torch.linspace(-KREL, KREL, NGRID, device=DEVICE)
    KXn, KYn = torch.meshgrid(axn, axn, indexing='xy')
    kxn, kyn = KXn[mask_t], KYn[mask_t]

    def batch(jj):
        P = torch.tensor(D['P'][jj], device=DEVICE)
        z = torch.tensor(D['z'][jj], device=DEVICE)
        kp = torch.tensor(D['meta'][jj, 0], device=DEVICE)
        Hs = torch.tensor(D['meta'][jj, 1], device=DEVICE)
        return P, z, kp, Hs

    def sample_loss(i, out, ls):
        P = torch.tensor(D['P'][i], device=DEVICE)
        z = torch.tensor(D['z'][i], device=DEVICE)
        Q = torch.tensor(D['Q'][i], device=DEVICE)
        gt = torch.tensor(D['gt_q'][i], device=DEVICE)
        kp_e, Hs, _, _, nfrac = D['meta'][i]
        sig0 = float(nfrac * Hs)
        return task_loss_one(out, P, z, Q, gt, float(kp_e), sig0,
                             ls, mask_t, kxn, kyn) / (Hs / 4) ** 2

    rng = np.random.default_rng(a.seed)
    vi = np.arange(n_val)
    ti = np.arange(n_val, n_all)

    def val_loss():
        net.eval()
        with torch.no_grad():
            vls, sms = [], []
            for i0 in range(0, 160, a.bs):
                jj = vi[i0:i0 + a.bs]
                ov, lsv = net(*batch(jj))
                vls += [float(sample_loss(int(i), ov[j], lsv[j]))
                        for j, i in enumerate(jj)]
                sms += [float(x) for x in torch.exp(lsv)]
        net.train()
        return float(np.mean(vls)), float(np.mean(sms)), float(np.std(sms))

    best = (np.inf, 0)
    t0 = time.time()
    v0, sm0, ss0 = val_loss()
    print(f"step     0 val {v0:.4f}  smult {sm0:.2f}±{ss0:.2f}", flush=True)
    for step in range(1, a.steps + 1):
        jj = rng.choice(ti, a.bs, replace=False)
        out, ls = net(*batch(jj))
        loss = torch.stack([sample_loss(int(i), out[j], ls[j])
                            for j, i in enumerate(jj)]).mean()
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        sched.step()
        if step % 500 == 0:
            v, sm, ss = val_loss()
            mark = ''
            if v < best[0]:
                best = (v, step)
                torch.save({'model': net.state_dict(),
                            'val': v, 'step': step, 'seed': a.seed,
                            'data': a.data}, a.ckpt)
                mark = '  *saved*'
            print(f"step {step:5d} val {v:.4f}  smult {sm:.2f}±{ss:.2f}  "
                  f"[{(time.time()-t0)/60:.0f}m]{mark}", flush=True)
    print(f"done. best val {best[0]:.4f} @ step {best[1]}  "
          f"({(time.time()-t0)/60:.0f} min)")
    return 0


if __name__ == '__main__':
    sys.exit(main())
