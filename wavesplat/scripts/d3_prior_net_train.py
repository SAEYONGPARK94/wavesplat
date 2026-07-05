"""
d3_prior_net_train.py — train the amortized prior net (module 1 of
DESIGN_amortized_prior_2026-07-05.md): 3ch normalized beamform power maps
-> log10 gamma map on the fixed 51x51 normalized bank grid.

Small U-Net-ish CNN; loss = masked SmoothL1 on log-gamma inside the
annulus band (0.35 <= |k|/kp <= 2.5) + light emphasis on high-energy cells
(weight 1 + Y_lin/max). Standing directive: train long enough before
judging (>= 20k steps, monitor val).

Usage: python3 scripts/d3_prior_net_train.py [--steps 30000]
       [--data tmp/prior_train/lin0] [--ckpt results/direction3/prior_net.pt]
"""
import os
import sys
import glob
import json
import time
import argparse

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
os.chdir(os.path.join(os.path.dirname(__file__), '..'))

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NGRID, KREL = 51, 2.5


def band_mask():
    ax = np.linspace(-KREL, KREL, NGRID)
    KX, KY = np.meshgrid(ax, ax)
    kk = np.hypot(KX, KY)
    return (kk > 0.35) & (kk < KREL)


class PriorNet(nn.Module):
    def __init__(self, ch=48):
        super().__init__()

        def blk(i, o):
            return nn.Sequential(nn.Conv2d(i, o, 3, padding=1),
                                 nn.GroupNorm(8, o), nn.SiLU(),
                                 nn.Conv2d(o, o, 3, padding=1),
                                 nn.GroupNorm(8, o), nn.SiLU())
        self.e1 = blk(6, ch)
        self.e2 = blk(ch, 2 * ch)
        self.e3 = blk(2 * ch, 4 * ch)
        self.pool = nn.AvgPool2d(2)
        self.d2 = blk(4 * ch + 2 * ch, 2 * ch)
        self.d1 = blk(2 * ch + ch, ch)
        self.out = nn.Conv2d(ch, 1, 1)

    def forward(self, x):
        import torch.nn.functional as F
        e1 = self.e1(x)                       # 51
        e2 = self.e2(self.pool(e1))           # 25
        e3 = self.e3(self.pool(e2))           # 12
        u3 = F.interpolate(e3, size=e2.shape[2:], mode='bilinear',
                           align_corners=False)
        d2 = self.d2(torch.cat([u3, e2], 1))
        u2 = F.interpolate(d2, size=e1.shape[2:], mode='bilinear',
                           align_corners=False)
        d1 = self.d1(torch.cat([u2, e1], 1))
        return self.out(d1)[:, 0]


def load_data(stem):
    Xs, Ys = [], []
    for p in sorted(glob.glob(f"{stem}_s*.npz")):
        d = np.load(p)
        Xs.append(d['X'])
        Ys.append(d['Y'])
    X = np.concatenate(Xs)
    Y = np.concatenate(Ys)
    return X, Y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--steps', type=int, default=30000)
    ap.add_argument('--bs', type=int, default=64)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--data', default='tmp/prior_train/lin0')
    ap.add_argument('--ckpt', default='results/direction3/prior_net.pt')
    a = ap.parse_args()

    X, Y = load_data(a.data)
    n_val = max(200, len(X) // 10)
    Xv, Yv = X[:n_val], Y[:n_val]
    Xt, Yt = X[n_val:], Y[n_val:]
    print(f"data {a.data}: train {len(Xt)} val {len(Xv)}")
    mask = torch.tensor(band_mask(), device=DEVICE)
    net = PriorNet().to(DEVICE)
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.steps)
    lossf = nn.SmoothL1Loss(reduction='none')
    rng = np.random.default_rng(0)
    Xv_t = torch.tensor(Xv, device=DEVICE)
    Yv_t = torch.tensor(Yv, device=DEVICE)
    best = (np.inf, 0)
    t0 = time.time()
    for step in range(1, a.steps + 1):
        ii = rng.choice(len(Xt), a.bs)
        xb = torch.tensor(Xt[ii], device=DEVICE)
        yb = torch.tensor(Yt[ii], device=DEVICE)
        pred = net(xb)
        w = 1.0 + 10.0 ** yb                     # emphasize energetic cells
        loss = (lossf(pred, yb) * w * mask[None]).sum() \
            / (w * mask[None]).sum()
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        if step % 1000 == 0 or step == a.steps:
            net.eval()
            with torch.no_grad():
                vp = []
                for i in range(0, len(Xv_t), 256):
                    vp.append(net(Xv_t[i:i + 256]))
                vp = torch.cat(vp)
                wv = 1.0 + 10.0 ** Yv_t
                vl = float((lossf(vp, Yv_t) * wv * mask[None]).sum()
                           / (wv * mask[None]).sum())
            net.train()
            tag = ''
            if vl < best[0]:
                best = (vl, step)
                torch.save(dict(model=net.state_dict(), step=step,
                                val=vl, ngrid=NGRID, krel=KREL), a.ckpt)
                tag = ' *saved'
            print(f"step {step} train {float(loss):.4f} val {vl:.4f} "
                  f"({time.time()-t0:.0f}s){tag}", flush=True)
    print(f"best val {best[0]:.4f} @ step {best[1]} -> {a.ckpt}")


if __name__ == '__main__':
    main()
