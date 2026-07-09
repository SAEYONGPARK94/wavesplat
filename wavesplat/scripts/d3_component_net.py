"""Discriminative component net (Stage-1 of the ★1 cascade) — Fourier-feature DETR.

d34's Model used nn.Linear on raw coords → spectral-bias mean-collapse (nRMSE ~0.25).
This fixes it with Fourier features (Tancik 2020, as in amortized_ff.py) on the obs/query
coords + a DETR-style atom-query decoder that outputs a SHARP off-grid component set
{(kx,ky,A,phi,presence)}. Trained on synthetic JONSWAP (nc∈{5,15,30}) with field + obs +
Chamfer + Hungarian losses (reused from d34). Gives a sharp POINT estimate (no posterior-
mean blur); the set-diffusion supplies UQ around it in the cascade.

Run: python3 scripts/d3_component_net.py [--steps 6000] [--eval_only ckpt.pt]
"""
import os, sys, json, time, argparse
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
os.chdir(os.path.join(os.path.dirname(__file__), '..'))

from amortized_ff import FourierFeatures
from d34_learned_offgrid import (make_batch, render, render_obs, match_loss, hungarian_reg,
                                 evaluate, DEVICE, KLIM, NQ, XS, YS, TS)


class CompNet(nn.Module):
    """obs (B,M,4)=(x,y,t,z) → sharp component set. Fourier-feature coords + DETR queries."""
    def __init__(self, d=192, heads=4, enc=4, dec=3, nff=48):
        super().__init__()
        self.ff = FourierFeatures(nff, learnable=True); fdim = 2 * nff
        self.oenc = nn.Sequential(nn.Linear(fdim + 1, d), nn.GELU(), nn.Linear(d, d))
        el = nn.TransformerEncoderLayer(d, heads, 4 * d, batch_first=True, activation='gelu', norm_first=True)
        self.enc = nn.TransformerEncoder(el, enc)
        self.qtok = nn.Parameter(torch.randn(NQ, d) * 0.02)
        dl = nn.TransformerDecoderLayer(d, heads, 4 * d, batch_first=True, activation='gelu', norm_first=True)
        self.dec = nn.TransformerDecoder(dl, dec)
        self.head = nn.Linear(d, 6)

    def forward(self, obs):
        B = obs.shape[0]
        xyt = obs[..., :3].clone()
        xyt[..., 0] /= XS; xyt[..., 1] /= YS; xyt[..., 2] /= TS      # FourierFeatures expects ~O(1)
        oin = torch.cat([self.ff(xyt), obs[..., 3:4]], -1)
        ctx = self.enc(self.oenc(oin))
        h = self.dec(self.qtok[None].expand(B, -1, -1), ctx)
        p = self.head(h)
        kx = torch.tanh(p[..., 0]) * KLIM; ky = torch.tanh(p[..., 1]) * KLIM
        A = torch.nn.functional.softplus(p[..., 2]) * 0.1
        cs = torch.tanh(p[..., 3]); sn = torch.tanh(p[..., 4])
        nrm = torch.sqrt(cs**2 + sn**2 + 1e-9); cs = cs / nrm; sn = sn / nrm
        pres = torch.sigmoid(p[..., 5])
        return kx, ky, A, cs, sn, pres


def train(steps, B=24, lr=3e-4):
    model = CompNet().to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    t0 = time.time()
    for s in range(steps):
        obs, gtf, gx, gy, Hs, tstar, atoms, amask = make_batch(B)
        kx, ky, A, cs, sn, pres = model(obs)
        fld = render(kx, ky, A, cs, sn, pres, gx, gy, tstar.view(B, 1, 1))
        fl = (((fld - gtf) ** 2).mean(dim=(1, 2)) / (Hs ** 2 + 1e-6)).mean()
        oz_hat = render_obs(kx, ky, A, cs, sn, pres, obs[..., 0], obs[..., 1], obs[..., 2])
        ol = (((oz_hat - obs[..., 3]) ** 2).mean(dim=1) / (Hs ** 2 + 1e-6)).mean()
        ml = match_loss(kx, ky, pres * A, atoms[..., 0], atoms[..., 1], atoms[..., 2], amask)
        hr = hungarian_reg(kx, ky, pres * A, atoms[..., 0], atoms[..., 1], atoms[..., 2], amask)
        loss = fl + 0.3 * ol + 1.0 * ml + 2.0 * hr
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        if s % 250 == 0 or s == steps - 1:
            print(f"  step {s:4d}  loss {loss.item():.4f}  field {fl.item():.4f}  obs {ol.item():.4f}  "
                  f"match {ml.item():.4f}  hung {hr.item():.4f}  ({time.time()-t0:.0f}s)", flush=True)
    return model


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--steps', type=int, default=6000)
    ap.add_argument('--ckpt', default='results/direction3/component_net_ff.pt')
    ap.add_argument('--eval_only', default=None)
    a = ap.parse_args()
    eps = ['BF4_15c_s0', 'BF4_30c_s0', 'BF5_15c_s0', 'BF5_30c_s1', 'BF3_5c_s0', 'BF4_5c_s0']
    if a.eval_only:
        model = CompNet().to(DEVICE); model.load_state_dict(torch.load(a.eval_only, map_location=DEVICE))
    else:
        print(f"training {a.steps} steps on {DEVICE} (Fourier-feature DETR component net)...")
        model = train(a.steps)
        torch.save(model.state_dict(), a.ckpt); print(f"saved {a.ckpt}")
    print("\n=== eval (FF component net vs DPS-BMA 0.176, RELAX_full 0.119) ===")
    res = evaluate(model, eps)
    print(f"mean nRMSE: {np.mean(list(res.values())):.3f}")
    json.dump(res, open('results/direction3/component_net_ff_eval.json', 'w'), indent=2)
