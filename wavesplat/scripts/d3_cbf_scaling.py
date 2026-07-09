"""d3_cbf_scaling.py — three questions the complex-beamform result raised.

The CNN beat the ridge-optimal linear map by 3% once it had 40k scenes
(COMPLEX_BEAMFORM_FIELD_2026-07-09 §7). That gain is real but small, and it
was attributed — as a hypothesis — to SCENE ADAPTATION: conditioned on
(Hs, Tp, direction, spread) the field and c(k) are jointly Gaussian and the
optimal estimator is linear; marginalized over those parameters the mixture
is not Gaussian, so E[eta|c] is nonlinear.

Three probes:

  arch    Does the gain need a CNN at all? If the hypothesis holds, all the
          network must do is READ the spectrum off c(k) and reweight the bank
          per scene. That is a Wiener filter:

              eta = Re sum_k w(k) c(k) e^{i k.x},   w(k) >= 0 predicted per scene

          The synthesis is a FIXED matrix (physics); only w is learned, and w
          is a function of |c| alone — exactly the gamma(k) that WavePrior
          already supplies. If `wiener` matches `resid`, the direct-field
          network's advantage IS spectral reweighting, and the whole
          direct-field detour returns to the prior.

  scale   Is the gain data-limited? Learning curves of the ridge (closed form,
          so its curve is the true linear ceiling at each N) and of the
          networks, over N = 2.5k .. 39k. A ridge curve that has flattened
          while the network's still falls means more data buys more nonlinear
          gain, not a better linear map.

  conv    Is it optimization-limited? Longer schedule + EMA on the best arch.

Usage: python3 scripts/d3_cbf_scaling.py --task {arch,scale,conv,all}
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
os.chdir(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, 'scripts')

from d3_complex_beamform_field import (CACHE, NK, NG, KLIM, DOM_X, DOM_Y,
                                       CNN, LinearMap, ResidNet, build_cache,
                                       N_CACHE)

DEV = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
OUT = 'results/direction3/cbf_scaling.json'
N_VAL = 1000


def synthesis_matrix():
    """Fixed physics: [Re c; Im c] -> field. eta(x) = Re sum_k c(k) e^{i k.x}
    = sum_k [Re c cos(k.x) - Im c sin(k.x)]. No learning in here."""
    ax = np.linspace(-KLIM, KLIM, NK)
    KX, KY = np.meshgrid(ax, ax, indexing='xy')
    gx = np.linspace(*DOM_X, NG)
    gy = np.linspace(*DOM_Y, NG)
    Xq, Yq = np.meshgrid(gx, gy)
    ph = (KX.ravel()[None, :] * Xq.ravel()[:, None]
          + KY.ravel()[None, :] * Yq.ravel()[:, None])       # (NG^2, NK^2)
    S = np.concatenate([np.cos(ph), -np.sin(ph)], 1)          # (NG^2, 2NK^2)
    return torch.tensor(S, dtype=torch.float32)


class Wiener(nn.Module):
    """Learn only the per-scene spectral gain w(k) >= 0, from |c(k)|.

    Phase is never touched: it rides through the fixed synthesis. This is the
    gamma(k) prior of WavePrior, expressed as a gain instead of a variance.
    """

    def __init__(self, S, hidden=64):
        super().__init__()
        self.register_buffer('S', S)
        def blk(i, o):
            return nn.Sequential(nn.Conv2d(i, o, 3, 1, 1),
                                 nn.GroupNorm(8, o), nn.SiLU())
        self.net = nn.Sequential(blk(1, hidden), blk(hidden, hidden),
                                 blk(hidden, hidden),
                                 nn.Conv2d(hidden, 1, 3, 1, 1))
        self.scale = nn.Parameter(torch.tensor(0.0))

    def forward(self, x):
        mag = torch.sqrt(x[:, 0:1] ** 2 + x[:, 1:2] ** 2 + 1e-12)
        mag = torch.log10(mag + 1e-6) / 6 + 1                  # same as prior net
        w = torch.nn.functional.softplus(self.net(mag))        # (B,1,NK,NK)
        cw = x * w                                             # gate Re and Im
        f = cw.flatten(1) @ self.S.T
        return f.view(-1, NG, NG) * torch.exp(self.scale)


def ridge_val(X, F, n, lam_grid=(1e-8, 1e-7, 1e-6, 1e-5, 1e-4)):
    Xtr = X[N_VAL:N_VAL + n].reshape(n, -1)
    Ftr = F[N_VAL:N_VAL + n].reshape(n, -1)
    Xva = X[:N_VAL].reshape(N_VAL, -1).astype(np.float64)
    Fva = F[:N_VAL].reshape(N_VAL, -1).astype(np.float64)
    p = Xtr.shape[1]
    A = np.zeros((p, p)); B = np.zeros((p, Ftr.shape[1]))
    for i in range(0, n, 4000):
        xb = Xtr[i:i + 4000].astype(np.float64)
        A += xb.T @ xb
        B += xb.T @ Ftr[i:i + 4000].astype(np.float64)
    tr = np.trace(A) / p
    best = np.inf
    for lam in lam_grid:
        W = np.linalg.solve(A + lam * tr * np.eye(p), B)
        best = min(best, float(np.mean((Xva @ W - Fva) ** 2)))
    return best


def train_net(kind, Xt, Ft, Xv, Fv, steps, S=None, W0=None, bs=64, lr=3e-4,
              wd=1e-2, ema_decay=None, seed=0, log_every=1000):
    if kind == 'wiener':
        net = Wiener(S).to(DEV)
    elif kind == 'resid':
        net = ResidNet(W0).to(DEV)
    else:
        net = CNN().to(DEV)
    params = [p for p in net.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    rng = np.random.default_rng(seed)
    ema = {k: v.detach().clone() for k, v in net.state_dict().items()} \
        if ema_decay else None

    best, t0 = np.inf, time.time()
    for step in range(1, steps + 1):
        net.train()
        ii = torch.tensor(rng.choice(len(Xt), bs, replace=False), device=DEV)
        xb, fb = Xt[ii], Ft[ii]
        if rng.random() < 0.5:
            xb, fb = torch.flip(xb, [2]), torch.flip(fb, [1])
        loss = ((net(xb) - fb) ** 2).mean()
        opt.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(params, 1.0)
        opt.step(); sched.step()
        if ema is not None:
            with torch.no_grad():
                for k, v in net.state_dict().items():
                    if v.dtype.is_floating_point:
                        ema[k].mul_(ema_decay).add_(v, alpha=1 - ema_decay)
                    else:
                        ema[k].copy_(v)

        if step % log_every == 0 or step == steps:
            net.eval()
            with torch.no_grad():
                vl = float(((net(Xv) - Fv) ** 2).mean())
            if ema is not None:
                bak = {k: v.detach().clone() for k, v in net.state_dict().items()}
                net.load_state_dict(ema)
                with torch.no_grad():
                    vl_e = float(((net(Xv) - Fv) ** 2).mean())
                net.load_state_dict(bak)
                vl = min(vl, vl_e)
            best = min(best, vl)
            if step % (log_every * 5) == 0 or step == steps:
                print(f"    step {step:6d} val {vl:.4f} best {best:.4f} "
                      f"({time.time()-t0:.0f}s)", flush=True)
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--task', default='all',
                    choices=['arch', 'scale', 'conv', 'all'])
    a = ap.parse_args()

    X, F = build_cache(N_CACHE, 4242, CACHE)
    Xv = torch.tensor(X[:N_VAL]).to(DEV)
    Fv = torch.tensor(F[:N_VAL]).to(DEV)
    S = synthesis_matrix().to(DEV)
    res = json.load(open(OUT)) if os.path.exists(OUT) else {}

    Wl = torch.load('results/direction3/cbf_field_linear.pt',
                    map_location=DEV)['model']['fc.weight']

    if a.task in ('arch', 'all'):
        print("\n=== arch: does the gain need a CNN, or just w(k)? ===")
        Xt = torch.tensor(X[N_VAL:]).to(DEV)
        Ft = torch.tensor(F[N_VAL:]).to(DEV)
        out = {}
        for kind in ('wiener', 'cnn', 'resid'):
            print(f"  [{kind}]", flush=True)
            out[kind] = train_net(kind, Xt, Ft, Xv, Fv, 20000, S=S, W0=Wl)
        out['linear_ridge'] = ridge_val(X, F, len(X) - N_VAL)
        res['arch'] = out
        for k, v in out.items():
            print(f"  {k:<14} val {v:.4f}")
        json.dump(res, open(OUT, 'w'), indent=1)

    if a.task in ('scale', 'all'):
        print("\n=== scale: learning curves vs training-set size ===")
        Ns = [2500, 5000, 10000, 20000, len(X) - N_VAL]
        out = {'N': Ns, 'ridge': [], 'resid': [], 'wiener': []}
        for n in Ns:
            Xt = torch.tensor(X[N_VAL:N_VAL + n]).to(DEV)
            Ft = torch.tensor(F[N_VAL:N_VAL + n]).to(DEV)
            r = ridge_val(X, F, n)
            rs = train_net('resid', Xt, Ft, Xv, Fv, 12000, W0=Wl,
                           log_every=2000)
            wi = train_net('wiener', Xt, Ft, Xv, Fv, 12000, S=S,
                           log_every=2000)
            out['ridge'].append(r); out['resid'].append(rs)
            out['wiener'].append(wi)
            print(f"  N={n:6d}  ridge {r:.4f}  resid {rs:.4f}  "
                  f"wiener {wi:.4f}", flush=True)
        res['scale'] = out
        json.dump(res, open(OUT, 'w'), indent=1)

    if a.task in ('conv', 'all'):
        print("\n=== conv: is it optimization-limited? (long + EMA) ===")
        Xt = torch.tensor(X[N_VAL:]).to(DEV)
        Ft = torch.tensor(F[N_VAL:]).to(DEV)
        base = res.get('arch', {}).get('resid')
        long_ = train_net('resid', Xt, Ft, Xv, Fv, 60000, W0=Wl,
                          ema_decay=0.999, log_every=2000)
        res['conv'] = {'resid_20k': base, 'resid_60k_ema': long_}
        print(f"  resid 20k {base}  ->  60k+EMA {long_:.4f}")
        json.dump(res, open(OUT, 'w'), indent=1)

    print(f"\n-> {OUT}")


if __name__ == '__main__':
    sys.exit(main())
