"""d3_cbf_scaling2.py — is the CNN's 3% a nonlinearity, or a starved baseline?

Three defects in the first scaling run (d3_cbf_scaling.py), all of which
inflate the network:

  1. LEAK. `resid` was always seeded with the ridge map fitted on ALL 39k
     training scenes, even when trained on 2.5k. Its linear part had seen
     data the ridge baseline at that N had not.
  2. STARVED BASELINE. The ridge is still falling steeply at 39k
     (0.5962 -> 0.5717 from 20k). A linear map with 2178x1024 free
     parameters is data-hungry; a CNN with weight sharing can beat it at
     finite N through inductive bias alone, with no nonlinearity present.
     If the gap closes as N grows, there is nothing nonlinear to find.
  3. WRONG WIENER. The scene-adaptation hypothesis was tested as
     `backprojection x w(k)`, which cannot deconvolve the aperture PSF at
     all (a diagonal gain cannot undo off-diagonal k-space leakage). It
     scored 0.79 — worse than linear — and said nothing about the hypothesis.

This script fixes all three:

  ridge curve   Gram accumulated once, snapshotted at each N (exact optimum
                per N, so the true linear ceiling at that sample size).
  resid curve   the ridge map for THAT N is what gets frozen underneath.
  wiener2       `ridge_map(w(k) * c)` with w == 1 at initialisation, so
                training starts exactly at the ridge optimum. Any gain is
                per-scene diagonal reweighting — the hypothesis, properly posed.

Extrapolates the ridge curve (val ~ a + b/N^p) to ask what the linear ceiling
converges to.

Usage: python3 scripts/d3_cbf_scaling2.py [--n_total 100000]
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

from d3_complex_beamform_field import (NK, NG, CNN, LinearMap, ResidNet,
                                       scene, complex_beamform)

DEV = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
OUT = 'results/direction3/cbf_scaling2.json'
CACHE40 = 'tmp/cbf_cache_40k.npz'
CACHE_BIG = 'tmp/cbf_cache_100k.npz'
N_VAL = 1000


def extend_cache(n_total):
    """Reuse the 40k cache and append i.i.d. scenes from a fresh stream."""
    if os.path.exists(CACHE_BIG):
        d = np.load(CACHE_BIG)
        if len(d['F']) >= n_total:
            return d['X'][:n_total], d['F'][:n_total]
    d = np.load(CACHE40)
    X, F = list(d['X']), list(d['F'])
    print(f"  have {len(X)}, generating {n_total - len(X)} more...", flush=True)
    rng = np.random.default_rng(5151)
    t0 = time.time()
    while len(X) < n_total:
        P, Z, Fi, Hs, kp, t0s, _ = scene(rng)
        Xi, s, _ = complex_beamform(P, Z, kp, t0s)
        X.append(Xi); F.append((Fi / s).astype(np.float32))
        if len(X) % 5000 == 0:
            print(f"    {len(X)}/{n_total} ({time.time()-t0:.0f}s)", flush=True)
    X, F = np.stack(X), np.stack(F)
    np.savez(CACHE_BIG, X=X, F=F)
    return X, F


def ridge_curve(X, F, Ns, lam_grid=(1e-8, 1e-7, 1e-6, 1e-5, 1e-4)):
    """Gram is additive, so one pass gives the exact optimum at every N."""
    Xva = X[:N_VAL].reshape(N_VAL, -1).astype(np.float64)
    Fva = F[:N_VAL].reshape(N_VAL, -1).astype(np.float64)
    p = X.shape[1] * X.shape[2] * X.shape[3]
    A = np.zeros((p, p)); B = np.zeros((p, NG * NG))
    out, Ws, done, i = {}, {}, sorted(Ns), 0
    for n in done:
        while i < n:
            j = min(i + 4000, n)
            xb = X[N_VAL + i:N_VAL + j].reshape(j - i, -1).astype(np.float64)
            fb = F[N_VAL + i:N_VAL + j].reshape(j - i, -1).astype(np.float64)
            A += xb.T @ xb; B += xb.T @ fb
            i = j
        tr = np.trace(A) / p
        best = (np.inf, None)
        for lam in lam_grid:
            W = np.linalg.solve(A + lam * tr * np.eye(p), B)
            vl = float(np.mean((Xva @ W - Fva) ** 2))
            if vl < best[0]:
                best = (vl, W)
        out[n] = best[0]; Ws[n] = best[1]
        print(f"  ridge N={n:6d}  val {best[0]:.4f}", flush=True)
    return out, Ws


class Wiener2(nn.Module):
    """ridge_map( w(k) * c ), with w == 1 at init.

    Training therefore STARTS at the ridge optimum. The only freedom is a
    per-scene, per-wavenumber real gain read off |c| — i.e. exactly the
    gamma(k) that WavePrior supplies to the physics solver. If this closes
    the gap to `resid`, the network's nonlinearity is spectral reweighting
    and nothing more.
    """

    def __init__(self, W):
        super().__init__()
        self.lin = LinearMap()
        with torch.no_grad():
            self.lin.fc.weight.copy_(W)
        for p_ in self.lin.parameters():
            p_.requires_grad = False
        h = 64
        def blk(i, o):
            return nn.Sequential(nn.Conv2d(i, o, 3, 1, 1),
                                 nn.GroupNorm(8, o), nn.SiLU())
        self.net = nn.Sequential(blk(1, h), blk(h, h), blk(h, h),
                                 nn.Conv2d(h, 1, 3, 1, 1))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.constant_(self.net[-1].bias, 0.54132)   # softplus(x)=1

    def forward(self, x):
        mag = torch.sqrt(x[:, 0:1] ** 2 + x[:, 1:2] ** 2 + 1e-12)
        mag = torch.log10(mag + 1e-6) / 6 + 1
        w = torch.nn.functional.softplus(self.net(mag))
        return self.lin(x * w)


def train(kind, W0, Xt, Ft, Xv, Fv, steps, seed=0, lr=3e-4, wd=1e-2, bs=64):
    net = (ResidNet(W0) if kind == 'resid' else Wiener2(W0)).to(DEV)
    params = [p for p in net.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    rng = np.random.default_rng(seed)
    best = np.inf
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
        if step % 1000 == 0 or step == steps:
            net.eval()
            with torch.no_grad():
                best = min(best, float(((net(Xv) - Fv) ** 2).mean()))
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n_total', type=int, default=100000)
    ap.add_argument('--steps', type=int, default=12000)
    a = ap.parse_args()

    X, F = extend_cache(a.n_total)
    print(f"cache {len(X)} scenes", flush=True)
    Ns = [n for n in (5000, 10000, 20000, 39000, 60000, len(X) - N_VAL)
          if n <= len(X) - N_VAL]
    Ns = sorted(set(Ns))

    rv, Ws = ridge_curve(X, F, Ns)

    Xv = torch.tensor(X[:N_VAL]).to(DEV)
    Fv = torch.tensor(F[:N_VAL]).to(DEV)
    res = {'N': Ns, 'ridge': [rv[n] for n in Ns], 'resid': [], 'wiener2': []}
    for n in Ns:
        Xt = torch.tensor(X[N_VAL:N_VAL + n]).to(DEV)
        Ft = torch.tensor(F[N_VAL:N_VAL + n]).to(DEV)
        W0 = torch.tensor(Ws[n].T, dtype=torch.float32, device=DEV)  # per-N!
        r = train('resid', W0, Xt, Ft, Xv, Fv, a.steps)
        w2 = train('wiener2', W0, Xt, Ft, Xv, Fv, a.steps)
        res['resid'].append(r); res['wiener2'].append(w2)
        print(f"  N={n:6d}  ridge {rv[n]:.4f}  resid {r:.4f}  "
              f"wiener2 {w2:.4f}  gap {rv[n]-r:+.4f}", flush=True)
        del Xt, Ft
        torch.cuda.empty_cache()

    # extrapolate the linear ceiling: val = a + b * N^-p
    n = np.array(Ns, float); y = np.array(res['ridge'])
    from scipy.optimize import curve_fit
    try:
        popt, _ = curve_fit(lambda x, A, B, P: A + B * x ** (-P), n, y,
                            p0=[0.5, 10, 0.5], maxfev=20000)
        res['ridge_fit'] = {'asymptote': float(popt[0]), 'B': float(popt[1]),
                            'p': float(popt[2])}
        print(f"\nridge extrapolation: val -> {popt[0]:.4f} as N -> inf "
              f"(b={popt[1]:.2f}, p={popt[2]:.3f})")
        print(f"resid at largest N:   {res['resid'][-1]:.4f}")
    except Exception as e:
        print('fit failed', e)

    json.dump(res, open(OUT, 'w'), indent=1)
    print(f"\n-> {OUT}")


if __name__ == '__main__':
    sys.exit(main())
