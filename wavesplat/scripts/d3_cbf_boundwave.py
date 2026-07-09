"""d3_cbf_boundwave.py — can a network synthesize the bound wave from c(k)?

The previous test (d3_cbf_nonlinear) asked whether a free CNN residual beats a
diagonal spectral gain once the sea carries 2nd-order bound waves. It answered
"no" — but it never could have answered "yes":

    bound wave / free wave std      = 0.064
    bound wave share of field var   = 0.0041   (0.39%)
    unexplained variance at val     = 0.53     (53%)
    wiener2 - resid gap             = 0.0223   (5x the entire bound signal)

A perfect bound-wave model would have improved the total-field MSE by at most
0.004. The experiment had no power. Falsifying a prediction with an instrument
that cannot see the effect is not falsification.

So point the instrument at the effect: make the bound wave THE target.

    target = eta_2 alone (Sharma & Dean cross terms), normalized

Now a model that cannot produce off-shell content scores exactly at the
zero-predictor (relative MSE = 1). Predictions:

    ridge, wiener2   ~ 1.0    diagonal / shell-constrained: cannot synthesize
                              content at (k1+k2, w1+w2), which is off the
                              dispersion shell c(k) is built on
    resid            < 1.0    the bound wave is a deterministic quadratic
                              functional of the free waves, whose amplitudes
                              and phases c(k) does carry

Also fixes an aliasing defect: eta_2 lives near 2k, and the 32x32 output grid
of the earlier runs has Nyquist 0.41 rad/m — the bound wave folds back onto
itself. Output grid is 64x64 here (Nyquist 0.82).

Usage: python3 scripts/d3_cbf_boundwave.py [--n 20000] [--steps 12000]
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

from wavesim.wave_generator import jonswap_components
from d3_sa_fim_identifiability import omega_of
from wavesplat.model.stokes2 import Stokes2Transfer
from d3_complex_beamform_field import (NK, DOM_X, DOM_Y, R_FAN, HALF_FOV,
                                       DEPTH, W_OBS, SCAN_DT, M_PER_SCAN,
                                       NOISE, M_CAP, complex_beamform, CNN)

G = 9.81
DEV = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
OUT = 'results/direction3/cbf_boundwave.json'
CACHE = 'tmp/cbf_cache_bw.npz'
NG2 = 64          # Nyquist 0.82 rad/m: the bound wave near 2k is resolvable
NC = 64
N_VAL = 1000


def scene(rng):
    Hs = rng.uniform(1.0, 3.0)
    Tp = rng.uniform(5.0, 11.0)
    kp = (2 * np.pi / Tp) ** 2 / G
    comps = jonswap_components(Hs=Hs, Tp=Tp, n_components=NC, depth=DEPTH,
                               direction_deg=180.0 + rng.uniform(-40, 40),
                               direction_spread_deg=rng.uniform(15, 45),
                               seed=int(rng.integers(1 << 30)))
    kx = np.array([c.kx for c in comps]); ky = np.array([c.ky for c in comps])
    A = np.array([c.A for c in comps]); ph = np.array([c.phi for c in comps])
    om = omega_of(kx, ky, DEPTH)
    S2T = Stokes2Transfer(kx, ky, DEPTH)

    def free(x, y, t):
        psi = (kx[None] * x[:, None] + ky[None] * y[:, None]
               - om[None] * t + ph[None])
        return (A[None] * np.cos(psi)).sum(1)

    def bound(x, y, t):
        return S2T.correction_np(x, y, float(t), A, kx, ky, ph, om)

    t0 = W_OBS
    P, Z = [], []
    for t_s in np.arange(0.0, t0 + 1e-9, SCAN_DT):
        rr = rng.uniform(*R_FAN, M_PER_SCAN)
        aa = rng.uniform(-HALF_FOV, HALF_FOV, M_PER_SCAN)
        xs, ys = rr * np.cos(aa), rr * np.sin(aa)
        keep = (xs >= DOM_X[0]) & (xs <= DOM_X[1]) & \
               (ys >= DOM_Y[0]) & (ys <= DOM_Y[1])
        if keep.sum() == 0:
            continue
        xs, ys = xs[keep], ys[keep]
        z = free(xs, ys, t_s) + bound(xs, ys, t_s)
        P.append(np.column_stack([xs, ys, np.full(len(xs), t_s)]))
        Z.append(z + rng.normal(0, NOISE, len(xs)))
    P, Z = np.vstack(P), np.concatenate(Z)
    if len(Z) > M_CAP:
        ii = rng.choice(len(Z), M_CAP, replace=False)
        P, Z = P[ii], Z[ii]

    gx = np.linspace(*DOM_X, NG2)
    gy = np.linspace(*DOM_Y, NG2)
    Xq, Yq = np.meshgrid(gx, gy)
    xr, yr = Xq.ravel(), Yq.ravel()
    Ffree = free(xr, yr, t0).reshape(NG2, NG2)
    Fbnd = bound(xr, yr, t0).reshape(NG2, NG2)
    return P, Z, Ffree, Fbnd, kp, t0


def build(n, seed=1234):
    if os.path.exists(CACHE):
        d = np.load(CACHE)
        if len(d['Fb']) >= n:
            return d['X'][:n], d['Ff'][:n], d['Fb'][:n]
    rng = np.random.default_rng(seed)
    Xs, Ffs, Fbs, t0 = [], [], [], time.time()
    for i in range(n):
        P, Z, Ff, Fb, kp, ts = scene(rng)
        X, s, _ = complex_beamform(P, Z, kp, ts)
        Xs.append(X)
        Ffs.append((Ff / s).astype(np.float32))
        Fbs.append((Fb / s).astype(np.float32))
        if (i + 1) % 2500 == 0:
            print(f"    {i+1}/{n} ({time.time()-t0:.0f}s)", flush=True)
    X, Ff, Fb = np.stack(Xs), np.stack(Ffs), np.stack(Fbs)
    np.savez(CACHE, X=X, Ff=Ff, Fb=Fb)
    return X, Ff, Fb


def ridge(X, Y, n, lam_grid=(1e-8, 1e-6, 1e-4, 1e-2)):
    p = X.shape[1] * X.shape[2] * X.shape[3]
    Xva = X[:N_VAL].reshape(N_VAL, -1).astype(np.float64)
    Yva = Y[:N_VAL].reshape(N_VAL, -1).astype(np.float64)
    A = np.zeros((p, p)); B = np.zeros((p, Y.shape[1] * Y.shape[2]))
    for i in range(0, n, 4000):
        xb = X[N_VAL + i:N_VAL + min(i + 4000, n)].reshape(-1, p).astype(np.float64)
        yb = Y[N_VAL + i:N_VAL + min(i + 4000, n)].reshape(len(xb), -1).astype(np.float64)
        A += xb.T @ xb; B += xb.T @ yb
    tr = np.trace(A) / p
    best = (np.inf, None)
    for lam in lam_grid:
        W = np.linalg.solve(A + lam * tr * np.eye(p), B)
        vl = float(np.mean((Xva @ W - Yva) ** 2))
        if vl < best[0]:
            best = (vl, W)
    return best


class LinMap(nn.Module):
    def __init__(self, W=None):
        super().__init__()
        self.fc = nn.Linear(2 * NK * NK, NG2 * NG2, bias=False)
        if W is not None:
            with torch.no_grad():
                self.fc.weight.copy_(W)

    def forward(self, x):
        return self.fc(x.flatten(1)).view(-1, NG2, NG2)


class Wiener2(nn.Module):
    def __init__(self, W):
        super().__init__()
        self.lin = LinMap(W)
        for p_ in self.lin.parameters():
            p_.requires_grad = False
        h = 64
        def blk(i, o):
            return nn.Sequential(nn.Conv2d(i, o, 3, 1, 1),
                                 nn.GroupNorm(8, o), nn.SiLU())
        self.net = nn.Sequential(blk(1, h), blk(h, h), blk(h, h),
                                 nn.Conv2d(h, 1, 3, 1, 1))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.constant_(self.net[-1].bias, 0.54132)

    def forward(self, x):
        mag = torch.sqrt(x[:, 0:1] ** 2 + x[:, 1:2] ** 2 + 1e-12)
        mag = torch.log10(mag + 1e-6) / 6 + 1
        w = torch.nn.functional.softplus(self.net(mag))
        return self.lin(x * w)


class Resid(nn.Module):
    def __init__(self, W):
        super().__init__()
        self.lin = LinMap(W)
        for p_ in self.lin.parameters():
            p_.requires_grad = False
        self.res = CNN()
        self.res.head[-1] = nn.Linear(1024, NG2 * NG2)
        nn.init.zeros_(self.res.head[-1].weight)
        nn.init.zeros_(self.res.head[-1].bias)

    def forward(self, x):
        return self.lin(x) + self.res.head(self.res.enc(x)).view(-1, NG2, NG2)


def train(net, Xt, Yt, Xv, Yv, steps, seed=0, lr=3e-4, wd=1e-2, bs=64):
    net = net.to(DEV)
    params = [p for p in net.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    rng = np.random.default_rng(seed)
    best = np.inf
    for step in range(1, steps + 1):
        net.train()
        ii = torch.tensor(rng.choice(len(Xt), bs, replace=False), device=DEV)
        xb, yb = Xt[ii], Yt[ii]
        if rng.random() < 0.5:
            xb, yb = torch.flip(xb, [2]), torch.flip(yb, [1])
        loss = ((net(xb) - yb) ** 2).mean()
        opt.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(params, 1.0)
        opt.step(); sched.step()
        if step % 1000 == 0 or step == steps:
            net.eval()
            with torch.no_grad():
                best = min(best, float(((net(Xv) - Yv) ** 2).mean()))
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=20000)
    ap.add_argument('--steps', type=int, default=12000)
    a = ap.parse_args()

    print(f"building {a.n} scenes (NG={NG2})...", flush=True)
    X, Ff, Fb = build(a.n)
    n = a.n - N_VAL
    res = {'n': n, 'NG': NG2}

    for tag, Y in (('bound', Fb), ('free', Ff)):
        var = float(np.var(Y[:N_VAL]))
        rv, W = ridge(X, Y, n)
        Xv = torch.tensor(X[:N_VAL]).to(DEV)
        Yv = torch.tensor(Y[:N_VAL]).to(DEV)
        Xt = torch.tensor(X[N_VAL:N_VAL + n]).to(DEV)
        Yt = torch.tensor(Y[N_VAL:N_VAL + n]).to(DEV)
        W0 = torch.tensor(W.T, dtype=torch.float32, device=DEV)
        w2 = train(Wiener2(W0), Xt, Yt, Xv, Yv, a.steps)
        rs = train(Resid(W0), Xt, Yt, Xv, Yv, a.steps)
        res[tag] = {'var': var, 'ridge': rv, 'wiener2': w2, 'resid': rs,
                    'ridge_rel': rv / var, 'wiener2_rel': w2 / var,
                    'resid_rel': rs / var}
        print(f"\n[{tag}]  var {var:.4f}   (relative MSE: 1.0 = zero-predictor)")
        print(f"  ridge   {rv:.4f}  rel {rv/var:.3f}")
        print(f"  wiener2 {w2:.4f}  rel {w2/var:.3f}")
        print(f"  resid   {rs:.4f}  rel {rs/var:.3f}")
        del Xt, Yt
        torch.cuda.empty_cache()
        json.dump(res, open(OUT, 'w'), indent=1)

    print(f"\n-> {OUT}")


if __name__ == '__main__':
    sys.exit(main())
