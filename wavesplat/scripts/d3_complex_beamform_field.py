"""d3_complex_beamform_field.py — can a network eat the beamform map and emit
the wave field, if we stop throwing the phase away?

Every direct field regressor in this project mean-collapsed. Two distinct
reasons were conflated:

  (a) the objective  — pixel MSE on an under-determined field is minimized by
      the posterior mean, which is flat wherever the phase is unknown.
  (b) the input      — our beamform map is a POWER map, |c(k)|^2. Squaring
      destroys the phase, and the field IS the phase
      (tmp/probe_beamform_phase.py: backprojecting |c| gives corr -0.13,
      indistinguishable from random phase).

This script removes (b): the network is fed the COMPLEX beamform coefficient
c(k) as two channels (Re, Im) on the normalized k/kp bank, and asked for the
field. Now the information is present. Does the network do anything a physicist
would not?

Three estimators on identical observations:

  backproj   eta = Re sum_k c(k) e^{i psi}          matched filter, no fitting
  linear     a single learned linear map (Re c, Im c) -> field
  cnn        a conv encoder + FC head

The linear probe is the point. Given Gaussian statistics, c(k) is a linear
summary of the observations and the optimal estimator of the field from it is
LINEAR. So if cnn ~= linear, the network has merely learned the physics
inverse and buys nothing. If cnn > linear, there is nonlinear structure (a
prior) worth learning. Both are compared against the physics MMSE.

Usage:
  python3 scripts/d3_complex_beamform_field.py --model cnn    --steps 6000
  python3 scripts/d3_complex_beamform_field.py --model linear --steps 6000
  python3 scripts/d3_complex_beamform_field.py --eval_only
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
from d3_sa_nomp import mmse_bank_estimate

G = 9.81
DEV = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CKPT = 'results/direction3/cbf_field_{}.pt'
OUT_JSON = 'results/direction3/complex_beamform_field.json'

# canonical geometry (episodes: gt_grid 10..250 x -100..100, FOV +-60 deg)
DOM_X, DOM_Y = (10.0, 250.0), (-100.0, 100.0)
R_FAN, HALF_FOV = (15.0, 200.0), np.pi / 3
DEPTH, W_OBS, SCAN_DT, M_PER_SCAN, NOISE, M_CAP = 50.0, 20.0, 0.5, 15, 0.02, 600
# ABSOLUTE k-grid, not k/kp. Normalizing k by the scene's own kp while the
# output field stays in metres makes the map scene-dependent: the phase is
# k.x = (k/kp)*(kp*x), so a shared linear map cannot exist across scenes with
# different kp. The tell is that the ridge-optimal linear map (0.964) scored
# WORSE than parameter-free backprojection (0.803) — impossible for a
# universal map, since backprojection IS linear. With an absolute grid the
# map c(k) -> eta(x) is one fixed Fourier synthesis for every scene.
# klim covers Tp in [5, 11] s: |k| in [0.012, 0.403] rad/m.
# dk = 0.0262 = 2*pi/240 m, exactly the domain's resolution limit.
KLIM = 0.42
NK = 33
NG = 32            # output field grid
N_CACHE = 12000    # ridge needs n >> 2*NK*NK = 2178 features
CACHE = 'tmp/cbf_cache_abs.npz'


def scene(rng):
    """One synthetic sea + its fan observations + the true field at t0.

    Waves approach the sensor (mean direction 180 deg +- 40), so the fan,
    which looks along +x, faces them. Everything is expressed relative to the
    sensor at the origin and to t0, or the phase of c(k) would be meaningless.
    """
    Hs = rng.uniform(1.0, 3.0)
    Tp = rng.uniform(5.0, 11.0)
    kp = (2 * np.pi / Tp) ** 2 / G
    comps = jonswap_components(Hs=Hs, Tp=Tp, n_components=200, depth=DEPTH,
                               direction_deg=180.0 + rng.uniform(-40, 40),
                               direction_spread_deg=rng.uniform(15, 45),
                               seed=int(rng.integers(1 << 30)))
    kx = np.array([c.kx for c in comps]); ky = np.array([c.ky for c in comps])
    A = np.array([c.A for c in comps]); ph = np.array([c.phi for c in comps])
    om = omega_of(kx, ky, DEPTH)

    def eta(x, y, t):
        ps = kx[None] * np.atleast_1d(x)[:, None] \
            + ky[None] * np.atleast_1d(y)[:, None] \
            - om[None] * np.atleast_1d(t)[:, None] + ph[None]
        return (A[None] * np.cos(ps)).sum(1)

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
        P.append(np.column_stack([xs, ys, np.full(len(xs), t_s)]))
        Z.append(eta(xs, ys, t_s) + rng.normal(0, NOISE, len(xs)))
    P, Z = np.vstack(P), np.concatenate(Z)
    if len(Z) > M_CAP:
        ii = rng.choice(len(Z), M_CAP, replace=False)
        P, Z = P[ii], Z[ii]

    gx = np.linspace(*DOM_X, NG)
    gy = np.linspace(*DOM_Y, NG)
    Xq, Yq = np.meshgrid(gx, gy)
    F = eta(Xq.ravel(), Yq.ravel(), t0).reshape(NG, NG)
    return P, Z, F, Hs, kp, t0, (Xq, Yq)


def complex_beamform(P, Z, kp, t0):
    """c(k) = (1/N) sum_i z_i exp(-i psi_i) on the normalized dispersion bank.

    Normalizing z by its own std makes the input scale-free (deployable), and
    the same std rescales the target, so the network never sees Hs.
    """
    s = np.std(Z) + 1e-9
    ax = np.linspace(-KLIM, KLIM, NK)
    KX, KY = np.meshgrid(ax, ax, indexing='xy')
    kk = np.hypot(KX, KY)
    om = np.sqrt(G * np.maximum(kk, 1e-9) * np.tanh(np.maximum(kk, 1e-9) * DEPTH))
    psi = (KX.ravel()[:, None] * P[:, 0][None]
           + KY.ravel()[:, None] * P[:, 1][None]
           - om.ravel()[:, None] * (P[:, 2] - t0)[None])
    zc = (Z / s)[None]
    c = (zc * np.exp(-1j * psi)).sum(1) / len(Z)
    band = ((kk > 0.010) & (kk < KLIM)).ravel()      # global, kp-independent
    c = c * band
    X = np.stack([c.real, c.imag]).reshape(2, NK, NK).astype(np.float32)
    return X, s, (KX, KY, om, band)


def backproject(P, Z, kp, t0, Xq, Yq, F_true):
    """Matched filter, with the scale it cannot know granted to it.

    Summing e^{i psi} over a k-grid has an arbitrary gain (grid spacing, band
    area, the 1/N in c). Rather than derive it, fit the single scalar by least
    squares against the truth: an ORACLE-GAIN matched filter, i.e. an upper
    bound on what backprojection alone can do. Its shape is all that is being
    tested here.
    """
    X, s, (KX, KY, om, band) = complex_beamform(P, Z, kp, t0)
    c = (X[0] + 1j * X[1]).ravel()
    psq = (KX.ravel()[:, None] * Xq.ravel()[None]
           + KY.ravel()[:, None] * Yq.ravel()[None])
    f = np.real((c[:, None] * np.exp(1j * psq)).sum(0)).reshape(Xq.shape)
    g = float((f * F_true).sum() / max((f * f).sum(), 1e-12))
    return f * g


class LinearMap(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(2 * NK * NK, NG * NG, bias=False)

    def forward(self, x):
        return self.fc(x.flatten(1)).view(-1, NG, NG)


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        def blk(i, o, s=1):
            return nn.Sequential(nn.Conv2d(i, o, 3, s, 1),
                                 nn.GroupNorm(8, o), nn.SiLU())
        self.enc = nn.Sequential(blk(2, 32), blk(32, 64, 2), blk(64, 64),
                                 blk(64, 128, 2), blk(128, 128),
                                 blk(128, 256, 2))
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(256 * 5 * 5, 1024),
                                  nn.SiLU(), nn.Linear(1024, NG * NG))

    def forward(self, x):
        return self.head(self.enc(x)).view(-1, NG, NG)


def make_batch(rng, bs):
    Xs, Fs = [], []
    for _ in range(bs):
        P, Z, F, Hs, kp, t0, _ = scene(rng)
        X, s, _ = complex_beamform(P, Z, kp, t0)
        Xs.append(X)
        Fs.append((F / s).astype(np.float32))
    return (torch.tensor(np.stack(Xs)), torch.tensor(np.stack(Fs)))


def fan_masks(Xq, Yq):
    R = np.hypot(Xq, Yq)
    ca = Xq / np.maximum(R, 1e-9)
    inside = (R >= R_FAN[0]) & (R <= R_FAN[1]) & (ca >= np.cos(HALF_FOV))
    return inside, ~inside


def evaluate(nets, n_val=64, n_mmse=16, seed=777):
    rng = np.random.default_rng(seed)
    acc = {k: {'in': [], 'out': [], 'amp': []}
           for k in list(nets) + ['backproj', 'MMSE']}
    for i in range(n_val):
        P, Z, F, Hs, kp, t0, (Xq, Yq) = scene(rng)
        X, s, _ = complex_beamform(P, Z, kp, t0)
        inside, outside = fan_masks(Xq, Yq)
        sg = F.std()

        preds = {'backproj': backproject(P, Z, kp, t0, Xq, Yq, F)}
        with torch.no_grad():
            xt = torch.tensor(X[None], device=DEV)
            for name, net in nets.items():
                preds[name] = net(xt)[0].cpu().numpy() * s
        if i < n_mmse:
            Pq = np.column_stack([Xq.ravel(), Yq.ravel(),
                                  np.full(Xq.size, t0)])
            preds['MMSE'] = mmse_bank_estimate(
                P, Z, DEPTH, Pq, sigma0=0.05 * Hs, kp_hint=kp,
                seed=0).reshape(NG, NG)

        for name, pr in preds.items():
            for key, m in (('in', inside), ('out', outside)):
                acc[name][key].append(
                    float(np.sqrt(np.mean((pr[m] - F[m]) ** 2)) / sg))
            acc[name]['amp'].append(float(pr.std() / sg))

    return {k: {kk: (float(np.mean(v)) if v else None)
                for kk, v in d.items()} for k, d in acc.items()}


def build_cache(n, seed, path):
    """Generate once, reuse. Scene synthesis dominates the step time."""
    if os.path.exists(path):
        d = np.load(path)
        if len(d['F']) >= n:
            return d['X'][:n], d['F'][:n]
    rng = np.random.default_rng(seed)
    Xs, Fs = [], []
    t0_ = time.time()
    for i in range(n):
        P, Z, F, Hs, kp, t0, _ = scene(rng)
        X, s, _ = complex_beamform(P, Z, kp, t0)
        Xs.append(X)
        Fs.append((F / s).astype(np.float32))
        if (i + 1) % 500 == 0:
            print(f"  cache {i+1}/{n} ({time.time()-t0_:.0f}s)", flush=True)
    X, F = np.stack(Xs), np.stack(Fs)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez_compressed(path, X=X, F=F)
    return X, F


def fit_linear_ridge(X, F, n_val=1000):
    """Closed-form best linear map — the estimator this experiment must beat.

    c(k) is a linear summary of the observations, so under Gaussian statistics
    the optimal field estimate from it is linear. Solving the normal equations
    gives that optimum exactly; training a linear layer by SGD would only
    approximate it (and did: amp ratio 0.15 after 200 steps).
    """
    Xf = X.reshape(len(X), -1).astype(np.float64)
    Ff = F.reshape(len(F), -1).astype(np.float64)
    Xtr, Ftr, Xva, Fva = Xf[n_val:], Ff[n_val:], Xf[:n_val], Ff[:n_val]
    A = Xtr.T @ Xtr
    B = Xtr.T @ Ftr
    best = (np.inf, None, None)
    for lam in [1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1]:
        W = np.linalg.solve(A + lam * np.trace(A) / len(A) * np.eye(len(A)), B)
        vl = float(np.mean((Xva @ W - Fva) ** 2))
        print(f"  ridge lam={lam:.0e}  val {vl:.4f}", flush=True)
        if vl < best[0]:
            best = (vl, lam, W)
    print(f"  -> lam={best[1]:.0e}, val {best[0]:.4f}")
    net = LinearMap().to(DEV)
    with torch.no_grad():
        net.fc.weight.copy_(torch.tensor(best[2].T, dtype=torch.float32))
    torch.save({'model': net.state_dict(), 'val': best[0], 'lam': best[1],
                'arch': 'linear'}, CKPT.format('linear'))
    return net, best[0]


def train(model, steps, bs=16, lr=3e-4, seed=0):
    net = (LinearMap() if model == 'linear' else CNN()).to(DEV)
    n_par = sum(p.numel() for p in net.parameters())
    print(f"[{model}] {n_par/1e6:.2f} M params, device {DEV}", flush=True)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    rng = np.random.default_rng(seed)
    Xc, Fc = build_cache(N_CACHE, 4242, CACHE)
    Xv = torch.tensor(Xc[:1000]).to(DEV)
    Fv = torch.tensor(Fc[:1000]).to(DEV)
    Xt = torch.tensor(Xc[1000:]).to(DEV)
    Ft = torch.tensor(Fc[1000:]).to(DEV)
    print(f"  train {len(Xt)} / val {len(Xv)} scenes", flush=True)

    best, hist, t_start = np.inf, [], time.time()
    for step in range(1, steps + 1):
        net.train()
        ii = torch.tensor(rng.choice(len(Xt), bs, replace=False), device=DEV)
        loss = ((net(Xt[ii]) - Ft[ii]) ** 2).mean()
        opt.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step(); sched.step()

        if step % 200 == 0 or step == steps:
            net.eval()
            with torch.no_grad():
                pv = net(Xv)
                vl = float(((pv - Fv) ** 2).mean())
                amp = float(pv.std() / Fv.std())
            hist.append({'step': step, 'train': float(loss), 'val': vl,
                         'amp_ratio': amp})
            el = time.time() - t_start
            print(f"  step {step:5d}  train {float(loss):.4f}  val {vl:.4f}  "
                  f"amp {amp:.3f}  ({el:.0f}s)", flush=True)
            if vl < best:
                best = vl
                torch.save({'model': net.state_dict(), 'step': step,
                            'val': vl, 'amp_ratio': amp, 'arch': model},
                           CKPT.format(model))
    return net, hist, best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='cnn', choices=['linear', 'cnn'])
    ap.add_argument('--steps', type=int, default=6000)
    ap.add_argument('--eval_only', action='store_true')
    a = ap.parse_args()

    res = json.load(open(OUT_JSON)) if os.path.exists(OUT_JSON) else {}
    if not a.eval_only:
        if a.model == 'linear':
            Xc, Fc = build_cache(N_CACHE, 4242, CACHE)
            print(f"[linear] closed-form ridge on {len(Xc)} scenes", flush=True)
            _, best = fit_linear_ridge(Xc, Fc)
            res.setdefault('train', {})['linear'] = {'best_val': best,
                                                     'method': 'ridge'}
        else:
            _, hist, best = train(a.model, a.steps)
            res.setdefault('train', {})[a.model] = {'history': hist,
                                                    'best_val': best,
                                                    'steps': a.steps}
        with open(OUT_JSON, 'w') as f:
            json.dump(res, f, indent=1)

    nets = {}
    for m in ('linear', 'cnn'):
        p = CKPT.format(m)
        if os.path.exists(p):
            net = (LinearMap() if m == 'linear' else CNN()).to(DEV)
            net.load_state_dict(torch.load(p, map_location=DEV)['model'])
            net.eval()
            nets[m] = net
    if nets:
        print("\nevaluating (field nRMSE, normalized by GT std)...", flush=True)
        ev = evaluate(nets)
        res['eval'] = ev
        print(f"\n{'estimator':<12} {'in-fan':>9} {'out-of-fan':>12} "
              f"{'amp ratio':>11}")
        print('-' * 48)
        for k, v in ev.items():
            f = lambda x: '   --' if x is None else f'{x:9.3f}'
            print(f"{k:<12} {f(v['in'])} {f(v['out']):>12} "
                  f"{v['amp']:>11.3f}")
        with open(OUT_JSON, 'w') as f:
            json.dump(res, f, indent=1)
        print(f"\n-> {OUT_JSON}")


if __name__ == '__main__':
    sys.exit(main())
