"""d3_cbf_nonlinear.py — where does a free network actually earn its freedom?

In LINEAR seas the structured model won: learning only a per-wavenumber gain
w(k) on top of the ridge-optimal map beat a free CNN residual at every sample
size (CBF_SCALING_ARCH_2026-07-09). The gain is diagonal in k, and that was
enough, because the whole field lives on the dispersion shell that our
beamform c(k) is built on.

Second-order Stokes bound waves do not. A component pair (k, omega) generates
a bound harmonic at (2k, 2 omega), and 2*omega(k) != omega(2k) — the energy
sits OFF the shell. Our c(k) is dispersion-constrained, so it cannot see the
bound wave, and a diagonal gain w(k) on the shell cannot synthesize it.

But the bound wave is a deterministic quadratic functional of the free waves,
so a network with cross-wavenumber freedom CAN predict it from c(k).

Prediction, registered before running:

    linear sea      wiener2 > resid     (structure wins; confirmed already)
    nonlinear sea   resid  > wiener2    (freedom is needed, and only here)

If it holds, the free residual earns its keep exactly where the physics bank
is misspecified — the same wall the storm record hit in C1'.

Usage: python3 scripts/d3_cbf_nonlinear.py [--n 40000] [--steps 12000]
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
os.chdir(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, 'scripts')

from wavesim.wave_generator import jonswap_components
from d3_sa_fim_identifiability import omega_of
from wavesplat.model.stokes2 import Stokes2Transfer
from d3_complex_beamform_field import (NK, NG, DOM_X, DOM_Y, R_FAN, HALF_FOV,
                                       DEPTH, W_OBS, SCAN_DT, M_PER_SCAN,
                                       NOISE, M_CAP, complex_beamform)
import d3_cbf_scaling2 as S2

G = 9.81
DEV = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
OUT = 'results/direction3/cbf_nonlinear.json'
CACHE = 'tmp/cbf_cache_nl_{}.npz'
N_VAL = 1000


NC = 64        # cross-term Stokes2 is O(NC^2); 200 components would cost 10x


def scene_nl(rng, nonlinear):
    """Identical to the linear scene, except the surface carries the 2nd-order
    bound wave. Same components, same fan, same normalization — the ONLY
    difference between the two datasets is that bound wave.

    Sharma & Dean (1981) CROSS terms, not the self-harmonic of
    nonlinear_forward.elevation. The self-harmonic sums sum_i c_i A_i^2, which
    is quadratic in the INDIVIDUAL amplitudes, so splitting a given Hs over N
    components suppresses it as 1/sqrt(N): at N=200 it is 0.4% of the surface
    and there is nothing to test. The cross terms sum coherently and give the
    physical ~k*a/2, independent of N (measured: 5.5% at N=32, 5.2% at N=64).
    """
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
    S2T = Stokes2Transfer(kx, ky, DEPTH) if nonlinear else None

    def eta(x, y, t):
        x = np.atleast_1d(x); y = np.atleast_1d(y)
        psi = (kx[None] * x[:, None] + ky[None] * y[:, None]
               - om[None] * t + ph[None])
        z = (A[None] * np.cos(psi)).sum(1)
        if S2T is not None:
            z = z + S2T.correction_np(x, y, float(t), A, kx, ky, ph, om)
        return z

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

    # diagnostic: how much of the surface is bound wave?
    lin = (A[None] * np.cos(kx[None] * Xq.ravel()[:, None]
                            + ky[None] * Yq.ravel()[:, None]
                            - om[None] * t0 + ph[None])).sum(1)
    bw_frac = float(np.std(F.ravel() - lin) / (np.std(lin) + 1e-9))
    return P, Z, F, kp, t0, bw_frac


def build(n, nonlinear, seed):
    path = CACHE.format('nl' if nonlinear else 'lin')
    if os.path.exists(path):
        d = np.load(path)
        if len(d['F']) >= n:
            return d['X'][:n], d['F'][:n], float(d['bw'])
    rng = np.random.default_rng(seed)
    Xs, Fs, bws, t0 = [], [], [], time.time()
    for i in range(n):
        P, Z, F, kp, ts, bw = scene_nl(rng, nonlinear)
        X, s, _ = complex_beamform(P, Z, kp, ts)
        Xs.append(X); Fs.append((F / s).astype(np.float32)); bws.append(bw)
        if (i + 1) % 5000 == 0:
            print(f"    {i+1}/{n} ({time.time()-t0:.0f}s)", flush=True)
    X, F = np.stack(Xs), np.stack(Fs)
    bw = float(np.mean(bws))
    np.savez(path, X=X, F=F, bw=bw)
    return X, F, bw


def run(tag, X, F, n, steps):
    rv, Ws = S2.ridge_curve(X, F, [n])
    Xv = torch.tensor(X[:N_VAL]).to(DEV)
    Fv = torch.tensor(F[:N_VAL]).to(DEV)
    Xt = torch.tensor(X[N_VAL:N_VAL + n]).to(DEV)
    Ft = torch.tensor(F[N_VAL:N_VAL + n]).to(DEV)
    W0 = torch.tensor(Ws[n].T, dtype=torch.float32, device=DEV)
    w2 = S2.train('wiener2', W0, Xt, Ft, Xv, Fv, steps)
    rs = S2.train('resid', W0, Xt, Ft, Xv, Fv, steps)
    print(f"  [{tag}] ridge {rv[n]:.4f}  wiener2 {w2:.4f}  resid {rs:.4f}",
          flush=True)
    del Xt, Ft
    torch.cuda.empty_cache()
    return {'ridge': rv[n], 'wiener2': w2, 'resid': rs}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=25000)
    ap.add_argument('--steps', type=int, default=12000)
    a = ap.parse_args()
    res = json.load(open(OUT)) if os.path.exists(OUT) else {}

    for nl in (False, True):
        tag = 'nonlinear' if nl else 'linear'
        print(f"[{tag}] building {a.n} scenes...", flush=True)
        X, F, bw = build(a.n, nl, 4242 if not nl else 8484)
        print(f"  bound-wave / free-wave std ratio = {bw:.4f}", flush=True)
        r = run(tag, X, F, a.n - N_VAL, a.steps)
        r['bw_frac'] = bw
        res[tag] = r
        json.dump(res, open(OUT, 'w'), indent=1)
        del X, F

    print("\n=== registered prediction ===")
    print("  linear:    wiener2 < resid  (structure wins)")
    print("  nonlinear: resid   < wiener2 (freedom needed)  [lower is better]")
    for tag in ('linear', 'nonlinear'):
        if tag in res:
            r = res[tag]
            win = 'wiener2' if r['wiener2'] < r['resid'] else 'resid'
            print(f"  {tag:<10} ridge {r['ridge']:.4f}  wiener2 {r['wiener2']:.4f}"
                  f"  resid {r['resid']:.4f}  -> winner: {win}")
    print(f"\n-> {OUT}")


if __name__ == '__main__':
    sys.exit(main())
