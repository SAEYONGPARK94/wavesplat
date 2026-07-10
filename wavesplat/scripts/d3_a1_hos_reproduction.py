"""A1: do today's results survive nonlinear (HOS) waves?

Everything found today -- the failure axis, the energy cap, the adaptive band
selector -- was measured on linear or second-order seas. The standing rule for
this project is that a headline must reproduce under HOS nonlinearity, so this
is the gate the rest of the work has to pass before anything is built on it.

The synthetic adversary cannot be used here: HOS is a stateful pseudo-spectral
integrator, not a differentiable function of the sea's amplitudes. That turns
out not to matter, because the adversary's sea was only ever a caricature of a
physical one -- energy below the bank's floor. The ocean supplies that for free
in the form of a SWELL running under a wind sea.

  wind        JONSWAP Tp = 8 s.  kp = 0.063, so lam_p = 100 m. All the energy
              sits inside the default band [0.35, 2.5] kp. Widening should
              cost, not pay -- this is the population that stands in for the
              real records, where widening hurt (LJ18g 0.550 -> 0.609).

  swell+wind  the same wind sea plus a Tp = 17 s swell at k = 0.020, which is
              0.32 kp -- BELOW the band's floor of 0.35 kp (lam = 314 m, and
              the observing aperture is 200 m). This is the failure axis,
              produced by physics rather than by gradient ascent.

Four estimators, as in G5: fixed floor 0.35, fixed floor 0.15, the oracle band
(chosen on the truth -- an upper bound we may not use), and the adaptive
selector that ranks candidate bands by their residual on 25% held-out
OBSERVATIONS. Plus the energy cap beta = min(1, std(z)/std(p)) on top.

The HOS box is 1000 x 600 m so the 314 m swell has room to exist; the scored
domain is the usual 240 x 200 m rectangle inside it.

WHAT WOULD FALSIFY TODAY'S RESULTS:
  - the fixed band is fine on swell+wind          -> the failure axis is an
                                                     artifact of the adversary
  - the selector picks wrong under nonlinearity   -> the residual ranking was
                                                     a linear-sea coincidence
  - the cap fires on the wind sea                 -> the safeguard is not free
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
os.chdir(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, 'scripts')

from wavesim.hos_simulator import HOSSimulator, HOSConfig
from wavesim.wave_generator import jonswap_components
from d3_gpu_mmse import mmse_gpu
import d3_static_field as SF

OUT = 'results/direction3/a1_hos_reproduction.json'
DEV = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
G = 9.81
DEPTH = 45.0
LX, LY, NX, NY = 1000.0, 600.0, 256, 128
TP_WIND, HS_WIND = 8.0, 2.0
TP_SWELL, HS_SWELL = 17.0, 1.6
CX, CY = 300.0, 300.0                 # sensor, so the domain sits in the box
NOISE = 0.02
CANDS = [(0.35, 2.5), (0.25, 2.5), (0.15, 2.5), (0.15, 4.0)]
HELD = 0.25


def wavenumber(Tp, depth):
    om = 2 * np.pi / Tp
    k = om ** 2 / G
    for _ in range(60):
        k = om ** 2 / (G * np.tanh(k * depth))
    return k


def make_sim(kind, seed):
    """A fresh simulator. HOS is stateful; every pass needs its own."""
    s = HOSSimulator(HOSConfig(Lx=LX, Ly=LY, Nx=NX, Ny=NY, depth=DEPTH, M=3))
    comps = jonswap_components(Hs=HS_WIND, Tp=TP_WIND, n_components=120,
                               depth=DEPTH, direction_deg=180.0,
                               direction_spread_deg=25.0, seed=9000 + seed)
    if kind == 'swell+wind':
        comps = comps + jonswap_components(
            Hs=HS_SWELL, Tp=TP_SWELL, n_components=40, depth=DEPTH,
            direction_deg=195.0, direction_spread_deg=12.0, seed=7000 + seed)
    s.initialize_from_components(comps)
    return s


def observe(sim, rng, t0):
    """Fan observations, strictly ascending in time (HOS cannot go back)."""
    P, z = [], []
    for t_s in np.arange(t0 - SF.W_OBS, t0 + 1e-9, SF.SCAN_DT):
        while sim.t < t_s - 1e-9:
            sim.step(min(0.1, t_s - sim.t))
        rr = rng.uniform(*SF.R_FAN, SF.M_PER_SCAN)
        aa = rng.uniform(-SF.HALF_FOV, SF.HALF_FOV, SF.M_PER_SCAN)
        xs, ys = CX + rr * np.cos(aa), CY + rr * np.sin(aa)
        keep = ((xs >= CX + SF.DOM_X[0]) & (xs <= CX + SF.DOM_X[1]) &
                (ys >= CY + SF.DOM_Y[0]) & (ys <= CY + SF.DOM_Y[1]))
        xs, ys = xs[keep], ys[keep]
        if len(xs) == 0:
            continue
        zz = sim.get_eta_at(np.mod(xs, LX), np.mod(ys, LY))
        P.append(np.column_stack([xs, ys, np.full(len(xs), t_s)]))
        z.append(zz + rng.normal(0, NOISE, len(xs)))
    P, z = np.vstack(P), np.concatenate(z)
    if len(z) > SF.M_CAP:
        ii = rng.choice(len(z), SF.M_CAP, replace=False)
        P, z = P[ii], z[ii]
    return P, z


def truth(sim, t0):
    while sim.t < t0 - 1e-9:
        sim.step(min(0.1, t0 - sim.t))
    gx = np.arange(CX + SF.DOM_X[0], CX + SF.DOM_X[1] + 0.1, SF.DX)
    gy = np.arange(CY + SF.DOM_Y[0], CY + SF.DOM_Y[1] + 0.1, SF.DY)
    X, Y = np.meshgrid(gx, gy)
    ins, _ = SF.fan_mask(X, Y, CX, CY, np.array([1.0, 0.0]))
    xr, yr = X.ravel()[ins.ravel()], Y.ravel()[ins.ravel()]
    gt = sim.get_eta_at(np.mod(xr, LX), np.mod(yr, LY))
    Pq = np.column_stack([xr, yr, np.full(len(xr), t0)])
    return Pq, gt


def solve(P, z, Pq, band, kp, sig0):
    pr = mmse_gpu(P, z, DEPTH, Pq, sigma0=sig0, kp_hint=kp, band=band)
    return pr - pr.mean()


def select_band(P, z, kp, sig0, gen):
    n = len(z)
    idx = torch.randperm(n, device=DEV, generator=gen)
    tr, he = idx[:int((1 - HELD) * n)], idx[int((1 - HELD) * n):]
    best, bb = None, None
    for b in CANDS:
        pv = mmse_gpu(P[tr], z[tr], DEPTH, P[he], sigma0=sig0, kp_hint=kp,
                      band=b)
        r = float(torch.sqrt(((pv - z[he]) ** 2).mean()))
        if best is None or r < best:
            best, bb = r, b
    return bb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', type=int, default=4)
    a_ = ap.parse_args()

    kp = wavenumber(TP_WIND, DEPTH)
    ks = wavenumber(TP_SWELL, DEPTH)
    print(f"wind  kp {kp:.4f}  (lam {2*np.pi/kp:.0f} m)")
    print(f"swell k  {ks:.4f}  (lam {2*np.pi/ks:.0f} m)  = {ks/kp:.2f} kp"
          f"   band floor is 0.35 kp -> lam {2*np.pi/(0.35*kp):.0f} m")
    print(f"observing aperture: {SF.R_FAN[1]-SF.R_FAN[0]:.0f} m\n")

    gen = torch.Generator(device=DEV).manual_seed(0)
    t0 = 10.0 + SF.W_OBS
    sig0 = 0.05 * HS_WIND
    res = {}

    print(f"{'sea':<14}{'fixed .35':>11}{'fixed .15':>11}{'oracle':>9}"
          f"{'ADAPTIVE':>10}{'+cap':>8}{'beta':>7}   picks")
    for kind in ('wind', 'swell+wind'):
        rows, picks, betas = [], [], []
        for s in range(a_.seeds):
            rng = np.random.default_rng(100 + s)
            sim = make_sim(kind, s)
            P, z = observe(sim, rng, t0)
            Pq, gt = truth(sim, t0)
            Pt = torch.tensor(P, device=DEV, dtype=torch.float32)
            zt = torch.tensor(z, device=DEV, dtype=torch.float32)
            Qt = torch.tensor(Pq, device=DEV, dtype=torch.float32)
            gtt = torch.tensor(gt, device=DEV, dtype=torch.float32)
            gtt = gtt - gtt.mean()
            zero = float(gtt.std())

            def ratio(b):
                pr = solve(Pt, zt, Qt, b, kp, sig0)
                return (float(torch.sqrt(((pr - gtt) ** 2).mean())) / zero, pr)

            r35 = ratio(CANDS[0])[0]
            r15 = ratio((0.15, 2.5))[0]
            orc = min(ratio(b)[0] for b in CANDS)
            sel = select_band(Pt, zt, kp, sig0, gen)
            rad, pr = ratio(sel)
            beta = min(1.0, float(zt.std()) / float(pr.std() + 1e-9))
            rcap = float(torch.sqrt(((beta * pr - gtt) ** 2).mean())) / zero
            rows.append((r35, r15, orc, rad, rcap))
            picks.append(sel)
            betas.append(beta)
        v = np.array(rows)
        wide = sum(1 for p in picks if p[0] <= 0.15) / len(picks)
        print(f"{kind:<14}{v[:,0].mean():>11.3f}{v[:,1].mean():>11.3f}"
              f"{v[:,2].mean():>9.3f}{v[:,3].mean():>10.3f}{v[:,4].mean():>8.3f}"
              f"{np.mean(betas):>7.3f}   floor<=0.15 in {wide*100:.0f}%")
        res[kind] = dict(fixed35=v[:, 0].mean(), fixed15=v[:, 1].mean(),
                         oracle=v[:, 2].mean(), adaptive=v[:, 3].mean(),
                         capped=v[:, 4].mean(), beta=float(np.mean(betas)),
                         frac_wide=wide)

    print("\nratios are nRMSE / zero-predictor; 1.00 = as good as doing nothing")
    w, sw = res['wind'], res['swell+wind']
    print(f"\nfailure axis under HOS:   fixed .35 on swell+wind = "
          f"{sw['fixed35']:.3f}  vs  {sw['fixed15']:.3f} with the floor lowered")
    print(f"selector's cost on wind:  {w['adaptive']-w['fixed35']:+.4f}   "
          f"(it widened on {w['frac_wide']*100:.0f}% of them)")
    print(f"selector vs oracle:       wind {w['adaptive']-w['oracle']:+.4f}   "
          f"swell+wind {sw['adaptive']-sw['oracle']:+.4f}")
    print(f"cap on the healthy sea:   beta {w['beta']:.3f}  "
          f"(1.000 = never fired)")

    os.makedirs('results/direction3', exist_ok=True)
    with open(OUT, 'w') as f:
        json.dump({k: {kk: float(vv) for kk, vv in v.items()}
                   for k, v in res.items()}, f, indent=2)
    return 0


if __name__ == '__main__':
    sys.exit(main())
