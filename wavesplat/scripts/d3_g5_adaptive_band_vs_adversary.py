"""G5: does the adaptive band selector survive the adversary?

Two of today's results point at the same prescription and disagree about which
way to move the band:

  synthetic adversary   lowering the floor 0.35 -> 0.15 kp kills the attack
                        (ratio 3.93 -> 0.27) and helps healthy seas too
  real LJ18g            widening the band HURT (in-fan 0.550 -> 0.609)

Both are right. Widen when there IS sub-band energy, do not otherwise. That is
exactly what `d3_band_selection.choose_band` does: it scores each candidate
band by its residual on 25% held-out OBSERVATIONS and never touches the truth.
Yesterday it matched the oracle band on real data (0.507 vs 0.504).

It has never been tested against a sea built to break it. The selector's
score is an observation residual, and G3b showed the adversarial sea's
observation residual is 69x the noise floor -- so the signal is there in
principle. The question is whether the residual RANKS the bands correctly when
the truth is unrepresentable.

Three populations, all demeaned, nRMSE normalised by the sea's fixed sigma and
reported as a ratio to the zero-predictor:

  healthy      64 random seas on the shell
  sub-band     seas with most energy below 0.35 kp (the adversary's region)
  ADVERSARIAL  found by ascent through the fixed-band solver

Four estimators: fixed 0.35 floor, fixed 0.15 floor, oracle-band (best of the
candidates, scored on the TRUTH -- an upper bound we may not use), and the
adaptive selector.

WHAT WOULD FALSIFY THE SELECTOR: it picks the narrow band on the adversarial
sea (residual fails to rank), or it pays a real cost on healthy seas.
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

from d3_gpu_mmse import mmse_gpu
from d3_g3_adversarial_sea import (geometry, eta, norm_amp, sea_shell, DEV,
                                   DEPTH, KP, NCOMP, SIGMA, NOISE)

OUT = 'results/direction3/g5_adaptive_band.json'
S0 = 0.05 * 4 * SIGMA
CANDS = [(0.35, 2.5), (0.25, 2.5), (0.15, 2.5), (0.15, 4.0)]
HELD = 0.25


def dm(v):
    return v - v.mean()


def field(P, z, Pq, band):
    return dm(mmse_gpu(P, z, DEPTH, Pq, sigma0=S0, kp_hint=KP, band=band))


def select_band(P, z, gen):
    """Rank candidate bands by residual on held-out observations. No truth."""
    n = len(z)
    idx = torch.randperm(n, device=DEV, generator=gen)
    tr, he = idx[:int((1 - HELD) * n)], idx[int((1 - HELD) * n):]
    best, bb = None, None
    for b in CANDS:
        pv = mmse_gpu(P[tr], z[tr], DEPTH, P[he], sigma0=S0, kp_hint=KP, band=b)
        r = float(torch.sqrt(((pv - z[he]) ** 2).mean()))
        if best is None or r < best:
            best, bb = r, b
    return bb, best


def evaluate(P, Pq, kx, ky, om, a, ph, gen):
    z = eta(P, kx, ky, om, a, ph)
    z = z + NOISE * torch.randn(len(z), device=DEV, generator=gen)
    gt = dm(eta(Pq, kx, ky, om, a, ph))
    zero = float(gt.std() / SIGMA)

    def ratio(b):
        pr = field(P, z, Pq, b)
        return float(torch.sqrt(((pr - gt) ** 2).mean()) / SIGMA) / max(zero, 1e-6)

    r_fix = ratio(CANDS[0])
    r_low = ratio((0.15, 2.5))
    r_orc = min(ratio(b) for b in CANDS)          # uses the truth: upper bound
    sel, _ = select_band(P, z, gen)
    r_ada = ratio(sel)
    return r_fix, r_low, r_orc, r_ada, sel


def sub_band_sea(rng):
    """Most of the energy below the default floor, like the adversary's."""
    n1, n2 = 64, 32
    k = np.concatenate([KP * np.exp(rng.uniform(np.log(0.15), np.log(0.33), n1)),
                        KP * np.exp(rng.uniform(np.log(0.5), np.log(2.0), n2))])
    th = rng.uniform(-np.pi, np.pi, len(k))
    a = np.abs(rng.normal(0, 1, len(k))) + 1e-3
    a[:n1] *= np.sqrt(0.85 * SIGMA ** 2 / ((a[:n1] ** 2).sum() / 2))
    a[n1:] *= np.sqrt(0.15 * SIGMA ** 2 / ((a[n1:] ** 2).sum() / 2))
    t = lambda x: torch.tensor(x, device=DEV, dtype=torch.float32)
    return (t(k * np.cos(th)), t(k * np.sin(th)),
            t(np.sqrt(9.81 * k * np.tanh(k * DEPTH))), t(a),
            t(rng.uniform(-np.pi, np.pi, len(k))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=48)
    ap.add_argument('--steps', type=int, default=1200)
    a_ = ap.parse_args()

    P, Pq = geometry()
    gen = torch.Generator(device=DEV).manual_seed(0)
    res = {}

    print(f"{'population':<16}{'fixed .35':>11}{'fixed .15':>11}"
          f"{'oracle':>9}{'ADAPTIVE':>10}   band picked")

    # THE CONTROL THE FIRST RUN LACKED. "healthy" seas are drawn on the full
    # [0.15, 5] kp shell, so they carry sub-band energy and the selector
    # correctly widened on 100% of them. That cannot show the selector's COST,
    # because no population in that run should have been narrowed. The real
    # records are the missing case: JONSWAP-like, energy concentrated near kp,
    # where widening HURT (LJ18g in-fan 0.550 -> 0.609). 'in-band' supplies it.
    for tag in ('healthy', 'in-band', 'sub-band'):
        rows, picks = [], []
        for s in range(a_.n):
            rng = np.random.default_rng(400 + s)
            if tag == 'healthy':
                kx, ky, om, k = sea_shell(rng)
                aa = norm_amp(torch.tensor(rng.normal(0, 1, NCOMP), device=DEV,
                                           dtype=torch.float32))
                ph = torch.tensor(rng.uniform(-np.pi, np.pi, NCOMP),
                                  device=DEV, dtype=torch.float32)
            elif tag == 'in-band':
                kk = KP * np.exp(rng.uniform(np.log(0.5), np.log(2.0), NCOMP))
                th = rng.uniform(-np.pi, np.pi, NCOMP)
                t = lambda x: torch.tensor(x, device=DEV, dtype=torch.float32)
                kx, ky = t(kk * np.cos(th)), t(kk * np.sin(th))
                om = t(np.sqrt(9.81 * kk * np.tanh(kk * DEPTH)))
                aa = norm_amp(t(rng.normal(0, 1, NCOMP)))
                ph = t(rng.uniform(-np.pi, np.pi, NCOMP))
            else:
                kx, ky, om, aa, ph = sub_band_sea(rng)
            with torch.no_grad():
                r = evaluate(P, Pq, kx, ky, om, aa, ph, gen)
            rows.append(r[:4])
            picks.append(r[4])
        v = np.array(rows)
        frac = sum(1 for p in picks if p[0] <= 0.15) / len(picks)
        print(f"{tag:<16}{v[:,0].mean():>11.3f}{v[:,1].mean():>11.3f}"
              f"{v[:,2].mean():>9.3f}{v[:,3].mean():>10.3f}"
              f"   floor<=0.15 in {frac*100:.0f}%")
        res[tag] = dict(fixed35=v[:, 0].mean(), fixed15=v[:, 1].mean(),
                        oracle=v[:, 2].mean(), adaptive=v[:, 3].mean(),
                        frac_low=frac)

    # the adversary, found against the FIXED narrow band
    rng = np.random.default_rng(101)
    kx, ky, om, k = sea_shell(rng)
    raw = torch.tensor(rng.normal(0, 1, NCOMP), device=DEV, dtype=torch.float32,
                       requires_grad=True)
    ph = torch.tensor(rng.uniform(-np.pi, np.pi, NCOMP), device=DEV,
                      dtype=torch.float32, requires_grad=True)
    opt = torch.optim.Adam([raw, ph], lr=0.05)
    for s in range(a_.steps):
        amp = norm_amp(raw)
        z = eta(P, kx, ky, om, amp, ph)
        z = z + NOISE * torch.randn(len(z), device=DEV, generator=gen)
        gt = dm(eta(Pq, kx, ky, om, amp, ph))
        pr = field(P, z, Pq, CANDS[0])
        loss = -torch.sqrt(((pr - gt) ** 2).mean()) / SIGMA
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if (s + 1) % 300 == 0:
            print(f"    ascent {s+1:4d}  nRMSE {-float(loss):.3f}", flush=True)
    with torch.no_grad():
        r = evaluate(P, Pq, kx, ky, om, norm_amp(raw).detach(), ph.detach(),
                     gen)
    print(f"{'ADVERSARIAL':<16}{r[0]:>11.3f}{r[1]:>11.3f}{r[2]:>9.3f}"
          f"{r[3]:>10.3f}   picked {r[4]}")
    res['adversarial'] = dict(fixed35=r[0], fixed15=r[1], oracle=r[2],
                              adaptive=r[3], band=list(r[4]))

    print("\nratios are nRMSE / zero-predictor; 1.00 = as good as doing nothing")
    ib = res['in-band']
    cost = ib['adaptive'] - ib['fixed35']
    print(f"\nselector's cost on the in-band population (where widening is "
          f"WRONG): {cost:+.4f}")
    print(f"  it narrowed on {(1-ib['frac_low'])*100:.0f}% of those seas")
    ok = r[3] < 1.3 and cost <= 0.02
    print("the selector holds: it widens when there is sub-band energy, and "
          "does not\npay for it when there is none." if ok else
          "the selector does NOT hold; see which population it fails on.")

    os.makedirs('results/direction3', exist_ok=True)
    with open(OUT, 'w') as f:
        json.dump({k: {kk: (float(vv) if not isinstance(vv, list) else vv)
                       for kk, vv in v.items()} for k, v in res.items()}, f,
                  indent=2)
    return 0


if __name__ == '__main__':
    sys.exit(main())
