"""A3: attack the DEFENDED estimator. A guard the adversary has not seen is not a guard.

The adversarial sea that started this arc was optimised against a solver with a
fixed [0.35, 2.5] kp bank and no safeguard. Since then two defences appeared:

  adaptive band   pick the bank's band by residual on 25% held-out observations
                  (G5: matches the oracle band; HOS-confirmed in A1)
  energy cap      beta = min(1, std(z) / std(p)), bounding the field's amplitude
                  by what the observed variance supports (G4: free on healthy
                  seas, cuts the blind attacker 4.82 -> 2.86)

Both were only ever tested against seas that predate them. That is the weakest
possible test. Here the whole defended pipeline sits inside the computation
graph and gradient ascent is free to defeat it -- including by driving the
selector into choosing the wrong band, which A1b suggests it can: at
lam/aperture ~ 3 the selector already missed the oracle by +0.092.

Three attackers, all on the DEMEANED field, all normalised by the sea's fixed
spectral std (never by a realised statistic of the truth -- that denominator
has now been attacked twice, errors #16 and #18):

  vs raw          fixed band, no cap.        The old result, for reference.
  vs cap          fixed band + energy cap.
  vs full         adaptive band + energy cap. The real question.

Each attacker's sea is then scored under ALL THREE defences, so we can see
whether a sea tuned against one transfers to the others.

WHAT WOULD ESTABLISH THE DEFENCE: the attacker against `full` cannot push the
ratio meaningfully past 1.0 (the zero-predictor), i.e. the worst it can do is
make the estimator useless, never harmful.
WHAT WOULD FALSIFY IT: it finds a sea whose defended ratio clearly exceeds the
worst physical cell of A1b (2.61x the oracle, 1.279 absolute).
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

OUT = 'results/direction3/a3_selfplay_vs_defense.json'
S0 = 0.05 * 4 * SIGMA
CANDS = [(0.35, 2.5), (0.25, 2.5), (0.15, 2.5), (0.15, 4.0)]
HELD = 0.25
MODES = ('raw', 'cap', 'full')


def dm(v):
    return v - v.mean()


def pick_band(P, z, gen):
    """Selector. Non-differentiable argmin, but it only chooses a branch; the
    gradient flows through whichever band it returns."""
    with torch.no_grad():
        n = len(z)
        idx = torch.randperm(n, device=DEV, generator=gen)
        tr, he = idx[:int((1 - HELD) * n)], idx[int((1 - HELD) * n):]
        best, bb = None, CANDS[0]
        for b in CANDS:
            pv = mmse_gpu(P[tr], z[tr], DEPTH, P[he], sigma0=S0, kp_hint=KP,
                          band=b)
            r = float(torch.sqrt(((pv - z[he]) ** 2).mean()))
            if best is None or r < best:
                best, bb = r, b
    return bb


def defended(P, z, Pq, mode, gen):
    band = pick_band(P, z, gen) if mode == 'full' else CANDS[0]
    pr = dm(mmse_gpu(P, z, DEPTH, Pq, sigma0=S0, kp_hint=KP, band=band))
    if mode == 'raw':
        return pr, torch.tensor(1.0, device=DEV), band
    beta = torch.clamp(dm(z).std() / (pr.std() + 1e-9), max=1.0)
    return beta * pr, beta, band


def score(P, Pq, kx, ky, om, a, ph, gen, mode):
    z = eta(P, kx, ky, om, a, ph)
    z = z + NOISE * torch.randn(len(z), device=DEV, generator=gen)
    gt = dm(eta(Pq, kx, ky, om, a, ph))
    pr, beta, band = defended(P, z, Pq, mode, gen)
    nr = torch.sqrt(((pr - gt) ** 2).mean()) / SIGMA
    return nr, beta, band, gt


def ascend(P, Pq, gen, mode, steps, seed):
    rng = np.random.default_rng(seed)
    kx, ky, om, k = sea_shell(rng)
    raw = torch.tensor(rng.normal(0, 1, NCOMP), device=DEV, dtype=torch.float32,
                       requires_grad=True)
    ph = torch.tensor(rng.uniform(-np.pi, np.pi, NCOMP), device=DEV,
                      dtype=torch.float32, requires_grad=True)
    opt = torch.optim.Adam([raw, ph], lr=0.05)
    hist = []
    for s in range(steps):
        nr, beta, band, _ = score(P, Pq, kx, ky, om, norm_amp(raw), ph, gen,
                                  mode)
        opt.zero_grad(set_to_none=True)
        (-nr).backward()
        opt.step()
        if (s + 1) % 250 == 0:
            hist.append(float(nr))
            print(f"    [{mode}] step {s+1:4d}  nRMSE {float(nr):.3f}  "
                  f"beta {float(beta):.3f}  band {band}", flush=True)
    return kx, ky, om, norm_amp(raw).detach(), ph.detach(), hist


def evaluate_all(P, Pq, kx, ky, om, a, ph, gen):
    """Score one sea under every defence, plus the zero-predictor."""
    out = {}
    with torch.no_grad():
        for m in MODES:
            nr, beta, band, gt = score(P, Pq, kx, ky, om, a, ph, gen, m)
            zero = float(gt.std() / SIGMA)
            out[m] = dict(nrmse=float(nr), ratio=float(nr) / max(zero, 1e-6),
                          beta=float(beta), band=list(band))
        out['zero'] = zero
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--steps', type=int, default=1500)
    ap.add_argument('--restarts', type=int, default=2)
    ap.add_argument('--null', type=int, default=64)
    a_ = ap.parse_args()

    P, Pq = geometry()
    gen = torch.Generator(device=DEV).manual_seed(0)

    rng = np.random.default_rng(5)
    null = {m: [] for m in MODES}
    for _ in range(a_.null):
        kx, ky, om, k = sea_shell(rng)
        a = norm_amp(torch.tensor(rng.normal(0, 1, NCOMP), device=DEV,
                                  dtype=torch.float32))
        ph = torch.tensor(rng.uniform(-np.pi, np.pi, NCOMP), device=DEV,
                          dtype=torch.float32)
        e = evaluate_all(P, Pq, kx, ky, om, a, ph, gen)
        for m in MODES:
            null[m].append(e[m]['ratio'])
    print("null (random seas), ratio to the zero-predictor:")
    for m in MODES:
        v = np.array(null[m])
        print(f"  {m:<5} {v.mean():.3f} ± {v.std():.3f}   p99 "
              f"{np.quantile(v, 0.99):.3f}")
    print(f"\ndefence cost on healthy seas: cap "
          f"{np.mean(null['cap'])-np.mean(null['raw']):+.4f}   full "
          f"{np.mean(null['full'])-np.mean(null['raw']):+.4f}\n")

    res = {}
    for mode in MODES:
        print(f"--- attacker optimising against `{mode}` ---", flush=True)
        best = None
        for r in range(a_.restarts):
            kx, ky, om, a, ph, hist = ascend(P, Pq, gen, mode, a_.steps,
                                             200 + r)
            e = evaluate_all(P, Pq, kx, ky, om, a, ph, gen)
            e['hist'] = hist
            if best is None or e[mode]['ratio'] > best[mode]['ratio']:
                best = e
        res[mode] = best
        print(f"  best sea (tuned vs {mode}), scored under each defence:")
        for m in MODES:
            print(f"    {m:<5} ratio {best[m]['ratio']:.3f}  "
                  f"(nRMSE {best[m]['nrmse']:.3f}, beta {best[m]['beta']:.3f}, "
                  f"band {tuple(best[m]['band'])})")
        print(f"    zero-predictor nRMSE {best['zero']:.3f}\n", flush=True)

    worst = res['full']['full']['ratio']
    print("=== A3 ===")
    print(f"  attacker aware of the full defence:  ratio {worst:.3f}")
    print(f"  worst PHYSICAL cell (A1b, HOS):      ratio 2.61 "
          f"(fixed band, no defence)")
    print(f"  zero-predictor:                      1.000")
    if worst < 1.3:
        print("  the defence holds: the worst an informed adversary achieves is "
              "to make\n  the estimator useless, not harmful.")
    else:
        print(f"  the defence does NOT bound the error: an informed adversary "
              f"reaches {worst:.2f}x\n  the zero-predictor.")

    os.makedirs('results/direction3', exist_ok=True)
    with open(OUT, 'w') as f:
        json.dump({'null': {m: [float(x) for x in null[m]] for m in MODES},
                   'attacks': res, 'worst_full': worst}, f, indent=2)
    return 0


if __name__ == '__main__':
    sys.exit(main())
