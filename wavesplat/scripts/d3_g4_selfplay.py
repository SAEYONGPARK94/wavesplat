"""G4: attack the guarded estimator, and see whether the guard survives.

G3b decomposed the 6.1-sigma failure: 55% of it was a DC offset from a 500 m
swell that the bank (lam <= 228 m) cannot represent at all, and 45% was a
genuine variance blow-up (estimated field std 4.26 vs the sea's 0.98). The
production pipeline demeans its observations, so the DC half is an artifact of
the test harness, not a threat. The variance half has no safeguard.

Two rounds, both on the DEMEANED field, so the adversary can no longer win by
lifting a long crest over the fan and must attack the variance directly.

  round 1   ascent through the RAW solver.        Does the blow-up survive
                                                  demeaning?
  round 2   ascent through the GUARDED solver.    The energy cap is inside the
                                                  computation graph, so the
                                                  attacker sees it and can try
                                                  to defeat it.

The guard, calibrated without ever touching the truth:

    beta = min(1, std(z_demeaned) / std(p_demeaned))     p <- beta . p

It caps the field's amplitude at what the observed variance supports. Round 2
is the honest test: a safeguard that only works against an adversary who did
not know about it is not a safeguard.

WHAT WOULD FALSIFY THE GUARD: round 2 finds a sea whose guarded nRMSE clearly
exceeds the zero-predictor. What would establish it: round 2 stalls near the
zero-predictor, which is the bound the guard is supposed to certify.
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
                                   DEPTH, KP, NCOMP, SIGMA, NOISE, BAND)

OUT = 'results/direction3/g4_selfplay.json'
S0 = 0.05 * 4 * SIGMA


def dm(v):
    return v - v.mean()


def estimate(P, z, Pq, guard):
    pr = mmse_gpu(P, z, DEPTH, Pq, sigma0=S0, kp_hint=KP, band=BAND)
    pr = dm(pr)
    if not guard:
        return pr, torch.tensor(1.0, device=DEV)
    beta = torch.clamp(dm(z).std() / (pr.std() + 1e-9), max=1.0)
    return beta * pr, beta


def score(P, Pq, kx, ky, om, a, ph, gen, guard):
    z = eta(P, kx, ky, om, a, ph)
    z = z + NOISE * torch.randn(len(z), device=DEV, generator=gen)
    gt = dm(eta(Pq, kx, ky, om, a, ph))
    pr, beta = estimate(P, z, Pq, guard)
    # Normalise by the sea's FIXED spectral std, never by a realised statistic
    # of the truth. Dividing by gt.std() -- which is what the first version of
    # this file did, one day after documenting the same error as #16 -- lets the
    # adversary win by driving the denominator to zero: it pumps the energy into
    # waves so long that the DEMEANED truth is nearly flat over the fan, and
    # scores nRMSE = 73 while the estimate is not large at all (beta stayed at
    # 1.000 while the "amplitude ratio" read 73).
    nr = torch.sqrt(((pr - gt) ** 2).mean()) / SIGMA
    return nr, beta, pr, gt


def ascend(P, Pq, gen, guard, steps, seed):
    rng = np.random.default_rng(seed)
    kx, ky, om, k = sea_shell(rng)
    raw = torch.tensor(rng.normal(0, 1, NCOMP), device=DEV, dtype=torch.float32,
                       requires_grad=True)
    ph = torch.tensor(rng.uniform(-np.pi, np.pi, NCOMP), device=DEV,
                      dtype=torch.float32, requires_grad=True)
    opt = torch.optim.Adam([raw, ph], lr=0.05)
    for s in range(steps):
        nr, beta, _, _ = score(P, Pq, kx, ky, om, norm_amp(raw), ph, gen, guard)
        opt.zero_grad(set_to_none=True)
        (-nr).backward()
        opt.step()
        if (s + 1) % 100 == 0:
            print(f"    step {s+1:4d}  nRMSE/zero {float(nr):.3f}  "
                  f"beta {float(beta):.3f}", flush=True)
    with torch.no_grad():
        a = norm_amp(raw).detach()
        ph = ph.detach()
        nr_g, beta, pr, gt = score(P, Pq, kx, ky, om, a, ph, gen, True)
        nr_r, _, prr, _ = score(P, Pq, kx, ky, om, a, ph, gen, False)
        zero = float(gt.std() / SIGMA)          # nRMSE of predicting zero
    return dict(nrmse_raw=float(nr_r), nrmse_guarded=float(nr_g), zero=zero,
                ratio_raw=float(nr_r) / max(zero, 1e-6),
                ratio_guarded=float(nr_g) / max(zero, 1e-6),
                beta=float(beta), amp_raw=float(prr.std() / SIGMA),
                amp_guarded=float(pr.std() / SIGMA))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--steps', type=int, default=300)
    ap.add_argument('--restarts', type=int, default=3)
    ap.add_argument('--null', type=int, default=96)
    a_ = ap.parse_args()

    P, Pq = geometry()
    gen = torch.Generator(device=DEV).manual_seed(0)
    print("all quantities are DEMEANED over the fan; nRMSE is normalised by "
          "the sea's\nfixed spectral std. The zero-predictor's score is "
          "reported per sea.\n")

    rng = np.random.default_rng(5)
    nr, ng = [], []
    for _ in range(a_.null):
        kx, ky, om, k = sea_shell(rng)
        a = norm_amp(torch.tensor(rng.normal(0, 1, NCOMP), device=DEV,
                                  dtype=torch.float32))
        ph = torch.tensor(rng.uniform(-np.pi, np.pi, NCOMP), device=DEV,
                          dtype=torch.float32)
        with torch.no_grad():
            nr.append(float(score(P, Pq, kx, ky, om, a, ph, gen, False)[0]))
            ng.append(float(score(P, Pq, kx, ky, om, a, ph, gen, True)[0]))
    nr, ng = np.array(nr), np.array(ng)
    print(f"null ({a_.null} random seas)  raw {nr.mean():.3f}±{nr.std():.3f} "
          f"(p99 {np.quantile(nr,0.99):.3f})   "
          f"guarded {ng.mean():.3f}±{ng.std():.3f}")
    print(f"guard's cost on healthy seas: {ng.mean()-nr.mean():+.4f}\n")

    res = {}
    for guard in (False, True):
        tag = 'GUARDED' if guard else 'RAW'
        print(f"--- round: ascent through the {tag} estimator ---", flush=True)
        runs = [ascend(P, Pq, gen, guard, a_.steps, 100 + r)
                for r in range(a_.restarts)]
        best = max(runs, key=lambda d: d['ratio_guarded' if guard
                                        else 'ratio_raw'])
        res[tag] = runs
        print(f"  best: raw {best['nrmse_raw']:.3f}   "
              f"guarded {best['nrmse_guarded']:.3f}   "
              f"zero-pred {best['zero']:.3f}   "
              f"ratio(guarded/zero) {best['ratio_guarded']:.2f}   "
              f"beta {best['beta']:.3f}   "
              f"amp {best['amp_raw']:.2f} -> {best['amp_guarded']:.2f}\n",
              flush=True)

    b_raw = max(r['ratio_guarded'] for r in res['RAW'])
    b_gua = max(r['ratio_guarded'] for r in res['GUARDED'])
    print("=== G4 ===")
    print(f"  attacker blind to the guard:  guarded/zero {b_raw:.2f}")
    print(f"  attacker aware of the guard:  guarded/zero {b_gua:.2f}")
    print(f"  1.00 = as good as predicting zero")
    if b_gua < 1.3:
        print("  the guard holds: an adversary optimising against it cannot "
              "push the\n  error meaningfully past doing nothing.")
    else:
        print("  the guard is defeated once the adversary can see it.")

    os.makedirs('results/direction3', exist_ok=True)
    with open(OUT, 'w') as f:
        json.dump({'null_raw_mean': float(nr.mean()),
                   'null_guarded_mean': float(ng.mean()),
                   'null_raw_p99': float(np.quantile(nr, 0.99)),
                   'rounds': res, 'best_blind': b_raw, 'best_aware': b_gua},
                  f, indent=2)
    return 0


if __name__ == '__main__':
    sys.exit(main())
