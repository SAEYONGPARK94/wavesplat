"""B1: a bound the estimator cannot break, and how tight it is.

A3 showed that an adversary who knows the defence, and differentiates through
it, cannot push the error past 1.090x the zero-predictor. That is an empirical
worst case over a non-convex search -- it is not a guarantee. A reviewer will
ask what happens on the sea we did not find.

The energy cap gives a guarantee, for free, with no assumptions on the sea:

    beta = min(1, std(z~) / std(p))       enforces   std(beta.p) <= std(z~)

where z~ is the demeaned observation vector. Both the capped field and the
truth are zero-mean over the query set, so by the triangle inequality in L2,

    RMSE(beta.p, gt)  <=  std(beta.p) + std(gt)  <=  std(z~) + std(gt)

and dividing by the zero-predictor's error, which is exactly std(gt):

    ratio  <=  1 + kappa,        kappa = std(z~) / std(gt)

kappa is the observed sea's std over the fan divided by the truth's std over
the scored domain. Both measure the same sea; kappa ~ 1 for any geometry that
observes the region it scores. The certificate is therefore ~2x, and it holds
for EVERY sea, including ones no search will ever find.

Without the cap there is no such bound: the raw solver reached 4.2x here and
its amplitude is limited only by the conditioning of the bank.

This script (a) verifies the cap really enforces the amplitude constraint,
(b) measures kappa and the certificate 1+kappa across populations, and
(c) reports how much slack the certificate leaves against the achieved worst
case -- the gap between what we can prove and what an attacker can do.
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
from d3_a3_selfplay_vs_defense import pick_band, dm, CANDS, S0

OUT = 'results/direction3/b1_certified_bound.json'


def capped(P, z, Pq, adaptive, gen):
    band = pick_band(P, z, gen) if adaptive else CANDS[0]
    pr = dm(mmse_gpu(P, z, DEPTH, Pq, sigma0=S0, kp_hint=KP, band=band))
    beta = torch.clamp(dm(z).std() / (pr.std() + 1e-9), max=1.0)
    return beta * pr, pr, float(beta)


def one(P, Pq, kx, ky, om, a, ph, gen, adaptive=True):
    with torch.no_grad():
        z = eta(P, kx, ky, om, a, ph)
        z = z + NOISE * torch.randn(len(z), device=DEV, generator=gen)
        gt = dm(eta(Pq, kx, ky, om, a, ph))
        pc, praw, beta = capped(P, z, Pq, adaptive, gen)

        s_z = float(dm(z).std())
        s_gt = float(gt.std())
        s_pc = float(pc.std())
        s_pr = float(praw.std())
        rmse = float(torch.sqrt(((pc - gt) ** 2).mean()))
        raw_rmse = float(torch.sqrt(((praw - gt) ** 2).mean()))
    kappa = s_z / max(s_gt, 1e-9)
    return dict(ratio=rmse / max(s_gt, 1e-9),
                ratio_raw=raw_rmse / max(s_gt, 1e-9),
                kappa=kappa, cert=1.0 + kappa, beta=beta,
                amp_ok=bool(s_pc <= s_z + 1e-5),
                amp_capped=s_pc / max(s_z, 1e-9),
                amp_raw=s_pr / max(s_z, 1e-9))


def adversary(P, Pq, gen, steps, seed, adaptive):
    rng = np.random.default_rng(seed)
    kx, ky, om, k = sea_shell(rng)
    raw = torch.tensor(rng.normal(0, 1, NCOMP), device=DEV, dtype=torch.float32,
                       requires_grad=True)
    ph = torch.tensor(rng.uniform(-np.pi, np.pi, NCOMP), device=DEV,
                      dtype=torch.float32, requires_grad=True)
    opt = torch.optim.Adam([raw, ph], lr=0.05)
    for s in range(steps):
        a = norm_amp(raw)
        z = eta(P, kx, ky, om, a, ph)
        z = z + NOISE * torch.randn(len(z), device=DEV, generator=gen)
        gt = dm(eta(Pq, kx, ky, om, a, ph))
        band = pick_band(P, z, gen) if adaptive else CANDS[0]
        pr = dm(mmse_gpu(P, z, DEPTH, Pq, sigma0=S0, kp_hint=KP, band=band))
        beta = torch.clamp(dm(z).std() / (pr.std() + 1e-9), max=1.0)
        loss = -torch.sqrt((((beta * pr) - gt) ** 2).mean()) / SIGMA
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    return kx, ky, om, norm_amp(raw).detach(), ph.detach()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=96)
    ap.add_argument('--steps', type=int, default=1200)
    a_ = ap.parse_args()

    P, Pq = geometry()
    gen = torch.Generator(device=DEV).manual_seed(0)
    res = {}

    print(f"{'population':<16}{'n':>5}{'ratio':>9}{'raw':>9}"
          f"{'kappa':>9}{'cert 1+k':>11}{'slack':>9}{'amp ok':>9}")

    rng = np.random.default_rng(11)
    rows = []
    for _ in range(a_.n):
        kx, ky, om, k = sea_shell(rng)
        a = norm_amp(torch.tensor(rng.normal(0, 1, NCOMP), device=DEV,
                                  dtype=torch.float32))
        ph = torch.tensor(rng.uniform(-np.pi, np.pi, NCOMP), device=DEV,
                          dtype=torch.float32)
        rows.append(one(P, Pq, kx, ky, om, a, ph, gen))
    def summ(tag, rows):
        r = np.array([x['ratio'] for x in rows])
        rr = np.array([x['ratio_raw'] for x in rows])
        kk = np.array([x['kappa'] for x in rows])
        ok = all(x['amp_ok'] for x in rows)
        slack = np.mean(1 + kk - r)
        print(f"{tag:<16}{len(rows):>5}{r.mean():>9.3f}{rr.mean():>9.3f}"
              f"{kk.mean():>9.3f}{np.mean(1+kk):>11.3f}{slack:>9.3f}"
              f"{str(ok):>9}")
        return dict(n=len(rows), ratio=r.mean(), ratio_max=r.max(),
                    ratio_raw=rr.mean(), kappa=kk.mean(),
                    cert=float(np.mean(1 + kk)), amp_ok=ok,
                    viol=int((r > 1 + kk).sum()))
    res['random'] = summ('random seas', rows)

    for adaptive, tag in ((False, 'ADV vs cap'), (True, 'ADV vs full')):
        seas = [adversary(P, Pq, gen, a_.steps, 300 + i, adaptive)
                for i in range(2)]
        rows = [one(P, Pq, *s, gen, adaptive=adaptive) for s in seas]
        res[tag] = summ(tag, rows)

    print("\ncert = 1 + kappa is a DETERMINISTIC upper bound on ratio, from the "
          "triangle\ninequality and std(beta.p) <= std(z~). It assumes nothing "
          "about the sea.")
    v = sum(res[k]['viol'] for k in res)
    print(f"violations of the certificate across all seas: {v}")
    print(f"\nworst achieved ratio, informed adversary: "
          f"{res['ADV vs full']['ratio_max']:.3f}")
    print(f"certificate there:                        "
          f"{res['ADV vs full']['cert']:.3f}")
    print(f"raw solver, same seas:                    "
          f"{res['ADV vs full']['ratio_raw']:.3f}  (no bound exists)")

    os.makedirs('results/direction3', exist_ok=True)
    with open(OUT, 'w') as f:
        json.dump({k: {kk: (float(vv) if not isinstance(vv, bool) else vv)
                       for kk, vv in v_.items()} for k, v_ in res.items()},
                  f, indent=2)
    return 0


if __name__ == '__main__':
    sys.exit(main())
