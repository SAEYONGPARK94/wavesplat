"""A safeguard against the blow-up, calibrated without ever seeing the truth.

The adversarial sea drives the field to 6.1 sea-sigma of error and 4.3x the
sea's amplitude -- five times worse than predicting zero. The null-space story
is dead: its observation residual is 1.39 m, sixty-nine times the 0.02 m noise
floor. The failure is visible from the observations alone.

So shrink on the observations. Hold out 25% of the fan points, evaluate the
estimate AT those points, and pick the scalar that best explains them:

    alpha* = <p_held, z_held> / <p_held, p_held>

then report alpha* . field. Nothing here touches the truth. If the estimate is
blown up, its predictions at the held-out points overshoot, alpha* < 1, and the
field is pulled back. If the estimate is good, alpha* ~ 1 and nothing happens.
As alpha -> 0 the answer degrades to the zero-predictor, whose nRMSE is bounded
by gt_std / sigma <= 1.4 -- so the safeguard CERTIFIES an upper bound that the
raw estimator does not have.

WHAT WOULD FALSIFY IT: alpha* < 1 on the healthy in-band sea (the safeguard
costs accuracy when nothing is wrong), or alpha* ~ 1 on the adversarial sea
(it does not fire when it must).

Reported over the adversarial sea, a healthy in-band sea, and 64 random seas.
"""
import os
import sys

import numpy as np
import torch

os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, 'scripts')

from d3_gpu_mmse import mmse_gpu
from d3_g3_adversarial_sea import (geometry, eta, norm_amp, sea_shell, DEV,
                                   DEPTH, KP, NCOMP, SIGMA, NOISE, BAND, G)

STEPS = 300
HELD = 0.25
S0 = 0.05 * 4 * SIGMA


def solve(P, z, Pq):
    return mmse_gpu(P, z, DEPTH, Pq, sigma0=S0, kp_hint=KP, band=BAND)


def guarded(P, z, Pq, gt):
    """Fit on 75% of the fan, shrink on the remaining 25%, score on the field."""
    n = len(z)
    idx = torch.randperm(n, device=DEV, generator=torch.Generator(device=DEV)
                         .manual_seed(3))
    tr, he = idx[:int((1 - HELD) * n)], idx[int((1 - HELD) * n):]
    with torch.no_grad():
        pr = solve(P[tr], z[tr], Pq)                 # field, fit on 75%
        ph = solve(P[tr], z[tr], P[he])              # same fit, at held-out obs
        num = float((ph * z[he]).sum())
        den = float((ph * ph).sum()) + 1e-12
        alpha = max(0.0, min(1.0, num / den))

        # Projection shrinkage cannot fix this failure, and the reason is
        # instructive: at the OBSERVED points the estimate still correlates
        # with z (alpha* = 0.68, residual below ||z||), while ON THE QUERY GRID
        # its amplitude is 4.3x the sea. The coefficients are large and
        # cancelling -- a Runge-type divergence between the samples, not a
        # uniform overshoot. A residual test is blind to it.
        #
        # Energy is not. The sea's variance is observable: std(z) estimates it.
        # Cap the field's amplitude at what the observations can support.
        beta = min(1.0, float(z.std()) / float(pr.std() + 1e-12))

        raw = float(torch.sqrt(((pr - gt) ** 2).mean()) / SIGMA)
        gua = float(torch.sqrt(((alpha * pr - gt) ** 2).mean()) / SIGMA)
        cap = float(torch.sqrt(((beta * pr - gt) ** 2).mean()) / SIGMA)
        both = float(torch.sqrt(((min(alpha, beta) * pr - gt) ** 2).mean())
                     / SIGMA)
        zero = float(gt.std() / SIGMA)
    return alpha, raw, gua, zero, beta, cap, both


def show(tag, r):
    al, rw, gu, zp, be, ca, bo = r
    print(f"{tag:<24}{al:>8.3f}{be:>7.3f}{rw:>8.3f}{gu:>8.3f}"
          f"{ca:>8.3f}{bo:>8.3f}{zp:>8.3f}")


def rand_sea(rng):
    kx, ky, om, k = sea_shell(rng)
    a = norm_amp(torch.tensor(rng.normal(0, 1, NCOMP), device=DEV,
                              dtype=torch.float32))
    ph = torch.tensor(rng.uniform(-np.pi, np.pi, NCOMP), device=DEV,
                      dtype=torch.float32)
    return kx, ky, om, a, ph


def main():
    P, Pq = geometry()
    gen = torch.Generator(device=DEV).manual_seed(0)
    print(f"{'sea':<24}{'alpha*':>8}{'beta':>7}{'raw':>8}{'proj':>8}"
          f"{'cap':>8}{'both':>8}{'zero':>8}")

    # healthy in-band sea
    rng = np.random.default_rng(7)
    kk = KP * np.exp(rng.uniform(np.log(0.5), np.log(2.0), NCOMP))
    th = rng.uniform(-np.pi, np.pi, NCOMP)
    t = lambda x: torch.tensor(x, device=DEV, dtype=torch.float32)
    a = norm_amp(t(np.abs(rng.normal(0, 1, NCOMP))))
    ph = t(rng.uniform(-np.pi, np.pi, NCOMP))
    kx, ky = t(kk * np.cos(th)), t(kk * np.sin(th))
    om = t(np.sqrt(G * kk * np.tanh(kk * DEPTH)))
    with torch.no_grad():
        z = eta(P, kx, ky, om, a, ph) + NOISE * torch.randn(
            len(P), device=DEV, generator=gen)
        gt = eta(Pq, kx, ky, om, a, ph)
    r = guarded(P, z, Pq, gt)
    show('in-band (healthy)', r)

    # random seas
    rows = []
    for s in range(64):
        rng = np.random.default_rng(200 + s)
        kx, ky, om, a, ph = rand_sea(rng)
        with torch.no_grad():
            z = eta(P, kx, ky, om, a, ph) + NOISE * torch.randn(
                len(P), device=DEV, generator=gen)
            gt = eta(Pq, kx, ky, om, a, ph)
        rows.append(guarded(P, z, Pq, gt))
    r = np.array(rows)
    show('random x64 (mean)', r.mean(0))
    show('random x64 (worst raw)', r[int(np.argmax(r[:, 1]))])

    # the adversary
    rng = np.random.default_rng(101)
    kx, ky, om, k = sea_shell(rng)
    raw = torch.tensor(rng.normal(0, 1, NCOMP), device=DEV, dtype=torch.float32,
                       requires_grad=True)
    php = torch.tensor(rng.uniform(-np.pi, np.pi, NCOMP), device=DEV,
                       dtype=torch.float32, requires_grad=True)
    opt = torch.optim.Adam([raw, php], lr=0.05)
    for _ in range(STEPS):
        amp = norm_amp(raw)
        z = eta(P, kx, ky, om, amp, php)
        z = z + NOISE * torch.randn(len(z), device=DEV, generator=gen)
        gt = eta(Pq, kx, ky, om, amp, php)
        pr = solve(P, z, Pq)
        loss = -torch.sqrt(((pr - gt) ** 2).mean()) / SIGMA
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    with torch.no_grad():
        amp, php_ = norm_amp(raw).detach(), php.detach()
        z = eta(P, kx, ky, om, amp, php_) + NOISE * torch.randn(
            len(P), device=DEV, generator=gen)
        gt = eta(Pq, kx, ky, om, amp, php_)
    r = guarded(P, z, Pq, gt)
    show('ADVERSARIAL', r)
    al, rw, gu, zp, be, ca, bo = r
    print(f"\nadversary:  raw {rw:.3f}  ->  projection {gu:.3f}  ->  "
          f"energy cap {ca:.3f}  (zero-predictor {zp:.3f})")
    print("projection shrinks along the estimate; the energy cap bounds its "
          "amplitude.")
    print("\nNOTE: the adversary was optimised against the RAW solver. A fair "
          "next\nstep is to re-run the ascent THROUGH the guarded estimator "
          "and see\nwhether it finds a new sea. That is the self-play "
          "experiment.")


if __name__ == '__main__':
    sys.exit(main())
