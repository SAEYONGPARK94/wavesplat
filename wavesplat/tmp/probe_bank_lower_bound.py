"""The diagnosis prescribes one thing: lower the bank's floor. Does it work?

G3b: the adversary lifts a lam ~ 500 m swell over the fan. The bank's lowest
atom is 0.35 kp, i.e. lam = 228 m -- longer than the 200 m aperture itself. The
component is not merely hard to estimate, it is not in the model.

So widen downward. But nothing is free: yesterday's band experiments showed
that widening costs accuracy on sparse observations (LJ18g in-fan 0.550 ->
0.609), a degrees-of-freedom vs information trade.

Measured on the SAME two populations, demeaned, normalised by the sea's fixed
sigma, zero-predictor reported per sea:

  healthy   64 random seas drawn on the shell   -- what widening COSTS
  attack    the G3 adversarial sea               -- what widening BUYS

If the floor is the whole story, the attack's ratio collapses toward 1 and the
healthy population barely moves. If widening buys nothing, the failure is not
representational and the DC diagnosis was incomplete.
"""
import os
import sys

import numpy as np
import torch

os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, 'scripts')

from d3_gpu_mmse import mmse_gpu
from d3_g3_adversarial_sea import (geometry, eta, norm_amp, sea_shell, DEV,
                                   DEPTH, KP, NCOMP, SIGMA, NOISE)

FLOORS = [0.35, 0.25, 0.15, 0.10]
HI = 2.5
S0 = 0.05 * 4 * SIGMA
STEPS = 400


def dm(v):
    return v - v.mean()


def run(P, Pq, kx, ky, om, a, ph, gen, band):
    z = eta(P, kx, ky, om, a, ph)
    z = z + NOISE * torch.randn(len(z), device=DEV, generator=gen)
    gt = dm(eta(Pq, kx, ky, om, a, ph))
    pr = dm(mmse_gpu(P, z, DEPTH, Pq, sigma0=S0, kp_hint=KP, band=band))
    nr = float(torch.sqrt(((pr - gt) ** 2).mean()) / SIGMA)
    zero = float(gt.std() / SIGMA)
    return nr / max(zero, 1e-6)


def main():
    P, Pq = geometry()
    gen = torch.Generator(device=DEV).manual_seed(0)

    # the adversary, found against the default band
    rng = np.random.default_rng(101)
    kx, ky, om, k = sea_shell(rng)
    raw = torch.tensor(rng.normal(0, 1, NCOMP), device=DEV, dtype=torch.float32,
                       requires_grad=True)
    ph = torch.tensor(rng.uniform(-np.pi, np.pi, NCOMP), device=DEV,
                      dtype=torch.float32, requires_grad=True)
    opt = torch.optim.Adam([raw, ph], lr=0.05)
    for _ in range(STEPS):
        amp = norm_amp(raw)
        z = eta(P, kx, ky, om, amp, ph)
        z = z + NOISE * torch.randn(len(z), device=DEV, generator=gen)
        gt = dm(eta(Pq, kx, ky, om, amp, ph))
        pr = dm(mmse_gpu(P, z, DEPTH, Pq, sigma0=S0, kp_hint=KP,
                         band=(0.35, HI)))
        loss = -torch.sqrt(((pr - gt) ** 2).mean()) / SIGMA
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    a_adv, ph_adv = norm_amp(raw).detach(), ph.detach()
    kadv = (kx, ky, om)

    print(f"{'bank floor':<14}{'atoms':>8}{'healthy x64':>14}{'adversary':>12}")
    for lo in FLOORS:
        band = (lo, HI)
        heal = []
        for s in range(64):
            r2 = np.random.default_rng(300 + s)
            kx2, ky2, om2, _ = sea_shell(r2)
            a2 = norm_amp(torch.tensor(r2.normal(0, 1, NCOMP), device=DEV,
                                       dtype=torch.float32))
            p2 = torch.tensor(r2.uniform(-np.pi, np.pi, NCOMP), device=DEV,
                              dtype=torch.float32)
            with torch.no_grad():
                heal.append(run(P, Pq, kx2, ky2, om2, a2, p2, gen, band))
        with torch.no_grad():
            adv = run(P, Pq, *kadv, a_adv, ph_adv, gen, band)
        # bank size, for the degrees-of-freedom cost
        dk = KP / 10
        ax = np.arange(-HI * KP, HI * KP + dk / 2, dk)
        KX, KY = np.meshgrid(ax, ax)
        kk = np.hypot(KX, KY)
        nb = int(((kk > lo * KP) & (kk < HI * KP)).sum())
        print(f"{f'{lo} kp':<14}{nb:>8}{np.mean(heal):>14.3f}{adv:>12.3f}")

    print("\nratios are nRMSE / zero-predictor. 1.00 = as good as doing nothing.")
    print("lam at the floor:  0.35 kp -> 228 m,  0.15 kp -> 533 m "
          "(aperture is 200 m).")


if __name__ == '__main__':
    sys.exit(main())
