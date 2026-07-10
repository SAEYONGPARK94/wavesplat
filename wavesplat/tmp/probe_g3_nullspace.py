"""Did the adversary find unexplainable signal, or the observation null space?

Two stories fit "nRMSE 6.3":

  A. BLOW-UP. The bank cannot explain the observations, and the fit runs away.
     Then the OBSERVATION residual is large, and any residual-triggered
     safeguard (rollback, adaptive band, gating on fit quality) will catch it.

  B. NULL SPACE. The bank explains the observations almost perfectly, with a
     large-amplitude in-band field that happens to agree with the true sea at
     the 600 fan points and disagree everywhere else. Then the residual is
     SMALL, no observation-based check can see it, and the prescription I was
     about to write (energy rollback on fit residual) is worthless.

The partial-explainability hypothesis already died: sweeping the in-band energy
fraction gives a monotone curve, max ratio 1.49, never 6x. So the adversary is
not exploiting "a little bit explainable". Story B is the remaining candidate.

Measured on the adversarial sea and on two controls (random sea, in-band sea):

  resid_all    RMS of (estimate - z) at the 600 observed points
  resid_held   same, on a 25% split the solver's hyper-selection never saw
  field_nrmse  what we actually care about
  amp_ratio    std of the estimated field / std of the true sea

Story B looks like: resid ~ noise floor (0.02 m), field_nrmse >> 1.
Story A looks like: resid >> noise floor.
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


def evaluate(P, Pq, kx, ky, om, a, ph, gen, tag):
    with torch.no_grad():
        z = eta(P, kx, ky, om, a, ph)
        z = z + NOISE * torch.randn(len(z), device=DEV, generator=gen)
        gt = eta(Pq, kx, ky, om, a, ph)

        # field
        pr = mmse_gpu(P, z, DEPTH, Pq, sigma0=0.05 * 4 * SIGMA, kp_hint=KP,
                      band=BAND)
        nr = float(torch.sqrt(((pr - gt) ** 2).mean()) / SIGMA)
        amp = float(pr.std() / SIGMA)

        # the same estimate, queried AT the observation points
        pz = mmse_gpu(P, z, DEPTH, P, sigma0=0.05 * 4 * SIGMA, kp_hint=KP,
                      band=BAND)
        res = pz - z
        r_all = float(torch.sqrt((res ** 2).mean()))

        # a split the fit never used for hyper-selection: last 25% of points
        n = len(z)
        h = slice(int(0.75 * n), n)
        r_held = float(torch.sqrt((res[h] ** 2).mean()))

    print(f"{tag:<22}{r_all:>11.4f}{r_held:>12.4f}{nr:>13.3f}{amp:>11.2f}")
    return r_all, nr


def main():
    P, Pq = geometry()
    gen = torch.Generator(device=DEV).manual_seed(0)
    print(f"observation noise floor sigma = {NOISE:.3f} m   sea std {SIGMA} m\n")
    print(f"{'sea':<22}{'resid_all':>11}{'resid_held':>12}"
          f"{'field_nrmse':>13}{'amp_ratio':>11}")

    # control 1: random sea on the full shell
    rng = np.random.default_rng(7)
    kx, ky, om, k = sea_shell(rng)
    a = norm_amp(torch.tensor(rng.normal(0, 1, NCOMP), device=DEV,
                              dtype=torch.float32))
    ph = torch.tensor(rng.uniform(-np.pi, np.pi, NCOMP), device=DEV,
                      dtype=torch.float32)
    evaluate(P, Pq, kx, ky, om, a, ph, gen, 'random (full shell)')

    # control 2: a well-posed in-band sea
    rng = np.random.default_rng(7)
    kk = KP * np.exp(rng.uniform(np.log(0.5), np.log(2.0), NCOMP))
    th = rng.uniform(-np.pi, np.pi, NCOMP)
    t = lambda x: torch.tensor(x, device=DEV, dtype=torch.float32)
    evaluate(P, Pq, t(kk * np.cos(th)), t(kk * np.sin(th)),
             t(np.sqrt(G * kk * np.tanh(kk * DEPTH))), a, ph, gen, 'in-band')

    # the adversary
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
        gt = eta(Pq, kx, ky, om, amp, ph)
        pr = mmse_gpu(P, z, DEPTH, Pq, sigma0=0.05 * 4 * SIGMA, kp_hint=KP,
                      band=BAND)
        loss = -torch.sqrt(((pr - gt) ** 2).mean()) / SIGMA
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    r_adv, nr_adv = evaluate(P, Pq, kx, ky, om, norm_amp(raw).detach(),
                             ph.detach(), gen, 'ADVERSARIAL')

    print(f"\nnoise floor {NOISE:.3f} m.  resid_all / noise = "
          f"{r_adv / NOISE:.1f}")
    if r_adv < 4 * NOISE and nr_adv > 3:
        print("NULL SPACE. The fit explains the observations to near the noise "
              "floor\nand the field is still wrong by "
              f"{nr_adv:.1f} sea-sigma. No residual-based\nsafeguard can see "
              "this. Energy rollback on fit quality is worthless here.")
    elif nr_adv > 3:
        print("BLOW-UP. The residual is large, so a residual-triggered rollback "
              "would fire.")
    else:
        print("the ascent did not reproduce the failure in this run.")


if __name__ == '__main__':
    sys.exit(main())
