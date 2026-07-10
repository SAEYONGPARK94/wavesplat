"""Is partial explainability the poison?

G3's adversarial sea put 88% of its energy below the bank's band and kept
9-24% inside it, and blew the estimate up to 6.3x the sea's own std -- five
times worse than predicting zero. Hand-built seas with ALL energy outside the
band only reached 1.57x. So it is not the out-of-band placement.

HYPOTHESIS. When the bank can explain nothing, it gives up and returns a small
field (a zero-predictor, ratio ~1). When it can explain everything, it is
accurate (ratio 0.17). The danger is in between: a few in-band atoms are asked
to account for a large unexplainable long-wave signal, and their coefficients
run away.

PREDICTION, and this is what makes it falsifiable: the ratio, as a function of
the in-band energy fraction f, must be NON-MONOTONE -- it must PEAK at small
positive f (the adversary chose f ~ 0.1) and fall on both sides. A monotone
curve kills the hypothesis: then the story is simply "more in-band energy is
better" and G3's sea was nothing special.

f = 0 and f = 1 are the two hand-built controls already measured.
"""
import os
import sys

import numpy as np
import torch

os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, 'scripts')

from d3_gpu_mmse import mmse_gpu
from d3_g3_adversarial_sea import (geometry, eta, DEV, DEPTH, KP, SIGMA,
                                   NOISE, BAND, G)

FRACS = [0.0, 0.02, 0.05, 0.10, 0.20, 0.35, 0.50, 0.75, 1.0]
SEEDS = (0, 1, 2, 3, 4, 5, 6, 7)
N_LONG, N_IN = 64, 32
LONG = (0.15, 0.30)          # below band
INB = (0.50, 2.00)           # inside [0.35, 2.5]


def comps(rng, klo, khi, n):
    k = KP * np.exp(rng.uniform(np.log(klo), np.log(khi), n))
    th = rng.uniform(-np.pi, np.pi, n)
    return k, th


def build(rng, f):
    """Sea with fraction f of the variance in-band, 1-f in long waves."""
    kl, tl = comps(rng, *LONG, N_LONG)
    ki, ti = comps(rng, *INB, N_IN)
    k = np.concatenate([kl, ki])
    th = np.concatenate([tl, ti])
    # amplitudes: random within each group, then scaled so var splits as f
    a = np.abs(rng.normal(0, 1, len(k))) + 1e-3
    gl, gi = slice(0, N_LONG), slice(N_LONG, None)
    for sl, share in ((gl, 1.0 - f), (gi, f)):
        e = (a[sl] ** 2).sum() / 2
        a[sl] = a[sl] * (np.sqrt(share * SIGMA ** 2 / e) if e > 0 else 0.0)
    ph = rng.uniform(-np.pi, np.pi, len(k))
    t = lambda x: torch.tensor(x, device=DEV, dtype=torch.float32)
    om = np.sqrt(G * k * np.tanh(k * DEPTH))
    return t(k * np.cos(th)), t(k * np.sin(th)), t(om), t(a), t(ph)


def main():
    P, Pq = geometry()
    gen = torch.Generator(device=DEV).manual_seed(0)
    print(f"in-band = [{INB[0]}, {INB[1]}] kp   long = [{LONG[0]}, {LONG[1]}] kp"
          f"   bank band = {BAND}   sea std {SIGMA} m\n")
    print(f"{'f (in-band var)':<17}{'nRMSE':>9}{'zero-pred':>11}{'ratio':>9}"
          f"{'pred sd':>10}{'z-std':>8}")

    out = []
    for f in FRACS:
        rows = []
        for sd in SEEDS:
            rng = np.random.default_rng(1000 + sd)
            kx, ky, om, a, ph = build(rng, f)
            with torch.no_grad():
                z = eta(P, kx, ky, om, a, ph)
                z = z + NOISE * torch.randn(len(z), device=DEV, generator=gen)
                gt = eta(Pq, kx, ky, om, a, ph)
                pr, var = mmse_gpu(P, z, DEPTH, Pq, sigma0=0.05 * 4 * SIGMA,
                                   kp_hint=KP, band=BAND, return_var=True)
                sdv = torch.sqrt(var.clamp_min(1e-12))
                nr = float(torch.sqrt(((pr - gt) ** 2).mean()) / SIGMA)
                zp = float(gt.std() / SIGMA)
                rows.append((nr, zp, float(sdv.mean()),
                             float(((pr - gt) / sdv).std())))
        v = np.array(rows)
        nr, zp, psd, zst = v.mean(0)
        out.append((f, nr / zp))
        print(f"{f:<17.2f}{nr:>9.3f}{zp:>11.3f}{nr/zp:>9.2f}"
              f"{psd:>10.3f}{zst:>8.2f}")

    r = np.array([x[1] for x in out])
    fr = np.array([x[0] for x in out])
    i = int(np.argmax(r))
    interior = 0 < i < len(r) - 1
    print(f"\npeak ratio {r[i]:.2f} at f = {fr[i]:.2f}   "
          f"(ends: f=0 -> {r[0]:.2f}, f=1 -> {r[-1]:.2f})")
    if interior and r[i] > 1.3 * max(r[0], r[-1]):
        print("NON-MONOTONE, interior peak: partial explainability IS the poison.")
    elif interior:
        print("interior peak but shallow; the hypothesis is not established.")
    else:
        print("MONOTONE: hypothesis FALSIFIED. The adversary's sea was not "
              "special for its in-band fraction.")


if __name__ == '__main__':
    sys.exit(main())
