"""Which side of the band breaks it, and does the UQ gate see it coming?

G3's attack drove in-band energy to 0.095 and nRMSE to 6.32, five times worse
than a zero-predictor. The spectrum was not saved, so the side is unknown, and
the prescriptions differ:

    long waves  (k -> 0.15 kp)   aperture: the fan is a fraction of a wavelength
    short waves (k -> 5 kp)      sampling: 600 fan points alias them

No optimisation needed to find out. Build seas with energy confined to each
end, and one in-band sea as the control, all at the same variance.

Second question, worth more than the first. Yesterday's UQ gate answered 70% of
the domain and cut nRMSE by 41%, using the MMSE's own predictive variance. If
that variance rises when the field diverges, the safeguard already exists. If
it stays calm while the estimate runs to 6 sigma, the UQ is confidently wrong
and the gate cannot be trusted off-distribution.

Reported against the zero-predictor (nRMSE = gt_std / sigma), because "worse
than doing nothing" is the only threshold that matters here.
"""
import os
import sys

import numpy as np
import torch

os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, 'scripts')

from d3_gpu_mmse import mmse_gpu
from d3_g3_adversarial_sea import (geometry, eta, norm_amp, DEV, DEPTH, KP,
                                   NCOMP, SIGMA, NOISE, BAND, G)

CASES = {
    'long  0.15-0.30 kp': (0.15, 0.30),
    'in-band 0.5-2.0 kp': (0.50, 2.00),
    'short 3.0-5.0 kp': (3.00, 5.00),
    'full  0.15-5.0 kp': (0.15, 5.00),
}
SEEDS = (0, 1, 2, 3, 4)


def sea(rng, klo, khi):
    k = KP * np.exp(rng.uniform(np.log(klo), np.log(khi), NCOMP))
    th = rng.uniform(-np.pi, np.pi, NCOMP)
    t = lambda a: torch.tensor(a, device=DEV, dtype=torch.float32)
    return (t(k * np.cos(th)), t(k * np.sin(th)),
            t(np.sqrt(G * k * np.tanh(k * DEPTH))), t(k))


def main():
    P, Pq = geometry()
    gen = torch.Generator(device=DEV).manual_seed(0)
    print(f"fan obs {len(P)}   query {len(Pq)}   sea std {SIGMA:.2f} m\n")
    print(f"{'sea':<22}{'nRMSE':>9}{'zero-pred':>11}{'ratio':>8}"
          f"{'pred sd':>10}{'z-std':>8}")

    for tag, (klo, khi) in CASES.items():
        vals = []
        for sd in SEEDS:
            rng = np.random.default_rng(sd)
            kx, ky, om, k = sea(rng, klo, khi)
            raw = torch.tensor(rng.normal(0, 1, NCOMP), device=DEV,
                               dtype=torch.float32)
            ph = torch.tensor(rng.uniform(-np.pi, np.pi, NCOMP), device=DEV,
                              dtype=torch.float32)
            with torch.no_grad():
                a = norm_amp(raw)
                z = eta(P, kx, ky, om, a, ph)
                z = z + NOISE * torch.randn(len(z), device=DEV, generator=gen)
                gt = eta(Pq, kx, ky, om, a, ph)
                pr, var = mmse_gpu(P, z, DEPTH, Pq, sigma0=0.05 * 4 * SIGMA,
                                   kp_hint=KP, band=BAND, return_var=True)
                err = pr - gt
                nr = float(torch.sqrt((err ** 2).mean()) / SIGMA)
                zero = float(gt.std() / SIGMA)
                psd = float(torch.sqrt(var.clamp_min(1e-12)).mean())
                zstd = float((err / torch.sqrt(var.clamp_min(1e-12))).std())
            vals.append((nr, zero, psd, zstd))
        v = np.array(vals)
        nr, zero, psd, zstd = v.mean(0)
        print(f"{tag:<22}{nr:>9.3f}{zero:>11.3f}{nr/zero:>8.2f}"
              f"{psd:>10.3f}{zstd:>8.2f}")

    print("\nratio > 1  = worse than predicting zero.")
    print("z-std      = err / predictive sd. Calibrated UQ gives ~1.")
    print("If a sea has ratio >> 1 while pred sd stays small (z-std >> 1),")
    print("the posterior is confidently wrong and the UQ gate will pass it.")


if __name__ == '__main__':
    sys.exit(main())
