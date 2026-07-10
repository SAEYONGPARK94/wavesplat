"""G0-coherent: the mask bit and the range measurement are the SAME event.

d3_s2b_true_f.py swept the return rate while holding the amplitude budget at
600 points. That is incoherent. A ray that returns does not hand you a bit
saying "I returned" -- it hands you a full range measurement. So the return
rate sets the amplitude count and the mask's Fisher information TOGETHER:

    N_amp(rate)  = rate * N_fired        (the returns you actually get)
    mask         = one Bernoulli bit per FIRED ray, informative because
                   p depends on the radial facet slope

The honest question is therefore not "how dense must the sensor be before the
mask is worth +0.02 contraction", but:

    at every sensor density, how much does the mask add ON TOP OF the
    amplitudes that arrive with it?

If the answer is "nothing, at any density", the mask channel is closed
structurally and not merely at our operating point -- a much stronger claim,
and one that no amount of sensor engineering repairs.

Modelled with the same additive-precision trick both places: n modelled points
stand for N real ones by scaling their precision contribution by N/n. Exact for
a linear-Gaussian model, because the precision is a sum over observations.

Reported: contraction with amplitudes only, and with amplitudes + mask, at each
return rate. No training. No ground truth scored.
"""
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
os.chdir(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, 'scripts')

from d3_gpu_mmse import build_bank, design
from d3_g3_adversarial_sea import DEV, DEPTH, KP, SIGMA
from d3_s2_mask_information import grad_design, rays, BINS, T_WIN
from d3_s2b_true_f import fisher_w, calibrate, N_FIRED, SIG0

OUT = 'results/direction3/s2c_coherent.json'
BAND = (0.10, 2.5)
DT = torch.float64
N_MODEL = 4000        # modelled points; each stands for N_real / N_MODEL real ones
RATES = [1.0e-3, 3.0e-3, 1.0e-2, 3.0e-2, 1.0e-1, 3.0e-1]


def main():
    kxb, kyb, om = build_bank(KP, BAND, KP / 10.0, DEPTH, DEV, DT)
    B = len(kxb)
    gm = torch.ones(B, device=DEV, dtype=DT) / B
    gam = torch.cat([gm, gm]) * (SIGMA ** 2)
    kmag = (torch.hypot(kxb, kyb) / KP).cpu().numpy()

    P_amp = rays(N_MODEL, T_WIN, np.random.default_rng(1))
    P_ray = rays(N_MODEL, T_WIN, np.random.default_rng(2))
    Phi = design(P_amp, kxb, kyb, om)
    Gm = grad_design(P_ray, kxb, kyb, om)
    r_ray = torch.hypot(P_ray[:, 0], P_ray[:, 1]).cpu().numpy()
    ray_scale = N_FIRED / N_MODEL

    PhiTPhi = Phi.T @ Phi
    prior = torch.diag(1.0 / gam)

    def bins(c):
        return {f'{x}-{y}': float(c[(kmag >= x) & (kmag < y)].mean()) for x, y in BINS}

    def contract(A):
        c = 1.0 - torch.diagonal(torch.linalg.inv(A)) / gam
        return c.reshape(2, B).mean(0).clamp(0, 1).cpu().numpy()

    hdr = ''.join(f"{f'{x}-{y}':>10}" for x, y in BINS)
    print(f"bank {B} atoms   fired rays/frame {N_FIRED}   sigma0 {SIG0:.2f} m")
    print("each rate sets BOTH the amplitude count and the mask's information.\n")
    print(f"{'rate':>7}{'N_amp':>8}  {'channel':<12}{hdr}")

    res = {'rates': {}, 'n_fired': N_FIRED}
    for rate in RATES:
        n_amp = rate * N_FIRED
        a_scale = n_amp / N_MODEL
        ec = calibrate(r_ray, rate)
        wv, _, _ = fisher_w(r_ray, ec)
        w = torch.tensor(wv * ray_scale, device=DEV, dtype=DT)

        A_amp = a_scale * PhiTPhi / (SIG0 ** 2) + prior
        A_both = A_amp + (Gm * w[:, None]).T @ Gm
        c_amp, c_both = contract(A_amp), contract(A_both)
        b_amp, b_both = bins(c_amp), bins(c_both)
        add = {k: b_both[k] - b_amp[k] for k in b_amp}
        res['rates'][f'{rate:g}'] = {'n_amp': n_amp, 'amp': b_amp,
                                     'amp+mask': b_both, 'added': add}

        print(f"{rate:>7.3g}{n_amp:>8.0f}  {'amplitudes':<12}" +
              ''.join(f"{b_amp[f'{x}-{y}']:>10.4f}" for x, y in BINS))
        print(f"{'':>15}  {'+ mask':<12}" +
              ''.join(f"{b_both[f'{x}-{y}']:>10.4f}" for x, y in BINS))
        print(f"{'':>15}  {'added':<12}" +
              ''.join(f"{add[f'{x}-{y}']:>+10.4f}" for x, y in BINS), flush=True)

    print("\nmask's added contraction, as a fraction of what the amplitudes give:")
    print(f"{'rate':>7}  " + ''.join(f"{f'{x}-{y}':>10}" for x, y in BINS))
    worst = 0.0
    for rate in RATES:
        row = res['rates'][f'{rate:g}']
        fr = {k: row['added'][k] / max(row['amp'][k], 1e-12) for k in row['amp']}
        worst = max(worst, max(fr.values()))
        print(f"{rate:>7.3g}  " + ''.join(f"{100*fr[f'{x}-{y}']:>9.2f}%" for x, y in BINS))

    best_abs = max(max(res['rates'][f'{r:g}']['added'].values()) for r in RATES)
    print(f"\nmax ABSOLUTE added contraction over every density: {best_abs:+.4f}")
    print(f"max RELATIVE (mask / amplitudes):                  {100*worst:.2f}%")
    print("\nPRE-REGISTERED (G0): >= +0.02 added in some band, at the sensor's own")
    print(f"amplitude budget.  ->  {'PASS' if best_abs >= 0.02 else 'FAIL'}")
    if best_abs < 0.02:
        print("\nThe mask is dominated by the measurements it arrives with. Making the")
        print("sensor denser raises the mask's information AND the amplitudes'; the")
        print("ratio does not move enough to matter. This closes the channel")
        print("STRUCTURALLY, not merely at our operating point.")
    res['max_added'] = best_abs
    res['max_relative'] = worst
    res['pass'] = bool(best_abs >= 0.02)

    os.makedirs('results/direction3', exist_ok=True)
    with open(OUT, 'w') as f:
        json.dump(res, f, indent=2)
    print(f"\nwrote {OUT}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
