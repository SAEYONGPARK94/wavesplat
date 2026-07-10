"""S2: how much information is in the dropout mask? Measured, not trained.

Door 1 says C1 (Gaussian optimality) does not bind because a LiDAR return exists
only when the facet reflects toward the sensor: the mask is a threshold of a
LINEAR functional of the field's gradient, so p(z, mask | eta) is not Gaussian
in eta and the posterior mean is not linear.

Before spending 10 GPU-days on an amortized posterior, measure the prize the
same way S1 measured the information limit: through the posterior precision,
with no ground truth scored and no network trained.

  amplitudes:  z_j  = Phi_j a + noise,           R = sigma0^2 I
  mask bits:   m_j ~ Bernoulli(p_j),  p_j = f(s_j),  s_j = G_j a
               s_j = d eta / d r at the ray's hit point -- linear in a, with
               G_j = (k . r_hat) * [ -sin psi ; cos psi ] per atom.

Fisher information of one Bernoulli bit whose probability depends on a through
s_j:  I_j = (p_j')^2 / (p_j (1 - p_j)) * G_j^T G_j,  p_j' = dp/ds.

Posterior precision with the mask:  Gamma^-1 + Phi^T R^-1 Phi + sum_j I_j.

TWO RETURN MODELS, and the difference is the whole experiment:

  one-sided   p = Phi((s - tau)/sigma_s).  The bit says "the facet tilts toward
              me". Sign-informative.
  two-sided   p = Phi((s - tau + d)/sigma_s) - Phi((s - tau - d)/sigma_s), a
              specular BAND. If the band is centred at tau = 0 the bit is blind
              to the sign of s and the information collapses. Real specular
              geometry centres the band at a nonzero tau set by the grazing
              angle, so it should stay informative -- but by how much is exactly
              what nobody has computed.

  MCAR        p = const. Then p' = 0, I_j = 0 EXACTLY. The negative control is
              not an experiment, it is an identity: an ignorable mask adds zero
              information, C1 binds, and any learner that beats the linear MMSE
              under MCAR is overfitting. This script prints it as a check.

Reported as the added contraction per k/kp band, against S1's baseline.
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

from d3_gpu_mmse import build_bank, design
from d3_g3_adversarial_sea import DEV, DEPTH, KP, SIGMA
import d3_static_field as SF

OUT = 'results/direction3/s2_mask_information.json'
BAND = (0.10, 2.5)
SIG0 = 0.05 * 4 * SIGMA          # amplitude noise, 0.20 m
N_AMP = 600                      # amplitude measurements (the returns)
N_RAY = 2000                     # rays FIRED (mask bits available)
T_WIN = 20.0
SIG_S = 0.05                     # slope softness of the return law
BINS = [(0.10, 0.20), (0.20, 0.35), (0.35, 0.60), (0.60, 1.00),
        (1.00, 1.60), (1.60, 2.50)]
DT = torch.float64


def rays(n, T, rng, rmax=200.0):
    ts = np.repeat(np.linspace(-T, 0.0, 40)[:, None], n // 40, 1).ravel()
    rr = rng.uniform(15.0, rmax, len(ts))
    aa = rng.uniform(-SF.HALF_FOV, SF.HALF_FOV, len(ts))
    P = np.column_stack([rr * np.cos(aa), rr * np.sin(aa), ts])
    return torch.tensor(P, device=DEV, dtype=DT)


def grad_design(P, kxb, kyb, om):
    """Rows of G: d eta / d r at each ray's hit point, linear in the coeffs.

    eta = a_c cos(psi) + a_s sin(psi),  psi = k.x - omega t
    d eta / d r = (k . r_hat) * ( -a_c sin psi + a_s cos psi )
    so the gradient design is the elevation design with cos/sin swapped, one
    column negated, and every column scaled by the atom's along-ray wavenumber.
    """
    x, y, t = P[:, 0], P[:, 1], P[:, 2]
    r = torch.hypot(x, y).clamp_min(1e-6)
    rx, ry = x / r, y / r
    psi = x[:, None] * kxb[None] + y[:, None] * kyb[None] - t[:, None] * om[None]
    kr = kxb[None] * rx[:, None] + kyb[None] * ry[:, None]     # k . r_hat
    return torch.cat([-kr * torch.sin(psi), kr * torch.cos(psi)], 1)


def precision(P_amp, P_ray, kxb, kyb, om, gam, model, tau, delta):
    B = len(kxb)
    Phi = design(P_amp, kxb, kyb, om)
    A = Phi.T @ Phi / (SIG0 ** 2) + torch.diag(1.0 / gam)
    info = None
    if model != 'none':
        G = grad_design(P_ray, kxb, kyb, om)
        # s has prior sd sqrt(sum gam_i G_i^2); evaluate the Fisher weight at
        # the prior mean s = 0, which is the correct expected information.
        sd_s = torch.sqrt((G ** 2 * gam[None]).sum(1)).clamp_min(1e-9)
        n = torch.distributions.Normal(0.0, 1.0)
        s0 = torch.zeros_like(sd_s)
        if model == 'mcar':
            w = torch.zeros_like(sd_s)                    # p' = 0 exactly
        elif model == 'one_sided':
            u = (s0 - tau) / SIG_S
            p = n.cdf(u).clamp(1e-6, 1 - 1e-6)
            dp = torch.exp(n.log_prob(u)) / SIG_S
            w = dp ** 2 / (p * (1 - p))
        else:                                             # two-sided band
            u1 = (s0 - tau + delta) / SIG_S
            u2 = (s0 - tau - delta) / SIG_S
            p = (n.cdf(u1) - n.cdf(u2)).clamp(1e-6, 1 - 1e-6)
            dp = (torch.exp(n.log_prob(u1)) - torch.exp(n.log_prob(u2))) / SIG_S
            w = dp ** 2 / (p * (1 - p))
        info = float(w.mean())
        A = A + (G * w[:, None]).T @ G
    c = 1.0 - torch.diagonal(torch.linalg.inv(A)) / gam
    c = c.reshape(2, B).mean(0).clamp(0, 1)
    return c.cpu().numpy(), info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tau', type=float, default=0.10)
    ap.add_argument('--delta', type=float, default=0.05)
    a_ = ap.parse_args()

    rng = np.random.default_rng(0)
    kxb, kyb, om = build_bank(KP, BAND, KP / 10.0, DEPTH, DEV, DT)
    B = len(kxb)
    gm = torch.ones(B, device=DEV, dtype=DT) / B
    gam = torch.cat([gm, gm]) * (SIGMA ** 2)
    kmag = (torch.hypot(kxb, kyb) / KP).cpu().numpy()

    P_amp = rays(N_AMP, T_WIN, np.random.default_rng(1))
    P_ray = rays(N_RAY, T_WIN, np.random.default_rng(2))

    print(f"bank {B} atoms ({2*B} coeffs)   {N_AMP} amplitudes @ sigma0="
          f"{SIG0:.2f} m   {N_RAY} rays fired")
    print(f"return law: tau = {a_.tau}, band half-width delta = {a_.delta}, "
          f"slope softness sigma_s = {SIG_S}\n")

    hdr = ''.join(f"{f'{x}-{y}':>11}" for x, y in BINS)
    print(f"{'mask model':<14}{hdr}{'mean Fisher w':>15}")
    res = {}
    base = None
    for model in ('none', 'mcar', 'two_sided', 'one_sided'):
        c, info = precision(P_amp, P_ray, kxb, kyb, om, gam, model,
                            a_.tau, a_.delta)
        row = f"{model:<14}"
        vals = {}
        for x, y in BINS:
            m = (kmag >= x) & (kmag < y)
            v = float(c[m].mean())
            vals[f'{x}-{y}'] = v
            row += f"{v:>11.4f}"
        row += f"{(info if info is not None else 0.0):>15.4g}"
        print(row, flush=True)
        res[model] = vals
        if model == 'none':
            base = vals

    print("\nadded contraction (model - none):")
    for model in ('mcar', 'two_sided', 'one_sided'):
        d = {k: res[model][k] - base[k] for k in base}
        print(f"  {model:<12}" + "".join(f"{d[f'{x}-{y}']:>+11.4f}"
                                         for x, y in BINS))

    mc = max(abs(res['mcar'][k] - base[k]) for k in base)
    print(f"\nMCAR control: max |added contraction| = {mc:.2e}   "
          f"(must be 0 -- an ignorable mask carries no information, and any\n"
          f"  learner that beats the linear MMSE under MCAR is overfitting)")

    ts = res['two_sided']
    lo = ts['0.1-0.2'] - base['0.1-0.2']
    hi = ts['1.0-1.6'] - base['1.0-1.6']
    print(f"\nspecular (two-sided) mask adds: {lo:+.4f} to long waves "
          f"(0.10-0.20 kp), {hi:+.4f} to short waves (1.0-1.6 kp)")
    print(f"the mask measures the GRADIENT, so its sensitivity scales with |k|:")
    print(f"the prediction to check is that it informs SHORT waves most -- "
          f"exactly where\nthe amplitudes are weakest (S1: c = 0.0831).")

    os.makedirs('results/direction3', exist_ok=True)
    with open(OUT, 'w') as f:
        json.dump({'contraction': res, 'tau': a_.tau, 'delta': a_.delta,
                   'sigma_s': SIG_S, 'n_amp': N_AMP, 'n_ray': N_RAY}, f,
                  indent=2)
    return 0


if __name__ == '__main__':
    sys.exit(main())
