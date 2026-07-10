"""S1: measure the information limit. No ground truth is scored.

D1 line 87 concedes it: "진짜 정보 한계는 아직 측정되지 않았다." Every number in
this project so far was an nRMSE against a stereo reconstruction that the
observations were themselves drawn from -- and today's audit showed part of the
headline gain was that reconstruction's own artifact (AA15a +0.058 -> +0.037
after detrending obs AND gt).

This experiment cannot be contaminated that way, because it never scores
anything. For a linear-Gaussian model the posterior covariance IS the achievable
floor, and it is a function of the design matrix and the prior alone:

    Sigma_post(coeffs) = (Phi^T R^-1 Phi + Gam^-1)^-1

Per mode, the fraction of prior variance the data removes is the CONTRACTION

    c_i = 1 - Var_post(a_i)/gamma_i = gamma_i * phi_i^T K^-1 phi_i,
    K = Phi Gam Phi^T + sigma^2 I

c_i = 0 means the observations say nothing about that mode: adding it to the
bank adds a free parameter and no information. c_i -> 1 means the mode is
pinned.

THE TEST OF D1. D1 claims the wall is the band floor (a modelling choice we can
remove), not the observing aperture (an information limit we cannot). Then the
long-wave modes BELOW the default floor of 0.35 kp must be informed -- their
contraction must be substantially above zero, and it must depend on the floor's
position not at all (the floor is our choice; the information is not).

  FALSIFIER: if contraction for k < 0.35 kp is ~0 and stays ~0 as the aperture
  grows, those atoms carry no information, the aperture binds after all, and
  D1's "lower the floor and win" must be retracted.

  CONFIRMATION: if contraction rises with aperture (spatial L, and temporal T)
  and there is a knee where it collapses, that knee is the information wall
  this project has circled all week and never measured.

Swept: max range (spatial aperture), observation window (temporal aperture),
budget. Reported per k/kp band, plus the field-level posterior sd over the fan.
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

OUT = 'results/direction3/s1_posterior_contraction.json'
BAND = (0.10, 2.5)
SIG = 0.05 * 4 * SIGMA                 # the sigma0 the project uses
BINS = [(0.10, 0.20), (0.20, 0.35), (0.35, 0.60), (0.60, 1.00),
        (1.00, 1.60), (1.60, 2.50)]


def fan(rmax, T, n, rng, rmin=15.0):
    ts = np.repeat(np.linspace(-T, 0.0, 40)[:, None], n // 40, 1).ravel()
    rr = rng.uniform(rmin, rmax, len(ts))
    aa = rng.uniform(-SF.HALF_FOV, SF.HALF_FOV, len(ts))
    return torch.tensor(np.column_stack([rr * np.cos(aa), rr * np.sin(aa), ts]),
                        device=DEV, dtype=torch.float32)


def contraction(P, prior_flat=True):
    """c_i = gamma_i * phi_i^T K^-1 phi_i, per bank atom (cos & sin merged)."""
    kxb, kyb, om = build_bank(KP, BAND, KP / 10.0, DEPTH, DEV, torch.float64)
    Phi = design(P.double(), kxb, kyb, om)                 # (n, 2B)
    B = len(kxb)
    gm = torch.ones(B, device=DEV, dtype=torch.float64) / B
    gam = torch.cat([gm, gm]) * (SIGMA ** 2)
    n = len(P)
    K = (Phi * gam[None]) @ Phi.T + (SIG ** 2) * torch.eye(
        n, device=DEV, dtype=torch.float64)
    M = torch.linalg.solve(K, Phi)                          # (n, 2B)
    c = gam * (Phi * M).sum(0)                              # (2B,)
    c = c.reshape(2, B).mean(0).clamp(0.0, 1.0)
    kmag = torch.hypot(kxb, kyb) / KP
    return kmag.cpu().numpy(), c.cpu().numpy(), K, Phi, gam


def field_sd(P, K, Phi, gam):
    gx = np.arange(SF.DOM_X[0], SF.DOM_X[1], 10.0)
    gy = np.arange(SF.DOM_Y[0], SF.DOM_Y[1], 10.0)
    X, Y = np.meshgrid(gx, gy)
    ins, _ = SF.fan_mask(X, Y, 0.0, 0.0, np.array([1.0, 0.0]))
    Pq = torch.tensor(np.column_stack([X.ravel()[ins.ravel()],
                                       Y.ravel()[ins.ravel()],
                                       np.zeros(ins.sum())]),
                      device=DEV, dtype=torch.float64)
    kxb, kyb, om = build_bank(KP, BAND, KP / 10.0, DEPTH, DEV, torch.float64)
    Pq_ = design(Pq, kxb, kyb, om)
    A = Pq_ * gam[None]
    prior = (A * Pq_).sum(1)
    Bq = A @ Phi.T
    post = prior - ((Bq @ torch.linalg.inv(K)) * Bq).sum(1)
    return float(torch.sqrt(post.clamp_min(0)).mean() / SIGMA)


def main():
    ap = argparse.ArgumentParser()
    ap.parse_args()
    rng = np.random.default_rng(0)
    print(f"prior: flat over the bank, total variance sigma^2 = {SIGMA**2:.2f}"
          f"   sigma0 = {SIG:.2f} m")
    print(f"contraction c = 1 - Var_post/Var_prior.  c=0: the data say nothing "
          f"about that mode.\n")

    cfgs = [(200.0, 20.0, 600), (100.0, 20.0, 600), (400.0, 20.0, 600),
            (200.0, 2.5, 600), (200.0, 40.0, 600), (200.0, 20.0, 1200)]
    res = {}
    hdr = ''.join(f"{f'{a}-{b}':>11}" for a, b in BINS)
    print(f"{'Rmax':>6}{'T':>6}{'N':>6}{hdr}{'field sd':>11}")
    for rmax, T, n in cfgs:
        P = fan(rmax, T, n, np.random.default_rng(1))
        km, c, K, Phi, gam = contraction(P)
        row = f"{rmax:>6.0f}{T:>6.1f}{n:>6d}"
        vals = {}
        for a, b in BINS:
            m = (km >= a) & (km < b)
            v = float(c[m].mean()) if m.any() else float('nan')
            vals[f'{a}-{b}'] = v
            row += f"{v:>11.4f}"
        fs = field_sd(P, K, Phi, gam)
        row += f"{fs:>11.3f}"
        print(row, flush=True)
        res[f"R{rmax:.0f}_T{T:.0f}_N{n}"] = dict(bins=vals, field_sd=fs)
        del K, Phi
        torch.cuda.empty_cache()

    base = res['R200_T20_N600']['bins']
    print(f"\nsub-floor modes (k < 0.35 kp) at the Falcon geometry: "
          f"c = {base['0.1-0.2']:.4f} (0.10-0.20 kp), "
          f"{base['0.2-0.35']:.4f} (0.20-0.35 kp)")
    print(f"in-band modes (1.0-1.6 kp):                        "
          f"c = {base['1.0-1.6']:.4f}")

    lo = base['0.1-0.2']
    lo_wide = res['R400_T20_N600']['bins']['0.1-0.2']
    lo_short = res['R200_T2_N600']['bins']['0.1-0.2']
    print(f"\nD1 test:")
    print(f"  sub-floor contraction, aperture 200 m -> 400 m: "
          f"{lo:.4f} -> {lo_wide:.4f}")
    print(f"  sub-floor contraction, window 20 s -> 2.5 s:    "
          f"{lo:.4f} -> {lo_short:.4f}")
    if lo < 0.05:
        print("\n  sub-floor modes are UNINFORMED. Adding them to the bank adds "
              "free\n  parameters and no information -- the aperture binds, and "
              "D1's\n  'lower the floor and win' must be retracted.")
    else:
        print(f"\n  sub-floor modes carry information (c = {lo:.3f}). D1's claim "
              f"that the\n  floor -- not the aperture -- is the wall is "
              f"supported, and now quantified.")

    os.makedirs('results/direction3', exist_ok=True)
    with open(OUT, 'w') as f:
        json.dump(res, f, indent=2)
    return 0


if __name__ == '__main__':
    sys.exit(main())
