"""G0 -- S2 redone with the simulator's OWN return law. No invented tau.

S2 admitted its flaw: tau, delta and sigma_s were chosen by me. Self-correction
#35 then closed the retro-specular channel on geometry (raydrop_slope.py wants
the facet normal along the ray: required slope tau = r/h = 3..40 against a
wave-slope sd of 0.08, so P(specular) = 0 and the code always takes its
constant-P fallback at raydrop_slope.py:115).

An external review found what I had missed: the OTHER dropout model is not a
constant either. wavesim/raydrop_uniform.py is

    P(return | hit) = eta * R_fresnel(theta_i) * cos(theta_i) / (d/d_ref)^2

with theta_i the incidence angle taken from BILINEARLY INTERPOLATED WAVY
NORMALS (lidar_simulator.py:321-379, wired in pipeline.py:156-169). A radial
facet slope s tilts the normal, so theta_i = pi/2 - g(r) - s with
g(r) = atan(h/r) the ray's depression. Therefore p depends on the field,
p' != 0, and by S2's own identity the mask carries Fisher information.

This also inverts what our episode filenames mean:
    *_slope.h5    specular gate never fires -> constant P among hits
                  => the OCCLUSION-ONLY control
    *_uniform.h5  Fresnel * cos / d^2 on wavy normals
                  => occlusion + radiometric selection
(self-correction #37)

Here we compute, with no training and no ground truth scored, the Fisher
information that the true f puts into the posterior precision:

    I_j = (p'_j)^2 / (p_j (1 - p_j)) * G_j^T G_j,   p'_j = dp/ds at s = 0

G_j is S2's along-ray gradient design. Reported as added posterior contraction
per k/kp band against the same flat-prior baseline.

PRE-REGISTERED (external reviewer's G0, frozen before the run):
    added contraction < +0.02 in EVERY k band  =>  the mask channel is dead at
    the source; no network is ever trained on it.
The MCAR identity (p' = 0 => I = 0 exactly) is printed as the negative control.
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

from wavesim.raydrop_uniform import fresnel_reflectance
from d3_gpu_mmse import build_bank, design
from d3_g3_adversarial_sea import DEV, DEPTH, KP, SIGMA
from d3_s2_mask_information import grad_design, rays, BINS, N_AMP, T_WIN
import d3_static_field as SF

OUT = 'results/direction3/s2b_true_f.json'
BAND = (0.10, 2.5)
SIG0 = 0.05 * 4 * SIGMA
H_SENS = 5.0            # wavesim/config.py LidarCfg.height
D_REF = 125.0           # wavesim/raydrop_uniform.py
DT = torch.float64
N_FIRED = 152 * 500     # the real fired-ray count per scan (config.py)


def p_of_s(s, r, eta_cal):
    """the simulator's own law, as a function of radial facet slope s."""
    g = np.arctan(H_SENS / r)                 # ray depression at range r
    th = np.pi / 2 - g - s                    # facet tilted toward sensor: th down
    th = np.clip(th, 0.0, np.pi / 2 - 1e-9)
    P = eta_cal * fresnel_reflectance(th) * np.cos(th) / ((r / D_REF) ** 2 + 1e-9)
    return np.clip(P, 1e-12, 1 - 1e-12)


def fisher_w(r, eta_cal, h=1e-4):
    p = p_of_s(0.0, r, eta_cal)
    dp = (p_of_s(h, r, eta_cal) - p_of_s(-h, r, eta_cal)) / (2 * h)
    return dp ** 2 / (p * (1 - p)), p, dp


def calibrate(r, rate):
    """eta such that the mean return probability over the fired rays == rate."""
    lo, hi = 1e-9, 1e9
    for _ in range(80):
        mid = np.sqrt(lo * hi)
        m = p_of_s(0.0, r, mid).mean()
        lo, hi = (mid, hi) if m < rate else (lo, mid)
    return np.sqrt(lo * hi)


def contraction(P_amp, G, w, gam, kxb):
    B = len(kxb)
    Phi = design(P_amp, kxb, kyb_g, om_g)
    A = Phi.T @ Phi / (SIG0 ** 2) + torch.diag(1.0 / gam)
    if w is not None:
        A = A + (G * w[:, None]).T @ G
    c = 1.0 - torch.diagonal(torch.linalg.inv(A)) / gam
    return c.reshape(2, B).mean(0).clamp(0, 1).cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--rates', type=float, nargs='+',
                    default=[0.001, 0.01, 0.05, 0.20])
    ap.add_argument('--n-ray', type=int, default=20000)
    # 50 returns / frame (raydrop_model.n_pts in every episode) out of
    # 0.657 x 76,000 occluded hits (the occlusion hit rate measured by
    # scripts/d3_door1b_field_bias.py on this same geometry) = 1.0e-3.
    ap.add_argument('--deploy-rate', type=float, default=1.0e-3)
    a_ = ap.parse_args()

    global kyb_g, om_g
    kxb, kyb_g, om_g = build_bank(KP, BAND, KP / 10.0, DEPTH, DEV, DT)
    B = len(kxb)
    gm = torch.ones(B, device=DEV, dtype=DT) / B
    gam = torch.cat([gm, gm]) * (SIGMA ** 2)
    kmag = (torch.hypot(kxb, kyb_g) / KP).cpu().numpy()

    P_amp = rays(N_AMP, T_WIN, np.random.default_rng(1))
    P_ray = rays(a_.n_ray, T_WIN, np.random.default_rng(2))
    G = grad_design(P_ray, kxb, kyb_g, om_g)
    r = torch.hypot(P_ray[:, 0], P_ray[:, 1]).cpu().numpy()
    scale = N_FIRED / a_.n_ray            # each modelled ray stands for this many

    print("the simulator's law, d ln P / ds at s = 0  (rate 0.05):")
    ec = calibrate(r, 0.05)
    for rr in (25, 50, 100, 150, 200):
        w, p, dp = fisher_w(np.array([float(rr)]), ec)
        print(f"  r = {rr:3d} m   depress {np.rad2deg(np.arctan(H_SENS/rr)):5.2f} deg"
              f"   p = {p[0]:.3e}   dlnP/ds = {dp[0]/p[0]:+7.2f}")

    base = contraction(P_amp, None, None, gam, kxb)
    hdr = ''.join(f"{f'{x}-{y}':>11}" for x, y in BINS)
    print(f"\n{'return law':<22}{hdr}{'mean w':>13}")
    print(f"{'none (S1 baseline)':<22}" + ''.join(
        f"{base[(kmag>=x)&(kmag<y)].mean():>11.4f}" for x, y in BINS) + f"{0.0:>13.4g}")

    res = {'baseline': {f'{x}-{y}': float(base[(kmag >= x) & (kmag < y)].mean())
                        for x, y in BINS}, 'rays_modelled': a_.n_ray,
           'ray_scale': scale, 'rows': {}}

    # MCAR identity: p' = 0 exactly -> I = 0 exactly.
    c_mcar = contraction(P_amp, G, torch.zeros(len(G), device=DEV, dtype=DT), gam, kxb)
    print(f"{'mcar (p'' = 0)':<22}" + ''.join(
        f"{c_mcar[(kmag>=x)&(kmag<y)].mean():>11.4f}" for x, y in BINS) + f"{0.0:>13.4g}")
    mc = float(np.abs(c_mcar - base).max())

    # The gate must be read at the DEPLOYED return rate, not at whichever rate
    # makes it pass. This project's episodes keep n_pts = 50 returns out of
    # ~45,000 occluded hits (152 x 500 rays), so the operating point is
    # rate ~ 1.1e-3. Sweeping upward only maps how the prize would grow with a
    # denser sensor -- that is a sensor-design curve (C1), not this thesis's
    # operating point.
    per_rate = {}
    for rate in sorted(set(a_.rates + [a_.deploy_rate])):
        ec = calibrate(r, rate)
        wv, p, dp = fisher_w(r, ec)
        w = torch.tensor(wv * scale, device=DEV, dtype=DT)   # per-fired-ray count
        c = contraction(P_amp, G, w, gam, kxb)
        row = {f'{x}-{y}': float(c[(kmag >= x) & (kmag < y)].mean()) for x, y in BINS}
        add = {k: row[k] - res['baseline'][k] for k in row}
        per_rate[rate] = max(add.values())
        res['rows'][f'rate={rate}'] = {'contraction': row, 'added': add,
                                       'eta': float(ec), 'mean_w': float(wv.mean())}
        print(f"{f'fresnel rate={rate:g}':<22}" + ''.join(
            f"{row[f'{x}-{y}']:>11.4f}" for x, y in BINS) + f"{wv.mean()*scale:>13.4g}")

    print("\nadded contraction vs baseline:")
    for k, v in res['rows'].items():
        print(f"  {k:<16}" + ''.join(f"{v['added'][f'{x}-{y}']:>+11.4f}"
                                     for x, y in BINS))

    print(f"\nMCAR control: max |added contraction| = {mc:.2e}  "
          f"(exactly 0 -- an ignorable mask carries no information)")

    dep = per_rate[a_.deploy_rate]
    # I_j ~ p_j (dlnp/ds)^2 for small p, so the prize is linear in the rate.
    rs = sorted(per_rate)
    slope = per_rate[rs[-1]] / rs[-1]
    cross = 0.02 / slope
    print(f"\nG0 (pre-registered: added contraction >= +0.02 in some band)")
    print(f"   at the DEPLOYED rate {a_.deploy_rate:g} "
          f"({a_.deploy_rate*N_FIRED:.0f} returns / frame): max added = {dep:+.4f}")
    print(f"   {'PASS' if dep >= 0.02 else 'FAIL -- the mask channel is DEAD at this sensor sparsity.'}")
    print(f"   the prize is linear in the return rate; it reaches +0.02 at "
          f"rate ~ {cross:.3f}  ({cross*N_FIRED:.0f} returns / frame, "
          f"{cross/a_.deploy_rate:.0f}x denser)")
    res['g0_added_at_deploy'] = dep
    res['g0_pass'] = bool(dep >= 0.02)
    res['rate_to_pass'] = float(cross)
    res['deploy_rate'] = a_.deploy_rate
    res['mcar_max'] = mc

    os.makedirs('results/direction3', exist_ok=True)
    with open(OUT, 'w') as f:
        json.dump(res, f, indent=2)
    print(f"wrote {OUT}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
