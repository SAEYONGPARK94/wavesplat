"""Two discriminators for the synthetic(+0.08~0.13) vs real(~0/neg)
net-MMSE motion-gain gap, after H2(bandwidth) was rejected:

H5 patch scale — real records are mostly 1.0~1.3 lam patches (BS11a 1.3,
AA ~1) vs the harness 2 lam: with side < ~1.3 lam the ping-pong track has
little room, so the aperture gain may vanish geometrically.
  -> sweep side in {1.0, 1.3, 2.0, 3.0} lam at default spectrum.

H6 model mismatch (current Doppler) — a uniform current U_c shifts the
true dispersion (omega -> omega + k . U_c) while the estimator model
assumes none; the moving track's coherent chirp gain should die faster
than the fixed-footprint fit.
  -> at side=2 lam, sweep |U_c| in {0, 0.15, 0.3} m/s, random direction
     per sea. (0.3 m/s ~ 3% of cp, typical coastal current.)

8 seas x {fixed, track100} x {hand, net}. (repo tmp probe, 2026-07-05 밤)"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
os.chdir(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, 'scripts')

from wavesim.wave_generator import jonswap_components
from d3_sa_fim_identifiability import omega_of
from d3_sa_nomp import mmse_bank_estimate
from d3_prior_net_eval import get_net, net_gamma_mmse
from d3_sa_achievability_gap import (dispersion_lam, sample_obs,
                                     masks_common_unobs, hashabs, corr,
                                     N_OBS, NOISE_FRAC, TP, HS, DEPTH)

NSEAS = 8
net = get_net()
lam = dispersion_lam(TP, DEPTH)
cp = lam / TP
T = 10 * TP
fp = 0.18 * lam
r_fp = fp / 2


def run_cell(side_lam, uc_mag):
    side = side_lam * lam
    gx = np.linspace(0, side, 96)
    gy = np.linspace(0, side, 96)
    Xg, Yg = np.meshgrid(gx, gy)
    mu = masks_common_unobs(gx, gy, r_fp, 0.6 * fp)
    Xs, Ys = Xg[mu], Yg[mu]
    if len(Xs) > 1000:
        take = np.random.default_rng(0).choice(len(Xs), 1000, replace=False)
        Xs, Ys = Xs[take], Ys[take]
    t_star = T
    Pq = np.column_stack([Xs, Ys, np.full(len(Xs), t_star)])
    out = {c: {'hand': [], 'net': []} for c in ('fixed', 'track100')}
    for si in range(NSEAS):
        c = jonswap_components(Hs=HS, Tp=TP, n_components=200, depth=DEPTH,
                               seed=4600 + si)
        th = (np.array([x.kx for x in c]), np.array([x.ky for x in c]),
              np.array([x.A for x in c]), np.array([x.phi for x in c]))
        om_t = omega_of(th[0], th[1], DEPTH)
        crng = np.random.default_rng(500 + si)
        ang = crng.uniform(0, 2 * np.pi)
        ucx, ucy = uc_mag * np.cos(ang), uc_mag * np.sin(ang)
        om_true = om_t + th[0] * ucx + th[1] * ucy      # Doppler-shifted
        psi_q = (th[0][None] * Xs[:, None] + th[1][None] * Ys[:, None]
                 - om_true[None] * t_star + th[3][None])
        gt_u = (th[2][None] * np.cos(psi_q)).sum(1)
        sigma = NOISE_FRAC * HS
        for cname in out:
            rng = np.random.default_rng(9000 + 17 * si + hashabs(cname))
            U = (1.0 if cname == 'track100' else 0) * cp
            P = sample_obs('fixed' if cname == 'fixed' else 'moving', U,
                           lam, cp, T, gx, gy, r_fp, rng)
            psi = (th[0][None] * P[:, 0:1] + th[1][None] * P[:, 1:2]
                   - om_true[None] * P[:, 2:3] + th[3][None])
            z = (th[2][None] * np.cos(psi)).sum(1) \
                + rng.normal(0, sigma, N_OBS)
            fh = mmse_bank_estimate(P, z, DEPTH, Pq, sigma0=sigma, seed=si)
            out[cname]['hand'].append(corr(fh, gt_u))
            fn = net_gamma_mmse(net, P, z, DEPTH, Pq, sigma, seed=si,
                                use_em=False)
            out[cname]['net'].append(corr(fn, gt_u))
    return out


def report(tag, out):
    for est in ('hand', 'net'):
        f0 = np.mean(out['fixed'][est])
        f1 = np.mean(out['track100'][est])
        d = np.array(out['track100'][est]) - np.array(out['fixed'][est])
        print(f"{tag:16s} {est:5s} fixed {f0:+.3f} track {f1:+.3f} "
              f"gain {d.mean():+.3f} (win+ {int((d>0).sum())}/{len(d)})",
              flush=True)


print("== H5: patch scale (no current) ==")
for sl in (1.0, 1.3, 2.0, 3.0):
    report(f"side={sl:g}lam", run_cell(sl, 0.0))
print("== H6: current Doppler mismatch (side=2lam) ==")
for uc in (0.15, 0.30):
    report(f"Uc={uc:g}m/s", run_cell(2.0, uc))
