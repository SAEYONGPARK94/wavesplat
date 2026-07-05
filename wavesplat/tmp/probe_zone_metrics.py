"""Zone-resolved accuracy for the GIF case (sea4, track100): corr and
nRMSE (vs Hs/4) as a function of distance to the obs track, for RELAX /
hand / net + oracle references (true top-48 k ridge = practical ceiling).
Answers: "how well does it ACTUALLY match, where?"  (repo tmp)"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
os.chdir(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, 'scripts')

from wavesim.wave_generator import jonswap_components
from wavesplat.model.tnse import fit_amplitudes_phases
from d18_relax_estimator import omega, ls_fit
from d17_recursive_estimator import refine_params
from estimator_improved import relax_cv, estimate_klim
from d3_sa_fim_identifiability import omega_of, render_modes
from d3_sa_nomp import mmse_bank_estimate
from d3_prior_net_eval import get_net, net_gamma_mmse
from d3_sa_achievability_gap import (dispersion_lam, sample_obs, hashabs,
                                     corr, N_OBS, NOISE_FRAC, TP, HS, DEPTH)

SEAS = [4, 0, 7]          # GIF sea + two more for robustness
COND = 'track100'
lam = dispersion_lam(TP, DEPTH)
cp = lam / TP
T = 10 * TP
fp = 0.18 * lam
r_fp = fp / 2
side = 2.0 * lam
NG = 96
gx = np.linspace(0, side, NG)
gy = np.linspace(0, side, NG)
Xg, Yg = np.meshgrid(gx, gy)
t_star = T
net = get_net()

BINS = [(0, 0.15), (0.15, 0.35), (0.35, 0.6), (0.6, 1.2)]  # dist in lam

acc = {}
for SI in SEAS:
    c = jonswap_components(Hs=HS, Tp=TP, n_components=200, depth=DEPTH,
                           seed=4000 + SI)
    th = (np.array([x.kx for x in c]), np.array([x.ky for x in c]),
          np.array([x.A for x in c]), np.array([x.phi for x in c]))
    om_t = omega_of(th[0], th[1], DEPTH)
    rng = np.random.default_rng(9000 + 17 * SI + hashabs(COND))
    P = sample_obs('moving', cp, lam, cp, T, gx, gy, r_fp, rng)
    psi = (th[0][None] * P[:, 0:1] + th[1][None] * P[:, 1:2]
           - om_t[None] * P[:, 2:3] + th[3][None])
    sigma = NOISE_FRAC * HS
    z = (th[2][None] * np.cos(psi)).sum(1) + rng.normal(0, sigma, N_OBS)
    gt = render_modes(*th, DEPTH, Xg.ravel(), Yg.ravel(), t_star)
    # distance of each pixel to nearest obs point (space only)
    d2 = ((Xg.ravel()[:, None] - P[None, :, 0]) ** 2
          + (Yg.ravel()[:, None] - P[None, :, 1]) ** 2)
    dmin = np.sqrt(d2.min(1)) / lam
    Pq = np.column_stack([Xg.ravel(), Yg.ravel(),
                          np.full(Xg.size, t_star)])
    est = {}
    # RELAX
    target = {'gx': gx, 'gy': gy, 't': t_star,
              'gt': gt.reshape(NG, NG), 'Hs': HS}
    klim = estimate_klim(P.astype(np.float32), z.astype(np.float32), DEPTH)
    kxr, kyr = relax_cv(P.astype(np.float32), z.astype(np.float32), DEPTH,
                        klim, seed=SI)
    for _ in range(2):
        kxr, kyr, _ = refine_params(kxr, kyr, P.astype(np.float32),
                                    z.astype(np.float32), DEPTH, target,
                                    iters=400)
    coef_r, _, _ = ls_fit(kxr, kyr, P, z, DEPTH, lam=1e-2)
    omr = omega(np.hypot(kxr, kyr) + 1e-12, DEPTH)
    ps = kxr[None] * Pq[:, 0:1] + kyr[None] * Pq[:, 1:2] \
        - omr[None] * t_star
    Nr = len(kxr)
    est['RELAX'] = np.cos(ps) @ coef_r[:Nr] + np.sin(ps) @ coef_r[Nr:]
    est['hand'] = mmse_bank_estimate(P, z, DEPTH, Pq, sigma0=sigma,
                                     seed=SI)
    est['net'] = net_gamma_mmse(net, P, z, DEPTH, Pq, sigma, seed=SI,
                                use_em=False)
    # oracle: true top-48 k + ridge (practical ceiling)
    order = np.argsort(th[2])[::-1][:48]
    obs = np.column_stack([P, z]).astype(np.float32)
    A_, p_ = fit_amplitudes_phases(th[0][order].astype(np.float32),
                                   th[1][order].astype(np.float32), obs,
                                   DEPTH, lambda_reg=1.0)
    est['oracle48'] = render_modes(th[0][order], th[1][order],
                                   np.asarray(A_, float),
                                   np.asarray(p_, float), DEPTH,
                                   Xg.ravel(), Yg.ravel(), t_star)
    for name, f in est.items():
        for lo, hi in BINS:
            m = (dmin >= lo) & (dmin < hi)
            if m.sum() < 30:
                continue
            cc = corr(f[m], gt[m])
            nr = float(np.sqrt(np.mean((f[m] - gt[m]) ** 2)) / (HS / 4))
            acc.setdefault(name, {}).setdefault((lo, hi), []).append(
                (cc, nr))
    print(f"sea {SI} done", flush=True)

print(f"\n{'zone (dist/lam)':16s}" + "".join(
    f"{n:>18s}" for n in ('RELAX', 'hand', 'net', 'oracle48')))
print(" " * 16 + "  corr / nRMSE" * 4)
for lo, hi in BINS:
    line = f"{lo:.2f}-{hi:.2f} lam    "
    for name in ('RELAX', 'hand', 'net', 'oracle48'):
        v = np.array(acc[name][(lo, hi)])
        line += f"   {v[:,0].mean():+.2f} / {v[:,1].mean():.2f}   "
    print(line)
