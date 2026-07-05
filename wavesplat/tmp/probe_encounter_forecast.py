"""Operational reframe (user: trajectory is exogenous; the goal is ADVANCE
detection of incoming waves): score skill ANISOTROPICALLY — split pixels
by their position relative to the observed corridor ALONG the mean
propagation direction (+x in the harness seas) — and at FORECAST lead
times t* + dt. Information advects downstream at c_g, so the operational
zone ("waves that will reach the ship") = downstream band, and it should
carry far more skill than the isotropic average suggested.
Zones by along-propagation offset dx from the nearest obs point:
  upstream  dx < -0.15 lam | ontrack |dx| <= 0.15 | down1 0.15-0.6 |
  down2 0.6-1.2 lam
Lead times dt in {0, 10, 20} s (c_g ~ 5.5 m/s -> info travels ~ 55-110 m).
3 seas, track100, net-MMSE (+oracle48 reference). (repo tmp probe)"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
os.chdir(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, 'scripts')

from wavesim.wave_generator import jonswap_components
from wavesplat.model.tnse import fit_amplitudes_phases
from d3_sa_fim_identifiability import omega_of, render_modes
from d3_prior_net_eval import get_net, net_gamma_mmse
from d3_sa_achievability_gap import (dispersion_lam, sample_obs, hashabs,
                                     corr, N_OBS, NOISE_FRAC, TP, HS, DEPTH)

SEAS = [4, 0, 7]
DTS = [0.0, 10.0, 20.0]
net = get_net()
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

ZONES = [('upstream', -1.3, -0.15), ('ontrack', -0.15, 0.15),
         ('down 0.15-0.6', 0.15, 0.6), ('down 0.6-1.2', 0.6, 1.3)]

acc = {}
for SI in SEAS:
    c = jonswap_components(Hs=HS, Tp=TP, n_components=200, depth=DEPTH,
                           seed=4000 + SI)
    th = (np.array([x.kx for x in c]), np.array([x.ky for x in c]),
          np.array([x.A for x in c]), np.array([x.phi for x in c]))
    om_t = omega_of(th[0], th[1], DEPTH)
    rng = np.random.default_rng(9000 + 17 * SI + hashabs('track100'))
    P = sample_obs('moving', cp, lam, cp, T, gx, gy, r_fp, rng)
    psi = (th[0][None] * P[:, 0:1] + th[1][None] * P[:, 1:2]
           - om_t[None] * P[:, 2:3] + th[3][None])
    sigma = NOISE_FRAC * HS
    z = (th[2][None] * np.cos(psi)).sum(1) + rng.normal(0, sigma, N_OBS)
    # along-propagation offset (+x) from nearest obs point (recent half)
    Prec = P[P[:, 2] > T / 2]
    dx = np.empty(Xg.size)
    dy_abs = np.empty(Xg.size)
    for i in range(0, Xg.size, 2048):
        sl = slice(i, min(i + 2048, Xg.size))
        d2 = ((Xg.ravel()[sl, None] - Prec[None, :, 0]) ** 2
              + (Yg.ravel()[sl, None] - Prec[None, :, 1]) ** 2)
        j = np.argmin(d2, 1)
        dx[sl] = Xg.ravel()[sl] - Prec[j, 0]
        dy_abs[sl] = np.abs(Yg.ravel()[sl] - Prec[j, 1])
    dx = dx / lam
    order = np.argsort(th[2])[::-1][:48]
    obs = np.column_stack([P, z]).astype(np.float32)
    A_, p_ = fit_amplitudes_phases(th[0][order].astype(np.float32),
                                   th[1][order].astype(np.float32), obs,
                                   DEPTH, lambda_reg=1.0)
    for dt in DTS:
        te = t_star + dt
        gt_e = render_modes(*th, DEPTH, Xg.ravel(), Yg.ravel(), te)
        Pq = np.column_stack([Xg.ravel(), Yg.ravel(),
                              np.full(Xg.size, te)])
        fn = net_gamma_mmse(net, P, z, DEPTH, Pq, sigma, seed=SI,
                            use_em=False)
        fo = render_modes(th[0][order], th[1][order],
                          np.asarray(A_, float), np.asarray(p_, float),
                          DEPTH, Xg.ravel(), Yg.ravel(), te)
        for zn, lo, hi in ZONES:
            m = (dx >= lo) & (dx < hi)
            if m.sum() < 50:
                continue
            acc.setdefault(('net', dt, zn), []).append(corr(fn[m], gt_e[m]))
            acc.setdefault(('orc', dt, zn), []).append(corr(fo[m], gt_e[m]))
    print(f"sea {SI} done", flush=True)

print(f"\n{'dt':>4s} {'zone':16s} {'net':>7s} {'oracle48':>9s}")
for dt in DTS:
    for zn, lo, hi in ZONES:
        vn = acc.get(('net', dt, zn))
        vo = acc.get(('orc', dt, zn))
        if vn is None:
            continue
        print(f"{dt:4.0f} {zn:16s} {np.mean(vn):+7.2f} {np.mean(vo):+9.2f}")
