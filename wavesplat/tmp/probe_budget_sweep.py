"""Does absolute accuracy rise with a REALISTIC observation budget?
The 400-obs protocol is a deliberately starved matched-budget control;
operational LiDAR delivers 10-100x. Sweep n_obs in {400, 1600, 4800},
track100, hand + net, zone-resolved corr. (repo tmp probe)"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
os.chdir(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, 'scripts')

from wavesim.wave_generator import jonswap_components
from d3_sa_fim_identifiability import omega_of, render_modes
from d3_sa_nomp import mmse_bank_estimate
from d3_prior_net_eval import get_net, net_gamma_mmse
from d3_sa_achievability_gap import (dispersion_lam, hashabs, corr,
                                     NOISE_FRAC, TP, HS, DEPTH)

SEAS = [4, 0, 7]
NS = [400, 1600, 4800]
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
BINS = [(0, 0.35), (0.35, 0.6), (0.6, 1.2)]


def sample_obs_n(n, rng):
    tt = np.linspace(0, T, n)
    th_j = rng.uniform(0, 2 * np.pi, n)
    rr = r_fp * np.sqrt(rng.uniform(0, 1, n))
    jx, jy = rr * np.cos(th_j), rr * np.sin(th_j)
    P0 = np.array([gx[0] + r_fp, gy[0] + r_fp])
    P1 = np.array([gx[-1] - r_fp, gy[-1] - r_fp])
    L = float(np.linalg.norm(P1 - P0))
    u = (P1 - P0) / L
    s = np.mod(cp * tt, 2 * L)
    d = np.where(s <= L, s, 2 * L - s)
    xy = P0[None] + d[:, None] * u[None]
    return np.column_stack([xy[:, 0] + jx, xy[:, 1] + jy, tt])


acc = {}
for SI in SEAS:
    c = jonswap_components(Hs=HS, Tp=TP, n_components=200, depth=DEPTH,
                           seed=4000 + SI)
    th = (np.array([x.kx for x in c]), np.array([x.ky for x in c]),
          np.array([x.A for x in c]), np.array([x.phi for x in c]))
    om_t = omega_of(th[0], th[1], DEPTH)
    gt = render_modes(*th, DEPTH, Xg.ravel(), Yg.ravel(), t_star)
    sigma = NOISE_FRAC * HS
    for n in NS:
        rng = np.random.default_rng(9000 + 17 * SI + hashabs('track100'))
        P = sample_obs_n(n, rng)
        psi = (th[0][None] * P[:, 0:1] + th[1][None] * P[:, 1:2]
               - om_t[None] * P[:, 2:3] + th[3][None])
        z = (th[2][None] * np.cos(psi)).sum(1) + rng.normal(0, sigma, n)
        dmin = np.sqrt(((Xg.ravel()[:, None] - P[None, ::4, 0]) ** 2
                        + (Yg.ravel()[:, None] - P[None, ::4, 1]) ** 2
                        ).min(1)) / lam
        Pq = np.column_stack([Xg.ravel(), Yg.ravel(),
                              np.full(Xg.size, t_star)])
        fh = mmse_bank_estimate(P, z, DEPTH, Pq, sigma0=sigma, seed=SI)
        fn = net_gamma_mmse(net, P, z, DEPTH, Pq, sigma, seed=SI,
                            use_em=False)
        for name, f in (('hand', fh), ('net', fn)):
            for lo, hi in BINS:
                m = (dmin >= lo) & (dmin < hi)
                if m.sum() < 30:
                    continue
                acc.setdefault((name, n), {}).setdefault(
                    (lo, hi), []).append(corr(f[m], gt[m]))
        print(f"sea {SI} n={n} done", flush=True)

print(f"\n{'est':5s} {'n_obs':>6s}" + "".join(
    f"  {lo:.2f}-{hi:.2f}L" for lo, hi in BINS))
for name in ('hand', 'net'):
    for n in NS:
        line = f"{name:5s} {n:6d}"
        for b in BINS:
            v = acc[(name, n)].get(b, [np.nan])
            line += f"     {np.nanmean(v):+.2f} "
        print(line)
