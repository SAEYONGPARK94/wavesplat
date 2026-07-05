"""CONTROL-compatible configuration, first faithful evaluation:
ship advances on an exogenous straight course INTO the waves (head seas),
forward fan LiDAR (+-30 deg, 20-100 m ahead, scan every 0.5 s, m pts/scan),
CAUSAL estimation at t0 using only the trailing W seconds of scans,
encounter forecast eta(t) at the ship's FUTURE positions, lead 0-30 s.

Sweep trailing memory W in {10, 20, 40, 70} s. If W=20 ~ W=70 for leads
<= 15-20 s, the control latency constraint costs nothing (old wave groups
have already passed the ship; memory requirement scales with lead).
hand-MMSE only (net is OOD for this operator until retrained).
8 seas. (repo tmp probe, 2026-07-05 밤 — 사용자 제어 제약 반영)"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
os.chdir(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, 'scripts')

from wavesim.wave_generator import jonswap_components
from d3_sa_fim_identifiability import omega_of
from d3_sa_nomp import mmse_bank_estimate
from d3_sa_achievability_gap import (dispersion_lam, corr, NOISE_FRAC,
                                     TP, HS, DEPTH)

NSEAS = 8
WS = [10.0, 20.0, 40.0, 70.0]
LEADS = np.arange(0.0, 31.0, 2.0)
SCAN_DT = 0.5
M_PER_SCAN = 12
U_SHIP = 3.0                      # m/s ahead (head seas, waves +x, ship -x)
lam = dispersion_lam(TP, DEPTH)
cp = lam / TP
sigma = NOISE_FRAC * HS

res = {W: [] for W in WS}
for si in range(NSEAS):
    c = jonswap_components(Hs=HS, Tp=TP, n_components=200, depth=DEPTH,
                           seed=8200 + si)
    th = (np.array([x.kx for x in c]), np.array([x.ky for x in c]),
          np.array([x.A for x in c]), np.array([x.phi for x in c]))
    om_t = omega_of(th[0], th[1], DEPTH)

    def eta(x, y, t):
        ps = th[0][None] * np.atleast_1d(x)[:, None] \
            + th[1][None] * np.atleast_1d(y)[:, None] \
            - om_t[None] * np.atleast_1d(t)[:, None] + th[3][None]
        return (th[2][None] * np.cos(ps)).sum(1)

    rng = np.random.default_rng(300 + si)
    t0 = 100.0                     # 'now'
    # ship course: heading -x (into +x-propagating waves)
    def ship_pos(t):
        return np.array([-U_SHIP * (t - t0), 0.0])

    # scans over the longest memory; each scan samples the forward fan
    ts_scan = np.arange(t0 - max(WS), t0 + 1e-9, SCAN_DT)
    Pall = []
    for t_s in ts_scan:
        sp = ship_pos(t_s)
        rr = rng.uniform(20.0, 100.0, M_PER_SCAN)
        aa = rng.uniform(-np.pi / 6, np.pi / 6, M_PER_SCAN)
        # fan points ahead of the ship = -x direction... ship heads -x,
        # waves come from -x?? waves propagate +x -> they come FROM -x.
        # Looking into oncoming waves = look toward -x.
        xs = sp[0] - rr * np.cos(aa)
        ys = sp[1] + rr * np.sin(aa)
        Pall.append(np.column_stack([xs, ys, np.full(M_PER_SCAN, t_s)]))
    Pall = np.vstack(Pall)
    zall = eta(Pall[:, 0], Pall[:, 1], Pall[:, 2]) \
        + rng.normal(0, sigma, len(Pall))

    # encounter targets: ship future positions
    tq = t0 + LEADS
    xq = np.array([ship_pos(t)[0] for t in tq])
    yq = np.zeros_like(xq)
    gt_q = eta(xq, yq, tq)
    Pq = np.column_stack([xq, yq, tq])
    for W in WS:
        m = Pall[:, 2] >= t0 - W
        P, z = Pall[m], zall[m]
        fq = mmse_bank_estimate(P, z, DEPTH, Pq, sigma0=sigma,
                                kp_hint=2 * np.pi / lam, seed=si)
        res[W].append((fq, gt_q))
    print(f"sea {si} done ({int(m.sum())} obs at W=70)", flush=True)

print(f"\ncausal forward-fan encounter forecast (hand-MMSE, {NSEAS} seas)")
print(f"{'W_mem':>6s} | corr by lead band: 0-10s   10-20s   20-30s | n_obs")
for W in WS:
    preds = np.stack([p for p, _ in res[W]])
    gts = np.stack([g for _, g in res[W]])
    line = f"{W:6.0f} |"
    for lo, hi in ((0, 10), (10, 20), (20, 30)):
        m = (LEADS >= lo) & (LEADS < hi)
        cc = corr(preds[:, m].ravel(), gts[:, m].ravel())
        line += f"          {cc:+.2f}"
    n_o = int(W / SCAN_DT) * M_PER_SCAN
    print(line + f" | {n_o}")
