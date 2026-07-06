"""
d3_bimodal_probe.py — close the forensic loop (2026-07-06): YS17
windows show skill collapsing with directional BIMODALITY (Spearman
-0.84, p=0.001, badwin_forensics). Reproduce causally in simulation and
test the OPERATIONAL gate.

Part 1 (cause): linear forward-fan encounter probe with a bimodal sea =
primary JONSWAP + secondary system rotated by theta2 carrying frac r2 of
the variance. If encounter corr drops with (theta2, r2) as on YS17, the
cause is confirmed.

Part 2 (gate): the ship can't see the stereo patch — but it CAN compute
the angular structure of its own fan-obs beamform power. For each config
(and each sea) compute bimod_obs = secondary-lobe fraction of beamform
power in the 0.5-2 kp annulus (folded axis, > 35 deg off primary), and
report corr(bimod_obs, realized skill) across runs -> causal gate signal.

Usage: python3 scripts/d3_bimodal_probe.py [--n_seas 4]
"""
import os
import sys
import json
import argparse

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
os.chdir(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, 'scripts')

from scipy.stats import spearmanr

from wavesim.wave_generator import jonswap_components
from d3_sa_fim_identifiability import omega_of
from d3_sa_nomp import mmse_bank_estimate

G = 9.81
OUT = 'results/direction3/bimodal_probe.json'

HS, TP, DEPTH = 2.0, 5.5, 50.0        # YS17-like
U, W = 2.0, 20.0
LEADS = np.arange(0.0, 11.0, 1.0)
SCAN_DT = 0.5
M_SCAN = 12
NOISE = 0.02

CONFIGS = [
    # (name, theta2 [deg], r2, Tp2) — Tp2 != TP tests the crossing-sea
    # frequency clipping hypothesis (bank annulus is kp-centered)
    ('unimodal', 0.0, 0.0, None),
    ('bi45_r03', 45.0, 0.3, None),
    ('bi90_r03', 90.0, 0.3, None),
    ('bi90_r05', 90.0, 0.5, None),
    ('cross9_r03', 90.0, 0.3, 9.0),
    ('cross9_r05', 90.0, 0.5, 9.0),
]


def beamform_bimod(P, z, kp, depth):
    """Secondary-lobe fraction of the obs beamform power (causal)."""
    dk = kp / 10.0
    ax = np.arange(-2.0 * kp, 2.0 * kp + dk / 2, dk)
    KX, KY = np.meshgrid(ax, ax)
    K = np.hypot(KX, KY)
    ann = (K > 0.5 * kp) & (K < 2.0 * kp)
    kxb, kyb = KX[ann], KY[ann]
    om = omega_of(kxb, kyb, depth)
    psi = (kxb[:, None] * P[None, :, 0] + kyb[:, None] * P[None, :, 1]
           - om[:, None] * P[None, :, 2])
    pw = (z[None] * np.cos(psi)).sum(1) ** 2 \
        + (z[None] * np.sin(psi)).sum(1) ** 2
    th = np.arctan2(kyb, kxb)
    C = np.sum(pw * np.cos(2 * th)) / pw.sum()
    S = np.sum(pw * np.sin(2 * th)) / pw.sum()
    th0 = 0.5 * np.arctan2(S, C)
    dth = np.abs((th - th0 + np.pi / 2) % np.pi - np.pi / 2)
    return float(pw[dth > np.radians(35)].sum() / pw.sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n_seas', type=int, default=4)
    a = ap.parse_args()
    kp = (2 * np.pi / TP) ** 2 / G
    rows = []
    for name, th2, r2, tp2 in CONFIGS:
        for si in range(a.n_seas):
            c1 = jonswap_components(Hs=HS * np.sqrt(1 - r2), Tp=TP,
                                    n_components=150, depth=DEPTH,
                                    direction_spread_deg=20.0,
                                    seed=9500 + si)
            comps = list(c1)
            if r2 > 0:
                c2 = jonswap_components(Hs=HS * np.sqrt(r2),
                                        Tp=(tp2 or TP),
                                        n_components=150, depth=DEPTH,
                                        direction_deg=th2,
                                        direction_spread_deg=20.0,
                                        seed=9700 + si)
                comps += list(c2)
            kx = np.array([x.kx for x in comps])
            ky = np.array([x.ky for x in comps])
            A = np.array([x.A for x in comps])
            ph = np.array([x.phi for x in comps])
            om = omega_of(kx, ky, DEPTH)

            def eta(x, y, t):
                ps = kx[None] * np.atleast_1d(x)[:, None] \
                    + ky[None] * np.atleast_1d(y)[:, None] \
                    - om[None] * np.atleast_1d(t)[:, None] + ph[None]
                return (A[None] * np.cos(ps)).sum(1)

            rng = np.random.default_rng(600 + si)
            t0 = W

            def ship_pos(t):
                return np.array([-U * (t - t0), 0.0])

            Pl, zl = [], []
            for t_s in np.arange(0.0, t0 + 1e-9, SCAN_DT):
                sp = ship_pos(t_s)
                rr = rng.uniform(20.0, 100.0, M_SCAN)
                aa = rng.uniform(-np.pi / 3, np.pi / 3, M_SCAN)
                xs = sp[0] - rr * np.cos(aa)
                ys = sp[1] + rr * np.sin(aa)
                Pl.append(np.column_stack([xs, ys,
                                           np.full(M_SCAN, t_s)]))
                zl.append(eta(xs, ys, t_s)
                          + rng.normal(0, NOISE, M_SCAN))
            P = np.vstack(Pl)
            z = np.concatenate(zl)
            tq = t0 + LEADS
            Pq = np.array([[*ship_pos(t), t] for t in tq])
            gt = eta(Pq[:, 0], Pq[:, 1], tq)
            fq = mmse_bank_estimate(P, z, DEPTH, Pq, sigma0=0.05 * HS,
                                    kp_hint=kp, seed=si)
            cc = float(np.corrcoef(fq, gt)[0, 1])
            bo = beamform_bimod(P, z, kp, DEPTH)
            rows.append(dict(config=name, si=si, corr=cc, bimod_obs=bo))
        cs = [r['corr'] for r in rows if r['config'] == name]
        print(f"[{name}] corr mean {np.mean(cs):+.2f} "
              f"({', '.join(f'{c:+.2f}' for c in cs)})", flush=True)

    cc = np.array([r['corr'] for r in rows])
    bo = np.array([r['bimod_obs'] for r in rows])
    rho, p = spearmanr(bo, cc)
    print(f"\ngate check: spearman(bimod_obs, realized corr) = "
          f"{rho:+.2f} (p={p:.4f}, n={len(rows)})")
    json.dump(dict(rows=rows, gate_spearman=float(rho), gate_p=float(p)),
              open(OUT, 'w'), indent=1)
    print(f"saved {OUT}")


if __name__ == '__main__':
    main()
