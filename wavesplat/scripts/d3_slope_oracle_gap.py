"""
d3_slope_oracle_gap.py — is the slope ceiling (corr 0.68 at the design
point, gamma-tilt ineffective) INFORMATION or ALGORITHM? Compare the
production estimator against an ORACLE that knows the true per-cell
spectrum (gamma* = binned A^2/2 of the true modes, the same construction
as the training labels) on a linear sea at the design geometry.

If oracle slope ~ achieved slope -> information limit of the 48-pt fan
at 2 cm noise; the remedy is controller-side (vel-weighted FF).
If oracle >> achieved -> the spectral prior is the gap and learning
(v7) has headroom.

Usage: python3 scripts/d3_slope_oracle_gap.py [--n_seas 4]
"""
import os
import sys
import json
import argparse

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
os.chdir(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, 'scripts')

from wavesim.wave_generator import jonswap_components
from d3_sa_fim_identifiability import omega_of
from d18_relax_estimator import omega
from d3_sa_nomp import mmse_bank_estimate

G = 9.81
OUT = 'results/direction3/slope_oracle_gap.json'
U, HS, TP, DEPTH = 11.3, 1.2, 7.0, 50.0
W = 10.0
LEADS = np.arange(0.0, 6.1, 0.5)
SCAN_DT = 0.5
M_SCAN = 48                       # realistic Falcon density
NOISE = 0.02
DX, DT_FD = 0.5, 0.05


def oracle_predict(kxm, kym, Am, om_m, P, z, Pq, kp, s2, dk_frac=10.0):
    """Posterior mean with the TRUE binned spectrum as gamma."""
    dk = kp / dk_frac
    ax = np.arange(-2.5 * kp, 2.5 * kp + dk / 2, dk)
    KX, KY = np.meshgrid(ax, ax)
    kk = np.hypot(KX, KY)
    sel = (kk > 0.35 * kp) & (kk < 2.5 * kp)
    kxb, kyb = KX[sel].astype(float), KY[sel].astype(float)
    B = len(kxb)
    omb = omega(np.hypot(kxb, kyb) + 1e-12, DEPTH)
    # bin true mode variances to nearest cell
    g = np.zeros(B)
    for kx_, ky_, A_ in zip(kxm, kym, Am):
        d2 = (kxb - kx_) ** 2 + (kyb - ky_) ** 2
        g[np.argmin(d2)] += A_ ** 2 / 2.0
    gam = np.concatenate([g, g])

    def design(Q):
        psi = kxb[None] * Q[:, 0:1] + kyb[None] * Q[:, 1:2] \
            - omb[None] * Q[:, 2:3]
        return np.concatenate([np.cos(psi), np.sin(psi)], axis=1)

    Phi = design(P)
    K = (Phi * gam[None]) @ Phi.T + s2 * np.eye(len(z))
    w = Phi.T @ np.linalg.solve(K, z)
    return (design(Pq) * gam[None]) @ w


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n_seas', type=int, default=4)
    ap.add_argument('--sampling', default='random',
                    choices=['random', 'lines'],
                    help='random points in fan vs Falcon-structured '
                         'clusters (0.18 deg azimuth pairs + elevation-'
                         'step range pairs) at the SAME budget')
    ap.add_argument('--dk_frac', type=float, default=10.0,
                    help='bank spacing kp/dk_frac (both achieved+oracle)')
    ap.add_argument('--tag', default='')
    a = ap.parse_args()
    kp = (2 * np.pi / TP) ** 2 / G
    acc = {m: {k: ([], []) for k in ('eta', 'slope', 'vel')}
           for m in ('achieved', 'oracle')}
    for si in range(a.n_seas):
        comps = jonswap_components(Hs=HS, Tp=TP, n_components=200,
                                   depth=DEPTH, direction_spread_deg=25.0,
                                   seed=9000 + si)
        kxm = np.array([c.kx for c in comps])
        kym = np.array([c.ky for c in comps])
        Am = np.array([c.A for c in comps])
        phm = np.array([c.phi for c in comps])
        om_m = omega_of(kxm, kym, DEPTH)

        def eta_st(Q):
            ps = (kxm[None] * Q[:, 0:1] + kym[None] * Q[:, 1:2]
                  - om_m[None] * Q[:, 2:3] + phm[None])
            return (Am[None] * np.cos(ps)).sum(1)

        rng = np.random.default_rng(400 + si)
        t0 = W

        def ship_pos(t):
            return np.array([-U * (t - t0), 0.0])

        Pl, zl = [], []
        for t_s in np.arange(0.0, t0 + 1e-9, SCAN_DT):
            sp = ship_pos(t_s)
            if a.sampling == 'random':
                rr = rng.uniform(20.0, 100.0, M_SCAN)
                aa = rng.uniform(-np.pi / 3, np.pi / 3, M_SCAN)
            else:
                # Falcon-structured: M_SCAN/4 clusters of 4 = center,
                # azimuth-neighbor (r * 0.18 deg lateral), elevation-
                # neighbor (~5 m range step at 3 m sensor height), both
                n_cl = M_SCAN // 4
                r0 = rng.uniform(20.0, 95.0, n_cl)
                a0 = rng.uniform(-np.pi / 3, np.pi / 3, n_cl)
                dlat = r0 * np.radians(0.18)
                dr = np.clip(r0 ** 2 * np.radians(0.24) / 3.0, 1.0, 8.0)
                rr = np.concatenate([r0, r0, r0 + dr, r0 + dr])
                aa = np.concatenate([a0, a0 + dlat / r0, a0,
                                     a0 + dlat / r0])
            xs = sp[0] - rr * np.cos(aa)
            ys = sp[1] + rr * np.sin(aa)
            Q = np.column_stack([xs, ys, np.full(len(rr), t_s)])
            Pl.append(Q)
            zl.append(eta_st(Q) + rng.normal(0, NOISE, len(rr)))
        P = np.vstack(Pl)
        z = np.concatenate(zl)

        cen = np.array([[*ship_pos(t0 + l_), t0 + l_] for l_ in LEADS])
        Pq = []
        for c in cen:
            Pq += [c, c + [DX, 0, 0], c - [DX, 0, 0],
                   c + [0, 0, DT_FD], c - [0, 0, DT_FD]]
        Pq = np.array(Pq)
        gt = eta_st(Pq).reshape(-1, 5)
        preds = {
            'achieved': mmse_bank_estimate(
                P, z, DEPTH, Pq, sigma0=0.05 * HS, kp_hint=kp,
                seed=si, dk_c=kp / a.dk_frac).reshape(-1, 5),
            'oracle': oracle_predict(kxm, kym, Am, om_m, P, z, Pq, kp,
                                     NOISE ** 2,
                                     dk_frac=a.dk_frac).reshape(-1, 5)}
        for mth, fq in preds.items():
            pe = fq[:, 0]
            ps = (fq[:, 1] - fq[:, 2]) / (2 * DX)
            pv = (fq[:, 3] - fq[:, 4]) / (2 * DT_FD)
            ge = gt[:, 0]
            gs = (gt[:, 1] - gt[:, 2]) / (2 * DX)
            gv = (gt[:, 3] - gt[:, 4]) / (2 * DT_FD)
            for k, p_, g_ in (('eta', pe, ge), ('slope', ps, gs),
                              ('vel', pv, gv)):
                acc[mth][k][0].append(p_)
                acc[mth][k][1].append(g_)
        print(f"sea {si} done", flush=True)

    res = {}
    for mth in acc:
        res[mth] = {}
        line = f"{mth:9s}:"
        for k in ('eta', 'slope', 'vel'):
            p_ = np.concatenate(acc[mth][k][0])
            g_ = np.concatenate(acc[mth][k][1])
            cc = float(np.corrcoef(p_, g_)[0, 1])
            amp = float(np.std(p_) / (np.std(g_) + 1e-12))
            res[mth][k] = dict(corr=cc, amp_ratio=amp)
            line += f"  {k} {cc:+.3f}({amp:.2f})"
        print(line)
    out_p = OUT.replace('.json', a.tag + '.json') if a.tag else OUT
    json.dump(res, open(out_p, 'w'), indent=1)
    print(f"saved {out_p}")


if __name__ == '__main__':
    main()
