"""
d3_gate_confirmatory.py — CONFIRMATORY gate test, protocol pre-registered
in docs/GATE_CONFIRMATORY_prereg_2026-07-06.md (commit c76994e) BEFORE
this script was written. Do not tune anything here.

Fixed protocol: 4 unseen records (BS11a/b, BS13a/b), per-window scaled
geometry r_fan = (0.15, 0.7) lambda_p, U = 0.13 c_p, W = 10 s,
lead_max = 6 s (1 s steps), 12 windows/record, seeds rng(700+wi),
m_scan 12, m_cap 600, noise 2 cm. Signal = mmse diag vr_rel,
record-level median/MAD normalization. Decision: P1 pooled Spearman < 0
one-sided p < 0.05; P2 median-split delta > +0.10.

Usage: python3 scripts/d3_gate_confirmatory.py
"""
import os
import sys
import json

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
os.chdir(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, 'scripts')

from scipy.stats import spearmanr

from realdata_analysis import directional_spread
from d3_falcon_real_operator import bilinear, resolve_direction
from d3_sa_multirecord import MRecord
from d3_sa_nomp import mmse_bank_estimate

G = 9.81
OUT = 'results/direction3/gate_confirmatory.json'
RECORDS = {
    'BS11a': 'data/real_bergamasco/BS_2011/Surfaces_20111004_113800_short.nc',
    'BS11b': 'data/real_bergamasco/BS_2011/Surfaces_20111001_161800_short.nc',
    'BS13a': 'data/real_bergamasco/BS_2013/Surfaces_20130925_121501_short.nc',
    'BS13b': 'data/real_bergamasco/BS_2013/Surfaces_20130930_102001_short.nc',
}
DEPTH = 30.0
W = 10.0
LEAD_MAX = 6.0


def main():
    cells = []
    for name, path in RECORDS.items():
        rec = MRecord(path, stride=1)
        leads = np.arange(0.0, LEAD_MAX + 1e-9, 1.0)
        nfw = int(round(W / rec.dt))
        n_lead = int(round(LEAD_MAX / rec.dt))
        cx = 0.5 * (rec.gx.min() + rec.gx.max())
        cy = 0.5 * (rec.gy.min() + rec.gy.max())
        Xg, Yg = np.meshgrid(rec.gx, rec.gy)
        i0s = np.linspace(nfw + 1, rec.T_all - n_lead - 2, 12).astype(int)
        for wi, i0 in enumerate(i0s):
            i0 = int(i0)
            rng = np.random.default_rng(700 + wi)
            Zmem = rec.window(i0 - nfw + 1, nfw)
            Zfut = rec.window(i0, n_lead + 1)
            t0 = float(rec.t[i0])
            Hs = 4.0 * float(np.nanstd(Zmem))
            spread, lam_p, pdir_ax, kp = directional_spread(
                np.nan_to_num(Zmem[-1]), rec.dx, rec.dy)
            om_p = np.sqrt(G * kp * np.tanh(kp * DEPTH))
            cp = om_p / kp
            pdir, _ = resolve_direction(Zmem, rec, kp, pdir_ax, cp)
            h = -np.array([np.cos(np.radians(pdir)),
                           np.sin(np.radians(pdir))])
            U = 0.13 * cp
            r0, r1 = 0.15 * lam_p, 0.7 * lam_p
            T_tr = W + LEAD_MAX

            def ship_pos(t):
                return (np.array([cx, cy])
                        + U * (t - (t0 - W) - T_tr / 2) * h)

            P_all, z_all = [], []
            for j in range(nfw):
                t_j = float(rec.t[i0 - nfw + 1 + j])
                sp = ship_pos(t_j)
                RX, RY = Xg - sp[0], Yg - sp[1]
                R = np.hypot(RX, RY)
                ca = (RX * h[0] + RY * h[1]) / np.maximum(R, 1e-9)
                m = ((R >= r0) & (R <= r1) & (ca >= 0.5)
                     & np.isfinite(Zmem[j]))
                idx = np.flatnonzero(m)
                if len(idx) == 0:
                    continue
                pick = rng.choice(idx, min(12, len(idx)), replace=False)
                iy, ix = np.unravel_index(pick, Zmem[j].shape)
                P_all.append(np.column_stack([rec.gx[ix], rec.gy[iy],
                                              np.full(len(pick), t_j)]))
                z_all.append(Zmem[j][iy, ix]
                             + rng.normal(0, 0.02, len(pick)))
            if not P_all:
                continue
            P = np.vstack(P_all)
            z = np.concatenate(z_all)
            if len(z) > 600:
                ii = np.random.default_rng(0).choice(len(z), 600, False)
                P, z = P[ii], z[ii]
            Pq = np.array([[*ship_pos(t0 + l_), t0 + l_] for l_ in leads])
            gt_q = np.array([bilinear(Zfut[int(round(l_ / rec.dt))],
                                      rec.gx, rec.gy, px, py)
                             for (px, py, _t), l_ in zip(Pq, leads)])
            v = np.isfinite(gt_q)
            if v.sum() < 5:
                continue
            diag = {}
            fq = mmse_bank_estimate(P, z, DEPTH, Pq, sigma0=0.05 * Hs,
                                    kp_hint=kp, seed=i0, diag_out=diag)
            cc = float(np.corrcoef(fq[v], gt_q[v])[0, 1])
            cells.append(dict(record=name, wi=wi, vr_rel=diag['vr_rel'],
                              realized=cc, n_obs=int(len(z)),
                              lam_p=float(lam_p)))
            print(f"[{name} win{wi}] vr_rel={diag['vr_rel']:.3f} "
                  f"realized={cc:+.2f} (lam={lam_p:.0f}m n={len(z)})",
                  flush=True)

    # pre-registered analysis
    out = []
    for name in RECORDS:
        cs = [c for c in cells if c['record'] == name]
        if len(cs) < 4:
            continue
        vv = np.array([c['vr_rel'] for c in cs])
        med, mad = np.median(vv), np.median(np.abs(vv - np.median(vv)))
        for c, vn in zip(cs, (vv - med) / (mad + 1e-9)):
            out.append((vn, c['realized']))
    x = np.array([a for a, _ in out])
    r = np.array([b for _, b in out])
    rho, p_two = spearmanr(x, r)
    p_one = p_two / 2 if rho < 0 else 1 - p_two / 2
    med = np.median(x)
    d_split = float(r[x <= med].mean() - r[x > med].mean())
    p1 = bool(rho < 0 and p_one < 0.05)
    p2 = bool(d_split > 0.10)
    print(f"\nCONFIRMATORY (pre-registered, n={len(out)} cells, "
          f"4 unseen records):")
    print(f"  P1: spearman {rho:+.3f}, one-sided p={p_one:.4f} -> "
          f"{'PASS' if p1 else 'FAIL'}")
    print(f"  P2: split delta {d_split:+.3f} "
          f"(conf {r[x <= med].mean():+.2f} / risky "
          f"{r[x > med].mean():+.2f}) -> {'PASS' if p2 else 'FAIL'}")
    json.dump(dict(cells=cells, spearman=float(rho),
                   p_one_sided=float(p_one), split_delta=d_split,
                   P1_pass=p1, P2_pass=p2),
              open(OUT, 'w'), indent=1)
    print(f"saved {OUT}")


if __name__ == '__main__':
    main()
