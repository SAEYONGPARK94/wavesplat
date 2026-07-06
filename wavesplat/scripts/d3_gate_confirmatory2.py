"""
d3_gate_confirmatory2.py — confirmatory gate test #2, protocol
pre-registered in docs/GATE_CONFIRMATORY2_prereg_2026-07-06.md (commit
23d45e5) BEFORE this script. Mid/large-patch condition: AA15b (unseen
record, 12 windows) + midpoint-offset windows (11 each) of the three
development records. Signal/normalization/decision identical to #1.

Usage: python3 scripts/d3_gate_confirmatory2.py
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
from d3_falcon_real_operator import RECORDS, bilinear, resolve_direction
from d3_sa_multirecord import MRecord
from d3_sa_nomp import mmse_bank_estimate

G = 9.81
OUT = 'results/direction3/gate_confirmatory2.json'
W = 20.0
AA15B = dict(path='data/real_bergamasco/AA_2015/Surfaces_20150515_090000.nc',
             depth=17.0, stride=1, r_fan=(8.0, 40.0), U=1.2, lead_max=14.0)


def run_windows(name, cfg, i0_list, seed_base):
    rec = MRecord(cfg['path'], stride=cfg['stride'])
    depth, U = cfg['depth'], cfg['U']
    leads = np.arange(0.0, cfg['lead_max'] + 1e-9, 1.0)
    W_max = 40.0
    nfw = int(round(W_max / rec.dt))
    n_lead = int(round(cfg['lead_max'] / rec.dt))
    cx = 0.5 * (rec.gx.min() + rec.gx.max())
    cy = 0.5 * (rec.gy.min() + rec.gy.max())
    Xg, Yg = np.meshgrid(rec.gx, rec.gy)
    cells = []
    for wi, i0 in enumerate(i0_list):
        i0 = int(i0)
        rng = np.random.default_rng(seed_base + wi)
        Zmem = rec.window(i0 - nfw + 1, nfw)
        Zfut = rec.window(i0, n_lead + 1)
        t0 = float(rec.t[i0])
        Hs = 4.0 * float(np.nanstd(Zmem))
        spread, lam_p, pdir_ax, kp = directional_spread(
            np.nan_to_num(Zmem[-1]), rec.dx, rec.dy)
        om_p = np.sqrt(G * kp * np.tanh(kp * depth))
        pdir, _ = resolve_direction(Zmem, rec, kp, pdir_ax, om_p / kp)
        h = -np.array([np.cos(np.radians(pdir)),
                       np.sin(np.radians(pdir))])
        T_tr = W_max + cfg['lead_max']

        def ship_pos(t):
            return (np.array([cx, cy])
                    + U * (t - (t0 - W_max) - T_tr / 2) * h)

        r0, r1 = cfg['r_fan']
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
            z_all.append(Zmem[j][iy, ix] + rng.normal(0, 0.02, len(pick)))
        if not P_all:
            continue
        P = np.vstack(P_all)
        z = np.concatenate(z_all)
        m = P[:, 2] >= t0 - W
        P, z = P[m], z[m]
        if len(z) > 600:
            ii = np.random.default_rng(0).choice(len(z), 600, False)
            P, z = P[ii], z[ii]
        Pq = np.array([[*ship_pos(t0 + l_), t0 + l_] for l_ in leads])
        gt_q = np.array([bilinear(Zfut[int(round(l_ / rec.dt))],
                                  rec.gx, rec.gy, px, py)
                         for (px, py, _t), l_ in zip(Pq, leads)])
        v = np.isfinite(gt_q) & (leads <= 10.0)
        if v.sum() < 5:
            continue
        diag = {}
        fq = mmse_bank_estimate(P, z, depth, Pq, sigma0=0.05 * Hs,
                                kp_hint=kp, seed=i0, diag_out=diag)
        cc = float(np.corrcoef(fq[v], gt_q[v])[0, 1])
        cells.append(dict(record=name, wi=wi, vr_rel=diag['vr_rel'],
                          realized=cc))
        print(f"[{name} w{wi}] vr={diag['vr_rel']:.3f} real={cc:+.2f}",
              flush=True)
    return cells


def main():
    cells = []
    # AA15b: standard 12-window tiling (unseen record)
    rec_b = MRecord(AA15B['path'], stride=1)
    nfw = int(round(40.0 / rec_b.dt))
    n_lead = int(round(AA15B['lead_max'] / rec_b.dt))
    i0s_b = np.linspace(nfw + 1, rec_b.T_all - n_lead - 2, 12).astype(int)
    del rec_b
    cells += run_windows('AA15b', AA15B, i0s_b, 700)
    # midpoint-offset windows of the three development records
    for name in ('YS17', 'AA15a', 'AA14'):
        cfg = RECORDS[name]
        rec = MRecord(cfg['path'], stride=cfg['stride'])
        nfw = int(round(40.0 / rec.dt))
        n_lead = int(round(cfg['lead_max'] / rec.dt))
        i0s = np.linspace(nfw + 1, rec.T_all - n_lead - 2, 12).astype(int)
        mids = ((i0s[:-1] + i0s[1:]) // 2).astype(int)
        del rec
        cells += run_windows(name + '_mid', cfg, mids, 1700)

    out = []
    for name in sorted(set(c['record'] for c in cells)):
        cs = [c for c in cells if c['record'] == name]
        vv = np.array([c['vr_rel'] for c in cs])
        med = np.median(vv)
        mad = np.median(np.abs(vv - med)) + 1e-9
        for c, vn in zip(cs, (vv - med) / mad):
            out.append((vn, c['realized']))
    x = np.array([a for a, _ in out])
    r = np.array([b for _, b in out])
    rho, p_two = spearmanr(x, r)
    p_one = p_two / 2 if rho < 0 else 1 - p_two / 2
    med = np.median(x)
    d_split = float(r[x <= med].mean() - r[x > med].mean())
    p1 = bool(rho < 0 and p_one < 0.05)
    p2 = bool(d_split > 0.10)
    print(f"\nCONFIRMATORY #2 (pre-registered, n={len(out)}):")
    print(f"  P1: spearman {rho:+.3f}, one-sided p={p_one:.4f} -> "
          f"{'PASS' if p1 else 'FAIL'}")
    print(f"  P2: split {d_split:+.3f} (conf {r[x <= med].mean():+.2f} / "
          f"risky {r[x > med].mean():+.2f}) -> {'PASS' if p2 else 'FAIL'}")
    json.dump(dict(cells=cells, spearman=float(rho),
                   p_one_sided=float(p_one), split_delta=d_split,
                   P1_pass=p1, P2_pass=p2), open(OUT, 'w'), indent=1)
    print(f"saved {OUT}")


if __name__ == '__main__':
    main()
