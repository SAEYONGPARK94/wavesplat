"""
d3_hull_value_test.py — re-examination of improvement lever #2
(2026-07-06, user: "2번 한 번 더 검토"): if the vessel could measure the
wave under itself (ride-height sensor + IMU complementary filter gives
eta at the hull in the encounter band, buoy principle; 5 cm noise
assumed), how much does adding that stream to the forward-fan obs help
the REAL encounter forecast? YS17, 12 windows, W=20 s, same seeds;
compare per-window corr vs the fan-only baselines in
val_gate_probe.json. If the delta is negligible the observability
debate is moot (user's rejection stands) -> proceed to lever #4.

Usage: python3 scripts/d3_hull_value_test.py
"""
import os
import sys
import json

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
os.chdir(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, 'scripts')

from realdata_analysis import directional_spread
from d3_falcon_real_operator import RECORDS, bilinear, resolve_direction
from d3_sa_multirecord import MRecord
from d3_sa_nomp import mmse_bank_estimate

G = 9.81
W = 20.0
OUT = 'results/direction3/hull_value_test.json'


def main():
    base = {(c['record'], c['wi']): c['realized']
            for c in json.load(
                open('results/direction3/val_gate_probe.json'))['cells']}
    rows = []
    name = 'YS17'
    cfg = RECORDS[name]
    rec = MRecord(cfg['path'], stride=cfg['stride'])
    depth, U = cfg['depth'], cfg['U']
    leads = np.arange(0.0, cfg['lead_max'] + 1e-9, 1.0)
    W_max = 40.0
    nfw = int(round(W_max / rec.dt))
    n_lead = int(round(cfg['lead_max'] / rec.dt))
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
        P_all = np.vstack(P_all)
        z_all = np.concatenate(z_all)
        m = P_all[:, 2] >= t0 - W
        P, z = P_all[m], z_all[m]
        if len(z) > 600:
            ii = np.random.default_rng(0).choice(len(z), 600, False)
            P, z = P[ii], z[ii]
        Ph, zh = [], []
        for j in range(nfw - int(W / rec.dt), nfw):
            t_j = float(rec.t[i0 - nfw + 1 + j])
            sp = ship_pos(t_j)
            v = bilinear(Zmem[j], rec.gx, rec.gy, sp[0], sp[1])
            if np.isfinite(v):
                Ph.append([sp[0], sp[1], t_j])
                zh.append(v + rng.normal(0, 0.05))
        P2 = np.vstack([P, np.array(Ph)])
        z2 = np.concatenate([z, np.array(zh)])
        Pq = np.array([[*ship_pos(t0 + l_), t0 + l_] for l_ in leads])
        gt_q = np.array([bilinear(Zfut[int(round(l_ / rec.dt))],
                                  rec.gx, rec.gy, px, py)
                         for (px, py, _t), l_ in zip(Pq, leads)])
        fq = mmse_bank_estimate(P2, z2, depth, Pq, sigma0=0.05 * Hs,
                                kp_hint=kp, seed=i0)
        v = np.isfinite(gt_q) & (leads <= 10.0)
        cc = float(np.corrcoef(fq[v], gt_q[v])[0, 1])
        b = base.get((name, wi), np.nan)
        rows.append(dict(record=name, wi=wi, hull=cc, fan_only=b,
                         delta=cc - b, n_hull=len(zh)))
        print(f"[{name} win{wi}] fan={b:+.2f} fan+hull={cc:+.2f} "
              f"delta={cc - b:+.2f} (n_hull={len(zh)})", flush=True)
    d = np.array([r['delta'] for r in rows])
    s_ = np.array([r['fan_only'] for r in rows])
    print(f"\nmean delta {d.mean():+.3f} | bad(fan<0.5, "
          f"n={(s_ < 0.5).sum()}): {d[s_ < 0.5].mean():+.3f} | "
          f"good: {d[s_ >= 0.5].mean():+.3f}")
    json.dump(rows, open(OUT, 'w'), indent=1)
    print(f"saved {OUT}")


if __name__ == '__main__':
    main()
