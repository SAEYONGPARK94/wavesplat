"""
d3_val_gate_probe.py — is the estimator's OWN held-out residual (val
RMSE, computed inside mmse_bank_estimate on 25% of the fan obs) a gate
for the real-data window-skill swings that the posterior variance missed
(d3_uq_gating: cell Spearman +0.05)?

12 cells: {YS17, AA15a} x 6 windows, W = 20 s, same seeds as the main
run. Gate signal = vr_rel (val RMSE / std of held-out obs, ~ relative
predictive misfit, causal: uses only in-window data). Score = Spearman
vs realized encounter corr (leads 0-10 s) + median-split means.

Usage: python3 scripts/d3_val_gate_probe.py
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
OUT = 'results/direction3/val_gate_probe.json'
W = 20.0


def main():
    cells = []
    for name in ('YS17', 'AA15a'):
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
        i0s = np.linspace(nfw + 1, rec.T_all - n_lead - 2, 6).astype(int)
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
                z_all.append(Zmem[j][iy, ix]
                             + rng.normal(0, 0.02, len(pick)))
            P_all = np.vstack(P_all)
            z_all = np.concatenate(z_all)
            m = P_all[:, 2] >= t0 - W
            P, z = P_all[m], z_all[m]
            if len(z) > 600:
                ii = np.random.default_rng(0).choice(len(z), 600, False)
                P, z = P[ii], z[ii]
            Pq = np.array([[*ship_pos(t0 + l_), t0 + l_] for l_ in leads])
            gt_q = np.array([bilinear(Zfut[int(round(l_ / rec.dt))],
                                      rec.gx, rec.gy, px, py)
                             for (px, py, _t), l_ in zip(Pq, leads)])
            diag = {}
            fq = mmse_bank_estimate(P, z, depth, Pq, sigma0=0.05 * Hs,
                                    kp_hint=kp, seed=i0, diag_out=diag)
            v = np.isfinite(gt_q) & (leads <= 10.0)
            cc = float(np.corrcoef(fq[v], gt_q[v])[0, 1])
            cells.append(dict(record=name, wi=wi, vr_rel=diag['vr_rel'],
                              realized=cc))
            print(f"[{name} win{wi}] vr_rel={diag['vr_rel']:.3f} "
                  f"realized={cc:+.2f}", flush=True)

    vr = np.array([c['vr_rel'] for c in cells])
    re = np.array([c['realized'] for c in cells])
    rho, p = spearmanr(vr, re)
    med = np.median(vr)
    print(f"\nval-residual gate (12 cells, W=20): "
          f"spearman(vr_rel, realized) = {rho:+.2f} (p={p:.3f})")
    print(f"  confident half (low vr): {re[vr <= med].mean():+.2f} | "
          f"risky half: {re[vr > med].mean():+.2f}")
    json.dump(dict(cells=cells, spearman=float(rho), p=float(p),
                   confident=float(re[vr <= med].mean()),
                   risky=float(re[vr > med].mean())),
              open(OUT, 'w'), indent=1)
    print(f"saved {OUT}")


if __name__ == '__main__':
    main()
