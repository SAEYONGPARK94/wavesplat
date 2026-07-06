"""
d3_net_uq_gate.py — does the AMORTIZED NET PRIOR's UQ generalize as a
gate across sites? (2026-07-06 night, user-directed #2 after all four
hand-crafted gate signals failed cross-site: sigma +0.05 / val-res
-0.31 / NIS +0.26 / bimod -0.78 on YS17 but +0.13 on AA15a.)

Past record (SA protocol): the net prior corrected moving-obs
overconfidence (z-std 1.47 -> 1.17, cov95 0.90+). Test here on the
forward-fan operator: 12 windows x {YS17, AA15a}, W = 20 s, net-gamma
MMSE with posterior variance ->
  calibration   z-std / cov95 vs hand-MMSE (2.05 / 0.80)
  cell gating   Spearman(mean sig_rel, realized corr) per site + pooled
  val-res       the net path's own held-out residual as alt signal

Usage: python3 scripts/d3_net_uq_gate.py
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
from d3_prior_net_eval import get_net, net_gamma_mmse

G = 9.81
OUT = 'results/direction3/net_uq_gate.json'
W = 20.0


def main():
    net = get_net()
    cells, z_all = [], []
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
            P_all, z_l = [], []
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
                z_l.append(Zmem[j][iy, ix]
                           + rng.normal(0, 0.02, len(pick)))
            P_all = np.vstack(P_all)
            z_l = np.concatenate(z_l)
            m = P_all[:, 2] >= t0 - W
            P, z = P_all[m], z_l[m]
            if len(z) > 600:
                ii = np.random.default_rng(0).choice(len(z), 600, False)
                P, z = P[ii], z[ii]
            Pq = np.array([[*ship_pos(t0 + l_), t0 + l_] for l_ in leads])
            gt_q = np.array([bilinear(Zfut[int(round(l_ / rec.dt))],
                                      rec.gx, rec.gy, px, py)
                             for (px, py, _t), l_ in zip(Pq, leads)])
            diag = {}
            pred, var = net_gamma_mmse(net, P, z, depth, Pq, 0.05 * Hs,
                                       seed=i0, kp_hint=kp, use_em=False,
                                       return_var=True, diag_out=diag)
            sig = np.sqrt(var)
            v = np.isfinite(gt_q) & (leads <= 10.0)
            cc = float(np.corrcoef(pred[v], gt_q[v])[0, 1])
            vfull = np.isfinite(gt_q)
            z_all.append(((pred - gt_q) / np.maximum(sig, 1e-9))[vfull])
            cells.append(dict(record=name, wi=wi, realized=cc,
                              risk=float(np.mean(sig[v]) / (Hs / 4)),
                              vr_rel=diag['vr_rel']))
            print(f"[{name} win{wi}] net corr={cc:+.2f} "
                  f"sig_rel={cells[-1]['risk']:.2f} "
                  f"vr_rel={diag['vr_rel']:.3f}", flush=True)

    z = np.concatenate(z_all)
    calib = dict(z_std=float(np.std(z)),
                 cov95=float(np.mean(np.abs(z) < 1.96)), n=int(len(z)))
    print(f"\nNET-UQ calibration: z-std {calib['z_std']:.2f} "
          f"(hand: 2.05), cov95 {calib['cov95']:.2f} (hand: 0.80)")
    res = dict(calibration=calib, cells=cells, gates={})
    for sig_name in ('risk', 'vr_rel'):
        for scope in ('YS17', 'AA15a', 'pooled'):
            cs = [c for c in cells
                  if scope == 'pooled' or c['record'] == scope]
            x = np.array([c[sig_name] for c in cs])
            r_ = np.array([c['realized'] for c in cs])
            rho, p = spearmanr(x, r_)
            med = np.median(x)
            res['gates'][f'{sig_name}_{scope}'] = dict(
                spearman=float(rho), p=float(p),
                confident=float(r_[x <= med].mean()),
                risky=float(r_[x > med].mean()))
            print(f"  {sig_name:7s} {scope:6s}: spearman {rho:+.2f} "
                  f"(p={p:.3f})  split {r_[x <= med].mean():+.2f}/"
                  f"{r_[x > med].mean():+.2f}")
    json.dump(res, open(OUT, 'w'), indent=1)
    print(f"saved {OUT}")


if __name__ == '__main__':
    main()
