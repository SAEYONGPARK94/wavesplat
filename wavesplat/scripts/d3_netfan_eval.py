"""
d3_netfan_eval.py — evaluate a retrained prior-net checkpoint on the
real forward-fan 24-window protocol (both records, W=20, same seeds),
against the saved hand (val_gate_probe.json) and v3 zero-shot
(net_uq_gate.json) baselines.

Usage: python3 scripts/d3_netfan_eval.py --ckpt results/direction3/prior_net_v6fan.pt --tag v6fan
"""
import os
import sys
import json
import argparse

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
os.chdir(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, 'scripts')

from realdata_analysis import directional_spread
from d3_falcon_real_operator import RECORDS, bilinear, resolve_direction
from d3_sa_multirecord import MRecord
from d3_prior_net_eval import get_net, net_gamma_mmse

G = 9.81
W = 20.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--tag', required=True)
    a = ap.parse_args()
    net = get_net(a.ckpt)
    hand = {(c['record'], c['wi']): c['realized'] for c in json.load(
        open('results/direction3/val_gate_probe.json'))['cells']}
    v3 = {(c['record'], c['wi']): c['realized'] for c in json.load(
        open('results/direction3/net_uq_gate.json'))['cells']}
    rows = []
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
            pdir, _ = resolve_direction(Zmem, rec, kp, pdir_ax,
                                        om_p / kp)
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
                P_all.append(np.column_stack([
                    rec.gx[ix], rec.gy[iy], np.full(len(pick), t_j)]))
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
            fq = net_gamma_mmse(net, P, z, depth, Pq, 0.05 * Hs,
                                seed=i0, kp_hint=kp, use_em=False)
            v = np.isfinite(gt_q) & (leads <= 10.0)
            cc = float(np.corrcoef(fq[v], gt_q[v])[0, 1])
            rows.append(dict(record=name, wi=wi, this=cc,
                             hand=hand.get((name, wi)),
                             v3=v3.get((name, wi))))
            print(f"[{name} win{wi}] hand={hand.get((name, wi)):+.2f} "
                  f"v3={v3.get((name, wi)):+.2f} {a.tag}={cc:+.2f}",
                  flush=True)
    hb = np.array([r['hand'] for r in rows])
    for k in ('hand', 'v3', 'this'):
        x = np.array([r[k] for r in rows])
        print(f"{k:5s}: mean {x.mean():+.3f}  bad-cells "
              f"{x[hb < 0.5].mean():+.3f}  good {x[hb >= 0.5].mean():+.3f}")
    json.dump(rows, open(
        f'results/direction3/{a.tag}_real_eval.json', 'w'), indent=1)
    print(f"saved results/direction3/{a.tag}_real_eval.json")


if __name__ == '__main__':
    main()
