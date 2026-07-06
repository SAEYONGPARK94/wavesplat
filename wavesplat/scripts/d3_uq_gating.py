"""
d3_uq_gating.py — UQ reliability gating for the real-data Falcon
encounter forecast (2026-07-06, user-directed follow-up #1).

Question: the pooled encounter corr is ~0.7 but per-window skill swings
0.15-0.88 — does the MMSE posterior variance KNOW when the forecast is
bad, so a controller can gate the feedforward (K_ff scheduling)?

Protocol: replicate d3_falcon_real_operator.py windows verbatim (same
seeds), but request the posterior predictive variance at the encounter
points (mmse_bank_estimate(..., return_var=True)). Then:

  calibration  z = (pred - gt) / sigma_pred pooled: std(z), cov68/95.
               (GT = stereo surface, has its own noise -> z inflated;
               reported as-is, honest lower bound on calibration.)
  cell gating  cell = (record, window, W): predicted risk = mean
               sigma_pred / (Hs/4) vs realized corr over leads 0-10 s;
               Spearman rank corr + median-split realized corr.
  point gating pooled points split by sigma_pred quantile: realized
               corr of the confident half vs the risky half.

Usage: python3 scripts/d3_uq_gating.py [--records YS17 AA15a]
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

from realdata_analysis import directional_spread
from d3_falcon_real_operator import RECORDS, bilinear, resolve_direction
from d3_sa_multirecord import MRecord
from d3_sa_nomp import mmse_bank_estimate

G = 9.81
OUT = 'results/direction3/uq_gating_real.json'


def run_record(name, w_mems, n_win=6, m_scan=12, m_cap=600, noise=0.02):
    cfg = RECORDS[name]
    rec = MRecord(cfg['path'], stride=cfg['stride'])
    depth = cfg['depth']
    U = cfg['U']
    leads = np.arange(0.0, cfg['lead_max'] + 1e-9, 1.0)
    W_max = max(w_mems)
    nfw = int(round(W_max / rec.dt))
    n_lead = int(round(cfg['lead_max'] / rec.dt))
    cx = 0.5 * (rec.gx.min() + rec.gx.max())
    cy = 0.5 * (rec.gy.min() + rec.gy.max())
    Xg, Yg = np.meshgrid(rec.gx, rec.gy)
    i0s = np.linspace(nfw + 1, rec.T_all - n_lead - 2, n_win).astype(int)
    rows = []
    for wi, i0 in enumerate(i0s):
        rng = np.random.default_rng(700 + wi)
        Zmem = rec.window(i0 - nfw + 1, nfw)
        Zfut = rec.window(i0, n_lead + 1)
        t0 = float(rec.t[i0])
        Hs = 4.0 * float(np.nanstd(Zmem))
        spread, lam_p, pdir_ax, kp = directional_spread(
            np.nan_to_num(Zmem[-1]), rec.dx, rec.dy)
        om_p = np.sqrt(G * kp * np.tanh(kp * depth))
        cp = om_p / kp
        pdir, _ = resolve_direction(Zmem, rec, kp, pdir_ax, cp)
        h = -np.array([np.cos(np.radians(pdir)), np.sin(np.radians(pdir))])
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
            m = ((R >= r0) & (R <= r1) & (ca >= 0.5) & np.isfinite(Zmem[j]))
            idx = np.flatnonzero(m)
            if len(idx) == 0:
                continue
            pick = rng.choice(idx, min(m_scan, len(idx)), replace=False)
            iy, ix = np.unravel_index(pick, Zmem[j].shape)
            P_all.append(np.column_stack([rec.gx[ix], rec.gy[iy],
                                          np.full(len(pick), t_j)]))
            z_all.append(Zmem[j][iy, ix] + rng.normal(0, noise, len(pick)))
        P_all = np.vstack(P_all)
        z_all = np.concatenate(z_all)
        Pq = np.array([[*ship_pos(t0 + l_), t0 + l_] for l_ in leads])
        gt_q = np.array([bilinear(Zfut[int(round(l_ / rec.dt))],
                                  rec.gx, rec.gy, px, py)
                         for (px, py, _t), l_ in zip(Pq, leads)])
        for W in w_mems:
            m = P_all[:, 2] >= t0 - W
            P, z = P_all[m], z_all[m]
            if len(z) > m_cap:
                ii = np.random.default_rng(0).choice(len(z), m_cap, False)
                P, z = P[ii], z[ii]
            pred, var = mmse_bank_estimate(P, z, depth, Pq,
                                           sigma0=0.05 * Hs, kp_hint=kp,
                                           seed=int(i0), return_var=True)
            rows.append(dict(record=name, wi=wi, W=W, Hs=Hs,
                             leads=leads.tolist(), pred=pred.tolist(),
                             sig=np.sqrt(var).tolist(), gt=gt_q.tolist()))
            v = np.isfinite(gt_q)
            cc = np.corrcoef(pred[v], gt_q[v])[0, 1]
            sr = float(np.mean(np.sqrt(var)) / (Hs / 4))
            print(f"  [{name} win{wi} W={W:.0f}] corr={cc:+.2f} "
                  f"mean sig_rel={sr:.2f}", flush=True)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--records', nargs='+', default=['YS17', 'AA15a'])
    ap.add_argument('--w_mems', type=float, nargs='+', default=[10., 20., 40.])
    a = ap.parse_args()
    rows = []
    for r in a.records:
        print(f"[{r}]", flush=True)
        rows += run_record(r, a.w_mems)

    # ---------------- calibration (pooled z-scores)
    z_all, cells = [], []
    for r_ in rows:
        pred = np.array(r_['pred'])
        gt = np.array(r_['gt'])
        sig = np.array(r_['sig'])
        leads = np.array(r_['leads'])
        v = np.isfinite(gt)
        z_all.append(((pred - gt) / np.maximum(sig, 1e-9))[v])
        b = v & (leads <= 10.0)
        cells.append(dict(
            record=r_['record'], wi=r_['wi'], W=r_['W'],
            risk=float(np.mean(sig[b]) / (r_['Hs'] / 4)),
            realized=float(np.corrcoef(pred[b], gt[b])[0, 1])))
    z = np.concatenate(z_all)
    calib = dict(z_std=float(np.std(z)),
                 cov68=float(np.mean(np.abs(z) < 1.0)),
                 cov95=float(np.mean(np.abs(z) < 1.96)), n=int(len(z)))
    print(f"\ncalibration: z-std {calib['z_std']:.2f} "
          f"(1=perfect), cov68 {calib['cov68']:.2f}, "
          f"cov95 {calib['cov95']:.2f}  [GT noise not subtracted]")

    # ---------------- cell-level gating
    risk = np.array([c['risk'] for c in cells])
    real = np.array([c['realized'] for c in cells])
    rho, pval = spearmanr(risk, real)
    med = np.median(risk)
    lo_r = real[risk <= med].mean()
    hi_r = real[risk > med].mean()
    print(f"cell gating ({len(cells)} cells): spearman(risk, realized corr)"
          f" = {rho:+.2f} (p={pval:.3f})")
    print(f"  confident half: realized corr {lo_r:+.2f} | "
          f"risky half: {hi_r:+.2f}")

    # ---------------- point-level gating (pooled, leads 0-10 s)
    P_, G_, S_ = [], [], []
    for r_ in rows:
        leads = np.array(r_['leads'])
        v = np.isfinite(np.array(r_['gt'])) & (leads <= 10.0)
        P_.append(np.array(r_['pred'])[v])
        G_.append(np.array(r_['gt'])[v])
        S_.append((np.array(r_['sig']) / (r_['Hs'] / 4))[v])
    P_, G_, S_ = map(np.concatenate, (P_, G_, S_))
    medS = np.median(S_)
    c_conf = float(np.corrcoef(P_[S_ <= medS], G_[S_ <= medS])[0, 1])
    c_risk = float(np.corrcoef(P_[S_ > medS], G_[S_ > medS])[0, 1])
    print(f"point gating ({len(P_)} pts, leads 0-10s): "
          f"confident half corr {c_conf:+.2f} | risky half {c_risk:+.2f}")

    json.dump(dict(calibration=calib, cells=cells,
                   cell_gating=dict(spearman=float(rho), p=float(pval),
                                    confident=float(lo_r), risky=float(hi_r)),
                   point_gating=dict(confident=c_conf, risky=c_risk,
                                     n=int(len(P_))),
                   rows=rows),
              open(OUT, 'w'), indent=1)
    print(f"saved {OUT}")


if __name__ == '__main__':
    main()
