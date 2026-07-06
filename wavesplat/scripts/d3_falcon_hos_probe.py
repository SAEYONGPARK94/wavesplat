"""
d3_falcon_hos_probe.py — GIVEN-SENSOR validation, HOS leg (2026-07-06,
standing instruction: headlines must reproduce under HOS nonlinearity).

Same protocol as tmp/probe_causal_forward.py (the 2026-07-05 synthetic
forward-fan encounter probe, corr 0.91 @ lead 0-10 s) but the sea is an
HOS M=3 nonlinear simulation instead of a linear component sum: ship on
an exogenous straight head-seas course, forward fan (+-60 deg, 20-100 m,
scan 0.5 s, 12 pts/scan, 2 cm noise), CAUSAL trailing memory W, hand-MMSE
(kp_hint = spectral peak, known Tp), encounter forecast at the ship's
FUTURE positions, lead 0-20 s. The estimator's linear dispersion prior is
now misspecified by design — the probe measures how much the bound
harmonics + nonlinear evolution cost.

Sea: JONSWAP Hs 2 m, Tp 8 s (lam_p ~ 100 m), spread 25 deg, depth 50 m,
domain 500 x 400 m periodic, 10 s nonlinear spin-up before the window.

Usage: python3 scripts/d3_falcon_hos_probe.py [--n_seas 4] [--M 3]
"""
import os
import sys
import json
import argparse

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
os.chdir(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, 'scripts')

from wavesim.hos_simulator import HOSSimulator, HOSConfig
from d3_sa_nomp import mmse_bank_estimate
from d3_sa_achievability_gap import corr

G = 9.81
OUT = 'results/direction3/falcon_hos_probe.json'

HS, TP, DEPTH = 2.0, 8.0, 50.0            # overridable (design pt: BF4)
WS = [10.0, 20.0, 40.0]
LEADS = np.arange(0.0, 21.0, 2.0)
SCAN_DT = 0.5
M_PER_SCAN = 12
U_SHIP = 3.0
NOISE = 0.02                       # Falcon 2 cm (1 sigma)
SPINUP = 10.0
DT_HOS = 0.1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n_seas', type=int, default=4)
    ap.add_argument('--M', type=int, default=3)
    ap.add_argument('--U', type=float, default=U_SHIP)
    ap.add_argument('--Hs', type=float, default=HS)
    ap.add_argument('--Tp', type=float, default=TP)
    ap.add_argument('--Lx', type=float, default=500.0)
    ap.add_argument('--Nx', type=int, default=128)
    ap.add_argument('--w_mems', type=float, nargs='+', default=WS)
    ap.add_argument('--lead_max', type=float, default=20.0)
    ap.add_argument('--lead_step', type=float, default=2.0)
    ap.add_argument('--tag', default='')
    ap.add_argument('--noise', type=float, default=NOISE,
                    help='range noise 1-sigma [m]')
    ap.add_argument('--noise_range', type=float, default=0.0,
                    help='range-proportional vertical error [rad] '
                         '(IMU attitude error: sigma_z = r * this)')
    ap.add_argument('--m_scan', type=int, default=M_PER_SCAN)
    ap.add_argument('--heading_deg', type=float, default=0.0,
                    help='ship course angle off the up-wave axis [deg]')
    ap.add_argument('--shadow_h', type=float, default=0.0,
                    help='sensor height above SWL [m]; >0 enables '
                         'geometric crest shadowing (occlusion raydrop)')
    a = ap.parse_args()
    global LEADS
    LEADS = np.arange(0.0, a.lead_max + 1e-9, a.lead_step)
    kp = (2 * np.pi / a.Tp) ** 2 / G          # deep-ish water peak k
    ws = list(a.w_mems)
    res = {W: [] for W in ws}
    for si in range(a.n_seas):
        cfg = HOSConfig(Lx=a.Lx, Ly=400.0, Nx=a.Nx, Ny=128,
                        depth=DEPTH, M=a.M)
        sim = HOSSimulator(cfg)
        sim.initialize_jonswap(Hs=a.Hs, Tp=a.Tp, direction_deg=0.0,
                               direction_spread_deg=25.0, seed=9000 + si)
        rng = np.random.default_rng(400 + si)
        t0 = SPINUP + max(ws)                  # 'now'
        cx, cy = 250.0, 200.0                  # domain center (periodic)
        hb = np.radians(a.heading_deg)
        hv = np.array([-np.cos(hb), np.sin(hb)])   # course unit vector

        def ship_pos(t):                       # course hv, into +x waves
            return np.array([cx, cy]) + a.U * (t - t0) * hv

        # ---- causal scans while stepping the nonlinear sea
        scan_ts = np.arange(t0 - max(ws), t0 + 1e-9, SCAN_DT)
        Pall, zall = [], []
        t_next = 0.0
        for t_s in scan_ts:
            while sim.t < t_s - 1e-9:
                sim.step(DT_HOS)
            sp = ship_pos(t_s)
            rr = rng.uniform(20.0, 100.0, a.m_scan)
            aa = rng.uniform(-np.pi / 3, np.pi / 3, a.m_scan)  # +-60 deg
            dx_ = hv[0] * np.cos(aa) - hv[1] * np.sin(aa)
            dy_ = hv[1] * np.cos(aa) + hv[0] * np.sin(aa)
            xs = sp[0] + rr * dx_              # fan ahead along course
            ys = sp[1] + rr * dy_
            z = sim.get_eta_at(np.mod(xs, cfg.Lx), np.mod(ys, cfg.Ly))
            keep = np.ones(a.m_scan, bool)
            if a.shadow_h > 0:                 # crest occlusion along rays
                r_chk = np.arange(10.0, 100.0, 4.0)
                Xc = sp[0] + r_chk[None, :] * dx_[:, None]
                Yc = sp[1] + r_chk[None, :] * dy_[:, None]
                zc = sim.get_eta_at(np.mod(Xc.ravel(), cfg.Lx),
                                    np.mod(Yc.ravel(), cfg.Ly)
                                    ).reshape(Xc.shape)
                el_c = (zc - a.shadow_h) / r_chk[None, :]
                el_t = (z - a.shadow_h) / rr
                for i_ in range(a.m_scan):
                    blk = r_chk < rr[i_]
                    if blk.any() and el_c[i_, blk].max() > el_t[i_]:
                        keep[i_] = False
            sig = a.noise + a.noise_range * rr
            Pall.append(np.column_stack(
                [xs, ys, np.full(a.m_scan, t_s)])[keep])
            zall.append((z + rng.normal(0, 1, a.m_scan) * sig)[keep])
        Pall = np.vstack(Pall)
        zall = np.concatenate(zall)

        # ---- future encounter GT from the continued nonlinear run
        gt_q = []
        for lead in LEADS:
            while sim.t < t0 + lead - 1e-9:
                sim.step(DT_HOS)
            sp = ship_pos(t0 + lead)
            gt_q.append(float(sim.get_eta_at(
                np.mod(sp[:1], cfg.Lx), np.mod(sp[1:2], cfg.Ly))[0]))
        gt_q = np.array(gt_q)
        Pq = np.array([[*ship_pos(t0 + l), t0 + l] for l in LEADS])

        for W in ws:
            m = Pall[:, 2] >= t0 - W
            fq = mmse_bank_estimate(Pall[m], zall[m], DEPTH, Pq,
                                    sigma0=0.05 * a.Hs, kp_hint=kp, seed=si)
            res[W].append((fq.tolist(), gt_q.tolist()))
        print(f"sea {si} done (Hs_run={4 * np.std(sim.get_eta()):.2f} m, "
              f"n_obs last-W: {int(m.sum())}, kept "
              f"{len(zall)}/{len(scan_ts) * a.m_scan})", flush=True)

    print(f"\nHOS M={a.M} causal forward-fan encounter forecast "
          f"(hand-MMSE, {a.n_seas} seas)")
    print(f"{'W_mem':>6s} | corr by lead band:   0-10s   10-20s")
    summary = {}
    for W in ws:
        preds = np.array([p for p, _ in res[W]])
        gts = np.array([g for _, g in res[W]])
        line = f"{W:6.0f} |"
        summary[W] = {}
        for lo, hi in ((0, 10), (10, 20)):
            m = (LEADS >= lo) & (LEADS <= hi)
            cc = corr(preds[:, m], gts[:, m])
            summary[W][f'{lo}-{hi}'] = cc
            line += f"          {cc:+.2f}"
        print(line)
    out_p = OUT.replace('.json', a.tag + '.json')
    json.dump(dict(M=a.M, n_seas=a.n_seas, Hs=a.Hs, Tp=a.Tp, U=a.U,
                   leads=LEADS.tolist(),
                   summary={str(k): v for k, v in summary.items()},
                   raw={str(k): v for k, v in res.items()}),
              open(out_p, 'w'), indent=1)
    print(f"saved {out_p}")


if __name__ == '__main__':
    main()
