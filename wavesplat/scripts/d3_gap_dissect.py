"""
d3_gap_dissect.py — dissect the synthetic-0.9 vs real-0.7 encounter gap
(2026-07-06, user-directed follow-up #2).

Suspects (directional spread already cleared: YS17 measured 25-35 deg ==
probe's 25 deg):

  noisefloor  GT = the stereo surface itself. Estimate its noise from
              the per-pixel temporal spectrum above 1.5 Hz (waves at
              Tp 5.5-7 s carry no energy there; white part = noise).
              corr ceiling = sqrt(SNR / (1 + SNR)). Caveats: bilinear
              GT sampling averages 4 px (noise down ~x2 if independent);
              obs share the same surface (common noise INFLATES corr) —
              so this is a bound, not a correction.

  current     YS17 is the current-suspect record. Re-run 3 windows
              (W=20) with the bank dispersion Doppler-shifted by Uc
              along the resolved wave axis, sweep |Uc| in
              {-0.4..+0.4} m/s: if some Uc != 0 heals the encounter
              corr, currents are (part of) the gap.

Usage: python3 scripts/d3_gap_dissect.py noisefloor
       python3 scripts/d3_gap_dissect.py current
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
from d3_sa_nomp import mmse_bank_estimate

G = 9.81
OUT = 'results/direction3/gap_dissect_{}.json'


# ------------------------------------------------------------- noise floor
def noisefloor(records=('YS17', 'AA15a'), n_px=400, n_t=2048, f_hi=1.5):
    out = {}
    for name in records:
        cfg = RECORDS[name]
        rec = MRecord(cfg['path'], stride=cfg['stride'])
        n_t_ = min(n_t, rec.T_all)
        Z = rec.window(0, n_t_)                       # (T, ny, nx)
        vy, vx = np.where(np.isfinite(Z).all(axis=0))
        rng = np.random.default_rng(0)
        pick = rng.choice(len(vy), min(n_px, len(vy)), replace=False)
        ts = Z[:, vy[pick], vx[pick]]                 # (T, n_px)
        ts = ts - ts.mean(axis=0, keepdims=True)
        F = np.fft.rfft(ts * np.hanning(n_t_)[:, None], axis=0)
        # Hann window power compensation: sum(w^2)/N
        wpow = float(np.mean(np.hanning(n_t_) ** 2))
        psd = (np.abs(F) ** 2).mean(axis=1) / (n_t_ * wpow) * 2 * rec.dt
        fr = np.fft.rfftfreq(n_t_, rec.dt)
        nyq = 0.5 / rec.dt
        hi = (fr >= f_hi) & (fr <= nyq - 0.2)
        noise_density = float(np.median(psd[hi]))     # [m^2/Hz], white part
        sig_n2 = noise_density * nyq                  # full-band white power
        var_tot = float(ts.var())
        sig_s2 = max(var_tot - sig_n2, 1e-12)
        snr = sig_s2 / sig_n2
        ceiling = float(np.sqrt(snr / (1 + snr)))
        out[name] = dict(sigma_n=float(np.sqrt(sig_n2)),
                         sigma_sig=float(np.sqrt(sig_s2)),
                         snr=float(snr), corr_ceiling=ceiling,
                         n_px=int(len(pick)), n_t=n_t_, f_hi=f_hi)
        print(f"[{name}] stereo noise sigma_n={np.sqrt(sig_n2)*100:.1f} cm "
              f"(signal {np.sqrt(sig_s2):.2f} m, SNR {snr:.0f}) -> "
              f"corr ceiling {ceiling:.3f}", flush=True)
    json.dump(out, open(OUT.format('noisefloor'), 'w'), indent=1)
    print(f"saved {OUT.format('noisefloor')}")


# ------------------------------------------------------------- current sweep
def current(wins=(0, 3, 5), W=20.0, ucs=(-0.4, -0.2, 0.0, 0.2, 0.4)):
    name = 'YS17'
    cfg = RECORDS[name]
    rec = MRecord(cfg['path'], stride=cfg['stride'])
    depth, U = cfg['depth'], cfg['U']
    leads = np.arange(0.0, cfg['lead_max'] + 1e-9, 1.0)
    W_max = 40.0                                       # match the main run
    nfw = int(round(W_max / rec.dt))
    n_lead = int(round(cfg['lead_max'] / rec.dt))
    cx = 0.5 * (rec.gx.min() + rec.gx.max())
    cy = 0.5 * (rec.gy.min() + rec.gy.max())
    Xg, Yg = np.meshgrid(rec.gx, rec.gy)
    i0s = np.linspace(nfw + 1, rec.T_all - n_lead - 2, 6).astype(int)
    res = []
    for wi in wins:
        i0 = int(i0s[wi])
        rng = np.random.default_rng(700 + wi)
        Zmem = rec.window(i0 - nfw + 1, nfw)
        Zfut = rec.window(i0, n_lead + 1)
        t0 = float(rec.t[i0])
        Hs = 4.0 * float(np.nanstd(Zmem))
        spread, lam_p, pdir_ax, kp = directional_spread(
            np.nan_to_num(Zmem[-1]), rec.dx, rec.dy)
        om_p = np.sqrt(G * kp * np.tanh(kp * depth))
        pdir, _ = resolve_direction(Zmem, rec, kp, pdir_ax, om_p / kp)
        wvec = np.array([np.cos(np.radians(pdir)), np.sin(np.radians(pdir))])
        h = -wvec
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
        Pq = np.array([[*ship_pos(t0 + l_), t0 + l_] for l_ in leads])
        gt_q = np.array([bilinear(Zfut[int(round(l_ / rec.dt))],
                                  rec.gx, rec.gy, px, py)
                         for (px, py, _t), l_ in zip(Pq, leads)])
        v = np.isfinite(gt_q) & (leads <= 10.0)
        row = dict(wi=wi, t0=t0, pdir=pdir, curve={})
        line = f"[win {wi}]"
        for uc in ucs:
            fq = mmse_bank_estimate(P, z, depth, Pq, sigma0=0.05 * Hs,
                                    kp_hint=kp, seed=i0,
                                    Uc=tuple(uc * wvec))
            cc = float(np.corrcoef(fq[v], gt_q[v])[0, 1])
            row['curve'][f"{uc:+.1f}"] = cc
            line += f"  Uc{uc:+.1f}: {cc:+.2f}"
        res.append(row)
        print(line, flush=True)
    json.dump(res, open(OUT.format('current'), 'w'), indent=1)
    print(f"saved {OUT.format('current')}")


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('mode', choices=['noisefloor', 'current'])
    a = ap.parse_args()
    noisefloor() if a.mode == 'noisefloor' else current()
