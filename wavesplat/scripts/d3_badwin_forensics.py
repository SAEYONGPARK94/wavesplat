"""
d3_badwin_forensics.py — why do half the real windows lose skill?
(2026-07-06 evening; naive drift metrics + 5 usual suspects already
rejected). Three causal-candidate metrics per YS17 window, Spearman'd
against the realized encounter skill of the 12-window val_gate run:

  bimod    directional bimodality: time-averaged 2D wavenumber spectrum
           (last 20 s of the memory window), energy in the annulus
           0.5-2 kp split by angle: secondary-lobe fraction = energy
           outside +-35 deg of the primary axis (folded).
  fanfrac  fan-coverage: mean fraction of the forward fan that lands on
           valid stereo pixels (obs geometry starvation).
  adv5     linear advectability: corr between the t0 frame and the
           t0-5s frame advected forward at c_p along the resolved
           direction (a window the linear model can't advect over 5 s
           can't be forecast by a dispersion-constrained estimator).

Usage: python3 scripts/d3_badwin_forensics.py
"""
import os
import sys
import json

import numpy as np
from scipy.ndimage import shift as ndshift

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
os.chdir(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, 'scripts')

from scipy.stats import spearmanr

from realdata_analysis import directional_spread
from d3_falcon_real_operator import RECORDS, resolve_direction
from d3_sa_gonogo import lowpass_field
from d3_sa_multirecord import MRecord

G = 9.81
OUT = 'results/direction3/badwin_forensics.json'


def bimodality(Zseg, dx, dy, kp):
    """Secondary-lobe energy fraction in the 0.5-2 kp annulus (folded)."""
    P = None
    for f in Zseg[::10]:
        s = np.nan_to_num(f - np.nanmean(f))
        F = np.fft.fftshift(np.abs(np.fft.fft2(s)) ** 2)
        P = F if P is None else P + F
    ny, nx = Zseg[0].shape
    kx = np.fft.fftshift(np.fft.fftfreq(nx, d=dx)) * 2 * np.pi
    ky = np.fft.fftshift(np.fft.fftfreq(ny, d=dy)) * 2 * np.pi
    KX, KY = np.meshgrid(kx, ky)
    K = np.hypot(KX, KY)
    ann = (K > 0.5 * kp) & (K < 2.0 * kp)
    th = np.arctan2(KY[ann], KX[ann])
    w = P[ann]
    # primary folded axis
    C = np.sum(w * np.cos(2 * th)) / w.sum()
    S = np.sum(w * np.sin(2 * th)) / w.sum()
    th0 = 0.5 * np.arctan2(S, C)
    dth = np.abs((th - th0 + np.pi / 2) % np.pi - np.pi / 2)
    return float(w[dth > np.radians(35)].sum() / w.sum())


def main():
    cfg = RECORDS['YS17']
    rec = MRecord(cfg['path'], stride=cfg['stride'])
    depth, U = cfg['depth'], cfg['U']
    W_max = 40.0
    nfw = int(round(W_max / rec.dt))
    n_lead = int(round(cfg['lead_max'] / rec.dt))
    cx = 0.5 * (rec.gx.min() + rec.gx.max())
    cy = 0.5 * (rec.gy.min() + rec.gy.max())
    Xg, Yg = np.meshgrid(rec.gx, rec.gy)
    i0s = np.linspace(nfw + 1, rec.T_all - n_lead - 2, 12).astype(int)

    gate = json.load(open('results/direction3/val_gate_probe.json'))
    realized = {c['wi']: c['realized'] for c in gate['cells']
                if c['record'] == 'YS17'}

    rows = []
    for wi, i0 in enumerate(i0s):
        i0 = int(i0)
        Zmem = rec.window(i0 - 200 + 1, 200)          # last 20 s
        t0 = float(rec.t[i0])
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

        # fan coverage over the last 20 s
        r0, r1 = cfg['r_fan']
        fr = []
        for j in range(0, 200, 20):
            sp = ship_pos(t0 - 20.0 + j * rec.dt)
            RX, RY = Xg - sp[0], Yg - sp[1]
            R = np.hypot(RX, RY)
            ca = (RX * h[0] + RY * h[1]) / np.maximum(R, 1e-9)
            fan = (R >= r0) & (R <= r1) & (ca >= 0.5)
            ok = fan & np.isfinite(Zmem[j])
            fr.append(ok.sum() / max(fan.sum(), 1))
        fanfrac = float(np.mean(fr))

        # 5-s advectability
        n5 = int(round(5.0 / rec.dt))
        k_cut = 1.5 * kp
        f0 = lowpass_field(np.nan_to_num(Zmem[-1 - n5]), rec.dx, rec.dy,
                           k_cut)
        f1 = lowpass_field(np.nan_to_num(Zmem[-1]), rec.dx, rec.dy, k_cut)
        th = np.radians(pdir)
        dxy = cp * n5 * rec.dt * np.array([np.cos(th), np.sin(th)])
        fs = ndshift(f0, (dxy[1] / rec.dy, dxy[0] / rec.dx), order=1,
                     mode='constant', cval=0.0)
        adv5 = float(np.corrcoef(fs.ravel(), f1.ravel())[0, 1])

        bim = bimodality(Zmem, rec.dx, rec.dy, kp)
        rows.append(dict(wi=wi, realized=realized.get(wi, np.nan),
                         bimod=bim, fanfrac=fanfrac, adv5=adv5,
                         Tp=float(2 * np.pi / om_p), spread=float(spread)))
        print(f"[win{wi}] realized={realized.get(wi, float('nan')):+.2f} "
              f"bimod={bim:.2f} fanfrac={fanfrac:.2f} adv5={adv5:+.2f}",
              flush=True)

    re = np.array([r['realized'] for r in rows])
    v = np.isfinite(re)
    print("\nSpearman vs realized skill (YS17, 12 windows):")
    stats = {}
    for k in ('bimod', 'fanfrac', 'adv5', 'spread'):
        x = np.array([r[k] for r in rows])
        rho, p = spearmanr(x[v], re[v])
        stats[k] = dict(spearman=float(rho), p=float(p))
        print(f"  {k:8s}: {rho:+.2f} (p={p:.3f})")
    json.dump(dict(rows=rows, stats=stats), open(OUT, 'w'), indent=1)
    print(f"saved {OUT}")


if __name__ == '__main__':
    main()
