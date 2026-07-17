"""e45 ATTACK probe 1 — does E-43's "buoy 89% / ww3 58%" survive
(a) CORRELATED (structured) external errors and (b) the axes E-43 froze at
arena-truth (Tp1, spread)?

E-43's bundles perturb {d1,d2,Tp2,r2} with INDEPENDENT draws per axis, and
never perturb Tp1 / spread / gamma (which a real buoy/WW3 must also supply,
and which the arena hardcodes to their true values 7.0 / 30 deg / 3.3).
Designer N1 already showed 59% of the value destruction lives on the
(ring2, w) JOINT axis. Real forecast errors are correlated across axes.

ARMS (each at buoy grade sd=5,sT=5,sw=0.05 and ww3 grade sd=15,sT=10,sw=0.10):
  iid      exact replication of e43's bundle (SAME rng: 91000+7i+{11,23})
           -> must reproduce +0.0639 / +0.0415 bit-for-bit or the harness is wrong
  cmode_d  d1,d2 share ONE error draw (rotated forecast / refraction bias)
  relgeo_d d1 += e, d2 -= e (partition separation error)
  jTw_p    Tp2 and r2 errors FULLY correlated, same sign (one g drives both)
  jTw_m    Tp2 and r2 errors fully correlated, opposite sign
  allcorr  single g drives d1=d2 shift, Tp2, r2 (fully structured error)
  pTp1     iid PLUS Tp1 perturbed at the same relative grade as Tp2
  pTp1spr  pTp1 PLUS spread perturbed (buoy +-5 deg, ww3 +-10 deg abs on 30)

Same seas as e43 (arena X, si=100..163, n=64), R=5 draws/cell, per-sea mean
before the paired test. cur = incumbent net (identical recipe).
Out: results-free; prints + tmp/e45_attack_structured.json
"""
import os
import sys
import json
import time

import numpy as np
from scipy import stats

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, '.')
sys.path.insert(0, 'scripts')

from e29_gate0_ladder import solve_with_grid
from e37_bimodal_ladder import build_sea_crosstp, net_gamma_map
from e41_regime_prize_ladder import lobe_grid

OUT = 'tmp/e45_attack_structured.json'
GRADES = {'buoy': (5.0, 5.0, 0.05), 'ww3': (15.0, 10.0, 0.10)}
E43_SEED = {'buoy': 11, 'ww3': 23}   # e43's BUNDLE_SEED, for bit-exact iid
ARM_SEED = {'cmode_d': 101, 'relgeo_d': 103, 'jTw_p': 107, 'jTw_m': 109,
            'allcorr': 113, 'pTp1': 127, 'pTp1spr': 131}
SPR_GRADE = {'buoy': 5.0, 'ww3': 10.0}   # deg absolute on true 30


def supplied_grid2(S, d1, d2, Tp2, r2, Tp1=7.0, spr1=30.0, spr2=30.0):
    kp = S['kp']
    L1 = lobe_grid(kp, [(float(Tp1), 3.3, 1.0)], [(d1, spr1, 1.0)])
    L2 = lobe_grid(kp, [(float(Tp2), 3.3, 1.0)], [(d2, spr2, 1.0)])
    return (1 - r2) * L1 + r2 * L2


def draw_theta(S, rng, arm, sd, sT, sw, spr):
    """One perturbed theta under the given error structure."""
    d1, d2, Tp2, r2 = S['d1'], S['d2'], S['Tp2'], S['r2']
    Tp1, s1, s2 = 7.0, 30.0, 30.0
    if arm == 'iid':
        d1 = d1 + np.deg2rad(rng.normal(0, sd))
        d2 = d2 + np.deg2rad(rng.normal(0, sd))
        Tp2 = Tp2 * (1 + rng.normal(0, sT / 100.0))
        r2 = r2 + rng.normal(0, sw)
    elif arm == 'cmode_d':
        e = np.deg2rad(rng.normal(0, sd))
        d1, d2 = d1 + e, d2 + e
        Tp2 = Tp2 * (1 + rng.normal(0, sT / 100.0))
        r2 = r2 + rng.normal(0, sw)
    elif arm == 'relgeo_d':
        e = np.deg2rad(rng.normal(0, sd))
        d1, d2 = d1 + e, d2 - e
        Tp2 = Tp2 * (1 + rng.normal(0, sT / 100.0))
        r2 = r2 + rng.normal(0, sw)
    elif arm in ('jTw_p', 'jTw_m'):
        d1 = d1 + np.deg2rad(rng.normal(0, sd))
        d2 = d2 + np.deg2rad(rng.normal(0, sd))
        g = rng.normal()
        Tp2 = Tp2 * (1 + (sT / 100.0) * g)
        r2 = r2 + sw * (g if arm == 'jTw_p' else -g)
    elif arm == 'allcorr':
        g = rng.normal()
        e = np.deg2rad(sd * g)
        d1, d2 = d1 + e, d2 + e
        Tp2 = Tp2 * (1 + (sT / 100.0) * g)
        r2 = r2 + sw * g
    elif arm == 'pTp1':
        d1 = d1 + np.deg2rad(rng.normal(0, sd))
        d2 = d2 + np.deg2rad(rng.normal(0, sd))
        Tp1 = 7.0 * (1 + rng.normal(0, sT / 100.0))
        Tp2 = Tp2 * (1 + rng.normal(0, sT / 100.0))
        r2 = r2 + rng.normal(0, sw)
    elif arm == 'pTp1spr':
        d1 = d1 + np.deg2rad(rng.normal(0, sd))
        d2 = d2 + np.deg2rad(rng.normal(0, sd))
        Tp1 = 7.0 * (1 + rng.normal(0, sT / 100.0))
        Tp2 = Tp2 * (1 + rng.normal(0, sT / 100.0))
        r2 = r2 + rng.normal(0, sw)
        s1 = 30.0 + rng.normal(0, spr)
        s2 = 30.0 + rng.normal(0, spr)
    else:
        raise ValueError(arm)
    Tp1 = float(np.clip(Tp1, 3.0, 16.0))
    Tp2 = float(np.clip(Tp2, 3.0, 16.0))
    r2 = float(np.clip(r2, 0.05, 0.6))
    s1 = float(np.clip(s1, 8.0, 80.0))
    s2 = float(np.clip(s2, 8.0, 80.0))
    return d1, d2, Tp2, r2, Tp1, s1, s2


def paired(a, b):
    d = np.asarray(a) - np.asarray(b)
    d = d[np.isfinite(d)]
    n = len(d)
    se = d.std(ddof=1) / np.sqrt(n)
    lo, hi = stats.t.interval(0.95, n - 1, loc=d.mean(), scale=se)
    return dict(mean=float(d.mean()), se=float(se), lo=float(lo),
                hi=float(hi), win=int((d > 0).sum()), n=n)


def main():
    t0 = time.time()
    R = 5
    seas, cur = [], []
    print('--- building 64 seas + incumbent (arena X, si=100..163) ---',
          flush=True)
    for si in range(100, 164):
        S = build_sea_crosstp(si)
        seas.append(S)
        cur.append(solve_with_grid(S, np.maximum(10.0 ** net_gamma_map(S),
                                                 1e-8)))
    cur = np.array(cur)
    print(f'  cur mean {np.nanmean(cur):+.4f} [{time.time()-t0:.0f}s]',
          flush=True)

    out = {}
    arms = ['iid', 'cmode_d', 'relgeo_d', 'jTw_p', 'jTw_m', 'allcorr',
            'pTp1', 'pTp1spr']
    for grade, (sd, sT, sw) in GRADES.items():
        for arm in arms:
            per_sea = []
            for i, S in enumerate(seas):
                if arm == 'iid':   # e43-exact rng stream
                    rng = np.random.default_rng(91000 + 7 * i
                                                + E43_SEED[grade])
                else:
                    rng = np.random.default_rng(92000 + 7 * i
                                                + ARM_SEED[arm]
                                                + (0 if grade == 'buoy'
                                                   else 500))
                vals = []
                for r in range(R):
                    if arm == 'iid':
                        # replicate e43.perturb draw order EXACTLY
                        d1 = S['d1'] + np.deg2rad(rng.normal(0, sd))
                        d2 = S['d2'] + np.deg2rad(rng.normal(0, sd))
                        Tp2 = S['Tp2'] * (1 + rng.normal(0, sT / 100.0))
                        Tp2 = float(np.clip(Tp2, 3.0, 16.0))
                        r2 = float(np.clip(S['r2'] + rng.normal(0, sw),
                                           0.05, 0.6))
                        g = supplied_grid2(S, d1, d2, Tp2, r2)
                    else:
                        d1, d2, Tp2, r2, Tp1, s1, s2 = draw_theta(
                            S, rng, arm, sd, sT, sw, SPR_GRADE[grade])
                        g = supplied_grid2(S, d1, d2, Tp2, r2, Tp1, s1, s2)
                    vals.append(solve_with_grid(S, g))
                per_sea.append(np.mean(vals))
            st = paired(per_sea, cur)
            out[f'{grade}_{arm}'] = st
            print(f"  {grade:4s} {arm:8s}: gain {st['mean']:+.4f} "
                  f"+- {st['se']:.4f} CI [{st['lo']:+.4f},{st['hi']:+.4f}] "
                  f"win {st['win']}/{st['n']} [{time.time()-t0:.0f}s]",
                  flush=True)

    print('\n=== reference: e43 iid bundles were buoy +0.0639 / ww3 +0.0415, '
          'dose-0 oracle +0.0715 ===')
    json.dump(out, open(OUT, 'w'), indent=1, default=float)
    print(f'saved -> {OUT} ({time.time()-t0:.0f}s)')


if __name__ == '__main__':
    main()
