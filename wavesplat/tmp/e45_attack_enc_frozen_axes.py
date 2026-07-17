"""e45 ATTACK probe 3 — does the frozen-axes collapse (Tp1/spread supplied
with error, not arena-perfect) transfer to the ENCOUNTER functional, i.e.
to the deliverable's actual numbers (enc10 buoy-cur +0.0371, ww3-cur +0.0284)?

Probe 1 (field level, e43 seas) showed: ww3 +0.0415 -> +0.0180 (Tp1 err)
-> +0.0058 (Tp1+spread err), CI includes 0. Here we measure the same arms on
the encounter task, on e44's own fresh seas (si=1000..1319, n=320), reusing
e44's stored enc10_cur / enc10_buoy / enc10_ww3 per sea (bit-identical seas).

NEW ARMS (R=2 draws, per-sea mean):
  buoy_f  d+-5,  Tp2+-5%,  r2+-0.05, PLUS Tp1+-5%,  spread+-5 deg
  ww3_f   d+-15, Tp2+-10%, r2+-0.10, PLUS Tp1+-10%, spread+-10 deg
Out: tmp/e45_attack_enc_frozen_axes.json
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

from e29_gennet_C import corr
from e37_bimodal_ladder import build_sea_crosstp, net_gamma_map
from e41_regime_prize_ladder import lobe_grid
from e44_encounter_transfer import field_of
from e44_probe_N2_encounter_transfer import (track_future, render_at, N_ENC,
                                             T_END, HORIZON)

OUT = 'tmp/e45_attack_enc_frozen_axes.json'
N = 320
R = 2
GRADES = {'buoy_f': (5.0, 5.0, 0.05, 5.0), 'ww3_f': (15.0, 10.0, 0.10, 10.0)}


def grid_full(S, d1, d2, Tp2, r2, Tp1, s1, s2):
    L1 = lobe_grid(S['kp'], [(float(Tp1), 3.3, 1.0)], [(d1, s1, 1.0)])
    L2 = lobe_grid(S['kp'], [(float(Tp2), 3.3, 1.0)], [(d2, s2, 1.0)])
    return (1 - r2) * L1 + r2 * L2


def paired(a, b):
    d = np.asarray(a) - np.asarray(b)
    d = d[np.isfinite(d)]
    n = len(d)
    se = d.std(ddof=1) / np.sqrt(n)
    lo, hi = stats.t.interval(0.95, n - 1, loc=d.mean(), scale=se)
    try:
        wp = float(stats.wilcoxon(d)[1])
    except Exception:
        wp = float('nan')
    return dict(mean=float(d.mean()), se=float(se), lo=float(lo),
                hi=float(hi), win=int((d > 0).sum()), n=n, wilcox=wp,
                median=float(np.median(d)))


def main():
    t0 = time.time()
    ref = {r['si']: r for r in json.load(
        open('results/direction3/e44_encounter_transfer.json'))['rows']}

    rows = []
    for si in range(1000, 1000 + N):
        S = build_sea_crosstp(si)
        te = T_END + np.linspace(HORIZON / N_ENC, HORIZON, N_ENC)
        Pq_e = track_future(S, te)
        gt_e = render_at(S, Pq_e)
        rng = np.random.default_rng(96000 + si)
        row = dict(si=si,
                   enc10_cur=ref[si]['enc10_cur'],
                   enc10_buoy=ref[si]['enc10_buoy'],
                   enc10_ww3=ref[si]['enc10_ww3'],
                   enc10_orc=ref[si]['enc10_orc'])
        for nm, (sd, sT, sw, sspr) in GRADES.items():
            vals = []
            for r in range(R):
                d1 = S['d1'] + np.deg2rad(rng.normal(0, sd))
                d2 = S['d2'] + np.deg2rad(rng.normal(0, sd))
                Tp1 = float(np.clip(7.0 * (1 + rng.normal(0, sT / 100)),
                                    3.0, 16.0))
                Tp2 = float(np.clip(S['Tp2'] * (1 + rng.normal(0, sT / 100)),
                                    3.0, 16.0))
                r2 = float(np.clip(S['r2'] + rng.normal(0, sw), 0.05, 0.6))
                s1 = float(np.clip(30.0 + rng.normal(0, sspr), 8.0, 80.0))
                s2 = float(np.clip(30.0 + rng.normal(0, sspr), 8.0, 80.0))
                g = grid_full(S, d1, d2, Tp2, r2, Tp1, s1, s2)
                fe = field_of(S, g, Pq_e)
                vals.append(np.nan if fe is None
                            else float(corr(fe, gt_e)))
            row[f'enc10_{nm}'] = float(np.nanmean(vals)) \
                if np.isfinite(vals).any() else np.nan
        rows.append(row)
        if (si - 1000) % 80 == 0:
            print(f"  si{si} cur {row['enc10_cur']:+.3f} "
                  f"ww3_f d {row['enc10_ww3_f']-row['enc10_cur']:+.3f} "
                  f"[{time.time()-t0:.0f}s]", flush=True)

    print(f'\n=== ENCOUNTER, n={len(rows)} seas (subset of e44 block) ===')
    res = {}
    for arm in ['orc', 'buoy', 'ww3', 'buoy_f', 'ww3_f']:
        st = paired([r[f'enc10_{arm}'] for r in rows],
                    [r['enc10_cur'] for r in rows])
        res[f'{arm}-cur'] = st
        star = '*' if st['lo'] > 0 else ' '
        print(f"  {arm:7s} - cur: {st['mean']:+.4f} +- {st['se']:.4f} "
              f"CI [{st['lo']:+.4f},{st['hi']:+.4f}]{star} "
              f"win {st['win']}/{st['n']} med {st['median']:+.4f} "
              f"wilcox {st['wilcox']:.2g}", flush=True)
    # direct paired drop: iid bundle minus frozen-axes bundle
    for a, b in [('buoy', 'buoy_f'), ('ww3', 'ww3_f')]:
        st = paired([r[f'enc10_{a}'] for r in rows],
                    [r[f'enc10_{b}'] for r in rows])
        res[f'{a}-{b}'] = st
        print(f"  {a:7s} - {b:6s}: {st['mean']:+.4f} "
              f"CI [{st['lo']:+.4f},{st['hi']:+.4f}]")
    json.dump(dict(rows=rows, contrasts=res), open(OUT, 'w'),
              indent=1, default=float)
    print(f'saved -> {OUT} ({time.time()-t0:.0f}s)')


if __name__ == '__main__':
    main()
