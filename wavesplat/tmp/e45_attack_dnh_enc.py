"""e45 ATTACK probe 2 — do-no-harm on the ENCOUNTER task, the control e44
pre-registered (UNI n=240) but silently never ran for the supplied arms
(its uni loop scores only 'cur').

Deployment reality: the external source must also decide WHETHER the sea is
cross-Tp bimodal. E-42 showed the 2-lobe family is high-gain/high-risk at the
FIELD level. Question: on a UNIMODAL sea, what does a buoy-grade FALSE
bimodal report (phantom second partition) cost on the encounter functional?
And what does a CORRECT unimodal report (single supplied lobe) buy?

Seas: build_sea_bimodal(4000+j, unimodal=True), j=0..239 == e44's uni block,
so our enc10_cur must match e44's stored uni values bit-for-bit (harness
validation).

ARMS
  cur        incumbent (validation vs e44 JSON)
  uni_orc    supplied single lobe, TRUE d1 (external source correct + perfect)
  uni_buoy   supplied single lobe, d1 +- 5 deg (correct regime, buoy noise)
  ph_buoy    PHANTOM bimodal at buoy grade: d1+-5deg; false partition with
             d2 = sea's plausible offset, Tp2 = 7*U(0.75,1.4), r2 ~ U(0.2,0.5)
  ph_ww3     same at ww3 grade (d+-15deg)
R = 3 phantom draws per sea, per-sea mean before pairing.
Out: tmp/e45_attack_dnh_enc.json
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
from e29_gate0_ladder import solve_with_grid
from e37_bimodal_ladder import build_sea_bimodal, net_gamma_map
from e41_regime_prize_ladder import lobe_grid
from e44_encounter_transfer import field_of
from e44_probe_N2_encounter_transfer import (track_future, render_at, N_ENC,
                                             T_END, HORIZON)

OUT = 'tmp/e45_attack_dnh_enc.json'
N_UNI = 240
R = 3


def one_lobe(S, d1):
    return lobe_grid(S['kp'], [(7.0, 3.3, 1.0)], [(d1, 30.0, 1.0)])


def two_lobe(S, d1, d2, Tp2, r2):
    L1 = lobe_grid(S['kp'], [(7.0, 3.3, 1.0)], [(d1, 30.0, 1.0)])
    L2 = lobe_grid(S['kp'], [(float(Tp2), 3.3, 1.0)], [(d2, 30.0, 1.0)])
    return (1 - r2) * L1 + r2 * L2


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
    ref = json.load(open('results/direction3/e44_encounter_transfer.json'))
    ref_uni = {r['j']: r['enc10_cur'] for r in ref['uni']}

    rows = []
    for j in range(N_UNI):
        S = build_sea_bimodal(4000 + j, unimodal=True)
        te = T_END + np.linspace(HORIZON / N_ENC, HORIZON, N_ENC)
        Pq_e = track_future(S, te)
        gt_e = render_at(S, Pq_e)
        g_cur = np.maximum(10.0 ** net_gamma_map(S), 1e-8)
        rng = np.random.default_rng(95000 + j)

        row = dict(j=j)

        def enc(g):
            fe = field_of(S, g, Pq_e)
            return np.nan if fe is None else float(corr(fe, gt_e))

        row['enc10_cur'] = enc(g_cur)
        row['fld_cur'] = solve_with_grid(S, g_cur)
        row['match_ref'] = bool(abs(row['enc10_cur']
                                    - ref_uni[j]) < 1e-9)

        row['enc10_uni_orc'] = enc(one_lobe(S, S['d1']))
        row['fld_uni_orc'] = solve_with_grid(S, one_lobe(S, S['d1']))

        # buoy-noise correct-unimodal + phantom arms, R draws each
        acc = {k: [] for k in ['enc10_uni_buoy', 'fld_uni_buoy',
                               'enc10_ph_buoy', 'fld_ph_buoy',
                               'enc10_ph_ww3', 'fld_ph_ww3']}
        for r in range(R):
            d1b = S['d1'] + np.deg2rad(rng.normal(0, 5.0))
            g = one_lobe(S, d1b)
            acc['enc10_uni_buoy'].append(enc(g))
            acc['fld_uni_buoy'].append(solve_with_grid(S, g))
            # phantom partition (same phantom for both grades, noise differs)
            Tp2 = 7.0 * rng.uniform(0.75, 1.4)
            r2 = rng.uniform(0.2, 0.5)
            for grade, sdg in [('ph_buoy', 5.0), ('ph_ww3', 15.0)]:
                d1p = S['d1'] + np.deg2rad(rng.normal(0, sdg))
                d2p = S['d2'] + np.deg2rad(rng.normal(0, sdg))
                Tp2p = float(np.clip(Tp2 * (1 + rng.normal(
                    0, (5.0 if grade == 'ph_buoy' else 10.0) / 100)),
                    3.0, 16.0))
                r2p = float(np.clip(r2 + rng.normal(
                    0, 0.05 if grade == 'ph_buoy' else 0.10), 0.05, 0.6))
                g2 = two_lobe(S, d1p, d2p, Tp2p, r2p)
                acc[f'enc10_{grade}'].append(enc(g2))
                acc[f'fld_{grade}'].append(solve_with_grid(S, g2))
        for k, v in acc.items():
            row[k] = float(np.nanmean(v))
        rows.append(row)
        if j % 40 == 0:
            print(f"  j{j} cur {row['enc10_cur']:+.3f} "
                  f"(ref match {row['match_ref']}) "
                  f"ph_buoy d {row['enc10_ph_buoy']-row['enc10_cur']:+.3f} "
                  f"[{time.time()-t0:.0f}s]", flush=True)

    n_match = sum(r['match_ref'] for r in rows)
    print(f'\nharness validation: enc10_cur matches e44 JSON on '
          f'{n_match}/{len(rows)} seas')

    res = {}
    print(f'\n=== UNIMODAL do-no-harm (n={len(rows)}) ===')
    for tag in ['enc10', 'fld']:
        for arm in ['uni_orc', 'uni_buoy', 'ph_buoy', 'ph_ww3']:
            st = paired([r[f'{tag}_{arm}'] for r in rows],
                        [r[f'{tag}_cur'] for r in rows])
            res[f'{tag}_{arm}-cur'] = st
            star = '*' if (st['lo'] > 0 or st['hi'] < 0) else ' '
            print(f"  {tag:6s} {arm:8s} - cur: {st['mean']:+.4f} "
                  f"+- {st['se']:.4f} CI [{st['lo']:+.4f},{st['hi']:+.4f}]"
                  f"{star} win {st['win']}/{st['n']}", flush=True)

    json.dump(dict(rows=rows, contrasts=res, n_match=n_match),
              open(OUT, 'w'), indent=1, default=float)
    print(f'saved -> {OUT} ({time.time()-t0:.0f}s)')


if __name__ == '__main__':
    main()
