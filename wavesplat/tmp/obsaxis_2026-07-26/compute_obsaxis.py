"""W9 — observable-coordinate phase diagram: the two Spearman correlations (crux A / F).

Frozen protocol: tmp/obsaxis_2026-07-26/PRESTATE.md (written before this ran).
Zero new estimator compute; all inputs are on disk. Writes obsaxis_stats.json.
"""
import json
import os
import re
import sys

import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..',
                                'figs_opus_diag_2026-07-26'))
import common as C  # noqa: E402  (palette + byte-faithful assembly)

OUT = os.path.join(C.ROOT, 'tmp/obsaxis_2026-07-26')
BF_REAL = os.path.join(C.ROOT, 'tmp/figs_opus_real_2026-07-26/bf_ratio_real.json')
HAND_AMP = os.path.join(C.ROOT, 'tmp/unified_2026-07-26/hand_amp.json')
WP_REAL = os.path.join(C.ROOT, 'tmp/wp_real_2026-07-25/wp_real.json')
TOL = 0.05  # pre-stated tolerance band


def synthetic():
    """Per-episode delta (learned - v2) under both aggregation orders + the two axes."""
    _, v2, _, learned, _, bfr = C.load_raw()
    eps = sorted({e for (e, _s) in v2})
    rows = []
    for ep in eps:
        v2l = [v2.get((ep, E)) for (E, D) in C.LANES]
        lnl = [learned.get((ep, E, D)) for (E, D) in C.LANES]
        pv, pl = C.med(v2l), C.med(lnl)
        lane_d = [l - v for l, v in zip(lnl, v2l) if l is not None and v is not None]
        c = C.comp(ep)
        rows.append(dict(episode=ep, short=C.short(ep), comp=c, regime=C.regime(c),
                         pure_v2=pv, pure_learned=pl,
                         delta=pl - pv,                       # PRIMARY (median-then-diff)
                         delta_lane=float(np.median(lane_d)),  # secondary (diff-then-median)
                         bfr=bfr[ep]))
    return rows


def real():
    """30 real windows: bf_ratio (GT-free, same helper) vs learned_now - hand_now."""
    bf = {(r['record'], r['win']): r for r in json.load(open(BF_REAL))['rows']}
    ha = {(r['record'], r['win']): r for r in json.load(open(HAND_AMP))['rows']}
    wr = {(r['record'], r['win']): r for r in json.load(open(WP_REAL))['results']}
    assert set(bf) == set(ha) == set(wr) == set(bf), 'window key mismatch'
    out = []
    for k in sorted(bf):
        assert abs(ha[k]['hand_now'] - wr[k]['hand_now']) < 1e-12
        assert abs(ha[k]['learned_now'] - wr[k]['learned_now']) < 1e-12
        out.append(dict(record=k[0], win=k[1], bfr=bf[k]['bf_ratio'],
                        hand_now=ha[k]['hand_now'], learned_now=ha[k]['learned_now'],
                        delta=ha[k]['learned_now'] - ha[k]['hand_now'],
                        collapse=bool(bf[k]['collapse'])))
    return out


def main():
    syn, rl = synthetic(), real()
    assert len(syn) == 100 and len(rl) == 30

    d = np.array([r['delta'] for r in syn])
    d_lane = np.array([r['delta_lane'] for r in syn])
    nc = np.array([r['comp'] for r in syn], float)
    bf = np.array([r['bfr'] for r in syn])

    def sp(x, y):
        r = spearmanr(x, y)
        return dict(rho=float(r.statistic), abs_rho=float(abs(r.statistic)),
                    p=float(r.pvalue), n=int(len(x)))

    stats = dict(
        primary=dict(ncomp=sp(nc, d), bf=sp(bf, d)),
        secondary_lane_delta=dict(ncomp=sp(nc, d_lane), bf=sp(bf, d_lane)),
        axis_collinearity=sp(nc, bf),
        real=dict(bf_vs_delta=sp([r['bfr'] for r in rl], [r['delta'] for r in rl]),
                  bfr_min=float(min(r['bfr'] for r in rl)),
                  bfr_max=float(max(r['bfr'] for r in rl)),
                  n_below_guard=int(sum(r['bfr'] < 0.05 for r in rl)),
                  median_delta=float(np.median([r['delta'] for r in rl])),
                  n_learned_wins=int(sum(r['delta'] < 0 for r in rl))),
    )
    a, b = stats['primary']['ncomp']['abs_rho'], stats['primary']['bf']['abs_rho']
    verdict = 'FOR_STUDENT' if b >= a - TOL else 'FOR_P3'
    stats['decision'] = dict(
        tol=TOL, abs_rho_ncomp=a, abs_rho_bf=b, diff=float(b - a), verdict=verdict,
        text=('the boundary is not a relabeling of the answer key: a GT-free observable '
              'orders the regime as well as the simulator knob'
              if verdict == 'FOR_STUDENT' else
              'n_components dominates; the phase diagram is drawn on a coordinate the '
              'runtime cannot read - P3 is right and Ch.5 says so'))
    # synthetic coverage of the real band (honesty clause)
    lo, hi = stats['real']['bfr_min'], stats['real']['bfr_max']
    stats['coverage'] = dict(
        real_band=[lo, hi],
        n_syn_in_real_band=int(sum(lo <= r['bfr'] <= hi for r in syn)),
        n_syn_total=100,
        syn_by_regime_in_band={g: int(sum(lo <= r['bfr'] <= hi
                                          for r in syn if r['regime'] == g))
                               for g in C.REGIME_ORDER},
        n_syn_below_guard=int(sum(r['bfr'] < C.GUARD_RATIO for r in syn)))

    json.dump(dict(stats=stats, synthetic=syn, real=rl),
              open(os.path.join(OUT, 'obsaxis_stats.json'), 'w'), indent=1)

    p = stats['primary']
    print('=' * 78)
    print('W9 OBSERVABLE-COORDINATE AXIS — Spearman on per-episode delta (learned - v2)')
    print('=' * 78)
    print(f"  (a) n_components  : rho={p['ncomp']['rho']:+.4f}  |rho|={a:.4f}  p={p['ncomp']['p']:.3g}")
    print(f"  (b) bf_peak_ratio : rho={p['bf']['rho']:+.4f}  |rho|={b:.4f}  p={p['bf']['p']:.3g}")
    print(f"  diff |rho_bf| - |rho_ncomp| = {b - a:+.4f}   (tolerance {TOL})")
    print(f"  PRE-STATED VERDICT = {verdict}")
    s = stats['secondary_lane_delta']
    print(f"  [secondary, per-lane delta] ncomp |rho|={s['ncomp']['abs_rho']:.4f}  "
          f"bf |rho|={s['bf']['abs_rho']:.4f}")
    print(f"  [axis collinearity] ncomp vs bf_ratio rho={stats['axis_collinearity']['rho']:+.4f}")
    r = stats['real']
    print(f"  real: bf_ratio in [{r['bfr_min']:.3f}, {r['bfr_max']:.3f}], "
          f"{r['n_below_guard']}/30 below guard 0.05, median delta={r['median_delta']:+.4f}, "
          f"learned wins {r['n_learned_wins']}/30")
    cv = stats['coverage']
    print(f"  coverage: {cv['n_syn_in_real_band']}/100 synthetic episodes inside the real band "
          f"{cv['syn_by_regime_in_band']}")


if __name__ == '__main__':
    main()
