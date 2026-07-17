"""e43d — complete the CONTROL SET for the e43c reversal (cheap, no re-search).

WHY. e43c scored a full-enumeration blind point-commit at FULL - cur_grid =
+0.0671 +- 0.0172 (CI [+0.0325,+0.1016], win 32/48), passing e37's pre-registered
"direction REOPENS" bar. But e37's bar had TWO more requirements that e43c never
measured, and one of them is the one that matters:

  GATE   cur_grid ~= m2 (the honest all-marginal) -- e41 measured m2 - cur =
         -0.005, e42 -0.026, e43b -0.023. So beating cur_grid WITHOUT beating m2
         means recovering NO information: you'd just be matching a smear. This is
         the primary defense against the E-37 failure (mistaking one's own
         detector's behaviour for information).
  cur_dep e37's bar demanded the win hold against BOTH cur_grid AND cur_dep.

Neither needs the 952,560-hypothesis enumeration re-run: the FULL score per sea
is already stored. m2 / cur_dep / ens1 are plain solver calls (~0.1 s each). So
this completes the control set on the SAME seas for ~2 min instead of ~30.

HONEST SCOPE. This adds CONTROLS to an existing arm on an existing seed block; it
does not add a new arm or a new test, so it is not a second bite at the apple.
But si=300..347 is now a SEEN block (e43c ran there, and its first 4 seas were
smoke-inspected first). A clean fresh-block replication of the whole battery is
still owed -- that is e43e, launched separately.

Out: results/direction3/e43d_gate.json
"""
import os
import sys
import json

import numpy as np
from scipy import stats

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, '.')
sys.path.insert(0, 'scripts')

from e29_gennet_C import corr, incumbent_field
from e29_gate0_ladder import solve_with_grid
from e37_bimodal_ladder import build_sea_crosstp, mu_hat_of
from e41_regime_prize_ladder import lobe_grid, TP2_RATIO_NODES, D2_OFF_NODES, \
    W_MARG

OUT = 'results/direction3/e43d_gate.json'
SRC = 'results/direction3/e43c_true_node_rank.json'


def x_marginals(S):
    """VERBATIM e42_blind_recovery.x_marginals (same convention as e41 score_x)."""
    kp = S['kp']
    mu_hat = mu_hat_of(S)
    L1 = lambda mu: lobe_grid(kp, [(7.0, 3.3, 1.0)], [(mu, 30.0, 1.0)])

    def L2(center=None, mu2=None, ring_true=False):
        rad = [(S['Tp2'], 3.3, 1.0)] if ring_true else \
            [(7.0 * r, 3.3, 1.0) for r in TP2_RATIO_NODES]
        ang = [(mu2, 30.0, 1.0)] if mu2 is not None else \
            [(center + s * np.deg2rad(o), 30.0, 1.0)
             for s in (+1, -1) for o in D2_OFF_NODES]
        return lobe_grid(kp, rad, ang)

    wm, r2 = W_MARG, S['r2']
    return dict(ens1=L1(mu_hat),
                m2=(1 - wm) * L1(mu_hat) + wm * L2(mu_hat),
                oracle_x=(1 - r2) * L1(S['d1'])
                + r2 * L2(mu2=S['d2'], ring_true=True))


def paired(rows, a, b):
    d = np.array([r[a] - r[b] for r in rows
                  if np.isfinite(r[a]) and np.isfinite(r[b])])
    n = len(d)
    se = d.std(ddof=1) / np.sqrt(n)
    t, p = stats.ttest_1samp(d, 0)
    lo, hi = stats.t.interval(0.95, n - 1, loc=d.mean(), scale=se)
    return dict(mean=float(d.mean()), se=float(se), t=float(t), p=float(p),
                lo=float(lo), hi=float(hi), win=int((d > 0).sum()), n=n)


def main():
    src = json.load(open(SRC))
    R = [r for r in src['rows'] if not r.get('excluded')]
    print(f'loaded {len(R)} scored seas from e43c', flush=True)
    rows = []
    for r in R:
        si = r['si']
        S = build_sea_crosstp(si)
        M = x_marginals(S)
        row = dict(si=si, FULL=r['sc_full'], cur_grid=r['sc_cur'],
                   node_oracle=r['sc_node_oracle'])
        for k, g in M.items():
            row[k] = solve_with_grid(S, g)
        row['cur_dep'] = float(corr(
            incumbent_field(S['P'], S['z'], S['Pq'], S['kp'], S['sigma0']),
            S['gt_u']))
        rows.append(row)
        print(f"{si} FULL {row['FULL']:+.3f} cur {row['cur_grid']:+.3f} "
              f"dep {row['cur_dep']:+.3f} m2 {row['m2']:+.3f} "
              f"orcX {row['oracle_x']:+.3f}", flush=True)

    print(f'\n=== arm means (n={len(rows)}) ===')
    for k in ['cur_grid', 'cur_dep', 'm2', 'ens1', 'FULL', 'oracle_x',
              'node_oracle']:
        print(f'  {k:11s}: {np.mean([r[k] for r in rows]):+.4f}')

    print('\n=== e37 REOPEN bar: FULL must beat BOTH curs, AND beat m2 ===')
    res = {}
    for aa, bb in [('FULL', 'cur_grid'), ('FULL', 'cur_dep'), ('FULL', 'm2'),
                   ('FULL', 'ens1'), ('m2', 'cur_grid'),
                   ('oracle_x', 'cur_grid'), ('node_oracle', 'cur_grid'),
                   ('node_oracle', 'oracle_x')]:
        r = paired(rows, aa, bb)
        res[f'{aa}-{bb}'] = r
        star = '*' if (r['lo'] > 0 or r['hi'] < 0) else ' '
        print(f"  {aa:11s} - {bb:9s}: {r['mean']:+.4f} +- {r['se']:.4f} "
              f"t {r['t']:+.2f} CI [{r['lo']:+.4f},{r['hi']:+.4f}]{star} "
              f"win {r['win']}/{r['n']}")

    A, B, G = res['FULL-cur_grid'], res['FULL-cur_dep'], res['FULL-m2']
    def bar(x):
        return x['mean'] >= 0.03 and x['lo'] > 0 and x['win'] / x['n'] >= 0.60
    reopen = bar(A) and bar(B) and G['lo'] > 0
    print('\n=== VERDICT ===')
    print(f"  vs cur_grid (>=+0.03, CI excl 0, win>=60%): {bar(A)}")
    print(f"  vs cur_dep  (same bar)                    : {bar(B)}")
    print(f"  GATE vs m2  (CI excludes 0)               : {G['lo'] > 0}")
    print(f"  -> {'REOPEN CONFIRMED on this block' if reopen else 'NOT confirmed'}")
    print('  (seed block si=300..347 is SEEN -- fresh-block replication owed)')
    json.dump(dict(rows=rows, contrasts=res, reopen=bool(reopen)),
              open(OUT, 'w'), indent=1, default=float)
    print(f'saved -> {OUT}')


if __name__ == '__main__':
    main()
