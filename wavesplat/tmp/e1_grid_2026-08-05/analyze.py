"""E1 analysis. Aggregation and tests are fixed by tmp/e1_grid_2026-08-05/PREREG.md sec 3-4."""
import json
import os
import sys
from math import comb
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from run_e1 import EPISODES, CELLS  # noqa: E402

# PREREG sec 3.3 -- pre-registered MDE(size), plug-in sigma_d from S1-A0 crowded arm.
MDE = {'init_nrmse': 0.0262, 'nRMSE': 0.0188, 'detection_rate': 0.0829,
       'energy_detection_rate': 0.1008, 'k_mag_rel_error_mean': 0.3132,
       'direction_error_deg_mean': 8.22, 'Skill_Score': 0.1800,
       'Pearson_rho': 0.1047, 'Directional_R2': 0.1434, 'Hs_error': 0.0235,
       'absVR1': 0.0482}
# +1 == treatment is BETTER when the paired difference is negative
LOWER_IS_BETTER = {'init_nrmse': 1, 'nRMSE': 1, 'k_mag_rel_error_mean': 1,
                   'direction_error_deg_mean': 1, 'Hs_error': 1, 'absVR1': 1,
                   'detection_rate': 0, 'energy_detection_rate': 0,
                   'Skill_Score': 0, 'Pearson_rho': 0, 'Directional_R2': 0}
SPEC = {'detection_rate', 'energy_detection_rate', 'k_mag_rel_error_mean',
        'direction_error_deg_mean'}

STRATA = {
    'sparse(1comp)': ['BF2_1comp', 'BF3_1comp', 'BF4_1comp'],
    'mid(2,5comp)': ['BF3_2comp_seed0_uniform', 'BF3_2comp_seed0_slope',
                     'BF3_5comp_seed0_uniform', 'BF3_5comp_seed0_slope', 'BF4_5comp'],
    'crowded(15,30)': ['BF4_15comp', 'BF4_30comp', 'BF5_15comp', 'BF5_30comp'],
}
# PREREG sec 6: episodes with NO true pair closer than MIN_DK=0.15 -> must not respond
CONTROL_EPS = ['BF2_1comp', 'BF3_1comp', 'BF4_1comp',
               'BF3_2comp_seed0_uniform', 'BF3_2comp_seed0_slope']
LENS_EPS = [e for e in EPISODES if not any(t in e for t in CONTROL_EPS)]


def sign_p(diffs, better):
    """Two-sided exact sign test; zeros dropped."""
    d = [x for x in diffs if np.isfinite(x) and x != 0]
    n = len(d)
    if n == 0:
        return float('nan'), 0, 0
    k = sum(1 for x in d if (x < 0) == bool(better))
    kk = max(k, n - k)
    p = min(1.0, 2 * sum(comb(n, i) for i in range(kk, n + 1)) / 2 ** n)
    return p, k, n


def load(path):
    per = defaultdict(dict)          # (cell, ep) -> seed -> record
    for line in open(path):
        r = json.loads(line)
        if 'error' in r:
            continue
        per[(r['cell'], r['episode'])][r['seed']] = r
    return per


def getval(rec, m):
    if m == 'absVR1':
        return abs(rec['Variance_Ratio'] - 1.0)
    if m in SPEC:
        return rec['spec'].get(m, float('nan'))
    return rec[m]


def medians(per, metrics):
    """(cell, ep) -> metric -> median over seeds; plus seed range."""
    med, rng = defaultdict(dict), defaultdict(dict)
    for key, byseed in per.items():
        for m in metrics:
            xs = [getval(byseed[s], m) for s in sorted(byseed)]
            xs = [x for x in xs if np.isfinite(x)]
            if xs:
                med[key][m] = float(np.median(xs))
                rng[key][m] = float(max(xs) - min(xs))
    return med, rng


def contrast(med, treat, ctrl, metric, eps):
    out = []
    for ep in eps:
        a, b = med.get((ctrl, ep), {}), med.get((treat, ep), {})
        if metric in a and metric in b:
            out.append((ep, b[metric] - a[metric]))
    return out


def verdict(diffs, metric):
    better = LOWER_IS_BETTER[metric]
    d = [x for _, x in diffs]
    med = float(np.median(d))
    p, k, n = sign_p(d, better)
    mde = MDE[metric]
    improved = (med < 0) if better else (med > 0)
    big = abs(med) > mde
    # two-sided exact sign test already covers BOTH directions (kk = max(k, n-k)),
    # so p <= 0.05 is the direction-agnostic form of the pre-registered ">=10/12" rule.
    consistent = (p <= 0.05)
    if improved and big and consistent:
        v = 'PASS'
    elif improved and (big or consistent):
        v = 'WEAK'
    elif (not improved) and big and consistent:
        v = 'HARM'
    elif (not improved) and (big or consistent):
        # symmetric counterpart of WEAK; the PREREG table only spelled out the
        # improvement side, so this is reported, not used to close an axis.
        v = 'WEAK-HARM'
    else:
        v = f'미탐지(MDE={mde:g})'
    return dict(median=med, p=p, k=k, n=n, mde=mde, verdict=v)


def main():
    metrics = list(MDE)
    per = load(os.path.join(HERE, 'results.jsonl'))
    med, rng = medians(per, metrics)
    cells = sorted({c for c, _ in med})
    print('cells present:', cells, '| episodes:', len({e for _, e in med}))

    print('\n=== per-cell pooled medians over 12 episodes (median-of-3 seeds) ===')
    hdr = ['init_nrmse', 'nRMSE', 'detection_rate', 'energy_detection_rate',
           'k_mag_rel_error_mean', 'direction_error_deg_mean']
    print(f"{'cell':<5s}" + ''.join(f'{h[:14]:>16s}' for h in hdr))
    for c in cells:
        row = []
        for m in hdr:
            v = [med[(c, e)][m] for e in EPISODES if m in med.get((c, e), {})]
            row.append(np.median(v) if v else float('nan'))
        print(f'{c:<5s}' + ''.join(f'{x:16.4f}' for x in row))

    print('\n=== realized seed noise (12ep, control cell A) vs pre-registered sigma_d ===')
    for m in metrics:
        rs = [rng[('A', e)][m] for e in EPISODES if m in rng.get(('A', e), {})]
        if not rs:
            continue
        rmed = float(np.median(rs))
        sd = 1.414 * 0.67 * rmed / 1.693
        print(f'  {m:<26s} range_med={rmed:.4f}  realized sigma_d={sd:.4f} '
              f'-> realized MDE={0.8881*sd:.4f}   (prereg MDE={MDE[m]:g})')

    contrasts = [('D', 'A', 'PRIMARY: narrow+recompute vs shipped'),
                 ('B', 'A', 'factor1 alone (narrow grid)'),
                 ('C', 'A', 'factor2 alone (recompute)'),
                 ('E', 'A', '2026-07-23 replication (MIN_DK 0.15->0.05, wide)')]
    for treat, ctrl, label in contrasts:
        if treat not in cells:
            continue
        print(f'\n=== {treat} vs {ctrl} — {label} ===')
        print(f"{'metric':<26s}{'medianΔ':>10s}{'k/n':>8s}{'p':>9s}{'MDE':>9s}  verdict")
        for m in metrics:
            d = contrast(med, treat, ctrl, m, EPISODES)
            if len(d) < 8:
                continue
            r = verdict(d, m)
            print(f"{m:<26s}{r['median']:10.4f}{f'{r[chr(107)]}/{r[chr(110)]}':>8s}"
                  f"{r['p']:9.4f}{r['mde']:9.4f}  {r['verdict']}")

    print('\n=== factorial main effects & interaction (init_nrmse, nRMSE, detection_rate) ===')
    for m in ['init_nrmse', 'nRMSE', 'detection_rate']:
        def g(c, e):
            return med.get((c, e), {}).get(m, float('nan'))
        f1, f2, inter = [], [], []
        for e in EPISODES:
            BA, DC = g('B', e) - g('A', e), g('D', e) - g('C', e)
            CA, DB = g('C', e) - g('A', e), g('D', e) - g('B', e)
            f1.append(0.5 * (BA + DC)); f2.append(0.5 * (CA + DB))
            inter.append(DC - BA)
        b = LOWER_IS_BETTER[m]
        for name, arr in (('GRID main', f1), ('PEAK main', f2), ('interaction', inter)):
            p, k, n = sign_p(arr, b)
            print(f'  {m:<22s} {name:<14s} median={np.median(arr):+.4f} '
                  f'k={k}/{n} p={p:.4f}')

    print('\n=== strata (DESCRIPTIVE ONLY — prereg sec 7) : Δ vs A, init_nrmse & nRMSE ===')
    for m in ['init_nrmse', 'nRMSE']:
        print(f'  -- {m} --')
        for sname, tags in STRATA.items():
            eps = [e for e in EPISODES if any(t in e for t in tags)]
            line = f'  {sname:<18s} n={len(eps)}'
            for c in ['B', 'C', 'D', 'E']:
                if c not in cells:
                    continue
                d = [x for _, x in contrast(med, c, 'A', m, eps)]
                line += f'  {c}-A={np.median(d):+.4f}'
            print(line)

    print('\n=== PREREG sec 6 negative control (5 eps with NO true pair < 0.15) ===')
    for m in ['init_nrmse', 'nRMSE']:
        for c in ['B', 'C', 'D', 'E']:
            if c not in cells:
                continue
            dc = [x for _, x in contrast(med, c, 'A', m,
                                         [e for e in EPISODES
                                          if any(t in e for t in CONTROL_EPS)])]
            dl = [x for _, x in contrast(med, c, 'A', m, LENS_EPS)]
            print(f'  {m:<12s} {c}-A: control(n={len(dc)}) median={np.median(dc):+.4f} | '
                  f'lens-applies(n={len(dl)}) median={np.median(dl):+.4f}')

    print('\n=== per-episode init_nrmse (median-of-3), all cells ===')
    print(f"{'episode':<38s}" + ''.join(f'{c:>10s}' for c in cells))
    for e in EPISODES:
        print(f'{e[8:-3]:<38s}' + ''.join(
            f"{med.get((c, e), {}).get('init_nrmse', float('nan')):10.4f}" for c in cells))
    print('\n=== per-episode avg nRMSE (median-of-3), all cells ===')
    print(f"{'episode':<38s}" + ''.join(f'{c:>10s}' for c in cells))
    for e in EPISODES:
        print(f'{e[8:-3]:<38s}' + ''.join(
            f"{med.get((c, e), {}).get('nRMSE', float('nan')):10.4f}" for c in cells))
    print('\n=== BF3_5comp — the 2026-07-23 "mid-density harm" episode (prereg sec 7) ===')
    for e in [x for x in EPISODES if 'BF3_5comp' in x]:
        for c in cells:
            k = (c, e)
            if k not in med:
                continue
            print(f"  {e[8:-3]:<28s} {c}: init={med[k]['init_nrmse']:.4f} "
                  f"(range {rng[k]['init_nrmse']:.4f})  nRMSE={med[k]['nRMSE']:.4f} "
                  f"(range {rng[k]['nRMSE']:.4f})  det={med[k]['detection_rate']:.3f}")

    print('\n=== final_n diagnostic (median-of-3) ===')
    print(f"{'episode':<38s}" + ''.join(f'{c:>8s}' for c in cells))
    for e in EPISODES:
        vals = []
        for c in cells:
            bs = per.get((c, e), {})
            vals.append(np.median([bs[s]['N'] for s in bs]) if bs else float('nan'))
        print(f'{e[8:-3]:<38s}' + ''.join(f'{v:8.0f}' for v in vals))


if __name__ == '__main__':
    main()
