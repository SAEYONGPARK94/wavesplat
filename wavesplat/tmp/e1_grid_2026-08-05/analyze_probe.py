"""E1 placement-probe analysis (PREREG sec 5 / 3.4). Deterministic -> sign axis only."""
import json
import os
import sys
from math import comb
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from run_e1 import EPISODES  # noqa: E402

CROWDED = [e for e in EPISODES if '15comp' in e or '30comp' in e]
CONTROL = ['BF2_1comp', 'BF3_1comp', 'BF4_1comp',
           'BF3_2comp_seed0_uniform', 'BF3_2comp_seed0_slope']
CTRL_EPS = [e for e in EPISODES if any(t in e for t in CONTROL)]
LENS_EPS = [e for e in EPISODES if e not in CTRL_EPS]


def sp(d, lower=True):
    d = [x for x in d if np.isfinite(x) and x != 0]
    n = len(d)
    if n == 0:
        return float('nan'), 0, 0
    k = sum(1 for x in d if (x < 0) == lower)
    kk = max(k, n - k)
    return min(1.0, 2 * sum(comb(n, i) for i in range(kk, n + 1)) / 2 ** n), k, n


def main():
    R = {}
    for line in open(os.path.join(HERE, 'probe.jsonl')):
        r = json.loads(line)
        if 'error' in r:
            print('ERR', r['job_id'], r['error'][:100]); continue
        R[(r['cell'], r['episode'], r['n_g'])] = r
    cells = sorted({k[0] for k in R})
    budgets = sorted({k[2] for k in R})

    for n_g in budgets:
        print(f'\n############ budget n_gaussians = {n_g} ############')
        print(f"{'episode':<30s}{'N_gt':>5s}" +
              ''.join(f'{c+" det":>9s}{c+" kerr":>10s}' for c in cells))
        for e in EPISODES:
            row = f'{e[8:-3]:<30s}'
            r0 = R.get((cells[0], e, n_g))
            row += f"{r0['n_gt']:5d}" if r0 else '    ?'
            for c in cells:
                r = R.get((c, e, n_g))
                if r:
                    row += f"{r['detection_rate']:9.3f}{r.get('k_mag_rel_error_mean', float('nan')):10.4f}"
                else:
                    row += ' ' * 19
            print(row)
        print(f'  {"pooled median":<28s}     ' + ''.join(
            f"{np.median([R[(c,e,n_g)]['detection_rate'] for e in EPISODES if (c,e,n_g) in R]):9.3f}"
            f"{np.nanmedian([R[(c,e,n_g)].get('k_mag_rel_error_mean',np.nan) for e in EPISODES if (c,e,n_g) in R]):10.4f}"
            for c in cells))

        for metric, lower in (('detection_rate', False),
                              ('energy_detection_rate', False),
                              ('k_mag_rel_error_mean', True)):
            print(f'  -- {metric} : Δ vs cell A --')
            for c in cells[1:]:
                for name, eps in (('all12', EPISODES), ('crowded4', CROWDED),
                                  ('lens7', LENS_EPS), ('control5', CTRL_EPS)):
                    d = [R[(c, e, n_g)].get(metric, np.nan) - R[('A', e, n_g)].get(metric, np.nan)
                         for e in eps if (c, e, n_g) in R and ('A', e, n_g) in R]
                    p, k, n = sp(d, lower)
                    print(f'     {c}-A {name:<9s} median={np.nanmedian(d):+9.4f} '
                          f'k={k}/{n} p={p:.4f}')

    print('\n############ placement diagnostics (n_gaussians = 15) ############')
    print(f"{'episode':<30s}" + ''.join(f'{c+":out/dup/kmax":>22s}' for c in cells))
    for e in EPISODES:
        row = f'{e[8:-3]:<30s}'
        for c in cells:
            r = R.get((c, e, 15))
            row += (f"{r['n_outside_narrow_box']:8d}/{r['n_placed_pairs_lt_002']:2d}/"
                    f"{r['kmag_max_placed']:8.3f}  " if r else ' ' * 22)
        print(row)


if __name__ == '__main__':
    main()
