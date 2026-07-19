"""E-B2K step 0b -- design checks that must settle BEFORE the k=9 run is frozen.

Two risks found while checking feasibility of fable's frozen record list:
  R1  aperture dilution: the candidate new records (BS11/BS13) sit at
      patch/lambda 0.43-1.22, and the learning gain net-hand is known to scale
      with aperture (rho +0.688, E-16). If Delta(T-L) shares that axis, adding
      low-aperture records dilutes the effect and the expansion back-fires.
  R2  cluster inflation: BS11b/c/d are all 2011-10-04 (11:38 / 13:07 / 15:30).
      Counting them as 3 independent clusters inflates k. Cluster must be
      sea-event (site-day), not file.
Also projects power for the k that each clustering choice actually buys.
"""
import json
import numpy as np
from collections import defaultdict
from scipy import stats

R3 = 'results/direction3/'

# patch/lambda from docs/E12_KP_FLOOR_AUDIT_prereg_2026-07-14.md kp exposure table
APERTURE = {'BS11a': 1.22, 'BS11b': 0.86, 'BS13a': 0.43, 'BS13b': 0.66,
            'LJ18n': 0.18, 'AA15a': 1.03, 'AA15b': 1.12, 'AA14': 2.23,
            'LJ18g': 1.14, 'YS17': 3.88}

# site-day of every candidate record (file date -> sea event)
# CORRECTED 2026-07-19 after fable red-team: BS11a and BS11b were SWAPPED in the
# first version of this dict. Primary source = scripts/d3_sa_multirecord.py:66,73
#   BS11a -> Surfaces_20111004_113800_short.nc  (Oct 4, measured 14.0 x 13.5 m)
#   BS11b -> Surfaces_20111001_161800_short.nc  (Oct 1, measured 13.5 x 13.0 m)
# cross-checked against the E-12 kp table (BS11a L=14 m, BS11b L=13 m).
# Consequence: the same-day triple is BS11a/c/d, NOT b/c/d, and the only
# in-range candidate (patch/lambda 1.22) is BS11a = the Oct-4 file.
SITE_DAY = {'YS17': 'YS_2017-05-13', 'AA15a': 'AA_2015-03-05',
            'AA14': 'AA_2014-03-27', 'AA15b': 'AA_2015-05-15',
            'LJ18g': 'LJ_2018-01-03', 'LJ18n': 'LJ_2018-01-03',
            'BS11a': 'BS_2011-10-04', 'BS11b': 'BS_2011-10-01',
            'BS11c': 'BS_2011-10-04', 'BS11d': 'BS_2011-10-04',
            'BS13a': 'BS_2013-09-22', 'BS13b': 'BS_2013-09-25',
            'BS13c': 'BS_2013-09-30'}


def per_record_delta():
    d1 = json.load(open(R3 + 'b2deep_eval.json'))['rows']
    d2 = json.load(open(R3 + 'b2deep2_eval.json'))['seed3']['rows']
    s2 = {(r['record'], r['wi']): (r['T_s2'], r['L_s2']) for r in d2}
    by = defaultdict(list)
    for r in d1:
        if r['hand'] >= 0.5:
            continue
        arms = [(r['T_s0'], r['L_s0']), (r['T_s1'], r['L_s1'])]
        k = (r['record'], r['wi'])
        if k in s2:
            arms.append(s2[k])
        by[r['record']].append(np.mean([t - l for t, l in arms]))
    return {k: float(np.mean(v)) for k, v in by.items()}


def main():
    print('=' * 68)
    print('R1 -- is Delta(T-L) on the aperture axis?  (if yes, expansion dilutes)')
    print('=' * 68)
    d = per_record_delta()
    recs = sorted(d, key=lambda r: APERTURE[r])
    ap = np.array([APERTURE[r] for r in recs])
    dv = np.array([d[r] for r in recs])
    for r in recs:
        print(f'  {r:8s} patch/lam {APERTURE[r]:5.2f}   Delta(T-L) {d[r]:+.4f}')
    rho, p = stats.spearmanr(ap, dv)
    pr, pp = stats.pearsonr(ap, dv)
    print(f'\n  Spearman(aperture, Delta) = {rho:+.3f}  p {p:.3f}')
    print(f'  Pearson  (aperture, Delta) = {pr:+.3f}  p {pp:.3f}')
    print('  reference: net-hand gain vs aperture (E-16) rho +0.688 p<1e-4')
    verdict = ('NOT on the aperture axis -> low-aperture records are admissible'
               if abs(rho) < 0.5 else
               'ON the aperture axis -> expanding with low-aperture records dilutes')
    print(f'  => {verdict}')

    print()
    print('=' * 68)
    print('R2 -- how many INDEPENDENT clusters does each record list actually buy?')
    print('=' * 68)
    lists = {
        'current (5 files)': ['YS17', 'AA15a', 'AA14', 'AA15b', 'LJ18g'],
        "fable draft 'BS11 x4' (9 files)":
            ['YS17', 'AA15a', 'AA14', 'AA15b', 'LJ18g',
             'BS11a', 'BS11b', 'BS11c', 'BS11d'],
        'reconciled: all usable, cluster=site-day (12 files)':
            ['YS17', 'AA15a', 'AA14', 'AA15b', 'LJ18g', 'BS11a', 'BS11b',
             'BS11c', 'BS11d', 'BS13a', 'BS13b', 'BS13c'],
    }
    for name, lst in lists.items():
        days = sorted({SITE_DAY[r] for r in lst})
        print(f'  {name}')
        print(f'      files {len(lst):2d}   INDEPENDENT site-day clusters {len(days):2d}')
        if len(lst) != len(days):
            dup = defaultdict(list)
            for r in lst:
                dup[SITE_DAY[r]].append(r)
            for k, v in dup.items():
                if len(v) > 1:
                    print(f'      collapsed: {k} <- {v}')

    print()
    print('=' * 68)
    print('POWER -- observed effect/sd, one-sided t, alpha 0.05')
    print('=' * 68)
    m = np.array(list(d.values()))
    eff, sd = m.mean(), m.std(ddof=1)
    print(f'  observed (k=5): mean {eff:+.4f}  sd {sd:.4f}')
    for k in (5, 6, 7, 8, 9, 10, 12):
        se = sd / np.sqrt(k)
        crit = stats.t.ppf(0.95, k - 1)
        mdes = crit * se
        ncp = eff / se
        pw = stats.nct.sf(crit, k - 1, ncp)
        flag = ' <-- current' if k == 5 else (' <-- reconciled' if k == 10 else '')
        print(f'  k={k:2d}  SE {se:.4f}  min detectable {mdes:+.4f}  '
              f'power@observed {pw:.2f}{flag}')
    print('\n  NOTE: power assumes new clusters share the observed mean/sd.')
    print('  If new (low-aperture) clusters have a smaller true effect, the mean')
    print('  falls faster than SE, and the expansion can LOWER the t-stat.')

    # what does the mean have to survive to still pass at k=10?
    se10 = sd / np.sqrt(10)
    need = stats.t.ppf(0.95, 9) * se10
    print(f'\n  at k=10 the record-mean must stay above {need:+.4f} to reach p<0.05')
    n_new = 5
    breakeven = (need * 10 - eff * 5) / n_new
    print(f'  => the {n_new} new clusters may average as low as {breakeven:+.4f} '
          f'and the test still passes')


if __name__ == '__main__':
    main()
