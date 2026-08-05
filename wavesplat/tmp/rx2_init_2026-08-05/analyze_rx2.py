"""Rx-2 PRO aggregation. Conventions frozen in PREREG.md + PREREG_ADDENDUM_PRO.md.

- per (cell, episode): median over the 3 seeds, INDEPENDENTLY per metric
- paired delta vs X-BASE per episode; report median(delta), k/n, exact two-sided sign p
- |VR-1| is formed PER FRAME then averaged (driver key `absVR_framewise`)
- gap closure is computed PER EPISODE against the ORACLE cell, then median-ed
- n=4 => sign-test floor p=0.125; alpha=0.05 unreachable; never rank cells
"""
import json
import os
import sys
from collections import defaultdict
from math import comb

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.environ.get('RX2_OUT', os.path.join(HERE, 'results.jsonl'))
EPISODES = ['episode_BF4_15comp_seed0_uniform.h5', 'episode_BF4_30comp_seed0_uniform.h5',
            'episode_BF5_15comp_seed0_uniform.h5', 'episode_BF5_30comp_seed0_uniform.h5']
SHORT = {e: e.replace('episode_', '').replace('_seed0_uniform.h5', '') for e in EPISODES}
# frozen single-seed LETKF reference (tmp/s1s2_2026-08-05/RESULT.md), NOT re-run here
LETKF = {'episode_BF4_15comp_seed0_uniform.h5': 0.2282,
         'episode_BF4_30comp_seed0_uniform.h5': 0.2347,
         'episode_BF5_15comp_seed0_uniform.h5': 0.2298,
         'episode_BF5_30comp_seed0_uniform.h5': 0.2651}
# lower is better / higher is better
LOWER = ['init_nrmse', 'nRMSE', 'Hs_error', 'absVR_framewise', 'avg_RMSE_unobserved']
HIGHER = ['Pearson_rho', 'Skill_Score']
METRICS = LOWER + HIGHER


def sign_p(deltas, better_is_lower=True):
    """Exact two-sided sign test. k = #episodes improved; zeros dropped."""
    d = [x for x in deltas if abs(x) > 1e-12]
    n = len(d)
    if n == 0:
        return 0, 0, 1.0
    k = sum(1 for x in d if (x < 0) == better_is_lower)
    m = max(k, n - k)
    p = min(1.0, 2 * sum(comb(n, i) for i in range(m, n + 1)) / 2 ** n)
    return k, n, p


def load(path=OUT):
    rows = []
    for line in open(path):
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        if 'error' in r:
            print('ERROR ROW', r['job_id'], r['error'][:120], file=sys.stderr)
            continue
        rows.append(r)
    return rows


def get(r, k):
    if k in r:
        return float(r[k])
    if k.startswith('spec.'):
        return float(r.get('spec', {}).get(k[5:], np.nan))
    return float('nan')


def build(rows):
    by = defaultdict(dict)
    for r in rows:
        by[(r['cell'], r['episode'])][r['seed']] = r
    cells = sorted({c for c, _ in by})
    med = {}
    for c in cells:
        for e in EPISODES:
            d = by.get((c, e))
            if not d:
                continue
            med[(c, e)] = {k: float(np.median([get(d[s], k) for s in sorted(d)]))
                           for k in METRICS + ['spec.detection_rate', 'spec.amp_rel_error_mean',
                                               'spec.k_mag_rel_error_mean', 'N',
                                               'decorr_mean', 'shrink_mean', 'm_mean']}
            med[(c, e)]['_seeds'] = sorted(d)
            med[(c, e)]['_range'] = {k: float(np.ptp([get(d[s], k) for s in sorted(d)]))
                                     for k in METRICS}
    return by, cells, med


def main():
    rows = load()
    by, cells, med = build(rows)
    have = [c for c in cells if all((c, e) in med for e in EPISODES)]
    print(f'# rows {len(rows)}   complete cells {have}\n')

    print('== per-cell medians (mean over the 4 crowded episodes) ==')
    hdr = f"{'cell':<8}" + ''.join(f'{k[:11]:>12}' for k in METRICS) + f"{'N':>6}"
    print(hdr)
    for c in have:
        v = [np.mean([med[(c, e)][k] for e in EPISODES]) for k in METRICS]
        n = np.mean([med[(c, e)]['N'] for e in EPISODES])
        print(f'{c:<8}' + ''.join(f'{x:>12.4f}' for x in v) + f'{n:>6.1f}')
    print()

    base = 'X-BASE'
    orc = 'ORACLE' if 'ORACLE' in have else None
    print('== paired deltas vs X-BASE (median over 4 eps, k/n improved, exact 2-sided sign p) ==')
    print('!! n=4 -> sign-test floor p=0.125. alpha=0.05 UNREACHABLE. no "significant", no ranking.')
    for k in METRICS:
        lower = k in LOWER
        print(f'\n-- {k} ({"lower" if lower else "higher"} is better) --')
        for c in have:
            if c == base:
                continue
            d = [med[(c, e)][k] - med[(base, e)][k] for e in EPISODES]
            kk, nn, p = sign_p(d, lower)
            per = ' '.join(f'{x:+.4f}' for x in d)
            print(f'  {c:<8} med {np.median(d):+.4f}  {kk}/{nn}  p={p:.3f}   [{per}]')

    if orc:
        print('\n== gap closure on init_nrmse (per-episode, denominator = X-BASE - ORACLE) ==')
        for c in have:
            if c in (base, orc):
                continue
            cl = []
            for e in EPISODES:
                gap = med[(base, e)]['init_nrmse'] - med[(orc, e)]['init_nrmse']
                cl.append((med[(base, e)]['init_nrmse'] - med[(c, e)]['init_nrmse']) / gap)
            print(f'  {c:<8} median {np.median(cl)*100:+6.1f}%  '
                  f'{sum(1 for x in cl if x>0)}/4 positive   '
                  f'[{" ".join(f"{x*100:+.1f}%" for x in cl)}]')

    print('\n== vs frozen LETKF (single-seed reference, NOT re-run) ==')
    for c in have:
        w = sum(1 for e in EPISODES if med[(c, e)]['nRMSE'] < LETKF[e])
        print(f'  {c:<8} nRMSE mean4 {np.mean([med[(c,e)]["nRMSE"] for e in EPISODES]):.4f}'
              f'   beats LETKF {w}/4')

    print('\n== rho-vs-v decomposition  m = (1-rho^2) + (sqrt(v)-rho)^2  (frame-wise, then mean) ==')
    print(f"{'cell':<8}{'m':>9}{'decorr':>9}{'shrink':>9}{'shrink%':>9}"
          f"{'d_decorr':>10}{'d_shrink':>10}")
    for c in have:
        m = np.mean([med[(c, e)]['m_mean'] for e in EPISODES])
        de = np.mean([med[(c, e)]['decorr_mean'] for e in EPISODES])
        sh = np.mean([med[(c, e)]['shrink_mean'] for e in EPISODES])
        dd = np.median([med[(c, e)]['decorr_mean'] - med[(base, e)]['decorr_mean']
                        for e in EPISODES])
        ds = np.median([med[(c, e)]['shrink_mean'] - med[(base, e)]['shrink_mean']
                        for e in EPISODES])
        print(f'{c:<8}{m:>9.4f}{de:>9.4f}{sh:>9.4f}{sh/max(m,1e-9)*100:>8.1f}%'
              f'{dd:>10.4f}{ds:>10.4f}')

    print('\n== realized sigma_d + realized MDE (from X-BASE seed ranges, post hoc) ==')
    d80 = 2.1279
    for k in METRICS:
        rm = float(np.median([med[(base, e)]['_range'][k] for e in EPISODES]))
        sd = np.sqrt(2) * 0.6698 * (rm / 1.693)
        print(f'  {k:<22} range_med {rm:.4f}  sigma_d {sd:.4f}  realized MDE {d80*sd:.4f}')

    print('\n== diagnostics (detection_rate is DISQUALIFIED as a placement metric; '
          'reported with n_est) ==')
    for c in have:
        dr = np.mean([med[(c, e)]['spec.detection_rate'] for e in EPISODES])
        ke = np.mean([med[(c, e)]['spec.k_mag_rel_error_mean'] for e in EPISODES])
        ae = np.mean([med[(c, e)]['spec.amp_rel_error_mean'] for e in EPISODES])
        n = np.mean([med[(c, e)]['N'] for e in EPISODES])
        print(f'  {c:<8} det {dr:.3f}  k_mag_err {ke:.3f}  amp_rel_err {ae:.3f}  N {n:.1f}')

    # interaction for the declared bundles
    print('\n== declared BUNDLE interactions (never given a single-cause name) ==')
    for bundle, a, b in [('G1P1', 'G1', 'P1'), ('G1P3', 'G1', 'P3')]:
        if not all(x in have for x in (bundle, a, b)):
            continue
        for k in ('init_nrmse', 'nRMSE'):
            dab = np.median([med[(bundle, e)][k] - med[(base, e)][k] for e in EPISODES])
            da = np.median([med[(a, e)][k] - med[(base, e)][k] for e in EPISODES])
            db = np.median([med[(b, e)][k] - med[(base, e)][k] for e in EPISODES])
            print(f'  {bundle} {k:<11} bundle {dab:+.4f} = {a} {da:+.4f} + {b} {db:+.4f} '
                  f'+ interaction {dab-da-db:+.4f}')


if __name__ == '__main__':
    main()
