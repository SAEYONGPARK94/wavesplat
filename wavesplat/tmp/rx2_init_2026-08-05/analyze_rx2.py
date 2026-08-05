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
# FLAT-SEA (zero-predictor) null, re-derived independently at the exact frames run.py scores
# (PREREG AMENDMENT 1). Mandatory alongside every table: init_nrmse is saturated at this
# floor in the shipping-fallback band, so it cannot separate "slightly wrong" placement from
# "completely wrong" placement.
NULL_INIT = {'episode_BF4_15comp_seed0_uniform.h5': 0.2489,
             'episode_BF4_30comp_seed0_uniform.h5': 0.2692,
             'episode_BF5_15comp_seed0_uniform.h5': 0.2602,
             'episode_BF5_30comp_seed0_uniform.h5': 0.2982}
NULL_TRACK = {'episode_BF4_15comp_seed0_uniform.h5': 0.2533,
              'episode_BF4_30comp_seed0_uniform.h5': 0.2736,
              'episode_BF5_15comp_seed0_uniform.h5': 0.2388,
              'episode_BF5_30comp_seed0_uniform.h5': 0.2732}
# seed-paired bootstrap MDE at the shipping-fallback point (PREREG AMENDMENT 1, 4000x, 80%)
BOOT_MDE = {'init_nrmse': 0.0183, 'nRMSE': 0.0309}
# corrected reachable prize (oracle placement with k TRAINABLE == s1s2 arm S2-T1)
PRIZE_NRMSE = 0.0637
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

    print('== per-cell, PREREG aggregation = median over the 4 crowded episodes ==')
    print('   (flat-sea null and LETKF printed at the bottom -- mandatory companions)')
    hdr = (f"{'cell':<9}" + ''.join(f'{k[:11]:>12}' for k in METRICS) +
           f"{'N':>6}{'>null_i':>9}{'>null_n':>9}{'<LETKF':>8}")
    print(hdr)
    for c in have:
        v = [np.median([med[(c, e)][k] for e in EPISODES]) for k in METRICS]
        n = np.median([med[(c, e)]['N'] for e in EPISODES])
        bi = sum(1 for e in EPISODES if med[(c, e)]['init_nrmse'] < NULL_INIT[e])
        bn = sum(1 for e in EPISODES if med[(c, e)]['nRMSE'] < NULL_TRACK[e])
        bl = sum(1 for e in EPISODES if med[(c, e)]['nRMSE'] < LETKF[e])
        print(f'{c:<9}' + ''.join(f'{x:>12.4f}' for x in v) +
              f'{n:>6.1f}{bi:>7}/4{bn:>7}/4{bl:>6}/4')
    print(f"{'FLAT-NULL':<9}{np.median(list(NULL_INIT.values())):>12.4f}"
          f"{np.median(list(NULL_TRACK.values())):>12.4f}")
    print(f"{'LETKF':<9}{'':>12}{np.median(list(LETKF.values())):>12.4f}"
          f"{'':>34}{sum(1 for e in EPISODES if LETKF[e] < NULL_TRACK[e]):>7}/4")
    print()
    print('== mean-over-episodes companion table (NOT the prereg statistic; for comparability'
          ' with the older mean-based tables only) ==')
    for c in have:
        v = [np.mean([med[(c, e)][k] for e in EPISODES]) for k in ('init_nrmse', 'nRMSE')]
        print(f'  {c:<9} init {v[0]:.4f}  nRMSE {v[1]:.4f}')
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
            flag = ''
            if k in BOOT_MDE:
                flag = ('  [< bootstrap MDE %.4f => UNINTERPRETABLE]' % BOOT_MDE[k]
                        if abs(np.median(d)) < BOOT_MDE[k] else
                        '  [> bootstrap MDE %.4f]' % BOOT_MDE[k])
            print(f'  {c:<8} med {np.median(d):+.4f}  {kk}/{nn}  p={p:.3f}   [{per}]{flag}')

    for denom, tag in [('ORACLE-K', 'REACHABLE ceiling, k trainable == s1s2 S2-T1  [PRIMARY]'),
                       ('ORACLE', 'k-FROZEN ceiling == s1s2 S2-T1f  [not deployable]')]:
        if denom not in have:
            continue
        for k in ('init_nrmse', 'nRMSE'):
            print(f'\n== gap closure on {k}: denominator = X-BASE - {denom}   ({tag}) ==')
            for c in have:
                if c in (base,) or c.startswith('ORACLE') or c.startswith('X-BASE'):
                    continue
                cl = []
                for e in EPISODES:
                    gap = med[(base, e)][k] - med[(denom, e)][k]
                    cl.append((med[(base, e)][k] - med[(c, e)][k]) / gap if abs(gap) > 1e-9
                              else np.nan)
                print(f'  {c:<8} median {np.nanmedian(cl)*100:+7.1f}%  '
                      f'{sum(1 for x in cl if x>0)}/4 positive   '
                      f'[{" ".join(f"{x*100:+.0f}%" for x in cl)}]')

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
          'ALWAYS printed with its same-n_est one-point-stacking null) ==')
    nulls = stacking_nulls({int(round(med[(c, e)]['N'])) for c in have for e in EPISODES})
    for c in have:
        dr = np.median([med[(c, e)]['spec.detection_rate'] for e in EPISODES])
        nl = np.median([nulls[(e, int(round(med[(c, e)]['N'])))] for e in EPISODES])
        over = sum(1 for e in EPISODES
                   if med[(c, e)]['spec.detection_rate'] >
                   nulls[(e, int(round(med[(c, e)]['N'])))])
        ke = np.median([med[(c, e)]['spec.k_mag_rel_error_mean'] for e in EPISODES])
        ae = np.median([med[(c, e)]['spec.amp_rel_error_mean'] for e in EPISODES])
        n = np.median([med[(c, e)]['N'] for e in EPISODES])
        print(f'  {c:<8} det {dr:.3f}  stack-null {nl:.3f}  beats-null {over}/4  '
              f'k_mag_err {ke:.3f}  amp_rel_err {ae:.3f}  N {n:.1f}')


def stacking_nulls(n_ests):
    """Discipline #1 of tmp/placement_field_debate_2026-08-05/VERDICT.md: the matcher's
    acceptance radius is so wide that stacking every atom on the GT energy centroid already
    scores 0.60-0.87. Every detection_rate must be printed next to that null at the SAME
    n_est."""
    import h5py
    from run_rx2 import DATA, match_components
    out = {}
    for e in EPISODES:
        with h5py.File(os.path.join(DATA, e), 'r') as f:
            comp = np.asarray(f['config/wave_components'])
        A, kx, ky = comp[:, 0], comp[:, 2], comp[:, 3]
        wsum = max(float(np.sum(A ** 2)), 1e-12)
        cx = float(np.sum(A ** 2 * kx) / wsum)
        cy = float(np.sum(A ** 2 * ky) / wsum)
        for n in n_ests:
            n = max(int(n), 1)
            ekx = np.full(n, cx); eky = np.full(n, cy)
            m, _, _ = match_components(ekx, eky, np.ones(n), kx, ky, A)
            out[(e, n)] = len(m) / max(len(A), 1)
    return out

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
