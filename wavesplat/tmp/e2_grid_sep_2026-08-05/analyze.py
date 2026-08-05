"""E2 aggregation, exactly as PREREG sections 3-4 fix it. No post-hoc choices."""
import json
import math
import os
import sys
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
E1 = os.path.join(HERE, '..', 'e1_grid_2026-08-05')

EPISODES = [
    'episode_BF2_1comp_seed0_uniform.h5', 'episode_BF3_1comp_seed0_uniform.h5',
    'episode_BF3_2comp_seed0_uniform.h5', 'episode_BF3_2comp_seed0_slope.h5',
    'episode_BF3_5comp_seed0_uniform.h5', 'episode_BF3_5comp_seed0_slope.h5',
    'episode_BF4_1comp_seed0_uniform.h5', 'episode_BF4_5comp_seed0_uniform.h5',
    'episode_BF4_15comp_seed0_uniform.h5', 'episode_BF4_30comp_seed0_uniform.h5',
    'episode_BF5_15comp_seed0_uniform.h5', 'episode_BF5_30comp_seed0_uniform.h5',
]
SHORT = {e: e[8:-3].replace('_seed0', '') for e in EPISODES}
LAYERS = {'sparse(1comp)': EPISODES[0:2] + [EPISODES[6]],
          'mid(2,5comp)': EPISODES[2:6] + [EPISODES[7]],
          'crowded(15,30)': EPISODES[8:12]}

# PREREG section 2.7 -- frozen contrasts, each changes exactly one thing.
CONTRASTS = [
    ('U0',   'A0', 'nuisance: sites2,3 grid 61->81'),
    ('U1',   'U0', '*** STEP alone      .100 -> .020'),
    ('U2',   'U1', '*** RADIUS alone @fine   .200 -> .100'),
    ('U3',   'U0', '*** RADIUS alone @coarse .200 -> .100'),
    ('U4',   'U1', '*** COVERAGE alone  +-4 -> +-0.8'),
    ('U1M2', 'U1', '*** MINSEP 0.02 @fine'),
    ('U1M5', 'U1', '*** MINSEP 0.05 @fine'),
    ('U0M5', 'U0', 'MINSEP 0.05 @coarse (negative control)'),
    ('U4M5', 'U4', '*** MINSEP 0.05 @narrow'),
]

# metric -> (better direction: -1 lower-is-better, +1 higher-is-better, 0 diagnostic,
#            pre-registered MDE from E1 realised sigma_d, E1 prior MDE)
FIELD = {
    'init_nrmse':     (-1, 0.0114, 0.0262),
    'Hs_error':       (-1, 0.0131, 0.0235),
    'absVR1':         (-1, 0.0249, 0.0482),
    'nRMSE':          (-1, 0.0118, 0.0188),
    'Skill_Score':    (+1, 0.0299, 0.1800),
    'Pearson_rho':    (+1, 0.0097, 0.1047),
    'Directional_R2': (+1, 0.0333, 0.1434),
    'spec_det':       (+1, None, None),
    'N':              (0, None, None),
    'final_nn_p10':   (0, None, None),
    'final_nn_median': (0, None, None),
    'final_n_pairs_lt_005': (0, None, None),
}
PROBE = {
    'detection_rate':        (+1,),
    'energy_detection_rate': (+1,),
    'k_mag_rel_error_mean':  (-1,),
    'placed_nn_p10':         (+1,),   # larger = less stacking
    'placed_nn_min':         (+1,),
    'n_outside_narrow_box':  (-1,),
    'n_found_est':           (0,),
}


def binom_two_sided(k, n):
    if n == 0:
        return float('nan')
    c = math.comb
    p = [c(n, i) / 2 ** n for i in range(n + 1)]
    return float(min(1.0, sum(x for x in p if x <= p[k] + 1e-15)))


def signtest(diffs, better):
    """diffs = treatment - control. Returns (median, k_improved, n_eff, p, floor_p)."""
    d = [x for x in diffs if x == x]
    nz = [x for x in d if abs(x) > 0]
    n = len(nz)
    if n == 0:
        return (float(np.median(d)) if d else float('nan'), 0, 0, float('nan'),
                float('nan'))
    k = sum(1 for x in nz if (x < 0) == (better < 0)) if better else 0
    return (float(np.median(d)), k, n, binom_two_sided(k, n), 2 * 0.5 ** n)


def load_e2e(path, cellmap=None):
    per = defaultdict(list)
    for line in open(path):
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if 'error' in r:
            continue
        c = r['cell'] if cellmap is None else cellmap.get(r['cell'], r['cell'])
        rec = dict(r)
        rec['absVR1'] = abs(r.get('Variance_Ratio', float('nan')) - 1.0)
        rec['spec_det'] = r.get('spec', {}).get('detection_rate', float('nan'))
        per[(c, r['episode'])].append(rec)
    med = {}
    for k, rows in per.items():
        med[k] = {m: float(np.median([x.get(m, float('nan')) for x in rows]))
                  for m in FIELD}
        med[k]['_n_seed'] = len(rows)
        med[k]['_range'] = {m: float(np.max([x.get(m, np.nan) for x in rows]) -
                                     np.min([x.get(m, np.nan) for x in rows]))
                            for m in ('init_nrmse', 'nRMSE', 'Hs_error', 'absVR1')}
    return med


def load_probe(path, cellmap=None):
    out = {}
    for line in open(path):
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if 'error' in r:
            continue
        c = r['cell'] if cellmap is None else cellmap.get(r['cell'], r['cell'])
        out[(c, r['episode'], r['n_g'])] = r
    return out


def main():
    med = load_e2e(os.path.join(HERE, 'results.jsonl'))
    prb = load_probe(os.path.join(HERE, 'probe.jsonl'))
    cells = sorted({k[0] for k in med})
    print('cells with e2e data:', cells)
    print('runs per cell/episode (should be 3):',
          sorted({v['_n_seed'] for v in med.values()}))

    # ---------------- realised sigma_d (reported alongside, not used to judge) --------
    print('\n' + '=' * 100)
    print('REALISED seed range (median over 12 ep) per cell -- noise floor sanity')
    print(f"{'cell':<6s}" + ''.join(f'{m:>14s}' for m in
                                    ('init_nrmse', 'nRMSE', 'Hs_error', 'absVR1')))
    for c in cells:
        rs = [med[(c, e)]['_range'] for e in EPISODES if (c, e) in med]
        if not rs:
            continue
        print(f'{c:<6s}' + ''.join(
            f"{np.median([r[m] for r in rs]):14.4f}"
            for m in ('init_nrmse', 'nRMSE', 'Hs_error', 'absVR1')))

    # ---------------- FIELD axis -----------------------------------------------------
    print('\n' + '=' * 100)
    print('FIELD AXIS -- paired difference (treatment - control), median over 12 episodes')
    print('  sign k/n = # episodes moving in the IMPROVING direction; p = exact two-sided')
    for tr, ct, name in CONTRASTS:
        if not any((tr, e) in med for e in EPISODES):
            continue
        print(f'\n--- {tr} - {ct}   {name}')
        print(f"  {'metric':<22s}{'medDelta':>11s}{'k/n':>8s}{'p':>9s}"
              f"{'MDE(E1real)':>13s}{'MDE(prior)':>12s}  verdict")
        for m, spec in FIELD.items():
            better = spec[0]
            d = [med[(tr, e)][m] - med[(ct, e)][m] for e in EPISODES
                 if (tr, e) in med and (ct, e) in med]
            if not d:
                continue
            mu, k, n, p, floor = signtest(d, better if better else -1)
            mde = spec[1]
            if better == 0:
                verdict = 'diagnostic'
            else:
                improving = (mu < 0) if better < 0 else (mu > 0)
                big = mde is not None and abs(mu) > mde
                sig = (p == p) and p < 0.05
                if mde is None:
                    verdict = ('improve' if improving else 'degrade') + (' SIG' if sig else '')
                elif improving:
                    verdict = 'PASS' if (big and sig) else ('WEAK' if (big or sig)
                                                            else f'undetected(MDE={mde})')
                else:
                    verdict = 'HARM' if (big and sig) else ('weak-harm' if (big or sig)
                                                            else f'undetected(MDE={mde})')
            print(f'  {m:<22s}{mu:>11.4f}{f"{k}/{n}":>8s}{p:>9.4f}'
                  f'{str(mde):>13s}{str(spec[2]):>12s}  {verdict}')

    # ---------------- 2x2 main effects / interaction ---------------------------------
    print('\n' + '=' * 100)
    print('2x2 {U0,U1,U2,U3} = STEP x RADIUS -- main effects and interaction (n=12)')
    print(f"  {'metric':<22s}{'STEP main':>12s}{'k/n':>7s}{'RADIUS main':>13s}{'k/n':>7s}"
          f"{'interaction':>13s}{'k/n':>7s}")
    for m, spec in FIELD.items():
        better = spec[0] or -1
        try:
            step = [0.5 * ((med[('U1', e)][m] - med[('U0', e)][m]) +
                           (med[('U2', e)][m] - med[('U3', e)][m])) for e in EPISODES]
            rad = [0.5 * ((med[('U3', e)][m] - med[('U0', e)][m]) +
                          (med[('U2', e)][m] - med[('U1', e)][m])) for e in EPISODES]
            inter = [(med[('U2', e)][m] - med[('U1', e)][m]) -
                     (med[('U3', e)][m] - med[('U0', e)][m]) for e in EPISODES]
        except KeyError:
            continue
        a = signtest(step, better); b = signtest(rad, better); c_ = signtest(inter, better)
        print(f'  {m:<22s}{a[0]:>12.4f}{f"{a[1]}/{a[2]}":>7s}{b[0]:>13.4f}'
              f'{f"{b[1]}/{b[2]}":>7s}{c_[0]:>13.4f}{f"{c_[1]}/{c_[2]}":>7s}')

    # ---------------- additivity vs E1 (PREREG 2.8) ----------------------------------
    print('\n' + '=' * 100)
    print('ADDITIVITY CHECK (PREREG 2.8): chain (U0-A0)+(U1-U0)+(U4-U1) vs E1 measured B-A')
    e1med = load_e2e(os.path.join(E1, 'results.jsonl'))
    print(f"  {'metric':<22s}{'chain sum':>12s}{'E1 B-A':>12s}{'residual':>12s}")
    for m in ('init_nrmse', 'nRMSE', 'Hs_error', 'absVR1'):
        try:
            chain = [med[('U4', e)][m] - med[('A0', e)][m] for e in EPISODES]
            e1d = [e1med[('B', e)][m] - e1med[('A', e)][m] for e in EPISODES]
        except KeyError:
            continue
        cs, e1s = float(np.median(chain)), float(np.median(e1d))
        print(f'  {m:<22s}{cs:>12.4f}{e1s:>12.4f}{cs - e1s:>12.4f}')

    # ---------------- PLACEMENT axis -------------------------------------------------
    print('\n' + '=' * 100)
    print('PLACEMENT AXIS (probe, deterministic) -- paired diff, TIES DROPPED')
    for ng in (15, 3):
        print(f'\n########## n_g = {ng}')
        for tr, ct, name in CONTRASTS:
            if not any((tr, e, ng) in prb for e in EPISODES):
                continue
            print(f'\n--- {tr} - {ct}   {name}')
            print(f"  {'metric':<24s}{'medDelta':>11s}{'k/n_eff':>10s}{'p':>9s}"
                  f"{'floor p':>9s}  note")
            for m, (better,) in PROBE.items():
                d = [prb[(tr, e, ng)].get(m, float('nan')) -
                     prb[(ct, e, ng)].get(m, float('nan'))
                     for e in EPISODES if (tr, e, ng) in prb and (ct, e, ng) in prb]
                if not d:
                    continue
                mu, k, n, p, floor = signtest(d, better if better else -1)
                note = 'diagnostic' if better == 0 else (
                    'alpha=.05 possible' if floor == floor and floor <= 0.05
                    else 'alpha=.05 IMPOSSIBLE at this n_eff')
                print(f'  {m:<24s}{mu:>11.4f}{f"{k}/{n}":>10s}{p:>9.4f}'
                      f'{floor:>9.4f}  {note}')

    # ---------------- per-episode tables --------------------------------------------
    for m in ('init_nrmse', 'Hs_error', 'absVR1', 'N'):
        print('\n' + '=' * 100)
        print(f'PER-EPISODE {m} (median-of-3)')
        print(f"{'episode':<24s}" + ''.join(f'{c:>9s}' for c in cells))
        for e in EPISODES:
            print(f'{SHORT[e]:<24s}' + ''.join(
                f"{med[(c, e)][m]:9.4f}" if (c, e) in med else f"{'--':>9s}"
                for c in cells))

    print('\n' + '=' * 100)
    print('PROBE detection_rate per episode (n_g=15 / n_g=3)')
    print(f"{'episode':<24s}" + ''.join(f'{c:>13s}' for c in cells))
    for e in EPISODES:
        row = f'{SHORT[e]:<24s}'
        for c in cells:
            a = prb.get((c, e, 15), {}).get('detection_rate')
            b = prb.get((c, e, 3), {}).get('detection_rate')
            row += f"{(f'{a:.2f}/{b:.2f}' if a is not None and b is not None else '--'):>13s}"
        print(row)

    print('\n' + '=' * 100)
    print('PROBE placed_nn_p10 per episode (n_g=15) -- direct stacking indicator')
    print(f"{'episode':<24s}" + ''.join(f'{c:>9s}' for c in cells))
    for e in EPISODES:
        print(f'{SHORT[e]:<24s}' + ''.join(
            f"{prb[(c, e, 15)]['placed_nn_p10']:9.4f}" if (c, e, 15) in prb
            else f"{'--':>9s}" for c in cells))

    # ---------------- layers (descriptive only) --------------------------------------
    print('\n' + '=' * 100)
    print('LAYERS (DESCRIPTIVE ONLY -- alpha=0.05 impossible at n=3/5/4; no mechanism)')
    for m in ('init_nrmse', 'nRMSE', 'Hs_error', 'absVR1'):
        print(f'\n-- {m}: median paired difference by layer')
        print(f"{'contrast':<14s}" + ''.join(f'{L:>18s}' for L in LAYERS))
        for tr, ct, _ in CONTRASTS:
            if not any((tr, e) in med for e in EPISODES):
                continue
            row = f'{tr+"-"+ct:<14s}'
            for L, eps in LAYERS.items():
                d = [med[(tr, e)][m] - med[(ct, e)][m] for e in eps
                     if (tr, e) in med and (ct, e) in med]
                row += f'{np.median(d):18.4f}' if d else f'{"--":>18s}'
            print(row)


if __name__ == '__main__':
    sys.path.insert(0, HERE)
    main()
