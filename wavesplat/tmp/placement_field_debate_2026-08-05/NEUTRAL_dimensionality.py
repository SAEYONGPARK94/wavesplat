"""NEUTRAL — (a) is "placement quality" one-dimensional?  (b) can the mediator be identified?

Re-analysis only.
(a) rank-correlate the placement metrics that E1/S1/S2 all record, over every arm x episode
    unit available, and ask whether they order the arms the same way.
(b) variance-decompose the proposed mediator (atom spacing) into between-treatment and
    within-treatment parts, and check its collinearity with the rival mediator (box coverage).
"""
import json
import os
from collections import defaultdict

import numpy as np
from scipy.stats import spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..'))
E1 = os.path.join(REPO, 'tmp', 'e1_grid_2026-08-05')
S12 = os.path.join(REPO, 'tmp', 's1s2_2026-08-05')
CROWDED = ['episode_BF4_15comp_seed0_uniform.h5', 'episode_BF4_30comp_seed0_uniform.h5',
           'episode_BF5_15comp_seed0_uniform.h5', 'episode_BF5_30comp_seed0_uniform.h5']


def med(v):
    v = [x for x in v if x is not None and np.isfinite(x)]
    return float(np.median(v)) if v else float('nan')


def main():
    out = {}
    e1 = [json.loads(l) for l in open(os.path.join(E1, 'results.jsonl'))]
    e1 = [r for r in e1 if 'error' not in r]   # NOTE: substring filter on the raw line would
    # have dropped every record, because 'k_mag_rel_error_mean' contains 'error'.
    probe = [json.loads(l) for l in open(os.path.join(E1, 'probe.jsonl'))]
    s12 = [json.loads(l) for l in open(os.path.join(S12, 'results.jsonl'))]

    # -------- unit = (experiment, arm, episode); median over seeds --------
    units = []
    grp = defaultdict(list)
    for r in e1:
        grp[('E1', r['cell'], r['episode'])].append(r)
    for r in s12:
        if 'spec' in r:
            grp[('S1S2', r['arm'], r['episode'])].append(r)
    for (exp, arm, ep), rows in grp.items():
        sp = [r['spec'] for r in rows if 'spec' in r and 'detection_rate' in r['spec']]
        if not sp:
            continue
        units.append(dict(
            exp=exp, arm=arm, ep=ep,
            det=med([s['detection_rate'] for s in sp]),
            edet=med([s.get('energy_detection_rate') for s in sp]),
            kerr=med([s.get('k_mag_rel_error_mean') for s in sp]),
            direrr=med([s.get('direction_error_deg_mean') for s in sp]),
            n_est=med([s['n_est'] for s in sp]),
            n_gt=sp[0]['n_gt'],
            nRMSE=med([r['nRMSE'] for r in rows]),
            init_nrmse=med([r['init_nrmse'] for r in rows]),
            Hs=med([r['Hs_error'] for r in rows]),
            absVR=med([abs(r['Variance_Ratio'] - 1) for r in rows])))
    out['units'] = units
    print(f'{len(units)} (experiment, arm, episode) units, seeds collapsed by median\n')

    print('=== (a1) do the placement metrics agree with each other? '
          '(Spearman over units, crowded episodes only) ===')
    cw = [u for u in units if u['ep'] in CROWDED and np.isfinite(u['kerr'])]
    print(f'  n = {len(cw)} units')
    pairs = [('det', 'kerr'), ('det', 'direrr'), ('edet', 'kerr'),
             ('det', 'edet'), ('kerr', 'direrr')]
    agree = {}
    for a, b in pairs:
        x = [u[a] for u in cw]
        y = [u[b] for u in cw]
        rho, p = spearmanr(x, y)
        agree[f'{a}~{b}'] = dict(rho=float(rho), p=float(p), n=len(cw))
        note = ''
        if (a, b) in (('det', 'kerr'), ('det', 'direrr'), ('edet', 'kerr'), ('kerr', 'direrr')):
            note = '   (lower kerr/direrr = better; so a NEGATIVE rho means the two metrics agree)'
        print(f'  spearman({a:>6s}, {b:>6s}) = {rho:+.3f}  p={p:.4f}{note}')
    out['metric_agreement'] = agree

    print('\n=== (a2) the two metrics rank these arms in OPPOSITE order ===')
    for arm in ['S1-A0', 'S2-C1', 'S2-T1', 'S2-T1f']:
        s = [u for u in units if u['arm'] == arm and u['ep'] in CROWDED]
        print(f'  {arm:<8s} det {med([u["det"] for u in s]):.3f}   '
              f'k_mag_rel_err {med([u["kerr"] for u in s]):.4f}   '
              f'dir_err {med([u["direrr"] for u in s]):6.2f}deg   '
              f'nRMSE {med([u["nRMSE"] for u in s]):.4f}')
    print('  -> S1-A0 beats S2-C1 on detection (0.400 vs 0.217) and LOSES to it on k-error.')

    print('\n=== (a3) which placement metric tracks the field? (Spearman vs nRMSE, crowded) ===')
    for k in ('det', 'edet', 'kerr', 'direrr', 'n_est'):
        x = [u[k] for u in cw]
        y = [u['nRMSE'] for u in cw]
        rho, p = spearmanr(x, y)
        print(f'  spearman({k:>7s}, nRMSE) = {rho:+.3f}  p={p:.4f}')
    # same, restricted to E1 only (where placement is the only thing that moved)
    cwe = [u for u in cw if u['exp'] == 'E1']
    print(f'  -- E1 cells only (n={len(cwe)}) --')
    for k in ('det', 'edet', 'kerr'):
        rho, p = spearmanr([u[k] for u in cwe], [u['nRMSE'] for u in cwe])
        print(f'  spearman({k:>7s}, nRMSE) = {rho:+.3f}  p={p:.4f}')

    # -------------------------------------------------------------- (b)
    print('\n=== (b1) mediator variance decomposition: is spacing free to vary '
          'INSIDE a treatment cell? ===')
    for n_g in (3, 15):
        rows = [r for r in probe if r['n_g'] == n_g]
        eps = sorted({r['episode'] for r in rows})
        # two-way: cell effect vs episode effect on log spacing
        M = np.array([[[r['placed_nn_median'] for r in rows
                        if r['cell'] == c and r['episode'] == e][0]
                       for e in eps] for c in 'ABCDE'])
        L = np.log10(M)
        gm = L.mean()
        ss_tot = ((L - gm) ** 2).sum()
        ss_cell = len(eps) * ((L.mean(1) - gm) ** 2).sum()
        ss_ep = 5 * ((L.mean(0) - gm) ** 2).sum()
        print(f'  n_g={n_g}: var(log10 spacing) explained by CELL {ss_cell / ss_tot * 100:5.1f}% | '
              f'by EPISODE {ss_ep / ss_tot * 100:5.1f}% | residual '
              f'{(ss_tot - ss_cell - ss_ep) / ss_tot * 100:5.1f}%')

    print('\n=== (b2) collinearity of the rival mediators (delta vs cell A, per episode) ===')
    pr = {(r['cell'], r['episode'], r['n_g']): r for r in probe}
    eps = sorted({r['episode'] for r in probe})
    for n_g in (3, 15):
        ds, do, dp = [], [], []
        for c in 'BCDE':
            for e in eps:
                a, x = pr[('A', e, n_g)], pr[(c, e, n_g)]
                ds.append(x['placed_nn_median'] - a['placed_nn_median'])
                do.append(x['n_outside_narrow_box'] - a['n_outside_narrow_box'])
                dp.append(x['n_placed_pairs_lt_002'] - a['n_placed_pairs_lt_002'])
        r1, p1 = spearmanr(ds, do)
        print(f'  n_g={n_g}: spearman(delta spacing, delta #atoms-outside-box) = {r1:+.3f} '
              f'p={p1:.2e}   (n={len(ds)})')
        if np.std(dp) > 0:
            r2, p2 = spearmanr(ds, dp)
            print(f'          spearman(delta spacing, delta #pairs<0.02)        = {r2:+.3f} '
                  f'p={p2:.2e}')
        else:
            print(f'          delta #pairs<0.02 is constant ({np.unique(dp)}) -> no variation')

    print('\n=== (b3) the mediator moved WITHOUT the outcome: cell C ===')
    e1med = defaultdict(dict)
    for c in 'ABCDE':
        for e in eps:
            rows = [r for r in e1 if r['cell'] == c and r['episode'] == e]
            e1med[c][e] = dict(Hs=med([r['Hs_error'] for r in rows]),
                               absVR=med([abs(r['Variance_Ratio'] - 1) for r in rows]))
    print(f'  {"cell":<5s} {"med spacing n_g=15":>19s} {"vs A":>8s} {"med dHs":>9s} '
          f'{"harm k/12":>10s} {"med dAbsVR":>11s}')
    tab = []
    for c in 'ABCDE':
        sp = med([pr[(c, e, 15)]['placed_nn_median'] for e in eps])
        dhs = [e1med[c][e]['Hs'] - e1med['A'][e]['Hs'] for e in eps]
        dvr = [e1med[c][e]['absVR'] - e1med['A'][e]['absVR'] for e in eps]
        spa = med([pr[('A', e, 15)]['placed_nn_median'] for e in eps])
        tab.append(dict(cell=c, spacing=sp, ratio_vs_A=sp / spa,
                        med_dHs=float(np.median(dhs)),
                        harm_k=int(sum(1 for v in dhs if v > 0)),
                        med_dabsVR=float(np.median(dvr))))
        print(f'  {c:<5s} {sp:>19.4f} {sp / spa:>8.2f}x {np.median(dhs):>9.4f} '
              f'{sum(1 for v in dhs if v > 0):>7d}/12 {np.median(dvr):>11.4f}')
    out['mediator_table'] = tab
    out['variance_note'] = ('spacing is a post-treatment quantity; it has no within-cell '
                            'variation that is not also episode variation')

    with open(os.path.join(HERE, 'NEUTRAL_dimensionality.json'), 'w') as f:
        json.dump(out, f, indent=1)


if __name__ == '__main__':
    main()
