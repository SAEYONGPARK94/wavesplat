"""Rx-1 analysis. PREREG sec.1 aggregation + AMENDMENT 2 (both fixed before results seen).

Aggregation (PREREG sec.1)
  - per (cell, episode): MEDIAN over the 3 seeds
  - primary statistic: MEDIAN of the PAIRED per-episode deltas (cell - X-BASE)
  - direction: nRMSE / init_nrmse / Hs_error / amp_rel_error / |VR-1| smaller better;
               rho / skill larger better.  VR is always reported as |VR-1|.
  - two-sided sign test; k/n and p always together; n=4 floor p=0.125 -> never "significant"

AMENDMENT 2
  - |VR-1| is formed PER FRAME then averaged (`absVR1_framemean`), never mean-then-transform
  - the mechanism split is the EXACT orthogonal one, per frame then averaged:
        m = (1 - rho^2)  +  (sqrt(v) - rho)^2
            decorrelation      shrinkage (removable by one scalar)
    and the ρ channel is PRIMARY, the v channel SECONDARY
  - the design is a 2x2 factorial in (lambda_l1, lambda_smooth); the interaction is reported
  - knobs are NOT ranked against each other (global-null P(spurious 4/4) = 0.865 at m=3)
"""
import json
import math
import os
import sys
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))

SMALLER_BETTER = ['nRMSE', 'init_nrmse', 'absVR_minus_1', 'Hs_error', 'amp_rel_error',
                  'decorr', 'shrink', 'm']
LARGER_BETTER = ['Pearson_rho', 'Skill_Score']
CELL_ORDER = ['X-BASE', 'K1', 'K2', 'K12', 'K3', 'K4', 'K5',
              'K6', 'K7', 'K12-S1', 'X-NOTRACK',
              'D-BASE', 'D-K12', 'D-K6', 'D-NOTRACK']
ORACLE_CELLS = ['K1', 'K2', 'K12', 'K3', 'K4', 'K5', 'K6', 'K7', 'K12-S1', 'X-NOTRACK']
DEPLOY_CELLS = ['D-K12', 'D-K6', 'D-NOTRACK']
REPORT = ['nRMSE', 'Pearson_rho', 'decorr', 'shrink', 'm', 'absVR_minus_1',
          'Hs_error', 'amp_rel_error', 'Skill_Score', 'init_nrmse',
          'Variance_Ratio', 'sum_w_ratio']
LEVEL_COLS = ['nRMSE', 'Pearson_rho', 'decorr', 'shrink', 'm', 'Variance_Ratio',
              'absVR_minus_1', 'Hs_error', 'amp_rel_error', 'Skill_Score',
              'init_nrmse', 'sum_w_ratio']


def binom_two_sided(k, n):
    if n == 0:
        return float('nan')
    from math import comb
    kk = min(k, n - k)
    return min(1.0, sum(comb(n, i) for i in range(0, kk + 1)) / 2 ** n * 2)


def load(paths):
    seen = {}
    for p in paths:
        if not os.path.exists(p):
            continue
        for line in open(p):
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if 'error' in r or 'cell' not in r:
                continue
            seen[r['job_id']] = r          # last wins
    return list(seen.values())


def enrich(r):
    spec = r.get('spec', {}) or {}
    r['amp_rel_error'] = spec.get('amp_rel_error_mean', float('nan'))
    sw, sg = spec.get('sum_w_est'), spec.get('sum_w_gt')
    r['sum_w_ratio'] = (sw / sg) if (sw is not None and sg) else float('nan')
    if 'absVR1_framemean' in r:            # AMENDMENT 2 path
        r['absVR_minus_1'] = r['absVR1_framemean']
        r['decorr'] = r['decorr_framemean']
        r['shrink'] = r['shrink_framemean']
        r['m'] = r['m_framemean']
        r['_perframe'] = True
    else:                                   # legacy pass-1 fallback (flagged in output)
        vr, rho = r.get('Variance_Ratio', np.nan), r.get('Pearson_rho', np.nan)
        s = math.sqrt(max(vr, 0)) if np.isfinite(vr) else np.nan
        r['absVR_minus_1'] = abs(vr - 1.0)
        r['decorr'] = 1 - rho ** 2
        r['shrink'] = (s - rho) ** 2
        r['m'] = 1 - 2 * rho * s + vr
        r['_perframe'] = False
    return r


def main():
    paths = sys.argv[1:] or [os.path.join(HERE, 'results2.jsonl'),
                             os.path.join(HERE, 'results.jsonl')]
    rows = [enrich(r) for r in load(paths)]
    pf = sum(1 for r in rows if r['_perframe'])
    print(f'rows={len(rows)}   per-frame |VR-1| available on {pf}/{len(rows)}'
          f'{"   *** SOME ROWS ARE MEAN-THEN-TRANSFORM ***" if pf < len(rows) else ""}')
    cells = defaultdict(lambda: defaultdict(list))
    for r in rows:
        cells[r['cell']][r['episode']].append(r)
    eps = sorted({r['episode'] for r in rows})
    for c in [x for x in CELL_ORDER if x in cells]:
        print(f'  {c:<8s} seeds/ep ' +
              str({e.split("_")[1] + "_" + e.split("_")[2]: len(v)
                   for e, v in sorted(cells[c].items())}))
    print()

    med = defaultdict(dict)
    for c, byep in cells.items():
        for e, rs in byep.items():
            med[c][e] = {m: float(np.nanmedian([r.get(m, np.nan) for r in rs]))
                         for m in LEVEL_COLS}

    # ---- levels ----
    print('=== LEVELS (median over 3 seeds, then median over the 4 episodes) ===')
    print('  ' + f'{"cell":>7s}' + ''.join(f'{h:>10s}' for h in
          ['nRMSE', 'rho', 'decorr', 'shrink', 'm', 'VR', '|VR-1|', 'Hs_err',
           'amp_err', 'skill', 'init', 'sumW/GT']))
    for c in [x for x in CELL_ORDER if x in med]:
        vals = [float(np.median([med[c][e][m] for e in eps if e in med[c]]))
                for m in LEVEL_COLS]
        print(f'  {c:>7s}' + ''.join(f'{v:10.4f}' for v in vals))
    print()

    out = {}

    def contrasts(cell_list, ref_name, title):
        ref = med.get(ref_name)
        if ref is None:
            return
        print(f'\n=== {title} (paired per-episode, median-of-3 each side) ===')
        for c in [x for x in cell_list if x in med]:
            out[c] = {'_ref': ref_name}
            print(f'\n  --- {c}  vs {ref_name} ---')
            for m in REPORT:
                per = [(e, med[c][e][m] - ref[e][m]) for e in eps
                       if e in med[c] and e in ref
                       and np.isfinite(med[c][e][m] - ref[e][m])]
                if not per:
                    continue
                d = [x for _, x in per]
                n = len(d)
                if m in SMALLER_BETTER:
                    k = sum(1 for x in d if x < 0)
                elif m in LARGER_BETTER:
                    k = sum(1 for x in d if x > 0)
                else:
                    k = sum(1 for x in d if x != 0)
                p = binom_two_sided(k, n)
                note = '' if m in SMALLER_BETTER + LARGER_BETTER else ' (no direction)'
                print(f'    {m:<15s} medD {np.median(d):+9.4f}  better {k}/{n} '
                      f'p={p:.4f}{note}   per-ep ' + ' '.join(f'{x:+.4f}' for x in d))
                out[c][m] = {'median_delta': float(np.median(d)), 'k': k, 'n': n, 'p': p,
                             'per_ep': {e: float(x) for e, x in per}}

    base = med.get('X-BASE')
    contrasts(ORACLE_CELLS, 'X-BASE', 'ORACLE OPERATING POINT: deltas vs X-BASE')
    contrasts(DEPLOY_CELLS, 'D-BASE', 'DEPLOYMENT OPERATING POINT: deltas vs D-BASE')

    # ---- harm-removed fraction, both operating points ----
    print('\n=== FRACTION OF TRACKER HARM REMOVED  (cell - BASE) / (NOTRACK - BASE) ===')
    print('  100% = the knob recovers exactly what freezing the tracker recovers;')
    print('  >100% = better than freezing the tracker.')
    for ref_name, floor_name, group in (('X-BASE', 'X-NOTRACK', ORACLE_CELLS),
                                        ('D-BASE', 'D-NOTRACK', DEPLOY_CELLS)):
        if ref_name not in med or floor_name not in med:
            continue
        harm = {e: med[ref_name][e]['nRMSE'] - med[floor_name][e]['nRMSE']
                for e in eps if e in med[ref_name] and e in med[floor_name]}
        print(f'  [{ref_name}] tracker harm per episode: ' +
              ' '.join(f'{harm[e]:+.4f}' for e in eps if e in harm) +
              f'   median {np.median(list(harm.values())):+.4f}')
        for c in [x for x in group if x in med]:
            fr = [100 * (med[ref_name][e]['nRMSE'] - med[c][e]['nRMSE']) / harm[e]
                  for e in eps if e in med[c] and e in harm and abs(harm[e]) > 1e-9]
            if fr:
                print(f'    {c:>10s} recovered {np.median(fr):7.1f}%   per-ep ' +
                      ' '.join(f'{x:6.1f}' for x in fr))

    # ---- 2x2 factorial (AMENDMENT 2) ----
    print('\n=== 2x2 FACTORIAL  lambda_l1 x lambda_smooth  (AMENDMENT 2) ===')
    print('  cells: X-BASE=(on,on)  K1=(off,on)  K2=(on,off)  K12=(off,off)')
    if all(x in med for x in ('X-BASE', 'K1', 'K2', 'K12')):
        for m in ['nRMSE', 'Pearson_rho', 'decorr', 'shrink', 'absVR_minus_1',
                  'init_nrmse']:
            mainA, mainB, inter, simpleB_on_A = [], [], [], []
            for e in eps:
                if not all(e in med[c] for c in ('X-BASE', 'K1', 'K2', 'K12')):
                    continue
                b, a1, a2, a12 = (med[c][e][m] for c in ('X-BASE', 'K1', 'K2', 'K12'))
                mainA.append(((a1 - b) + (a12 - a2)) / 2)     # effect of l1->0
                mainB.append(((a2 - b) + (a12 - a1)) / 2)     # effect of lsmooth->0
                inter.append((a12 - a2) - (a1 - b))
                simpleB_on_A.append(a12 - a1)                 # lsmooth->0 given l1=0
            if not mainA:
                continue
            kA = sum(1 for x in mainA if x < 0) if m in SMALLER_BETTER \
                else sum(1 for x in mainA if x > 0)
            kB = sum(1 for x in mainB if x < 0) if m in SMALLER_BETTER \
                else sum(1 for x in mainB if x > 0)
            n = len(mainA)
            print(f'  {m:<15s} main(l1) {np.median(mainA):+8.4f} {kA}/{n} '
                  f'p={binom_two_sided(kA, n):.4f} | main(lsm) {np.median(mainB):+8.4f} '
                  f'{kB}/{n} p={binom_two_sided(kB, n):.4f} | interaction '
                  f'{np.median(inter):+8.4f} | lsm|l1=0 {np.median(simpleB_on_A):+8.4f}')
            out.setdefault('_2x2', {})[m] = {
                'main_l1': float(np.median(mainA)), 'k_l1': kA,
                'main_lsmooth': float(np.median(mainB)), 'k_lsmooth': kB,
                'interaction': float(np.median(inter)),
                'simple_lsmooth_given_l1_0': float(np.median(simpleB_on_A)), 'n': n}

    # ---- 2x2 in (lambda, displacement budget): the ATTRIBUTABILITY test (AMENDMENT 3) ----
    print('\n=== ATTRIBUTABILITY 2x2:  lambda(on/off) x tracking_steps(10/1) ===')
    print('  X-BASE=(lam,10)  K12=(0,10)  K6=(lam,1)  K12-S1=(0,1)')
    print('  If zeroing lambda and shortening the trajectory are the SAME channel, the')
    print('  double cell buys nothing beyond the better single cell (interaction ~ +|main|).')
    if all(x in med for x in ('X-BASE', 'K12', 'K6', 'K12-S1')):
        for m in ['nRMSE', 'Pearson_rho', 'decorr', 'shrink']:
            mA, mB, inter, dbl = [], [], [], []
            for e in eps:
                if not all(e in med[c] for c in ('X-BASE', 'K12', 'K6', 'K12-S1')):
                    continue
                b, a, s, ab = (med[c][e][m] for c in ('X-BASE', 'K12', 'K6', 'K12-S1'))
                mA.append(a - b); mB.append(s - b)
                inter.append((ab - s) - (a - b))
                dbl.append(ab - b)
            if not mA:
                continue
            n = len(mA)
            print(f'  {m:<12s} lam->0 {np.median(mA):+8.4f} | steps->1 {np.median(mB):+8.4f}'
                  f' | both {np.median(dbl):+8.4f} | interaction {np.median(inter):+8.4f}'
                  f' | additive-prediction {np.median(mA)+np.median(mB):+8.4f}')
            out.setdefault('_2x2_attrib', {})[m] = {
                'lam_off': float(np.median(mA)), 'steps1': float(np.median(mB)),
                'both': float(np.median(dbl)), 'interaction': float(np.median(inter)),
                'n': n}
    # budget-matched pair: K6 (steps 10->1) vs K7 (lr 0.005->0.0005), same steps*lr
    if 'K6' in med and 'K7' in med:
        d = [med['K6'][e]['nRMSE'] - med['K7'][e]['nRMSE'] for e in eps
             if e in med['K6'] and e in med['K7']]
        if d:
            print(f'  budget-matched K6 (steps=1) vs K7 (lr x0.1): medD {np.median(d):+.4f}'
                  f'   per-ep ' + ' '.join(f'{x:+.4f}' for x in d))

    # ---- mechanism: which axis carries the change ----
    print('\n=== MECHANISM: m = (1-rho^2) + (sqrt(v)-rho)^2, per frame then averaged ===')
    print('  AMENDMENT 2: the rho channel is PRIMARY. shrink% = share of the LEVEL that a')
    print('  perfect single-scalar rescale could remove.')
    for c in [x for x in CELL_ORDER if x in med]:
        ref = med.get('D-BASE' if c.startswith('D-') else 'X-BASE')
        lm = float(np.median([med[c][e]['m'] for e in eps if e in med[c]]))
        ld = float(np.median([med[c][e]['decorr'] for e in eps if e in med[c]]))
        ls = float(np.median([med[c][e]['shrink'] for e in eps if e in med[c]]))
        line = f'  {c:>10s}  m {lm:7.4f} = decorr {ld:7.4f} + shrink {ls:7.4f}   ' \
               f'shrink share {100 * ls / lm:5.1f}%'
        if c not in ('X-BASE', 'D-BASE') and ref is not None:
            dm = [med[c][e]['m'] - ref[e]['m'] for e in eps if e in med[c] and e in ref]
            dd = [med[c][e]['decorr'] - ref[e]['decorr'] for e in eps
                  if e in med[c] and e in ref]
            ds = [med[c][e]['shrink'] - ref[e]['shrink'] for e in eps
                  if e in med[c] and e in ref]
            tot = np.median(dm)
            line += (f'   | d_m {tot:+7.4f} = d_decorr {np.median(dd):+7.4f} '
                     f'({100 * np.median(dd) / tot if abs(tot) > 1e-9 else float("nan"):5.1f}%)'
                     f' + d_shrink {np.median(ds):+7.4f} '
                     f'({100 * np.median(ds) / tot if abs(tot) > 1e-9 else float("nan"):5.1f}%)')
        print(line)

    # ---- seed noise / MDE ----
    print('\n=== seed noise (this run) ===')
    sp = [np.std([r['nRMSE'] for r in rs], ddof=1)
          for byep in cells.values() for rs in byep.values() if len(rs) >= 2]
    if sp:
        print(f'  per-(cell,ep) seed sd of nRMSE: median {np.median(sp):.4f} '
              f'max {np.max(sp):.4f} over {len(sp)} cells')
        print(f'  s.e. of a median-of-3 ~ {np.median(sp)/math.sqrt(3):.4f}; '
              f'paired-delta noise ~ {np.median(sp)*math.sqrt(2/3):.4f}')
    print('  AMENDMENT 2: MDE(80%) = 0.011 (pairing held) to 0.030 (Phase-1 contamination);')
    print('  reachable two-sided p at n=4 are {0.125, 0.625, 1.0} -> MDE at alpha=0.05 is +inf.')
    print('  Global-null P(some spurious 4/4) = 0.865 at 5 knobs x m=3 -> NO knob ranking.')

    # ---- reproduction control ----
    print('\n=== X-BASE reproduction control (AMENDMENT 2 hygiene) ===')
    for f in ('bitid_post.jsonl', 'bitid_pre.jsonl'):
        p = os.path.join(HERE, f)
        if not os.path.exists(p):
            continue
        ref = {json.loads(l)['job_id']: json.loads(l) for l in open(p)}
        cur = {r['job_id']: r for r in rows}
        for j, a in ref.items():
            if j in cur:
                d = abs(a['nRMSE'] - cur[j]['nRMSE'])
                print(f'  {f:<18s} {j:<52s} |d nRMSE| {d:.2e} '
                      f'{"IDENTICAL" if d == 0 else "DIFFERS"}')

    with open(os.path.join(HERE, 'rx1_summary.json'), 'w') as fh:
        json.dump({'levels': {c: med[c] for c in med}, 'contrasts': out}, fh, indent=1)
    print('\nwrote rx1_summary.json')


if __name__ == '__main__':
    main()
