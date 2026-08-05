"""NEUTRAL — do E1 / S1 / S2 mean the same thing by "placement got better"?
Plus: can "stacking" be identified as a mediator from these data?

Re-analysis only. Inputs:
  tmp/e1_grid_2026-08-05/{results.jsonl, probe.jsonl, survey_k.json}
  tmp/s1s2_2026-08-05/results.jsonl
"""
import json
import os
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..'))
E1 = os.path.join(REPO, 'tmp', 'e1_grid_2026-08-05')
S12 = os.path.join(REPO, 'tmp', 's1s2_2026-08-05')
CROWDED = ['episode_BF4_15comp_seed0_uniform.h5', 'episode_BF4_30comp_seed0_uniform.h5',
           'episode_BF5_15comp_seed0_uniform.h5', 'episode_BF5_30comp_seed0_uniform.h5']


def med3(rows, key, getter=None):
    v = [(getter(r) if getter else r.get(key)) for r in rows]
    v = [x for x in v if x is not None and np.isfinite(x)]
    return float(np.median(v)) if v else float('nan')


def signtest(d):
    """two-sided exact sign test, zeros dropped"""
    from math import comb
    d = [x for x in d if x != 0]
    n = len(d)
    if n == 0:
        return 0, 0, 1.0
    k = sum(1 for x in d if x < 0)
    p = sum(comb(n, i) for i in range(0, min(k, n - k) + 1)) / 2 ** n * 2
    return k, n, min(p, 1.0)


def main():
    out = {}
    e1 = [json.loads(l) for l in open(os.path.join(E1, 'results.jsonl'))]
    probe = [json.loads(l) for l in open(os.path.join(E1, 'probe.jsonl'))]
    s12 = [json.loads(l) for l in open(os.path.join(S12, 'results.jsonl'))]
    survey = {r['ep']: r for r in json.load(open(os.path.join(E1, 'survey_k.json')))['rows']}

    # ---------------------------------------------------------------- 1
    # S1: is the detection gain a BUDGET-CEILING move?
    print('=== 1. S1 — detection_rate vs the ceiling min(n_est, n_gt)/n_gt ===')
    print('(n_est = components surviving to the END of the run; spec is computed on the '
          'FINAL spectrum_params, not on the initialization)')
    s1rows = defaultdict(list)
    for r in s12:
        if r.get('arm', '').startswith('S1') and 'spec' in r:
            s1rows[(r['arm'], r['episode'])].append(r)
    print(f'{"arm":<8s} {"episode":<14s} {"n_gt":>4s} {"n_est":>5s} {"ceiling":>7s} '
          f'{"det":>6s} {"det/ceil":>8s} {"nRMSE":>7s}')
    s1tab = []
    for arm in ['S1-A0', 'S1-A1', 'S1-A2', 'S1-A3']:
        for ep in CROWDED:
            rows = s1rows[(arm, ep)]
            if not rows:
                continue
            n_gt = rows[0]['spec']['n_gt']
            n_est = med3(rows, None, lambda r: r['spec']['n_est'])
            det = med3(rows, None, lambda r: r['spec']['detection_rate'])
            nr = med3(rows, 'nRMSE')
            ceil = min(n_est, n_gt) / n_gt
            s1tab.append(dict(arm=arm, ep=ep, n_gt=n_gt, n_est=n_est, ceiling=ceil,
                              det=det, det_over_ceiling=det / ceil if ceil else None,
                              nRMSE=nr))
            print(f'{arm:<8s} {ep[8:-14]:<14s} {n_gt:>4d} {n_est:>5.0f} {ceil:>7.3f} '
                  f'{det:>6.3f} {det / ceil:>8.3f} {nr:>7.4f}')
    out['S1_ceiling'] = s1tab
    d_ceil = [r['ceiling'] for r in s1tab if r['arm'] == 'S1-A1']
    a0_ceil = [r['ceiling'] for r in s1tab if r['arm'] == 'S1-A0']
    a0_det = [r['det'] for r in s1tab if r['arm'] == 'S1-A0']
    a1_det = [r['det'] for r in s1tab if r['arm'] == 'S1-A1']
    print(f'  A0 ceiling mean {np.mean(a0_ceil):.3f} -> A1 ceiling mean {np.mean(d_ceil):.3f} '
          f'(delta {np.mean(d_ceil) - np.mean(a0_ceil):+.3f})')
    print(f'  A0 det     mean {np.mean(a0_det):.3f} -> A1 det     mean {np.mean(a1_det):.3f} '
          f'(delta {np.mean(a1_det) - np.mean(a0_det):+.3f})')
    print(f'  share of the detection gain explainable by the ceiling move alone: '
          f'{(np.mean(d_ceil) - np.mean(a0_ceil)) / (np.mean(a1_det) - np.mean(a0_det)) * 100:.0f}%')

    # ---------------------------------------------------------------- 2
    # matched distance vs the true component spacing (a resolution criterion the
    # 0.5*k_scale threshold does not impose)
    print('\n=== 2. is a "detected" component RESOLVED? '
          'mean matched distance vs half the true spacing ===')
    print(f'{"cell":<5s} {"episode":<14s} {"mean|dk|":>9s} {"GTnn/2":>8s} {"ratio":>7s} '
          f'{"det":>6s}   resolved?')
    res = []
    for cell in 'ABCDE':
        for ep in CROWDED:
            p = [r for r in probe if r['cell'] == cell and r['episode'] == ep and r['n_g'] == 15]
            if not p:
                continue
            p = p[0]
            s = survey[ep]
            k_scale = s['kmag_max']
            dk = p['match_cost_mean'] * k_scale
            half = s['nn_med'] / 2
            res.append(dict(cell=cell, ep=ep, mean_matched_dk=dk, gt_nn_half=half,
                            ratio=dk / half, det=p['detection_rate']))
            print(f'{cell:<5s} {ep[8:-14]:<14s} {dk:>9.4f} {half:>8.4f} {dk / half:>7.2f} '
                  f'{p["detection_rate"]:>6.3f}   {"NO" if dk > half else "yes"}')
    out['resolution_check'] = res
    for cell in 'ABCDE':
        sub = [r for r in res if r['cell'] == cell]
        print(f'  cell {cell}: {sum(1 for r in sub if r["ratio"] > 1)}/{len(sub)} episodes where '
              f'the mean matched atom sits FARTHER than half the true spacing')

    # ---------------------------------------------------------------- 3
    # mediation: is spacing separable from the intervention?
    print('\n=== 3. mediator identification — spacing vs the two harm metrics ===')
    e1med = defaultdict(dict)
    for cell in 'ABCDE':
        for ep in sorted({r['episode'] for r in e1}):
            rows = [r for r in e1 if r['cell'] == cell and r['episode'] == ep and 'error' not in r]
            if not rows:
                continue
            e1med[cell][ep] = dict(
                Hs_error=med3(rows, 'Hs_error'),
                absVR=med3(rows, None, lambda r: abs(r['Variance_Ratio'] - 1)),
                init_nrmse=med3(rows, 'init_nrmse'),
                nRMSE=med3(rows, 'nRMSE'))
    pr = {(r['cell'], r['episode'], r['n_g']): r for r in probe}
    eps = sorted(e1med['A'])
    rows_m = []
    for cell in 'BCDE':
        for ep in eps:
            for n_g in (3, 15):
                a, d = pr.get(('A', ep, n_g)), pr.get((cell, ep, n_g))
                if not a or not d:
                    continue
                rows_m.append(dict(
                    cell=cell, ep=ep, n_g=n_g,
                    d_spacing=d['placed_nn_median'] - a['placed_nn_median'],
                    d_outside=d['n_outside_narrow_box'] - a['n_outside_narrow_box'],
                    d_pairs002=d['n_placed_pairs_lt_002'] - a['n_placed_pairs_lt_002'],
                    d_Hs=e1med[cell][ep]['Hs_error'] - e1med['A'][ep]['Hs_error'],
                    d_absVR=e1med[cell][ep]['absVR'] - e1med['A'][ep]['absVR'],
                    d_init=e1med[cell][ep]['init_nrmse'] - e1med['A'][ep]['init_nrmse']))
    out['mediation_rows'] = rows_m
    for n_g in (3, 15):
        sub = [r for r in rows_m if r['n_g'] == n_g]
        print(f'--- probe budget n_g={n_g}  (n={len(sub)} cell x episode pairs vs A) ---')
        for xk in ('d_spacing', 'd_outside', 'd_pairs002'):
            x = np.array([r[xk] for r in sub])
            for yk in ('d_Hs', 'd_absVR'):
                y = np.array([r[yk] for r in sub])
                m = np.isfinite(x) & np.isfinite(y)
                rho = np.corrcoef(x[m], y[m])[0, 1] if m.sum() > 2 and x[m].std() > 0 else np.nan
                print(f'   corr({xk:<11s}, {yk:<8s}) = {rho:+.3f}')
        # per-cell means: is the mediator moved without the harm anywhere?
        print(f'   {"cell":<5s} {"mean d_spacing":>15s} {"mean d_outside":>15s} '
              f'{"mean d_Hs":>10s} {"mean d_absVR":>13s}')
        for cell in 'BCDE':
            s = [r for r in sub if r['cell'] == cell]
            print(f'   {cell:<5s} {np.nanmean([r["d_spacing"] for r in s]):>15.4f} '
                  f'{np.nanmean([r["d_outside"] for r in s]):>15.2f} '
                  f'{np.nanmean([r["d_Hs"] for r in s]):>10.4f} '
                  f'{np.nanmean([r["d_absVR"] for r in s]):>13.4f}')

    # cell C = the natural "mediator moved, treatment differs" contrast
    print('\n   --- C vs D: same direction on the mediator, opposite on the harm? ---')
    for n_g in (3, 15):
        for cell in 'CD':
            s = [r for r in rows_m if r['cell'] == cell and r['n_g'] == n_g]
            sp = [r['d_spacing'] for r in s]
            hs = [r['d_Hs'] for r in s]
            k, n, p = signtest(hs)
            print(f'   n_g={n_g} cell {cell}: d_spacing median {np.nanmedian(sp):+.4f} '
                  f'(worse-spacing in {sum(1 for v in sp if v < 0)}/{len(sp)} ep) | '
                  f'd_Hs median {np.nanmedian(hs):+.4f}  harm in {n - k}/{n} (sign p={p:.4f})')

    # ---------------------------------------------------------------- 4
    # does the e2e's ACTUAL init budget (=3) show the placement change at all?
    print('\n=== 4. what the deployed ladder actually calls: n_gaussians=3 at this site ===')
    print(f'{"episode":<16s} {"detA":>6s} {"detD":>6s} {"D-A":>6s} | '
          f'{"spacingA":>9s} {"spacingD":>9s} | {"outA":>5s} {"outD":>5s}')
    n3 = []
    for ep in eps:
        a, d = pr[('A', ep, 3)], pr[('D', ep, 3)]
        n3.append(dict(ep=ep, detA=a['detection_rate'], detD=d['detection_rate'],
                       spA=a['placed_nn_median'], spD=d['placed_nn_median'],
                       outA=a['n_outside_narrow_box'], outD=d['n_outside_narrow_box']))
        print(f'{ep[8:-14]:<16s} {a["detection_rate"]:>6.3f} {d["detection_rate"]:>6.3f} '
              f'{d["detection_rate"] - a["detection_rate"]:>+6.3f} | '
              f'{a["placed_nn_median"]:>9.4f} {d["placed_nn_median"]:>9.4f} | '
              f'{a["n_outside_narrow_box"]:>5d} {d["n_outside_narrow_box"]:>5d}')
    out['budget3'] = n3
    dd = [r['detD'] - r['detA'] for r in n3]
    print(f'  D-A detection at the deployed budget: median {np.median(dd):+.4f}, '
          f'nonzero in {sum(1 for v in dd if v != 0)}/12 episodes, max {max(dd):+.3f}')

    # ---------------------------------------------------------------- 5
    # S2: what exactly was "improved placement" there?
    print('\n=== 5. S2 — the placement metrics of the oracle arm ===')
    for arm in ['S1-A0', 'S2-C1', 'S2-T1', 'S2-T1f', 'X-T1f-notrack']:
        rows = [r for r in s12 if r.get('arm') == arm and 'spec' in r]
        if not rows:
            continue
        det = np.median([r['spec']['detection_rate'] for r in rows])
        kerr = np.median([r['spec'].get('k_mag_rel_error_mean', np.nan) for r in rows])
        nest = np.median([r['spec']['n_est'] for r in rows])
        nr = np.median([r['nRMSE'] for r in rows])
        ini = np.median([r['init_nrmse'] for r in rows])
        print(f'  {arm:<15s} n_est {nest:>4.0f}  det {det:.3f}  k_mag_rel_err {kerr:.4f}  '
              f'init_nrmse {ini:.4f}  nRMSE {nr:.4f}')

    with open(os.path.join(HERE, 'NEUTRAL_three_definitions.json'), 'w') as f:
        json.dump(out, f, indent=1)


if __name__ == '__main__':
    main()
