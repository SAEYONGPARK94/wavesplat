"""NEUTRAL — is `detection_rate` a measure of PLACEMENT QUALITY, or of BOX COVERAGE?

Re-analysis only. No estimator is run. Inputs:
  * tmp/e1_grid_2026-08-05/probe.jsonl   (180 deterministic placement probes)
  * tmp/e1_grid_2026-08-05/survey_k.json (ground-truth component geometry)
  * data_finemesh_2026-07-23/*.h5        (config/wave_components table only, tiny read)

The matcher under audit is `run_e1.match_components`, verbatim:
    k_scale = max|k_gt| ; cost_ij = |k_est_i - k_gt_j| / k_scale ; accept if cost < 0.5
    Hungarian one-to-one, detection_rate = n_matched / n_gt

Three degenerate placements are scored with THAT SAME matcher:
  NULL-STACK1 : every atom at the GT centroid (zero resolution, maximal stacking)
  NULL-UNIF-N : atoms uniform at random in the NARROW box (-0.8,0.8)^2
  NULL-UNIF-W : atoms uniform at random in the WIDE   box (-4,4)^2
  NULL-STACK3 : atoms stacked 3-deep on the ceil(n/3) strongest GT components
"""
import json
import os
import sys

import numpy as np
from scipy.optimize import linear_sum_assignment

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..'))
E1 = os.path.join(REPO, 'tmp', 'e1_grid_2026-08-05')
DATA = os.path.join(REPO, 'data_finemesh_2026-07-23')
RNG = np.random.default_rng(20260805)
NDRAW = 2000


def match(est_kx, est_ky, gt_kx, gt_ky):
    """verbatim cost/threshold of run_e1.match_components (amplitudes unused there)"""
    if len(est_kx) == 0 or len(gt_kx) == 0:
        return 0
    k_scale = max(np.max(np.hypot(gt_kx, gt_ky)), 0.01)
    cost = np.hypot(est_kx[:, None] - gt_kx[None, :],
                    est_ky[:, None] - gt_ky[None, :]) / k_scale
    r, c = linear_sum_assignment(cost)
    return int(np.sum(cost[r, c] < 0.5))


def main():
    import h5py
    probe = [json.loads(l) for l in open(os.path.join(E1, 'probe.jsonl'))]
    survey = {r['ep']: r for r in json.load(open(os.path.join(E1, 'survey_k.json')))['rows']}
    eps = sorted(survey)

    gt = {}
    for ep in eps:
        with h5py.File(os.path.join(DATA, ep), 'r') as f:
            w = np.asarray(f['config/wave_components'])
        gt[ep] = dict(A=w[:, 0], kx=w[:, 2], ky=w[:, 3])

    out = {'geometry': [], 'ceiling': [], 'nulls': [], 'cost_margin': []}

    # ---------- 1. matcher acceptance radius vs true component spacing ----------
    for ep in eps:
        g = gt[ep]
        k_scale = max(np.max(np.hypot(g['kx'], g['ky'])), 0.01)
        radius = 0.5 * k_scale
        s = survey[ep]
        nn = s['nn_med']
        out['geometry'].append(dict(
            ep=ep, n_gt=s['N'], k_scale=k_scale, accept_radius_radm=radius,
            gt_nn_median=nn, gt_nn_min=s['nn_min'],
            radius_over_nn=(radius / nn) if nn else None,
            # how many GT components sit inside one acceptance disc, on average
            mean_gt_within_radius=float(np.mean([
                np.sum(np.hypot(g['kx'] - g['kx'][j], g['ky'] - g['ky'][j]) < radius)
                for j in range(len(g['kx']))])),
            box_halfwidth_narrow=0.8))

    # ---------- 2. is the observed detection_rate at the combinatorial ceiling? ----------
    for r in probe:
        ceil = min(r['n_g'], r['n_gt']) / r['n_gt']
        out['ceiling'].append(dict(
            cell=r['cell'], ep=r['episode'], n_g=r['n_g'], n_gt=r['n_gt'],
            det=r['detection_rate'], ceiling=ceil,
            at_ceiling=abs(r['detection_rate'] - ceil) < 1e-9,
            gap=ceil - r['detection_rate'],
            match_cost_mean=r.get('match_cost_mean'),
            nn_med=r.get('placed_nn_median'),
            n_outside_box=r.get('n_outside_narrow_box'),
            n_pairs_lt_002=r.get('n_placed_pairs_lt_002')))

    # ---------- 3. degenerate placements scored by the SAME matcher ----------
    for ep in eps:
        g = gt[ep]
        gkx, gky, gA = g['kx'], g['ky'], g['A']
        n_gt = len(gkx)
        for n_est in (3, 15, 30):
            cx, cy = float(np.mean(gkx)), float(np.mean(gky))
            d_stack1 = match(np.full(n_est, cx), np.full(n_est, cy), gkx, gky) / n_gt

            # stack 3 atoms on each of the strongest ceil(n/3) GT components
            order = np.argsort(-gA)
            n_sites = int(np.ceil(n_est / 3))
            sites = order[:min(n_sites, n_gt)]
            ekx = np.concatenate([np.full(3, gkx[s]) for s in sites])[:n_est]
            eky = np.concatenate([np.full(3, gky[s]) for s in sites])[:n_est]
            d_stack3 = match(ekx, eky, gkx, gky) / n_gt

            dn, dw = [], []
            for _ in range(NDRAW):
                dn.append(match(RNG.uniform(-0.8, 0.8, n_est),
                                RNG.uniform(-0.8, 0.8, n_est), gkx, gky) / n_gt)
                dw.append(match(RNG.uniform(-4.0, 4.0, n_est),
                                RNG.uniform(-4.0, 4.0, n_est), gkx, gky) / n_gt)
            obs = {r['cell']: r['detection_rate'] for r in probe
                   if r['episode'] == ep and r['n_g'] == n_est}
            out['nulls'].append(dict(
                ep=ep, n_gt=n_gt, n_est=n_est,
                ceiling=min(n_est, n_gt) / n_gt,
                null_stack1_centroid=d_stack1,
                null_stack3_on_strongest=d_stack3,
                null_uniform_narrow_mean=float(np.mean(dn)),
                null_uniform_narrow_p05=float(np.percentile(dn, 5)),
                null_uniform_narrow_p95=float(np.percentile(dn, 95)),
                null_uniform_wide_mean=float(np.mean(dw)),
                obs_A=obs.get('A'), obs_B=obs.get('B'),
                obs_C=obs.get('C'), obs_D=obs.get('D'), obs_E=obs.get('E')))

    # ---------- 4. how much slack do the accepted matches have? ----------
    for r in probe:
        if r.get('match_cost_mean') is not None and np.isfinite(r.get('match_cost_mean', np.nan)):
            out['cost_margin'].append(dict(
                cell=r['cell'], ep=r['episode'], n_g=r['n_g'],
                match_cost_mean=r['match_cost_mean'],
                threshold=0.5,
                frac_of_threshold=r['match_cost_mean'] / 0.5))

    with open(os.path.join(HERE, 'NEUTRAL_metric_audit.json'), 'w') as f:
        json.dump(out, f, indent=1)

    # ---------- console report ----------
    print('=== 1. matcher acceptance radius vs GT spacing ===')
    print(f'{"episode":<38s} {"n_gt":>4s} {"k_scale":>8s} {"radius":>8s} '
          f'{"GTnn_med":>9s} {"R/nn":>6s} {"#GT in disc":>11s}')
    for r in out['geometry']:
        print(f'{r["ep"][8:-3]:<38s} {r["n_gt"]:>4d} {r["k_scale"]:>8.3f} '
              f'{r["accept_radius_radm"]:>8.3f} '
              f'{(r["gt_nn_median"] if r["gt_nn_median"] else float("nan")):>9.4f} '
              f'{(r["radius_over_nn"] if r["radius_over_nn"] else float("nan")):>6.1f} '
              f'{r["mean_gt_within_radius"]:>11.1f}')

    print('\n=== 2. detection_rate vs combinatorial ceiling min(n_est,n_gt)/n_gt ===')
    for n_g in (3, 15, 30):
        print(f'--- budget n_gaussians={n_g} ---')
        print(f'{"episode":<26s} {"ceil":>5s} ' + ' '.join(f'{c:>17s}' for c in 'ABCDE'))
        for ep in eps:
            rows = {r['cell']: r for r in out['ceiling'] if r['ep'] == ep and r['n_g'] == n_g}
            if not rows:
                continue
            ceil = list(rows.values())[0]['ceiling']
            cells = ' '.join(
                f'{rows[c]["det"]:>10.3f}{"*" if rows[c]["at_ceiling"] else " "}      '
                for c in 'ABCDE')
            print(f'{ep[8:-3]:<26s} {ceil:>5.2f} {cells}')
    n_at = sum(1 for r in out['ceiling'] if r['at_ceiling'])
    print(f'(* = exactly at ceiling)   {n_at}/{len(out["ceiling"])} probes at ceiling')
    for c in 'ABCDE':
        sub = [r for r in out['ceiling'] if r['cell'] == c]
        print(f'   cell {c}: {sum(1 for r in sub if r["at_ceiling"])}/{len(sub)} at ceiling')

    print('\n=== 3. degenerate placements scored by the SAME matcher ===')
    print(f'{"episode":<20s} {"n_est":>5s} {"ceil":>5s} {"stack1":>7s} {"stack3":>7s} '
          f'{"unifN":>7s} {"unifW":>7s} | {"obsA":>6s} {"obsD":>6s}')
    for r in out['nulls']:
        print(f'{r["ep"][8:-14]:<20s} {r["n_est"]:>5d} {r["ceiling"]:>5.2f} '
              f'{r["null_stack1_centroid"]:>7.3f} {r["null_stack3_on_strongest"]:>7.3f} '
              f'{r["null_uniform_narrow_mean"]:>7.3f} {r["null_uniform_wide_mean"]:>7.3f} | '
              f'{(r["obs_A"] if r["obs_A"] is not None else float("nan")):>6.3f} '
              f'{(r["obs_D"] if r["obs_D"] is not None else float("nan")):>6.3f}')

    print('\n=== 4. accepted-match slack (cost/threshold; 1.0 = at the accept boundary) ===')
    for c in 'ABCDE':
        for n_g in (3, 15):
            sub = [r['frac_of_threshold'] for r in out['cost_margin']
                   if r['cell'] == c and r['n_g'] == n_g]
            if sub:
                print(f'  cell {c} n_g={n_g:<3d} median {np.median(sub):.3f} '
                      f'max {np.max(sub):.3f}  (n={len(sub)})')


if __name__ == '__main__':
    sys.exit(main())
