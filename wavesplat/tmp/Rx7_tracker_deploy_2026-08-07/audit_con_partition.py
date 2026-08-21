"""CON audit — (a) does the observed/unobserved partition actually reconstruct the
total RMSE on REAL geometry?  (b) how does the new block compare with Rx-6's block?

Read-only.  No estimator, no GPU.
"""
import glob, json, os, sys
import statistics as st
import numpy as np
import h5py

sys.path.insert(0, '/home/vexit/wavesplat/wavesplat')
from wavesplat.evaluation.metrics import rmse, rmse_observed, rmse_unobserved

print('=' * 78)
print('A. PARTITION IDENTITY on REAL grid+obs geometry (PREREG 5.6, never run by analyze_rx7)')
print('=' * 78)
p = '/home/vexit/wavesplat/wavesplat/data_indep4_2026-08-07/episode_BF4_15comp_seed8_uniform_ra_rf_ds.h5'
with h5py.File(p, 'r') as f:
    gx = f['frames/gt_grid_x'][:]; gy = f['frames/gt_grid_y'][:]
    pts = f['frames/lidar_points'][:]; msk = f['frames/lidar_valid_mask'][:]
    gt_all = f['frames/gt_wave_field'][:]

rng = np.random.default_rng(0)
worst = 0.0
for t in (0, 40, 120, 200, 300):
    m = msk[t]
    ox, oy = pts[t, m, 0], pts[t, m, 1]
    gt = gt_all[t]
    for trial in range(3):
        pred = gt + rng.normal(0, 0.3, gt.shape) * (1 + rng.random(gt.shape))
        ro = rmse_observed(pred, gt, gx, gy, ox, oy)
        ru = rmse_unobserved(pred, gt, gx, gy, ox, oy)
        rt = rmse(pred, gt)
        # recover the mask sizes exactly the same way the metric does
        Xg, Yg = np.meshgrid(gx, gy)
        d = np.sqrt((Xg[..., None] - ox[None, None, :]) ** 2 +
                    (Yg[..., None] - oy[None, None, :]) ** 2).min(axis=2)
        no = int((d < 10.0).sum()); nu = int((d >= 10.0).sum())
        recon = np.sqrt((no * ro ** 2 + nu * ru ** 2) / (no + nu))
        err = abs(recon - rt)
        worst = max(worst, err)
        if trial == 0:
            print(f'  t={t:3d} n_obs_grid={no:4d} n_unobs={nu:4d} sum={no+nu} '
                  f'RMSE_obs={ro:.5f} RMSE_unobs={ru:.5f} recon={recon:.8f} true={rt:.8f} '
                  f'|err|={err:.2e}')
print(f'  worst |recon - true| over 15 checks = {worst:.3e}  -> '
      f'{"PARTITION EXACT" if worst < 1e-12 else "PARTITION BROKEN"}')

print('\n--- mask complementarity / off-by-one at the boundary ---')
m = msk[100]; ox, oy = pts[100, m, 0], pts[100, m, 1]
Xg, Yg = np.meshgrid(gx, gy)
d = np.sqrt((Xg[..., None] - ox[None, None, :]) ** 2 + (Yg[..., None] - oy[None, None, :]) ** 2).min(axis=2)
print(f'  points with d exactly == 10.0: {(d == 10.0).sum()}   '
      f'(< : {(d<10).sum()}, >= : {(d>=10).sum()}, total {d.size})')
print(f'  batch loop: len(obs)={len(ox)} < batch=1000 -> single batch, inner loop covers all. OK')

print('\n' + '=' * 78)
print('B. min_dist SENSITIVITY — what CAN and CANNOT be redone from raw data')
print('=' * 78)
print('  results.jsonl fields:', sorted(json.loads(open(
    '/home/vexit/wavesplat/wavesplat/tmp/Rx7_tracker_deploy_2026-08-07/results.jsonl'
).readline()).keys()))
print('  -> no predicted field, no per-frame RMSE, no grid, no obs coords are stored.')
print('  -> RMSE_observed at min_dist != 10 m CANNOT be recomputed without re-running the estimator.')
print('  -> only the MASK SIZE is re-derivable (audit_con_mask.py).')

print('\n' + '=' * 78)
print('C. BLOCK DIFFICULTY: Rx-7 indep4 (seeds 8-11) vs Rx-6 indep2+indep3 (seeds 0-7)')
print('=' * 78)


def load(path, cellmap):
    rows = [json.loads(l) for l in open(path) if l.strip()]
    rows = [r for r in rows if 'error' not in r]
    idx = {}
    for r in rows:
        idx.setdefault((cellmap.get(r['cell'], r['cell']), r['episode']), []).append(r)
    return rows, idx


R7 = '/home/vexit/wavesplat/wavesplat/tmp/Rx7_tracker_deploy_2026-08-07'
R6 = '/home/vexit/wavesplat/wavesplat/tmp/Rx6_channel_curve_2026-08-07'
r7, i7 = load(f'{R7}/results.jsonl', {})
r6, i6 = load(f'{R6}/results.jsonl', {})
print('  Rx-6 cells:', sorted({r['cell'] for r in r6}))
print('  Rx-6 seeds:', sorted({r['seed'] for r in r6}), ' Rx-7 seeds:', sorted({r['seed'] for r in r7}))


def blockmed(idx, cell, key='nRMSE'):
    eps = sorted({e for (c, e) in idx if c == cell})
    out = {}
    for e in eps:
        v = [x[key] for x in idx[(cell, e)] if x.get(key) is not None and x[key] == x[key]]
        out[e] = st.median(v)
    return out


for cell6, cell7, lbl in (('X-BASE', 'AUTO', 'shipped AUTO'),
                          ('NOTRACK', 'AUTO-NOTRACK', 'floor')):
    c6 = cell6 if any(c == cell6 for (c, e) in i6) else None
    if c6 is None:
        cands = sorted({c for (c, e) in i6})
        c6 = [c for c in cands if 'NOTRACK' in c or c == 'AUTO'] or cands[:1]
        c6 = c6[0]
    m6 = blockmed(i6, c6); m7 = blockmed(i7, cell7)
    print(f'  {lbl:14s} Rx-6[{c6}] medOfMed={st.median(list(m6.values())):.4f} (n={len(m6)}) | '
          f'Rx-7[{cell7}] medOfMed={st.median(list(m7.values())):.4f} (n={len(m7)})')

# nulls
n6 = json.load(open(f'{R6}/nulls_rx6.json'))
n7 = json.load(open(f'{R7}/nulls_rx7.json'))
k6 = [x['zero_null_tracking_nRMSE_mean'] for x in n6]
k7 = [x['zero_null_tracking_nRMSE_mean'] for x in n7]
print(f'\n  flat-sea null: Rx-6 median={st.median(k6):.4f} (n={len(k6)}) | '
      f'Rx-7 median={st.median(k7):.4f} (n={len(k7)})')
# init nrmse (Phase-1 only, tracker-independent difficulty proxy)
for cell6, cell7 in (('X-BASE', 'AUTO'),):
    c6 = cell6 if any(c == cell6 for (c, e) in i6) else sorted({c for (c, e) in i6})[0]
    a6 = blockmed(i6, c6, 'init_nrmse'); a7 = blockmed(i7, cell7, 'init_nrmse')
    print(f'  init_nrmse (Phase 1, tracker-independent): Rx-6={st.median(list(a6.values())):.4f} | '
          f'Rx-7={st.median(list(a7.values())):.4f}')
    print(f'  init_nrmse per-episode range: Rx-6 [{min(a6.values()):.4f},{max(a6.values()):.4f}] | '
          f'Rx-7 [{min(a7.values()):.4f},{max(a7.values()):.4f}]')

print('\n  Rx-6 unobserved-region levels for the same two arms:')
for c in sorted({c for (c, e) in i6}):
    v = blockmed(i6, c, 'avg_RMSE_unobserved')
    if v:
        print(f'    {c:12s} unobs medOfMed={st.median(list(v.values())):.5f}')
print('  Rx-6 has NO avg_RMSE_observed field:',
      'avg_RMSE_observed' not in r6[0])
