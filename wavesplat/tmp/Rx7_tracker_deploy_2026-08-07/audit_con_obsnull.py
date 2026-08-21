"""CON audit — the MISSING null for the new metric.

nulls_rx7.json only carries a flat-sea null for the WHOLE-grid nRMSE.  The new
avg_RMSE_observed / avg_RMSE_unobserved were scored with no null at all, which is
exactly the defect feedback_detection_rate_needs_null.md was written about.
This builds the flat-sea (z=0) null in BOTH regions with the run's own conventions:
mean over tracking frames of rms(gt) restricted to the mask.  Read-only.
"""
import json, os
import statistics as st
import numpy as np
import h5py

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = '/home/vexit/wavesplat/wavesplat/data_indep4_2026-08-07'
NULLJ = json.load(open(f'{HERE}/nulls_rx7.json'))
HS = {n['episode']: n['Hs_true'] for n in NULLJ}
TP = {n['episode']: n['Tp_true'] for n in NULLJ}
NULL = {n['episode']: n['zero_null_tracking_nRMSE_mean'] for n in NULLJ}

rows = [json.loads(l) for l in open(f'{HERE}/results.jsonl') if l.strip()]
rows = [r for r in rows if 'error' not in r]
IDX = {}
for r in rows:
    IDX.setdefault((r['cell'], r['episode']), []).append(r)
EPS = sorted({r['episode'] for r in rows})


def med(c, e, k):
    v = [x[k] for x in IDX[(c, e)] if x.get(k) is not None and x[k] == x[k]]
    return st.median(v)


KEPT = [e for e in EPS if med('AUTO', e, 'nRMSE') < NULL[e]]

print(f'{"episode":24s} {"nullOBS":>8s} {"nullUNOBS":>10s} {"nullALL":>8s} | '
      f'{"AUTOobs":>8s} {"FLOORobs":>9s} | {"AUTOun":>8s} {"FLOORun":>8s}')
res = {}
for e in EPS:
    with h5py.File(os.path.join(DATA, e), 'r') as f:
        gx = f['frames/gt_grid_x'][:]; gy = f['frames/gt_grid_y'][:]
        pts = f['frames/lidar_points'][:]; msk = f['frames/lidar_valid_mask'][:]
        gt = f['frames/gt_wave_field'][:]
        ts = f['frames/timestamps'][:]
    Xg, Yg = np.meshgrid(gx, gy)
    keep = np.where(ts > 2.0 * TP[e])[0]          # get_tracking_frames convention
    ro, ru, ra = [], [], []
    for t in keep:
        m = msk[t]
        if not m.sum():
            continue
        ox, oy = pts[t, m, 0], pts[t, m, 1]
        d = np.sqrt((Xg[..., None] - ox[None, None, :]) ** 2 +
                    (Yg[..., None] - oy[None, None, :]) ** 2).min(axis=2)
        s = d < 10.0
        g = gt[t]
        ro.append(float(np.sqrt(np.mean(g[s] ** 2))))
        ru.append(float(np.sqrt(np.mean(g[~s] ** 2))))
        ra.append(float(np.sqrt(np.mean(g ** 2))))
    res[e] = (np.mean(ro), np.mean(ru), np.mean(ra))
    print(f'  {e[8:-25]:22s} {res[e][0]:8.4f} {res[e][1]:10.4f} {res[e][2]:8.4f} | '
          f'{med("AUTO",e,"avg_RMSE_observed"):8.4f} {med("AUTO-NOTRACK",e,"avg_RMSE_observed"):9.4f} | '
          f'{med("AUTO",e,"avg_RMSE_unobserved"):8.4f} {med("AUTO-NOTRACK",e,"avg_RMSE_unobserved"):8.4f}')

print('\n=== flat-sea null pass rate in each region (kept block n={}) ==='.format(len(KEPT)))
CELLS = ['AUTO', 'AUTO-LR10', 'AUTO-LR100', 'AUTO-LR1000', 'AUTO-NOTRACK']
print(f'  {"cell":14s} {"obs beats null":>16s} {"unobs beats null":>18s}')
for c in CELLS:
    po = sum(1 for e in KEPT if med(c, e, 'avg_RMSE_observed') < res[e][0])
    pu = sum(1 for e in KEPT if med(c, e, 'avg_RMSE_unobserved') < res[e][1])
    print(f'  {c:14s} {po:10d}/{len(KEPT):<4d} {pu:12d}/{len(KEPT):<4d}')

print('\n=== block medians ===')
print(f'  flat-sea null : obs={st.median([res[e][0] for e in KEPT]):.4f}  '
      f'unobs={st.median([res[e][1] for e in KEPT]):.4f}')
for c in CELLS:
    print(f'  {c:14s}: obs={st.median([med(c,e,"avg_RMSE_observed") for e in KEPT]):.4f}  '
          f'unobs={st.median([med(c,e,"avg_RMSE_unobserved") for e in KEPT]):.4f}')

print('\n=== how much of the observed-region gap to the flat sea does each cell close? ===')
for c in CELLS:
    frac = []
    for e in KEPT:
        n0 = res[e][0]
        frac.append(1.0 - med(c, e, 'avg_RMSE_observed') / n0)
    print(f'  {c:14s} observed-region skill vs flat sea: median {st.median(frac)*100:5.1f}%')

print('\n=== is the observed region intrinsically easier?  (flat-sea null obs vs unobs) ===')
r = [res[e][0] / res[e][1] for e in KEPT]
print(f'  median null_obs/null_unobs = {st.median(r):.4f}  -> the two regions have '
      f'{"the same" if 0.95 < st.median(r) < 1.05 else "different"} signal energy')
