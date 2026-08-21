"""CON audit — the decisive test of the "OVERFIT is a tautology" attack.

If RMSE_observed is essentially the tracker's own training residual, then a DUMB local
interpolator of the same LiDAR returns should reach roughly the same error at the same
grid nodes.  If the tracker beats local interpolation, the observed-region gain carries
information beyond the local data and the tautology charge is too strong.
Read-only, CPU, no estimator.
"""
import json, os
import statistics as st
import numpy as np
import h5py
from scipy.interpolate import griddata

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = '/home/vexit/wavesplat/wavesplat/data_indep4_2026-08-07'
NULLJ = json.load(open(f'{HERE}/nulls_rx7.json'))
TP = {n['episode']: n['Tp_true'] for n in NULLJ}
NULL = {n['episode']: n['zero_null_tracking_nRMSE_mean'] for n in NULLJ}
rows = [json.loads(l) for l in open(f'{HERE}/results.jsonl') if l.strip()]
IDX = {}
for r in rows:
    IDX.setdefault((r['cell'], r['episode']), []).append(r)
EPS = sorted({r['episode'] for r in rows})


def med(c, e, k):
    return st.median([x[k] for x in IDX[(c, e)]])


KEPT = [e for e in EPS if med('AUTO', e, 'nRMSE') < NULL[e]]

print(f'{"episode":24s} {"localINTERP":>12s} {"AUTO obs":>10s} {"FLOOR obs":>10s} {"nullOBS":>9s}')
out = {}
for e in EPS:
    with h5py.File(os.path.join(DATA, e), 'r') as f:
        gx = f['frames/gt_grid_x'][:]; gy = f['frames/gt_grid_y'][:]
        pts = f['frames/lidar_points'][:]; msk = f['frames/lidar_valid_mask'][:]
        gt = f['frames/gt_wave_field'][:]; ts = f['frames/timestamps'][:]
    Xg, Yg = np.meshgrid(gx, gy)
    keep = np.where(ts > 2.0 * TP[e])[0]
    errs = []
    for t in keep[::3]:
        m = msk[t]
        if m.sum() < 4:
            continue
        ox, oy, oz = pts[t, m, 0], pts[t, m, 1], pts[t, m, 2]
        d = np.sqrt((Xg[..., None] - ox[None, None, :]) ** 2 +
                    (Yg[..., None] - oy[None, None, :]) ** 2).min(axis=2)
        s = d < 10.0
        if not s.any():
            continue
        tgt = np.column_stack([Xg[s], Yg[s]])
        z = griddata(np.column_stack([ox, oy]), oz, tgt, method='linear')
        zn = griddata(np.column_stack([ox, oy]), oz, tgt, method='nearest')
        z = np.where(np.isfinite(z), z, zn)
        errs.append(float(np.sqrt(np.mean((z - gt[t][s]) ** 2))))
    out[e] = float(np.mean(errs))
    print(f'  {e[8:-25]:22s} {out[e]:12.4f} {med("AUTO",e,"avg_RMSE_observed"):10.4f} '
          f'{med("AUTO-NOTRACK",e,"avg_RMSE_observed"):10.4f}')

print('\nblock medians (kept n={}):'.format(len(KEPT)))
print(f'  local LiDAR interpolation (linear+nearest) : {st.median([out[e] for e in KEPT]):.4f} m')
print(f'  AUTO         observed-region RMSE          : '
      f'{st.median([med("AUTO",e,"avg_RMSE_observed") for e in KEPT]):.4f} m')
print(f'  AUTO-NOTRACK observed-region RMSE          : '
      f'{st.median([med("AUTO-NOTRACK",e,"avg_RMSE_observed") for e in KEPT]):.4f} m')
w = sum(1 for e in KEPT if med('AUTO', e, 'avg_RMSE_observed') < out[e])
print(f'  AUTO beats the dumb local interpolator in {w}/{len(KEPT)} episodes')
w2 = sum(1 for e in KEPT if med('AUTO-NOTRACK', e, 'avg_RMSE_observed') < out[e])
print(f'  AUTO-NOTRACK beats it in {w2}/{len(KEPT)}')
