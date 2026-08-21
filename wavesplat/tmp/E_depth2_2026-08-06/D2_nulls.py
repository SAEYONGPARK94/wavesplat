"""Depth-2: flat-sea (zero-predictor) null for avg_nRMSE and init_nrmse, per episode.

avg_nRMSE (run.py:277) = mean over tracking frames of RMSE(pred,gt)/Hs_true.
init_nrmse (run.py:179) = RMSE(pred,gt)/Hs_true at the LAST init frame.
Zero predictor => RMSE = sqrt(mean(gt**2)).
Tracking frames = timestamps > 2*Tp (data_loader.get_tracking_frames, tp_factor=2.0).
Run from /home/vexit/wavesplat/wavesplat.
"""
import json, os, sys
import numpy as np
import h5py

OUT = 'tmp/E_depth2_2026-08-06/D2_nulls.json'


def nulls(path):
    with h5py.File(path, 'r') as h:
        Hs = float(h['gt_params'].attrs['Hs_true'])
        Tp = float(h['gt_params'].attrs['Tp_true'])
        ts = h['frames/timestamps'][:]
        t_init = 2.0 * Tp
        idx_tr = np.where(ts > t_init)[0]
        idx_in = np.where(ts <= t_init)[0]
        gt = h['frames/gt_wave_field']
        tr = np.mean([float(np.sqrt(np.mean(gt[i] ** 2)) / max(Hs, 1e-6)) for i in idx_tr])
        last_init = int(idx_in[-1])
        ini = float(np.sqrt(np.mean(gt[last_init] ** 2)) / max(Hs, 1e-6))
    return dict(Hs=Hs, Tp=Tp, n_track=len(idx_tr), null_avg_nRMSE=float(tr),
                null_init_nrmse=ini)


def main():
    rows = {}
    for d in ('data_finemesh_2026-07-23', 'data_heldout40_2026-07-26'):
        for e in sorted(os.listdir(d)):
            if not e.endswith('.h5'):
                continue
            try:
                rows[d + '/' + e] = nulls(os.path.join(d, e))
            except Exception as ex:
                rows[d + '/' + e] = dict(err=repr(ex))
    json.dump(rows, open(OUT, 'w'), indent=1)
    print(len(rows), 'episodes ->', OUT)
    v = [r['null_avg_nRMSE'] for r in rows.values() if 'null_avg_nRMSE' in r]
    print('null_avg_nRMSE  median %.4f  min %.4f  max %.4f' % (np.median(v), min(v), max(v)))
    v2 = [r['null_init_nrmse'] for r in rows.values() if 'null_init_nrmse' in r]
    print('null_init_nrmse median %.4f  min %.4f  max %.4f' % (np.median(v2), min(v2), max(v2)))


main()
