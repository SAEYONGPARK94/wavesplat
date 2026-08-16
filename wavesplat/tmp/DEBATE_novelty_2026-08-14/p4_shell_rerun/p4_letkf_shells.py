#!/usr/bin/env python3
"""P4 red-team probe (READ-ONLY w.r.t. repo; prints to stdout only).

Question: the S13/HOSGATE headline compares SHIP vs LETKF_PLUS on a GLOBAL nRMSE.
Gaspari-Cohn localisation with loc_radius=50 m is identically zero beyond 2c=100 m,
so on every grid point farther than 100 m from all observations the LETKF analysis
increment is exactly zero.  How much of the scored domain is that, and does the
headline survive when restricted to the region the baseline is allowed to update?

Re-runs LETKF_PLUS exactly as tmp/HOSGATE_2026-08-13/run_hosgate.py:run_letkf does
(same seed, same config) and additionally records obs-distance shell RMSE + area
fractions, matching wavesplat/run.py:358 edges, so it is directly comparable with the
SHIP shell_rmse already stored in results.jsonl.  Also records the ZERO floor per shell.
"""
import json
import os
import sys

import numpy as np

ROOT = '/home/vexit/wavesplat/wavesplat'
sys.path.insert(0, ROOT)
os.chdir(ROOT)

HERE = os.path.join(ROOT, 'tmp/HOSGATE_2026-08-13')
EDGES = [0.0, 5.0, 10.0, 20.0, 30.0, 50.0, 75.0, 100.0, 150.0, np.inf]
KEYS = [f'{int(a)}_{"inf" if b == np.inf else int(b)}' for a, b in zip(EDGES[:-1], EDGES[1:])]


def run_letkf_shells(path, n_ensemble=600):
    from scipy.interpolate import RegularGridInterpolator
    from wavesplat.data_loader import EpisodeDataLoader
    from wavesplat.baselines.letkf_grid import LETKFGrid
    from wavesplat.evaluation.metrics import compute_all_metrics
    np.random.seed(42)
    loader = EpisodeDataLoader(path)
    gx = loader.gt_grid_x.cpu().numpy()
    gy = loader.gt_grid_y.cpu().numpy()
    egx = np.linspace(gx[0], gx[-1], 40)
    egy = np.linspace(gy[0], gy[-1], 30)
    letkf = LETKFGrid(egx, egy, depth=loader.depth, n_ensemble=n_ensemble,
                      mult_inflation=1.0, add_inflation=0.005, loc_radius=50.0)
    Xg2, Yg2 = np.meshgrid(gx, gy)
    q = np.column_stack([Yg2.ravel(), Xg2.ravel()])
    nr, zr, t = [], [], []
    sh_l = {k: [] for k in KEYS}
    sh_z = {k: [] for k in KEYS}
    fr = {k: [] for k in KEYS}
    for i, frame in enumerate(loader):
        x = frame.x_obs.cpu().numpy()
        y = frame.y_obs.cpu().numpy()
        z = frame.z_obs.cpu().numpy()
        letkf.update(x, y, z)
        pred = letkf.predict_grid()
        interp = RegularGridInterpolator((egy, egx), pred, method='linear',
                                         bounds_error=False, fill_value=0.0)
        pf = interp(q).reshape(len(gy), len(gx))
        gt = frame.gt_field.cpu().numpy()
        m = compute_all_metrics(pf, gt, gx, gy, loader.Hs_true,
                                obs_x=x if len(x) else None,
                                obs_y=y if len(y) else None)
        nr.append(float(m['nRMSE']))
        zr.append(float(np.sqrt(np.mean(gt ** 2)) / loader.Hs_true))
        t.append(float(loader.timestamps[i]))
        if len(x):
            md = np.full(Xg2.shape, np.inf)
            for j in range(len(x)):
                md = np.minimum(md, np.sqrt((Xg2 - x[j]) ** 2 + (Yg2 - y[j]) ** 2))
            for a, b, k in zip(EDGES[:-1], EDGES[1:], KEYS):
                msk = (md >= a) & (md < b)
                fr[k].append(float(np.mean(msk)))
                if np.any(msk):
                    sh_l[k].append(float(np.sqrt(np.mean((pf[msk] - gt[msk]) ** 2))))
                    sh_z[k].append(float(np.sqrt(np.mean(gt[msk] ** 2))))
    sel = np.array(t) > 2.0 * float(loader.Tp_true)
    # shells were only appended on frames with observations; approximate the same mask
    n_sh = len(sh_l[KEYS[0]])
    sel_sh = sel[:n_sh] if n_sh <= len(sel) else sel
    out = dict(avg_nRMSE=float(np.mean(np.array(nr)[sel])),
               zero_nRMSE=float(np.mean(np.array(zr)[sel])),
               n_match=int(sel.sum()), Hs=float(loader.Hs_true))
    out['letkf_shell'] = {k: (float(np.mean(np.array(v)[sel_sh[:len(v)]]))
                              if len(v) else float('nan')) for k, v in sh_l.items()}
    out['zero_shell'] = {k: (float(np.mean(np.array(v)[sel_sh[:len(v)]]))
                             if len(v) else float('nan')) for k, v in sh_z.items()}
    out['frac'] = {k: (float(np.mean(np.array(v)[sel_sh[:len(v)]]))
                       if len(v) else float('nan')) for k, v in fr.items()}
    return out


def main():
    rows = [json.loads(l) for l in open(os.path.join(HERE, 'results.jsonl'))]
    D = {(r['gt'], r['pair'], r['arm']): r for r in rows}
    pairs = sorted(set(r['pair'] for r in rows
                       if D[('LIN', r['pair'], 'SHIP')]['stratum'] == 'crowded'))
    res = {}
    for p in pairs:
        path = os.path.join(HERE, 'data_lin', 'episode_%s.h5' % p)
        o = run_letkf_shells(path)
        stored = D[('LIN', p, 'LETKF_PLUS')]['avg_nRMSE']
        o['stored'] = stored
        o['repro_abs_delta'] = abs(o['avg_nRMSE'] - stored)
        res[p] = o
        print(json.dumps({'pair': p, 'avg_nRMSE': o['avg_nRMSE'],
                          'stored': stored, 'd': o['repro_abs_delta']}), flush=True)
    print('===SUMMARY===')
    print(json.dumps(res))


if __name__ == '__main__':
    main()
