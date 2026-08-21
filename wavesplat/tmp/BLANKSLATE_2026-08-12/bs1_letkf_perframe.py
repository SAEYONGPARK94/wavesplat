#!/usr/bin/env python3
"""BS1 fix: re-run the LETKF baseline recording PER-FRAME metrics.

Why: scripts/run_100ep_letkf_and_ws_metrics.py scores LETKF with
`for frame in loader:` (ALL 301 frames, including its cold start) while it
scores WaveSplat only on `get_tracking_frames(tp_factor=2.0)` (t > 2*Tp).
results/100ep_ws_letkf.json therefore compares the two methods on DIFFERENT
frame sets.  This script re-runs LETKF identically but keeps every frame's
metrics so the same episode can be aggregated on either frame set, plus the
per-frame zero-field floor on the identical frames.

Output: tmp/BLANKSLATE_2026-08-12/letkf_perframe.json
"""
import os, sys, glob, json, re, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)

from scipy.interpolate import RegularGridInterpolator
from wavesplat.data_loader import EpisodeDataLoader
from wavesplat.baselines.letkf_grid import LETKFGrid
from wavesplat.evaluation.metrics import compute_all_metrics

DATA = os.path.join(ROOT, 'data')
OUT = os.path.join(HERE, 'letkf_perframe.json')
KEYS = ['nRMSE', 'Variance_Ratio', 'Pearson_rho', 'Skill_Score',
        'Directional_R2', 'Hs_error']


def run_one(h5_path):
    np.random.seed(42)
    loader = EpisodeDataLoader(h5_path)
    gx = loader.gt_grid_x.cpu().numpy()
    gy = loader.gt_grid_y.cpu().numpy()
    egx = np.linspace(gx[0], gx[-1], 40)
    egy = np.linspace(gy[0], gy[-1], 30)
    letkf = LETKFGrid(egx, egy, depth=loader.depth, n_ensemble=100,
                      mult_inflation=1.0, add_inflation=0.005, loc_radius=50.0)
    Xg, Yg = np.meshgrid(gx, gy)
    q = np.column_stack([Yg.ravel(), Xg.ravel()])
    recs = []
    for i, frame in enumerate(loader):
        x = frame.x_obs.cpu().numpy()
        y = frame.y_obs.cpu().numpy()
        z = frame.z_obs.cpu().numpy()
        elapsed = letkf.update(x, y, z)
        pred = letkf.predict_grid()
        interp = RegularGridInterpolator((egy, egx), pred, method='linear',
                                         bounds_error=False, fill_value=0.0)
        pf = interp(q).reshape(len(gy), len(gx))
        gt = frame.gt_field.cpu().numpy()
        m = compute_all_metrics(pf, gt, gx, gy, loader.Hs_true,
                                obs_x=x if len(x) else None,
                                obs_y=y if len(y) else None)
        rec = {k: float(m[k]) for k in KEYS if k in m}
        rec['t'] = float(loader.timestamps[i])
        rec['ms'] = float(elapsed)
        # zero-field floor on the identical frame
        rec['zero_nRMSE'] = float(np.sqrt(np.mean(gt ** 2)) / loader.Hs_true)
        recs.append(rec)
    return dict(Tp=float(loader.Tp_true), Hs=float(loader.Hs_true), frames=recs)


def main():
    eps = sorted(glob.glob(os.path.join(DATA, 'episode_*.h5')))
    eps = [e for e in eps if re.match(
        r'episode_BF[2-5]_(1|2|5|15|30)comp_seed[0-4]_(uniform|slope)\.h5',
        os.path.basename(e))]
    print(f'{len(eps)} episodes', flush=True)
    out = {}
    if os.path.exists(OUT):
        out = json.load(open(OUT))
    for i, e in enumerate(eps):
        key = os.path.basename(e)[len('episode_'):-3]
        if key in out:
            continue
        t0 = time.time()
        out[key] = run_one(e)
        print(f'[{i+1}/{len(eps)}] {key}  {time.time()-t0:.1f}s', flush=True)
        json.dump(out, open(OUT, 'w'))
    json.dump(out, open(OUT, 'w'))
    print('done', len(out))


if __name__ == '__main__':
    main()
