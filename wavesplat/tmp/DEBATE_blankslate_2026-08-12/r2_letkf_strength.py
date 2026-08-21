#!/usr/bin/env python3
"""R2/P3 task B: how weak is the LETKF baseline?

The published baseline is LETKFGrid(40x30, n_ens=100, mult_infl=1.0,
add_infl=0.005, obs_noise=0.05, loc_radius=50, **max_obs=40**).

max_obs=40 is a DEFAULT (letkf_grid.py:47) that thins each frame's ~8000-15000
LiDAR returns down to 40 random points.  The 2026-07-24 tuning campaign
(tmp/etkf_tune_stage*.json) swept nx/ny, n_ensemble, inflation and obs_noise
but held max_obs=40 fixed in all 16 configs -- so the single knob that controls
how much data the baseline is allowed to see was never tuned.

This script re-runs the baseline with max_obs raised (and a couple of the
usual DA knobs) on a handful of episodes, scored on the SAME frame set
(t > 2*Tp) as the WaveSplat arm.
"""
import os, sys, json, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from scipy.interpolate import RegularGridInterpolator
from wavesplat.data_loader import EpisodeDataLoader
from wavesplat.baselines.letkf_grid import LETKFGrid

OUT = os.path.join(HERE, 'r2_letkf_strength.json')

EPS = ['BF2_1comp_seed0_uniform', 'BF3_5comp_seed0_uniform',
       'BF4_15comp_seed0_uniform', 'BF5_30comp_seed0_uniform']

CFGS = {
    'published':      dict(nx=40, ny=30, n_ensemble=100, max_obs=40),
    'obs200':         dict(nx=40, ny=30, n_ensemble=100, max_obs=200),
    'obs800':         dict(nx=40, ny=30, n_ensemble=100, max_obs=800),
    'obs800_g80':     dict(nx=80, ny=60, n_ensemble=100, max_obs=800),
    'obs800_g80_l25': dict(nx=80, ny=60, n_ensemble=100, max_obs=800,
                           loc_radius=25.0),
    'obs800_g80_inf': dict(nx=80, ny=60, n_ensemble=100, max_obs=800,
                           mult_inflation=1.02),
}


def run(h5, cfg):
    np.random.seed(42)
    loader = EpisodeDataLoader(h5)
    gx = loader.gt_grid_x.cpu().numpy()
    gy = loader.gt_grid_y.cpu().numpy()
    nx, ny = cfg.pop('nx'), cfg.pop('ny')
    egx, egy = np.linspace(gx[0], gx[-1], nx), np.linspace(gy[0], gy[-1], ny)
    kw = dict(depth=loader.depth, n_ensemble=100, mult_inflation=1.0,
              add_inflation=0.005, obs_noise=0.05, loc_radius=50.0, max_obs=40)
    kw.update(cfg)
    lk = LETKFGrid(egx, egy, **kw)
    Xg, Yg = np.meshgrid(gx, gy)
    q = np.column_stack([Yg.ravel(), Xg.ravel()])
    t_cut = 2.0 * float(loader.Tp_true)
    nr, na, ms, nobs = [], [], [], []
    for i, frame in enumerate(loader):
        x = frame.x_obs.cpu().numpy(); y = frame.y_obs.cpu().numpy()
        z = frame.z_obs.cpu().numpy()
        nobs.append(len(x))
        ms.append(lk.update(x, y, z))
        pred = lk.predict_grid()
        pf = RegularGridInterpolator((egy, egx), pred, method='linear',
                                     bounds_error=False, fill_value=0.0)(q)
        gt = frame.gt_field.cpu().numpy()
        e = float(np.sqrt(np.mean((pf.reshape(gt.shape) - gt) ** 2))
                  / loader.Hs_true)
        na.append(e)
        if float(loader.timestamps[i]) > t_cut:
            nr.append(e)
    return dict(nrmse_match=float(np.mean(nr)), nrmse_all=float(np.mean(na)),
                n_match=len(nr), n_all=len(na),
                ms_median=float(np.median(ms)),
                obs_median=float(np.median(nobs)))


def main():
    out = json.load(open(OUT)) if os.path.exists(OUT) else {}
    for ep in EPS:
        for name, cfg in CFGS.items():
            k = f'{ep}|{name}'
            if k in out:
                continue
            t0 = time.time()
            try:
                out[k] = run(os.path.join('data', f'episode_{ep}.h5'), dict(cfg))
            except Exception as e:
                out[k] = {'error': f'{type(e).__name__}: {e}'}
            print(f'{k:<50s} {out[k].get("nrmse_match")} '
                  f'({time.time()-t0:.0f}s)', flush=True)
            json.dump(out, open(OUT, 'w'), indent=1)
    print('done')


if __name__ == '__main__':
    main()
