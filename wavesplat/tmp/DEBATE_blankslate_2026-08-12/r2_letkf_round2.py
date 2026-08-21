#!/usr/bin/env python3
"""R2/P3 task B round 2: the knobs that actually matter for this corpus.

Round 1 showed the standard data/ corpus carries only ~50 valid LiDAR returns
per frame (71 rays, median 49-50 valid), so max_obs=40 thins away only ~20%
and raising it changes almost nothing.  Round 2 sweeps the knobs that a DA
practitioner would actually turn: localization radius, observation-error
variance, ensemble size, additive inflation -- plus max_obs=1e9 (no thinning).
"""
import os, sys, json, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)
os.chdir(ROOT)
sys.path.insert(0, HERE)
from r2_letkf_strength import run, EPS

OUT = os.path.join(HERE, 'r2_letkf_round2.json')

CFGS = {}
for r in (10.0, 25.0, 75.0, 100.0, 200.0):
    CFGS[f'loc{int(r)}'] = dict(nx=40, ny=30, loc_radius=r, max_obs=10 ** 9)
for on in (0.01, 0.02, 0.1, 0.2):
    CFGS[f'noise{on}'] = dict(nx=40, ny=30, obs_noise=on, max_obs=10 ** 9)
CFGS['nothin'] = dict(nx=40, ny=30, max_obs=10 ** 9)
CFGS['ne300'] = dict(nx=40, ny=30, n_ensemble=300, max_obs=10 ** 9)
CFGS['ai0'] = dict(nx=40, ny=30, add_inflation=0.0, max_obs=10 ** 9)
CFGS['ai002'] = dict(nx=40, ny=30, add_inflation=0.02, max_obs=10 ** 9)
CFGS['best_guess'] = dict(nx=40, ny=30, loc_radius=25.0, obs_noise=0.02,
                          add_inflation=0.0, n_ensemble=300, max_obs=10 ** 9)


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
