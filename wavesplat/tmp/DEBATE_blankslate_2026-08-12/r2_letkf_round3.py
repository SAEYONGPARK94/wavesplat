#!/usr/bin/env python3
"""R2/P3 task B round 3: is the ensemble-size axis saturated?
Round 2's best config was n_ensemble=300 (-3.0%).  If nRMSE keeps falling with
Ne the baseline is NOT at its ceiling and 'we beat LETKF' is worth less.
"""
import os, sys, json, time
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT); os.chdir(ROOT); sys.path.insert(0, HERE)
from r2_letkf_strength import run, EPS
OUT = os.path.join(HERE, 'r2_letkf_round3.json')
CFGS = {
    'ne600':        dict(nx=40, ny=30, n_ensemble=600, max_obs=10**9),
    'ne1200':       dict(nx=40, ny=30, n_ensemble=1200, max_obs=10**9),
    'ne600_l25':    dict(nx=40, ny=30, n_ensemble=600, loc_radius=25.0,
                         max_obs=10**9),
    'ne600_l25_n02': dict(nx=40, ny=30, n_ensemble=600, loc_radius=25.0,
                          obs_noise=0.02, add_inflation=0.0, max_obs=10**9),
}
out = json.load(open(OUT)) if os.path.exists(OUT) else {}
for ep in EPS:
    for name, cfg in CFGS.items():
        k = f'{ep}|{name}'
        if k in out: continue
        t0 = time.time()
        try: out[k] = run(os.path.join('data', f'episode_{ep}.h5'), dict(cfg))
        except Exception as e: out[k] = {'error': f'{type(e).__name__}: {e}'}
        print(f'{k:<44s} {out[k].get("nrmse_match")} ({time.time()-t0:.0f}s)', flush=True)
        json.dump(out, open(OUT, 'w'), indent=1)
print('done')
