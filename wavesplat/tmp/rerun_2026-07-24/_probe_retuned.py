import os, sys, time, json
import numpy as np
HERE = os.path.dirname(os.path.abspath('tmp/rerun_2026-07-24/_probe_retuned.py'))
REPO = os.path.abspath('.')
sys.path.insert(0, os.path.join(REPO, 'tmp'))
import etkf_lib
from etkf_lib import run_etkf, get_episode

DATA = os.path.join(REPO, 'data_finemesh_2026-07-23')
PAIRS = [('BF2','1comp'),('BF3','1comp'),('BF3','2comp'),('BF3','5comp'),
         ('BF4','1comp'),('BF4','5comp'),('BF4','15comp'),('BF4','30comp'),
         ('BF5','15comp'),('BF5','30comp')]

# 1) integrity: open all 20 files
print("=== INTEGRITY CHECK (20 files) ===", flush=True)
bad=[]
for bf,c in PAIRS:
    for cond in ('uniform','slope'):
        ep=f'episode_{bf}_{c}_seed0_{cond}.h5'
        p=os.path.join(DATA,ep)
        try:
            d=get_episode(p)
            nfr=len(d['obs']); npts=int(np.mean([len(o[0]) for o in d['obs']]))
            print(f"  OK  {ep:<42s} frames={nfr} mean_pts={npts} Hs={d['Hs']:.2f}", flush=True)
            etkf_lib._CACHE.clear()
        except Exception as e:
            print(f"  BAD {ep}: {e!r}", flush=True); bad.append(ep)
print("bad:", bad, flush=True)

# 2) timing + seed spread on 2 episodes with RETUNED config
CFG = dict(nx=160, ny=120, n_ensemble=50, mult_inflation=1.0,
           add_inflation=0.0, obs_noise=0.20, max_obs=20)
print("\n=== TIMING + SEED SPREAD (retuned cfg) ===", flush=True)
for ep in ('episode_BF2_1comp_seed0_uniform.h5','episode_BF4_15comp_seed0_slope.h5'):
    p=os.path.join(DATA,ep)
    vals=[]
    for s in (42,1,2,3,4):
        t0=time.time()
        r=run_etkf(p, seed=s, diagnostics=False, **CFG)
        vals.append(r['avg_nRMSE'])
        print(f"  {ep} seed{s}: nRMSE={r['avg_nRMSE']:.5f} ({time.time()-t0:.1f}s)", flush=True)
    v=np.array(vals)
    print(f"  -> {ep}: median={np.median(v):.5f} min={v.min():.5f} max={v.max():.5f} spread={v.max()-v.min():.5f}", flush=True)
    etkf_lib._CACHE.clear()
