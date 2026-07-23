"""Ceiling probe: can ANY reachable ETKF/LETKF configuration acquire real field
skill (rho) on the fine-mesh data?  Tests the best-case combination:
fine analysis grid + prior-shaped background covariance + no additive inflation
+ large ensemble + no observation thinning, and the repo's LETKF."""
import os
os.environ.setdefault('OMP_NUM_THREADS','1'); os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import sys, json, numpy as np
from multiprocessing import Pool
HERE=os.path.dirname(os.path.abspath(__file__)); ROOT=os.path.dirname(HERE); sys.path.insert(0,HERE)
EPS=['episode_BF3_1comp_seed0_uniform.h5','episode_BF3_5comp_seed0_uniform.h5',
     'episode_BF4_15comp_seed0_uniform.h5']
CFGS=[
 ('white 40x30 shipped',      dict(nx=40,ny=30,n_ensemble=50,add_inflation=0.005,max_obs=40), 'w'),
 ('white 160x120 ai0 Ne200',  dict(nx=160,ny=120,n_ensemble=200,add_inflation=0.0,max_obs=100,mult_inflation=1.0), 'w'),
 ('prior 40x30 ai0 Ne100',    dict(nx=40,ny=30,n_ensemble=100,add_inflation=0.0,max_obs=100,mult_inflation=1.0), 'p'),
 ('prior 80x60 ai0 Ne200',    dict(nx=80,ny=60,n_ensemble=200,add_inflation=0.0,max_obs=100,mult_inflation=1.0), 'p'),
 ('prior 160x120 ai0 Ne200',  dict(nx=160,ny=120,n_ensemble=200,add_inflation=0.0,max_obs=100,mult_inflation=1.0), 'p'),
 ('prior 80x60 ai0 Ne200 R.02',dict(nx=80,ny=60,n_ensemble=200,add_inflation=0.0,max_obs=100,mult_inflation=1.0,obs_noise=0.02), 'p'),
]
def job(a):
    from etkf_lib import run_etkf, run_prior_etkf
    ep,name,cfg,kind=a
    p=os.path.join(ROOT,'data_finemesh_2026-07-23',ep)
    r=(run_prior_etkf if kind=='p' else run_etkf)(p, seed=0, diagnostics=True, **cfg)
    return dict(ep=ep,name=name,nRMSE=r['avg_nRMSE'],rho=r['avg_rho'])
if __name__=='__main__':
    jobs=[(e,n,c,k) for e in EPS for n,c,k in CFGS]
    with Pool(4) as p: res=p.map(job,jobs,chunksize=1)
    json.dump(res, open(os.path.join(ROOT,'tmp','etkf_ceiling_2026-07-24.json'),'w'), indent=1)
    for e in EPS:
        print(e)
        for r in res:
            if r['ep']==e: print(f"   {r['name']:30s} nRMSE {r['nRMSE']:.4f}  rho {r['rho']:+.3f}")
