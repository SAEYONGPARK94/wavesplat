"""Supplementary: the repo's LETKF(loc50) "stronger baseline" on both meshes.
Same driver shape as scripts/run_strong_baseline_comparison.py::run_grid_baseline
(grid 40x30, Ne=100, mi=1.0, ai=0.005, loc_radius=50), but median over 5 filter seeds
and also recording rho. CPU/numpy only."""
import os
os.environ.setdefault('OMP_NUM_THREADS','1'); os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import sys, json, numpy as np
from multiprocessing import Pool
HERE=os.path.dirname(os.path.abspath(__file__)); ROOT=os.path.dirname(HERE)
sys.path.insert(0,HERE); sys.path.insert(0,ROOT)
SCEN=[('BF2_1comp','uniform'),('BF3_1comp','uniform'),('BF3_2comp','uniform'),('BF3_2comp','slope'),
      ('BF3_5comp','uniform'),('BF3_5comp','slope'),('BF4_1comp','uniform'),('BF4_5comp','uniform'),
      ('BF4_15comp','uniform'),('BF4_30comp','uniform'),('BF5_15comp','uniform'),('BF5_30comp','uniform')]
EPS=[f'episode_{s}_seed0_{r}.h5' for s,r in SCEN]
KW=dict(n_ensemble=100, mult_inflation=1.0, add_inflation=0.005, loc_radius=50.0)

def job(a):
    from etkf_lib import get_episode, _grid_to_gt
    from wavesplat.baselines.letkf_grid import LETKFGrid
    ds, ep, seed = a
    e = get_episode(os.path.join(ROOT, ds, ep))
    np.random.seed(seed)
    gx=np.linspace(e['gt_grid_x'][0],e['gt_grid_x'][-1],40)
    gy=np.linspace(e['gt_grid_y'][0],e['gt_grid_y'][-1],30)
    f=LETKFGrid(gx,gy,depth=e['depth'],**KW)
    nr=[]; rh=[]
    for i,(x,y,z) in enumerate(e['obs']):
        f.update(x,y,z)
        pred=_grid_to_gt(f.predict_grid(),gx,gy,e['gt_grid_x'],e['gt_grid_y'])
        gt=e['gt'][i]
        nr.append(np.sqrt(np.mean((pred-gt)**2))/e['Hs'])
        p,g=pred.ravel(),gt.ravel(); den=np.sqrt((p@p)*(g@g))
        rh.append(float(p@g)/den if den>1e-14 else 0.0)
    return dict(ds=ds, ep=ep, seed=seed, nRMSE=float(np.mean(nr)), rho=float(np.mean(rh)))

if __name__=='__main__':
    jobs=[(ds,ep,s) for ds in ('data','data_finemesh_2026-07-23') for ep in EPS for s in range(5)]
    with Pool(4) as p: res=p.map(job,jobs,chunksize=1)
    json.dump(res, open(os.path.join(ROOT,'tmp','letkf_finemesh_2026-07-24.json'),'w'), indent=1)
    print('done', len(res))
