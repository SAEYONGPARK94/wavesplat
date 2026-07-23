"""Reproduce the paper's tab:etkf_sweep winning row on the OLD data with this harness."""
import os
os.environ.setdefault('OMP_NUM_THREADS','1'); os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import sys, json, numpy as np
from multiprocessing import Pool
HERE=os.path.dirname(os.path.abspath(__file__)); ROOT=os.path.dirname(HERE); sys.path.insert(0,HERE)
SCEN=[('BF2_1comp','uniform'),('BF3_1comp','uniform'),('BF3_2comp','uniform'),('BF3_2comp','slope'),
      ('BF3_5comp','uniform'),('BF3_5comp','slope'),('BF4_1comp','uniform'),('BF4_5comp','uniform'),
      ('BF4_15comp','uniform'),('BF4_30comp','uniform'),('BF5_15comp','uniform'),('BF5_30comp','uniform')]
EPS=[f'episode_{s}_seed0_{r}.h5' for s,r in SCEN]
CFGS={'paper_tuned_40x30_Ne100_mi1.00_ai0.005':dict(nx=40,ny=30,n_ensemble=100,mult_inflation=1.0,add_inflation=0.005),
      'worst_80x60_Ne50_mi1.05_ai0.020':dict(nx=80,ny=60,n_ensemble=50,mult_inflation=1.05,add_inflation=0.02)}
def job(a):
    from etkf_lib import run_etkf
    name,ep=a
    r=run_etkf(os.path.join(ROOT,'data',ep), seed=42, diagnostics=True, **CFGS[name])
    return dict(name=name,ep=ep,nRMSE=r['avg_nRMSE'],rho=r['avg_rho'])
if __name__=='__main__':
    with Pool(4) as p: res=p.map(job,[(n,e) for n in CFGS for e in EPS],chunksize=1)
    json.dump(res,open(os.path.join(ROOT,'tmp','etkf_repro_paper_2026-07-24.json'),'w'),indent=1)
    for n in CFGS:
        v=[r['nRMSE'] for r in res if r['name']==n]
        rr=[r['rho'] for r in res if r['name']==n]
        print(f"{n:44s} mean nRMSE {np.mean(v):.4f}  mean rho {np.mean(rr):.3f}")
