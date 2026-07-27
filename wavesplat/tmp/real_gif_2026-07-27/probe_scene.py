import os, sys, time
os.environ.setdefault('OMP_NUM_THREADS','4')
import warnings; warnings.filterwarnings('ignore')
import numpy as np
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__),'..','..'))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT,'scripts'))
import d3_compare_gifs as cg
for rec,win in [('AA15a',5),('AA15b',1),('AA14',2)]:
    t=time.time(); sc=cg.scene_real(rec,win=win)
    P=sc['P']
    print(rec,win,'n_obs',len(sc['z']),'t range',P[:,2].min(),P[:,2].max(),
          'ntimes',len(np.unique(P[:,2])),'Hs',round(sc['Hs'],3),'kp',round(sc['kp'],4),
          'depth',sc['depth'],'leads',sc['leads'][-1],'gx',sc['gxq'].min(),sc['gxq'].max(),
          'gy',sc['gyq'].min(),sc['gyq'].max(),'gridshape',sc['maps_gt'].shape,
          'gt_nan_frac',float(np.mean(~np.isfinite(sc['maps_gt']))),
          f'({time.time()-t:.0f}s)', flush=True)
