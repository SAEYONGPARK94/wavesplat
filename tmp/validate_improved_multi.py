"""Robustness: default vs improved (CV) estimator, multi-frame x seed, synthetic + real."""
import os, sys, numpy as np
sys.path.insert(0, '/home/vexit/wavesplat/wavesplat')
os.chdir('/home/vexit/wavesplat/wavesplat'); sys.path.insert(0, 'scripts')
from d26_relax_iterative import iterative_relax
from estimator_improved import iterative_relax_cv
from wavesplat.evaluation.metrics import pearson_correlation
from wavesplat.data_loader import EpisodeDataLoader
from d6_integrated_eval import tag_to_file
from d15_sequential_fusion import gather_frames

def sc(f, gt, Hs):
    v=~np.isnan(gt); return float(np.sqrt(np.mean((f[v]-gt[v])**2))/Hs), float(pearson_correlation(f[v],gt[v]))

print("=== SYNTHETIC default vs improved — 4 frames x 2 seeds ===")
print(f"{'episode':12s} | {'default nRMSE/corr':>20} | {'improved nRMSE/corr':>20} | win%")
for tag in ['BF4_15c_s0','BF4_30c_s0','BF5_30c_s1']:
    L=EpisodeDataLoader('data/'+tag_to_file(tag)); depth=L.depth; Hs=L.Hs_true
    n_back=int(round(2.2*L.Tp_true/0.1)); nfr=len(L)
    frames=np.linspace(n_back+3, nfr-2, 4).astype(int)
    dd,ii,wins=[],[],0; cd,ci=[],[]
    for i_star in frames:
        fstar=L[int(i_star)]
        target={'gx':L.gt_grid_x.numpy(),'gy':L.gt_grid_y.numpy(),'t':float(fstar.t),'gt':fstar.gt_field.numpy(),'Hs':Hs}
        P,z=gather_frames(L,int(i_star),n_back); P=P.astype(np.float32); z=z.astype(np.float32)
        fd,_=iterative_relax(P,z,depth,target,3); a=sc(fd,target['gt'],Hs)
        for seed in range(2):
            fi,_,_=iterative_relax_cv(P,z,depth,target,seed=seed); b=sc(fi,target['gt'],Hs)
            dd.append(a[0]); ii.append(b[0]); cd.append(a[1]); ci.append(b[1]); wins+=int(b[0]<a[0])
    dd,ii=np.array(dd),np.array(ii)
    print(f"{tag:12s} | {np.mean(dd):.3f}/{np.mean(cd):.2f}          | {np.mean(ii):.3f}/{np.mean(ci):.2f}          | {100*wins/len(ii):.0f}%")
