"""
C validation: default iterative_relax vs iterative_relax_cv (adaptive klim + CV model order).
Must NOT regress synthetic headline; SHOULD help smooth/sparse (La Jument) and short-wave (BS).
"""
import os, sys, numpy as np
sys.path.insert(0, '/home/vexit/wavesplat/wavesplat')
os.chdir('/home/vexit/wavesplat/wavesplat'); sys.path.insert(0, 'scripts')
from d26_relax_iterative import iterative_relax
from estimator_improved import iterative_relax_cv
from wavesplat.evaluation.metrics import pearson_correlation

def sc(f, gt, Hs):
    v = ~np.isnan(gt)
    return float(np.sqrt(np.mean((f[v]-gt[v])**2))/Hs), float(pearson_correlation(f[v], gt[v]))

# ---------- synthetic headline ----------
from wavesplat.data_loader import EpisodeDataLoader
from d6_integrated_eval import tag_to_file
from d15_sequential_fusion import gather_frames
print("=== SYNTHETIC headline (must not regress) ===")
print(f"{'episode':12s} | {'default N/nRMSE/corr':>22} | {'improved N/nRMSE/corr/klim':>26}")
for tag in ['BF4_15c_s0','BF4_30c_s0','BF5_30c_s1']:
    L = EpisodeDataLoader('data/'+tag_to_file(tag)); depth=L.depth; Hs=L.Hs_true
    n_back=int(round(2.2*L.Tp_true/0.1)); fstar=L[250]
    target={'gx':L.gt_grid_x.numpy(),'gy':L.gt_grid_y.numpy(),'t':float(fstar.t),'gt':fstar.gt_field.numpy(),'Hs':Hs}
    P,z=gather_frames(L,250,n_back); P=P.astype(np.float32); z=z.astype(np.float32)
    fd,Nd=iterative_relax(P,z,depth,target,3); nr_d,cr_d=sc(fd,target['gt'],Hs)
    fi,Ni,kl=iterative_relax_cv(P,z,depth,target); nr_i,cr_i=sc(fi,target['gt'],Hs)
    print(f"{tag:12s} | {Nd:3d} {nr_d:.3f} {cr_d:.2f}          | {Ni:3d} {nr_i:.3f} {cr_i:.2f}  klim={kl:.2f}")

# ---------- real: La Jument (smooth/sparse) & BS (short-wave) ----------
from scripts.realdata_analysis import load_surface_nc
def real_case(F, depth, Tp, nper, label):
    Z,gx,gy,t,meta=load_surface_nc(F,tmax_frames=650); T,Ny,Nx=Z.shape; dt=float(np.median(np.diff(t)))
    Xg,Yg=np.meshgrid(gx,gy); rng=np.random.default_rng(0); nb=min(int(4*Tp/dt),T-2); i=T-2
    gt=Z[i]; Hs=4*np.nanstd(gt); Ps,zs=[],[]
    for k in range(i-nb,i+1):
        idx=rng.choice(Nx*Ny,nper,replace=False)
        Ps.append(np.column_stack([Xg.ravel()[idx],Yg.ravel()[idx],np.full(nper,k*dt)])); zs.append(Z[k].ravel()[idx])
    P=np.vstack(Ps).astype(np.float32); z=np.concatenate(zs).astype(np.float32)
    target={'gx':gx,'gy':gy,'t':float(i*dt),'gt':np.nan_to_num(gt),'Hs':Hs}
    fd,Nd=iterative_relax(P,z,depth,target,3); nr_d,cr_d=sc(fd,gt,Hs)
    fi,Ni,kl=iterative_relax_cv(P,z,depth,target); nr_i,cr_i=sc(fi,gt,Hs)
    print(f"{label:12s} | {Nd:3d} {nr_d:.3f} {cr_d:.2f}          | {Ni:3d} {nr_i:.3f} {cr_i:.2f}  klim={kl:.2f}")

print("\n=== REAL (should help) ===")
print(f"{'record':12s} | {'default N/nRMSE/corr':>22} | {'improved N/nRMSE/corr/klim':>26}")
real_case('data/real_bergamasco/LJ_2018/Surfaces_20180103_093938_Grid1_short.nc', 40.0, 12.8, 30, 'La Jument')
real_case('data/real_bergamasco/BS_2011/Surfaces_20111004_113800_short.nc', 30.0, 2.4, 40, 'BS_2011')
