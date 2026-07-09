"""DPS guidance sweep: does tuning guidance push DPS toward the RELAX ceiling (corr 0.5-0.9)?"""
import os, sys, numpy as np, torch
sys.path.insert(0,'/home/vexit/wavesplat/wavesplat'); os.chdir('/home/vexit/wavesplat/wavesplat'); sys.path.insert(0,'scripts')
from d3_setdiff_posterior import gather_window, dps_sample, atoms_to_field, CKPT
from d3_setdiff_prior import SetDenoiser
from d52_F4_neural_operator import DEVICE
from wavesplat.data_loader import EpisodeDataLoader
ck=torch.load(CKPT,map_location=DEVICE); stats=ck['stats']
model=SetDenoiser(**ck.get('arch',{})).to(DEVICE); model.load_state_dict(ck['model']); model.eval()
abar=ck['abar'].to(DEVICE); betas=ck['betas'].to(DEVICE)
EPS=['episode_BF4_5comp_seed0_uniform.h5','episode_BF4_15comp_seed0_uniform.h5','episode_BF3_5comp_seed1_uniform.h5']
GUID=[0.1,0.3,0.6,1.0,2.0,4.0]
for ep in EPS:
    L=EpisodeDataLoader('data/'+ep); depth=L.depth; Hs=L.Hs_true
    gx=L.gt_grid_x.numpy(); gy=L.gt_grid_y.numpy(); ti=25
    P,z,frames=gather_window(L,ti); tgt=frames[ti+1]
    target={'gx':gx,'gy':gy,'t':float(tgt.t),'gt':tgt.gt_field.numpy(),'Hs':Hs}; gt=target['gt']
    row=[]
    for g in GUID:
        Xd=dps_sample(model,stats,abar,betas,P,z,depth,guidance=g,n=6,seed=0)
        fields=[]
        for b in range(6):
            f,_=atoms_to_field(Xd[b],stats,P,z,depth,target)
            if f is not None: fields.append(f)
        bma=np.mean(fields,axis=0) if fields else np.zeros_like(gt)
        nr=np.sqrt(np.mean((bma-gt)**2))/Hs; cr=np.corrcoef(bma.ravel(),gt.ravel())[0,1]
        row.append((g,nr,cr))
    best=max(row,key=lambda r:r[2])
    print(f"{ep[:28]}: "+"  ".join(f"g{g}:{cr:.2f}" for g,nr,cr in row)+f"  || best corr {best[2]:.2f}@g{best[0]} (nRMSE {best[1]:.3f})",flush=True)
