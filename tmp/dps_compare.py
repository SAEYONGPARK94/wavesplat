"""Fair review: RELAX (physics) vs DPS-old(260ep) vs DPS-new(600ep bigger), guidance=0.1, same obs."""
import os, sys, numpy as np, torch
sys.path.insert(0,'/home/vexit/wavesplat/wavesplat'); os.chdir('/home/vexit/wavesplat/wavesplat'); sys.path.insert(0,'scripts')
from d3_setdiff_posterior import gather_window, dps_sample, atoms_to_field
from d3_setdiff_prior import SetDenoiser
from d52_F4_neural_operator import DEVICE
from estimator_improved import relax_cv, estimate_klim
from d17_recursive_estimator import refine_params
from wavesplat.data_loader import EpisodeDataLoader
EPS=['episode_BF4_5comp_seed0_uniform.h5','episode_BF4_15comp_seed0_uniform.h5','episode_BF3_5comp_seed1_uniform.h5',
     'episode_BF4_30comp_seed0_uniform.h5','episode_BF5_15comp_seed0_uniform.h5']
def load_dps(path):
    ck=torch.load(path,map_location=DEVICE); m=SetDenoiser(**ck.get('arch',{})).to(DEVICE); m.load_state_dict(ck['model']); m.eval()
    return m,ck['stats'],ck['abar'].to(DEVICE),ck['betas'].to(DEVICE)
def dps_eval(prior,P,z,depth,target,g=0.1,n=6):
    m,stats,abar,betas=prior; Xd=dps_sample(m,stats,abar,betas,P,z,depth,guidance=g,n=n,seed=0)
    fs=[atoms_to_field(Xd[b],stats,P,z,depth,target)[0] for b in range(n)]; fs=[f for f in fs if f is not None]
    if not fs: return 1.0,0.0,0.0
    bma=np.mean(fs,axis=0); nr=np.sqrt(np.mean((bma-target['gt'])**2))/target['Hs']; cr=np.corrcoef(bma.ravel(),target['gt'].ravel())[0,1]
    return nr,cr,bma.std()
old=load_dps('results/direction3/setdiff_prior.pt')
newp=load_dps('results/direction3/setdiff_prior_long.pt') if os.path.exists('results/direction3/setdiff_prior_long.pt') else None
print(f"{'episode':30s} {'RELAX nr/cr':>14} {'DPSold nr/cr':>14} {'DPSnew nr/cr':>14}")
for ep in EPS:
    L=EpisodeDataLoader('data/'+ep); depth=L.depth; Hs=L.Hs_true
    gx=L.gt_grid_x.numpy(); gy=L.gt_grid_y.numpy(); ti=25
    P,z,frames=gather_window(L,ti); tgt=frames[ti+1]
    target={'gx':gx,'gy':gy,'t':float(tgt.t),'gt':tgt.gt_field.numpy(),'Hs':Hs}
    klim=estimate_klim(P,z,depth); kx,ky=relax_cv(P,z,depth,klim); kx,ky,fld=refine_params(kx,ky,P,z,depth,target,iters=400)
    rnr=np.sqrt(np.mean((fld-target['gt'])**2))/Hs; rcr=np.corrcoef(fld.ravel(),target['gt'].ravel())[0,1]
    onr,ocr,_=dps_eval(old,P,z,depth,target)
    nnr,ncr,_=(dps_eval(newp,P,z,depth,target) if newp else (float('nan'),float('nan'),0))
    print(f"{ep[:30]:30s} {rnr:6.3f}/{rcr:5.2f} {onr:6.3f}/{ocr:5.2f} {nnr:6.3f}/{ncr:5.2f}",flush=True)
print("\n(g=0.1; RELAX=물리 ceiling, DPSold=260ep, DPSnew=600ep d256l8)")
