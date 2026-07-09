"""Demo: learned set-diffusion generative prior (DPS) produces STRUCTURED wave fields from sparse obs + UQ.
Figure: GT | DPS-BMA(mean of samples) | sample#1 | sample#2 | diff(BMA-GT), for a few dataset episodes."""
import os, sys, numpy as np, torch
sys.path.insert(0,'/home/vexit/wavesplat/wavesplat'); os.chdir('/home/vexit/wavesplat/wavesplat'); sys.path.insert(0,'scripts')
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from d3_setdiff_posterior import gather_window, dps_sample, atoms_to_field, CKPT
from d3_setdiff_prior import SetDenoiser
from d52_F4_neural_operator import DEVICE
from wavesplat.data_loader import EpisodeDataLoader
ck=torch.load(CKPT,map_location=DEVICE); stats=ck['stats']
model=SetDenoiser(**ck.get('arch',{})).to(DEVICE); model.load_state_dict(ck['model']); model.eval()
abar=ck['abar'].to(DEVICE); betas=ck['betas'].to(DEVICE)
EPS=['episode_BF4_5comp_seed0_uniform.h5','episode_BF4_15comp_seed0_uniform.h5','episode_BF3_5comp_seed1_uniform.h5']
fig,axes=plt.subplots(len(EPS),5,figsize=(15,3.0*len(EPS)))
for ei,ep in enumerate(EPS):
    L=EpisodeDataLoader('data/'+ep); depth=L.depth; Hs=L.Hs_true
    gx=L.gt_grid_x.numpy(); gy=L.gt_grid_y.numpy(); ti=25
    P,z,frames=gather_window(L,ti); tgt=frames[ti+1]
    target={'gx':gx,'gy':gy,'t':float(tgt.t),'gt':tgt.gt_field.numpy(),'Hs':Hs}
    gt=target['gt']
    Xd=dps_sample(model,stats,abar,betas,P,z,depth,guidance=0.3,n=4,seed=0)
    fields=[]
    for b in range(4):
        f,_=atoms_to_field(Xd[b],stats,P,z,depth,target)
        if f is not None: fields.append(f)
    bma=np.mean(fields,axis=0)
    def sc(f): return np.sqrt(np.mean((f-gt)**2))/max(Hs,1e-6), np.corrcoef(f.ravel(),gt.ravel())[0,1]
    nrb,crb=sc(bma)
    vmax=2*gt.std()
    panels=[(gt,'GT'),(bma,f'DPS-BMA nRMSE{nrb:.2f} r{crb:.2f}'),(fields[0],'sample #1'),(fields[1] if len(fields)>1 else fields[0],'sample #2'),(bma-gt,'diff (BMA-GT)')]
    for j,(F,ti_) in enumerate(panels):
        ax=axes[ei,j] if len(EPS)>1 else axes[j]
        cmap='PuOr' if j==4 else 'RdBu_r'
        ax.imshow(F,vmin=-vmax,vmax=vmax,cmap=cmap,origin='lower',aspect='auto'); ax.set_xticks([]); ax.set_yticks([])
        if ei==0: ax.set_title(ti_,fontsize=10)
        if j==0: ax.set_ylabel(ep.replace('episode_','').replace('_uniform.h5',''),fontsize=9)
    print(f"{ep}: DPS-BMA nRMSE {nrb:.3f} corr {crb:.3f}  field std {bma.std():.3f} (GT {gt.std():.3f})  n_samples {len(fields)}",flush=True)
fig.suptitle('Learned set-diffusion generative prior (DPS): structured wave fields from sparse obs + posterior samples (UQ)',fontsize=12)
plt.tight_layout(); os.makedirs('results/gifs',exist_ok=True)
plt.savefig('results/gifs/DPS_generative_demo.png',dpi=110,bbox_inches='tight'); print("saved results/gifs/DPS_generative_demo.png")
