"""Run the ALREADY-trained mixed component-net (synthetic domain) DIRECTLY on real YS_2017 points
(scale-matching: 220m patch, kp~0.06 ~ synthetic domain). Coords shifted to training range (physical scale kept).
net+refine vs RELAX on real YS. No retraining."""
import os, sys, numpy as np, torch, torch.nn as nn
sys.path.insert(0,'/home/vexit/wavesplat/wavesplat'); os.chdir('/home/vexit/wavesplat/wavesplat'); sys.path.insert(0,'scripts')
from d52_F4_neural_operator import DEVICE
from amortized_ff import FourierFeatures
from scripts.realdata_analysis import load_surface_nc, peak_period
from wavesplat.model.tnse import fit_amplitudes_phases, render_wave_field
from estimator_improved import relax_cv, estimate_klim
from d17_recursive_estimator import refine_params
from scipy.interpolate import griddata
G=9.81; KMAX=0.6; MOUT=16; DEPTH=30.0
class Net(nn.Module):   # SAME arch as component_distill_mixed
    def __init__(s,d=192,heads=6,enc=4):
        super().__init__(); s.ff=FourierFeatures(48,learnable=False); fd=2*48
        s.oenc=nn.Sequential(nn.Linear(fd+1,d),nn.GELU(),nn.Linear(d,d))
        el=nn.TransformerEncoderLayer(d,heads,4*d,batch_first=True,activation='gelu',norm_first=True); s.enc=nn.TransformerEncoder(el,enc)
        s.qtok=nn.Parameter(torch.randn(1,MOUT,d)*0.02); s.attn=nn.MultiheadAttention(d,heads,batch_first=True)
        s.head=nn.Sequential(nn.Linear(d,d),nn.GELU(),nn.Linear(d,2))
    def forward(s,obs):
        h=s.enc(s.oenc(torch.cat([s.ff(obs[...,:3]),obs[...,3:4]],-1)))
        q=s.qtok.expand(obs.shape[0],-1,-1); r,_=s.attn(q,h,h); return torch.tanh(s.head(r))*KMAX
M=Net().to(DEVICE); M.load_state_dict(torch.load('results/direction3/component_distill_mixed.pt',map_location=DEVICE)); M.eval()
F='data/real_bergamasco/BS_2011/Surfaces_20111004_113800_short.nc'
Z,gxf,gyf,t,meta=load_surface_nc(F,tmax_frames=400); T,Ny,Nx=Z.shape; dt=float(np.median(np.diff(t)))
# shift coords to synthetic training range (x from 10, y centered), keep physical scale (wavenumbers preserved)
gxs=gxf-gxf.min()+10.0; gys=gyf-(gyf.min()+gyf.max())/2
print(f"BS: patch {gxf.max()-gxf.min():.0f}x{gyf.max()-gyf.min():.0f}m -> shifted x[{gxs.min():.0f},{gxs.max():.0f}] y[{gys.min():.0f},{gys.max():.0f}]  fps{1/dt:.0f}")
Tp,_=peak_period(Z,t); Tp=max(Tp,3.0); n_back=min(int(2.2*Tp/dt),T-2)
# coarse eval grid over shifted patch
gx=np.linspace(gxs.min(),gxs.max(),80); gy=np.linspace(gys.min(),gys.max(),60); Xg,Yg=np.meshgrid(gx,gy)
Xf,Yf=np.meshgrid(gxs,gys); Xfr,Yfr=Xf.ravel(),Yf.ravel(); rng=np.random.default_rng(0)
NPER=8; frames=np.linspace(n_back+2,T-2,3).astype(int)
def score(kx,ky,obs,Tobs,gt_coarse):
    A,phi=fit_amplitudes_phases(kx,ky,obs.astype(np.float32),DEPTH,lambda_reg=0.5)
    f=render_wave_field(kx,ky,A,phi,DEPTH,gx,gy,float(Tobs)); v=~np.isnan(gt_coarse)
    return float(np.corrcoef(f[v],gt_coarse[v])[0,1])
print(f"Tp~{Tp:.1f} n_back{n_back} | frames {list(frames)}")
print("=== REAL BS_2011 (already-trained net, coords shifted): neural-guided vs RELAX ===")
for i_star in frames:
    gtf=Z[i_star]  # full-res
    gt_coarse=griddata(np.column_stack([Xfr,Yfr]),gtf.ravel(),(Xg,Yg),method='linear')  # downsample to eval grid
    Hs=4*np.nanstd(gtf); Ps,zs=[],[]
    for i in range(i_star-n_back,i_star+1):
        fr=Z[i].ravel(); idx=rng.choice(len(fr),NPER,replace=False)
        Ps.append(np.column_stack([Xfr[idx],Yfr[idx],np.full(NPER,i*dt)])); zs.append(fr[idx])
    P=np.vstack(Ps).astype(np.float32); z=np.concatenate(zs).astype(np.float32); mvalid=~np.isnan(z); P,z=P[mvalid],z[mvalid]
    obs=np.column_stack([P,z]).astype(np.float32); tgt={'gx':gx,'gy':gy,'t':float(i_star*dt),'gt':np.nan_to_num(gt_coarse),'Hs':Hs}
    with torch.no_grad(): pk=M(torch.tensor(obs[None],device=DEVICE))[0].cpu().numpy()
    c_rawnet=score(pk[:,0],pk[:,1],obs,i_star*dt,gt_coarse)
    nkx,nky,_=refine_params(pk[:,0].astype(np.float32),pk[:,1].astype(np.float32),P,z,DEPTH,tgt,iters=400); c_net=score(nkx,nky,obs,i_star*dt,gt_coarse)
    klim=estimate_klim(P,z,DEPTH); rkx,rky=relax_cv(P,z,DEPTH,klim); rkx,rky,_=refine_params(rkx,rky,P,z,DEPTH,tgt,iters=400); c_rel=score(rkx,rky,obs,i_star*dt,gt_coarse)
    print(f"  frame{i_star} nobs={len(z)}: raw-net {c_rawnet:.3f} | net+refine {c_net:.3f} | RELAX {c_rel:.3f}",flush=True)
