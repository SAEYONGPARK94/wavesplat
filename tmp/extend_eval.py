"""Extension eval: net+refine vs RELAX across (ncomp, nobs) grid (3 seeds each). Maps generalization of the
mixed component-net beyond training range (comp up to 30, obs down to 150 = sparse/info-limited)."""
import os, sys, numpy as np, torch, torch.nn as nn
sys.path.insert(0,'/home/vexit/wavesplat/wavesplat'); os.chdir('/home/vexit/wavesplat/wavesplat'); sys.path.insert(0,'scripts')
from d52_F4_neural_operator import DEVICE
from amortized_ff import FourierFeatures
from wavesim.wave_generator import jonswap_components
from d40_learned_acquisition import pool_xy
from wavesplat.model.tnse import fit_amplitudes_phases, render_wave_field
from estimator_improved import relax_cv, estimate_klim
from d17_recursive_estimator import refine_params
G=9.81; DEPTH=50.0; K_TP=2.2; KMAX=0.6; MOUT=16
gx=np.linspace(10,250,80); gy=np.linspace(-100,100,60); Xg,Yg=np.meshgrid(gx,gy)
class Net(nn.Module):
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
def epi(NC,MOB,seed):
    Hs=0.6+1.4*((seed*7)%5)/5; Tp=5.5+4*((seed*3)%5)/5
    c=jonswap_components(Hs=Hs,Tp=Tp,n_components=NC,depth=DEPTH,seed=seed)
    A=np.array([x.A for x in c]);om=np.array([x.omega for x in c]);kx=np.array([x.kx for x in c]);ky=np.array([x.ky for x in c]);ph=np.array([x.phi for x in c])
    Tobs=K_TP*Tp;pool=pool_xy();rng=np.random.default_rng(seed);oi=rng.integers(0,len(pool),MOB);ox,oy=pool[oi,0],pool[oi,1];ot=rng.uniform(0,Tobs,MOB)
    oz=(A[None]*np.cos(kx[None]*ox[:,None]+ky[None]*oy[:,None]-om[None]*ot[:,None]+ph[None])).sum(1)+rng.normal(0,0.02,MOB)
    gt=(A[None]*np.cos(kx[None]*Xg.ravel()[:,None]+ky[None]*Yg.ravel()[:,None]-om[None]*Tobs+ph[None])).sum(1).reshape(Xg.shape)
    return np.column_stack([ox,oy,ot,oz]).astype(np.float32),Tobs,gt
def corr(kx,ky,obs,Tobs,gt):
    A,phi=fit_amplitudes_phases(kx,ky,obs.astype(np.float32),DEPTH,lambda_reg=0.5)
    f=render_wave_field(kx,ky,A,phi,DEPTH,gx,gy,float(Tobs)); return np.corrcoef(f.ravel(),gt.ravel())[0,1]
print(f"{'ncomp':>5} {'nobs':>5} | {'net+refine':>11} {'RELAX':>7}  (corr, 3-seed mean)")
for NC in [4,8,12,20,30]:
    for MOB in [600,300,150]:
        nh=[]; rr=[]
        for sd in [1,2,3]:
            obs,Tobs,gt=epi(NC,MOB,sd); P=obs[:,:3].astype(np.float32); z=obs[:,3].astype(np.float32); Hs4=4*gt.std()
            tgt={'gx':gx,'gy':gy,'t':float(Tobs),'gt':gt,'Hs':Hs4}
            with torch.no_grad(): pk=M(torch.tensor(obs[None],device=DEVICE))[0].cpu().numpy()
            nkx,nky,_=refine_params(pk[:,0].astype(np.float32),pk[:,1].astype(np.float32),P,z,DEPTH,tgt,iters=400); nh.append(corr(nkx,nky,obs,Tobs,gt))
            klim=estimate_klim(P,z,DEPTH); rkx,rky=relax_cv(P,z,DEPTH,klim); rkx,rky,_=refine_params(rkx,rky,P,z,DEPTH,tgt,iters=400); rr.append(corr(rkx,rky,obs,Tobs,gt))
        print(f"{NC:5d} {MOB:5d} | {np.mean(nh):11.3f} {np.mean(rr):7.3f}",flush=True)
