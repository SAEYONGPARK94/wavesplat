"""Hybrid test: component-net proposals + physics off-grid refinement (refine_params) -> RELAX quality?
net amortizes the beamform SEARCH (proposal); physics refines to precision. Compare raw-net / net+refine / RELAX."""
import os, sys, numpy as np, torch, torch.nn as nn, time
sys.path.insert(0,'/home/vexit/wavesplat/wavesplat'); os.chdir('/home/vexit/wavesplat/wavesplat'); sys.path.insert(0,'scripts')
from d52_F4_neural_operator import DEVICE
from amortized_ff import FourierFeatures
from wavesim.wave_generator import jonswap_components
from d40_learned_acquisition import pool_xy
from wavesplat.model.tnse import fit_amplitudes_phases, render_wave_field
from estimator_improved import relax_cv, estimate_klim
from d17_recursive_estimator import refine_params
G=9.81; DEPTH=50.0; K_TP=2.2; KMAX=0.6; NC=4; MO=600; MOUT=10
gx=np.linspace(10,250,80); gy=np.linspace(-100,100,60); Xg,Yg=np.meshgrid(gx,gy); grid=np.column_stack([Xg.ravel(),Yg.ravel()])
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
M=Net().to(DEVICE); M.load_state_dict(torch.load('results/direction3/component_distill.pt',map_location=DEVICE)); M.eval()
def comps(Hs,Tp,seed):
    c=jonswap_components(Hs=Hs,Tp=Tp,n_components=NC,depth=DEPTH,seed=seed)
    return (np.array([x.A for x in c]),np.array([x.omega for x in c]),np.array([x.kx for x in c]),np.array([x.ky for x in c]),np.array([x.phi for x in c]))
def make(Hs,Tp,seed):
    A,om,kx,ky,ph=comps(Hs,Tp,seed); Tobs=K_TP*Tp; pool=pool_xy(); rng=np.random.default_rng(seed)
    oi=rng.integers(0,len(pool),MO); ox,oy=pool[oi,0],pool[oi,1]; ot=rng.uniform(0,Tobs,MO)
    oz=(A[None]*np.cos(kx[None]*ox[:,None]+ky[None]*oy[:,None]-om[None]*ot[:,None]+ph[None])).sum(1)+rng.normal(0,0.02,MO)
    gt=(A[None]*np.cos(kx[None]*Xg.ravel()[:,None]+ky[None]*Yg.ravel()[:,None]-om[None]*Tobs+ph[None])).sum(1).reshape(Xg.shape)
    return np.column_stack([ox,oy,ot,oz]).astype(np.float32),Tobs,gt
def field(kx,ky,obs,Tobs):
    A,phi=fit_amplitudes_phases(kx,ky,obs.astype(np.float32),DEPTH,lambda_reg=0.5)
    return render_wave_field(kx,ky,A,phi,DEPTH,gx,gy,float(Tobs))
print(f"{'episode':10s} {'raw-net':>8} {'net+refine':>11} {'RELAX':>7}  (corr, determined 4comp/600obs)")
import time
tnet=trel=0
for Hs,Tp,seed in [(0.8,6.0,1),(1.2,7.0,2),(2.0,9.0,4),(1.5,8.0,7),(1.0,6.5,9)]:
    obs,Tobs,gt=make(Hs,Tp,seed); P=obs[:,:3].astype(np.float32); z=obs[:,3].astype(np.float32); Hs4=4*gt.std()
    target={'gx':gx,'gy':gy,'t':float(Tobs),'gt':gt,'Hs':Hs4}
    t0=time.time();
    with torch.no_grad(): pk=M(torch.tensor(obs[None],device=DEVICE))[0].cpu().numpy()
    # raw net
    fr=field(pk[:,0],pk[:,1],obs,Tobs); c_raw=np.corrcoef(fr.ravel(),gt.ravel())[0,1]
    # net + physics refine (seed refine_params with net proposals)
    kx,ky,fldr=refine_params(pk[:,0].astype(np.float32),pk[:,1].astype(np.float32),P,z,DEPTH,target,iters=400)
    fr2=field(kx,ky,obs,Tobs); c_hyb=np.corrcoef(fr2.ravel(),gt.ravel())[0,1]; tnet+=time.time()-t0
    # RELAX (full)
    t0=time.time(); klim=estimate_klim(P,z,DEPTH); rkx,rky=relax_cv(P,z,DEPTH,klim); rkx,rky,rfld=refine_params(rkx,rky,P,z,DEPTH,target,iters=400)
    c_rel=np.corrcoef(rfld.ravel(),gt.ravel())[0,1]; trel+=time.time()-t0
    print(f"Hs{Hs}Tp{Tp:<4} {c_raw:8.3f} {c_hyb:11.3f} {c_rel:7.3f}",flush=True)
print(f"\nmean time: net+refine {tnet/5*1000:.0f}ms  RELAX {trel/5*1000:.0f}ms")
