"""DESIGNED improved learned method (based on this session's lessons):
  physics-informed amortized estimator = wave-band FF encoder + dispersion-constrained output
  (predict coefficients on the FIXED dispersion basis -> output lives on wave manifold, cannot mean-collapse)
  + CLS-token set readout (not mean-pool) + obs-fit training (not pixel-MSE GT).
Combines: FF-scale fix + dispersion physics (why RELAX wins) + structure objective. Target: beat plain net 0.6 -> RELAX."""
import os, sys, numpy as np, torch, torch.nn as nn, time, argparse
sys.path.insert(0,'/home/vexit/wavesplat/wavesplat'); os.chdir('/home/vexit/wavesplat/wavesplat'); sys.path.insert(0,'scripts')
from d51_F1_continuous_density import disp_grid
from d52_F4_neural_operator import DEVICE
from amortized_ff import FourierFeatures   # FIXED wave-band FF
from wavesim.wave_generator import jonswap_components
from d40_learned_acquisition import pool_xy
from estimator_improved import relax_cv, estimate_klim
from d17_recursive_estimator import refine_params
G=9.81; DEPTH=50.0; K_TP=2.2
gx=np.linspace(10,250,80); gy=np.linspace(-100,100,60); Xg,Yg=np.meshgrid(gx,gy); grid=np.column_stack([Xg.ravel(),Yg.ravel()])
KX,KY,OM=disp_grid(16,kmax=0.6); N=len(KX)
KXt=torch.tensor(KX,dtype=torch.float32,device=DEVICE);KYt=torch.tensor(KY,dtype=torch.float32,device=DEVICE);OMt=torch.tensor(OM,dtype=torch.float32,device=DEVICE)
ap=argparse.ArgumentParser(); ap.add_argument('--ncomp',type=int,default=4); ap.add_argument('--nobs',type=int,default=600); ap.add_argument('--steps',type=int,default=12000); a=ap.parse_args()
NC,MO,STEPS=a.ncomp,a.nobs,a.steps
def Bd(P):
    psi=P[...,0:1]*KXt+P[...,1:2]*KYt-P[...,2:3]*OMt
    return torch.cat([torch.cos(psi),-torch.sin(psi)],-1)
class PhysAmort(nn.Module):
    """obs -> (wave-band FF) -> encoder -> CLS cross-attn readout -> dispersion-basis coefficients."""
    def __init__(s,d=192,heads=6,enc=4,nff=48):
        super().__init__(); s.ff=FourierFeatures(nff,learnable=False); fd=2*nff
        s.oenc=nn.Sequential(nn.Linear(fd+1,d),nn.GELU(),nn.Linear(d,d))
        el=nn.TransformerEncoderLayer(d,heads,4*d,batch_first=True,activation='gelu',norm_first=True); s.enc=nn.TransformerEncoder(el,enc)
        s.cls=nn.Parameter(torch.randn(1,1,d)*0.02)
        s.attn=nn.MultiheadAttention(d,heads,batch_first=True)
        s.head=nn.Sequential(nn.Linear(d,d),nn.GELU(),nn.Linear(d,2*N))
    def forward(s,obs):
        h=s.enc(s.oenc(torch.cat([s.ff(obs[...,:3]),obs[...,3:4]],-1)))          # (B,M,d)
        q=s.cls.expand(obs.shape[0],-1,-1); r,_=s.attn(q,h,h)                     # (B,1,d) CLS readout
        return s.head(r[:,0])                                                     # (B,2N) coefficients
def comps(Hs,Tp,seed):
    c=jonswap_components(Hs=Hs,Tp=Tp,n_components=NC,depth=DEPTH,seed=seed)
    return (np.array([x.A for x in c]),np.array([x.omega for x in c]),np.array([x.kx for x in c]),np.array([x.ky for x in c]),np.array([x.phi for x in c]))
def make(Hs,Tp,seed):
    A,om,kx,ky,ph=comps(Hs,Tp,seed); Tobs=K_TP*Tp; pool=pool_xy(); rng=np.random.default_rng(seed)
    oi=rng.integers(0,len(pool),MO); ox,oy=pool[oi,0],pool[oi,1]; ot=rng.uniform(0,Tobs,MO)
    oz=(A[None]*np.cos(kx[None]*ox[:,None]+ky[None]*oy[:,None]-om[None]*ot[:,None]+ph[None])).sum(1)+rng.normal(0,0.02,MO)
    gt=(A[None]*np.cos(kx[None]*Xg.ravel()[:,None]+ky[None]*Yg.ravel()[:,None]-om[None]*Tobs+ph[None])).sum(1).reshape(Xg.shape)
    return np.column_stack([ox,oy,ot,oz]).astype(np.float32),Tobs,gt
def makeq(Hs,Tp,seed,Q=400,fc=6.0):
    A,om,kx,ky,ph=comps(Hs,Tp,seed); Tobs=K_TP*Tp; pool=pool_xy(); rng=np.random.default_rng(seed)
    oi=rng.integers(0,len(pool),MO); ox,oy=pool[oi,0],pool[oi,1]; ot=rng.uniform(0,Tobs,MO)
    oz=(A[None]*np.cos(kx[None]*ox[:,None]+ky[None]*oy[:,None]-om[None]*ot[:,None]+ph[None])).sum(1)+rng.normal(0,0.02,MO)
    obs=np.column_stack([ox,oy,ot,oz]).astype(np.float32)
    qx=rng.uniform(10,250,Q);qy=rng.uniform(-100,100,Q);qt=rng.uniform(0.5*Tobs,Tobs+fc,Q)
    qz=(A[None]*np.cos(kx[None]*qx[:,None]+ky[None]*qy[:,None]-om[None]*qt[:,None]+ph[None])).sum(1)
    return obs,np.column_stack([qx,qy,qt]).astype(np.float32),qz.astype(np.float32),Hs
def gen(B,Q=400):
    obs=np.zeros((B,MO,4),np.float32);qp=np.zeros((B,Q,3),np.float32);qz=np.zeros((B,Q),np.float32);Hsb=np.zeros(B,np.float32)
    for b in range(B):
        Hs=np.random.uniform(0.6,2.0);Tp=np.random.uniform(5.5,9.5);o,q,z,_=makeq(Hs,Tp,int(np.random.randint(1e9)));obs[b]=o;qp[b]=q;qz[b]=z;Hsb[b]=Hs
    t=lambda x: torch.tensor(x,device=DEVICE); return t(obs),t(qp),t(qz),t(Hsb)
EVS=[make(0.8,6.0,1),make(1.2,7.0,2),make(2.0,9.0,4)]
relax_cr=[]
for obs,Tobs,gt in EVS:
    P=obs[:,:3].astype(np.float32);z=obs[:,3].astype(np.float32);Hs=4*gt.std();target={'gx':gx,'gy':gy,'t':float(Tobs),'gt':gt,'Hs':Hs}
    klim=estimate_klim(P,z,DEPTH);kx,ky=relax_cv(P,z,DEPTH,klim);kx,ky,fld=refine_params(kx,ky,P,z,DEPTH,target,iters=400)
    relax_cr.append(np.corrcoef(fld.ravel(),gt.ravel())[0,1])
RREF=np.mean(relax_cr)
def ev(M):
    o=[]
    for obs,Tobs,gt in EVS:
        with torch.no_grad(): coef=M(torch.tensor(obs[None],device=DEVICE))[0]
        q=torch.tensor(np.column_stack([grid,np.full(len(grid),Tobs)]),dtype=torch.float32,device=DEVICE)
        fld=(Bd(q)*coef[None,:]).sum(-1).cpu().numpy().reshape(Xg.shape)
        o.append(np.corrcoef(fld.ravel(),gt.ravel())[0,1])
    return np.mean(o)
M=PhysAmort().to(DEVICE); opt=torch.optim.AdamW(M.parameters(),lr=3e-4,weight_decay=1e-4); sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,STEPS); REG=3e-4; t0=time.time()
print(f"[PhysAmort ncomp={NC} nobs={MO}] N_basis={N}  RELAX ref corr {RREF:.3f}  (plain-net was 0.60)")
for s in range(STEPS):
    obs,q,qz,Hs=gen(24); coef=M(obs); predq=(Bd(q)*coef[:,None,:]).sum(-1)
    loss=(((predq-qz)**2).mean(1)/(Hs**2+1e-6)).mean()+REG*(coef**2).mean()
    opt.zero_grad(); loss.backward(); opt.step(); sch.step()
    if s%1500==0 or s==STEPS-1:
        ofr=0.0
        print(f"  step {s:5d} fieldMSE | field corr {ev(M):.3f} (RELAX {RREF:.3f}) ({time.time()-t0:.0f}s)",flush=True)
print(f"VERDICT PhysAmort-field: field corr {ev(M):.3f} vs RELAX {RREF:.3f} vs plain-net 0.60")
