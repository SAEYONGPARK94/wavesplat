"""DESIGNED method: Neural Matching Pursuit = LISTA (learned ISTA, soft-threshold=SPARSITY) on dispersion basis.
c_{k+1} = softThreshold(c_k - alpha_k * B^T(Bc_k - z), theta_k). Learned alpha_k, theta_k (+ per-coef scale).
Sparsity emulates RELAX's matching pursuit (the missing ingredient in dense nets). Trained on FIELD-MSE (determined)."""
import os, sys, numpy as np, torch, torch.nn as nn, time, argparse
sys.path.insert(0,'/home/vexit/wavesplat/wavesplat'); os.chdir('/home/vexit/wavesplat/wavesplat'); sys.path.insert(0,'scripts')
from d51_F1_continuous_density import disp_grid
from d52_F4_neural_operator import DEVICE
from wavesim.wave_generator import jonswap_components
from d40_learned_acquisition import pool_xy
from estimator_improved import relax_cv, estimate_klim
from d17_recursive_estimator import refine_params
G=9.81; DEPTH=50.0; K_TP=2.2
gx=np.linspace(10,250,80); gy=np.linspace(-100,100,60); Xg,Yg=np.meshgrid(gx,gy); grid=np.column_stack([Xg.ravel(),Yg.ravel()])
KX,KY,OM=disp_grid(16,kmax=0.6); N=len(KX)
KXt=torch.tensor(KX,dtype=torch.float32,device=DEVICE);KYt=torch.tensor(KY,dtype=torch.float32,device=DEVICE);OMt=torch.tensor(OM,dtype=torch.float32,device=DEVICE)
ap=argparse.ArgumentParser(); ap.add_argument('--ncomp',type=int,default=4); ap.add_argument('--nobs',type=int,default=600); ap.add_argument('--steps',type=int,default=12000); ap.add_argument('--K',type=int,default=20); a=ap.parse_args()
NC,MO,STEPS,K=a.ncomp,a.nobs,a.steps,a.K
def Bd(P):
    psi=P[...,0:1]*KXt+P[...,1:2]*KYt-P[...,2:3]*OMt
    return torch.cat([torch.cos(psi),-torch.sin(psi)],-1)
class LISTA(nn.Module):
    """Unrolled soft-thresholding on the dispersion basis. Learned per-step step-size & threshold + coef scale."""
    def __init__(s,K=20):
        super().__init__(); s.K=K
        s.log_a=nn.Parameter(torch.full((K,),np.log(1e-3)))      # step sizes
        s.log_th=nn.Parameter(torch.full((K,),np.log(1e-3)))     # soft-threshold levels
        s.log_pre=nn.Parameter(torch.zeros(2*N))                 # per-coef preconditioner
    def forward(s,obs):
        Bo=Bd(obs[...,:3]); z=obs[...,3]                         # (B,M,2N),(B,M)
        c=torch.zeros(obs.shape[0],2*N,device=obs.device); pre=torch.exp(s.log_pre)
        for k in range(s.K):
            r=(Bo*c[:,None,:]).sum(-1)-z
            g=torch.einsum('bm,bmn->bn',r,Bo)
            u=c-s.log_a[k].exp()*pre[None,:]*g
            th=s.log_th[k].exp()
            c=torch.sign(u)*torch.clamp(u.abs()-th,min=0.0)      # soft-threshold => SPARSE
        return c
def comps(Hs,Tp,seed):
    c=jonswap_components(Hs=Hs,Tp=Tp,n_components=NC,depth=DEPTH,seed=seed)
    return (np.array([x.A for x in c]),np.array([x.omega for x in c]),np.array([x.kx for x in c]),np.array([x.ky for x in c]),np.array([x.phi for x in c]))
def make(Hs,Tp,seed,Q=400,fc=6.0):
    A,om,kx,ky,ph=comps(Hs,Tp,seed); Tobs=K_TP*Tp; pool=pool_xy(); rng=np.random.default_rng(seed)
    oi=rng.integers(0,len(pool),MO); ox,oy=pool[oi,0],pool[oi,1]; ot=rng.uniform(0,Tobs,MO)
    oz=(A[None]*np.cos(kx[None]*ox[:,None]+ky[None]*oy[:,None]-om[None]*ot[:,None]+ph[None])).sum(1)+rng.normal(0,0.02,MO)
    obs=np.column_stack([ox,oy,ot,oz]).astype(np.float32)
    gt=(A[None]*np.cos(kx[None]*Xg.ravel()[:,None]+ky[None]*Yg.ravel()[:,None]-om[None]*Tobs+ph[None])).sum(1).reshape(Xg.shape)
    qx=rng.uniform(10,250,Q);qy=rng.uniform(-100,100,Q);qt=rng.uniform(0.5*Tobs,Tobs+fc,Q)
    qz=(A[None]*np.cos(kx[None]*qx[:,None]+ky[None]*qy[:,None]-om[None]*qt[:,None]+ph[None])).sum(1)
    return obs,np.column_stack([qx,qy,qt]).astype(np.float32),qz.astype(np.float32),Tobs,gt
def gen(B):
    obs=np.zeros((B,MO,4),np.float32);Q=400;qp=np.zeros((B,Q,3),np.float32);qz=np.zeros((B,Q),np.float32);Hsb=np.zeros(B,np.float32)
    for b in range(B):
        Hs=np.random.uniform(0.6,2.0);Tp=np.random.uniform(5.5,9.5);o,q,z,_,_=make(Hs,Tp,int(np.random.randint(1e9)));obs[b]=o;qp[b]=q;qz[b]=z;Hsb[b]=Hs
    t=lambda x: torch.tensor(x,device=DEVICE); return t(obs),t(qp),t(qz),t(Hsb)
EVS=[make(0.8,6.0,1),make(1.2,7.0,2),make(2.0,9.0,4)]
relax_cr=[]
for obs,q,qz,Tobs,gt in EVS:
    P=obs[:,:3].astype(np.float32);z=obs[:,3].astype(np.float32);Hs=4*gt.std();target={'gx':gx,'gy':gy,'t':float(Tobs),'gt':gt,'Hs':Hs}
    klim=estimate_klim(P,z,DEPTH);kx,ky=relax_cv(P,z,DEPTH,klim);kx,ky,fld=refine_params(kx,ky,P,z,DEPTH,target,iters=400)
    relax_cr.append(np.corrcoef(fld.ravel(),gt.ravel())[0,1])
RREF=np.mean(relax_cr)
def ev(M):
    o=[]; ncoef=[]
    for obs,q,qz,Tobs,gt in EVS:
        with torch.no_grad(): coef=M(torch.tensor(obs[None],device=DEVICE))[0]
        qg=torch.tensor(np.column_stack([grid,np.full(len(grid),Tobs)]),dtype=torch.float32,device=DEVICE)
        fld=(Bd(qg)*coef[None,:]).sum(-1).cpu().numpy().reshape(Xg.shape)
        o.append(np.corrcoef(fld.ravel(),gt.ravel())[0,1]); ncoef.append(int((coef.abs()>1e-3).sum().item()))
    return np.mean(o), np.mean(ncoef)
M=LISTA(K).to(DEVICE); opt=torch.optim.AdamW(M.parameters(),lr=5e-3); sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,STEPS); t0=time.time()
print(f"[LISTA K={K} ncomp={NC} nobs={MO}] N_basis={N}(2N={2*N}) RELAX ref {RREF:.3f} (plain-net 0.60)")
for s in range(STEPS):
    obs,q,qz,Hs=gen(24); coef=M(obs); predo=(Bd(obs[...,:3])*coef[:,None,:]).sum(-1)
    loss=(((predo-obs[...,3])**2).mean(1)/(Hs**2+1e-6)).mean()
    opt.zero_grad(); loss.backward(); opt.step(); sch.step()
    if s%1500==0 or s==STEPS-1:
        cr,nc=ev(M); print(f"  step {s:5d} loss {loss.item():.4f} | field corr {cr:.3f} (RELAX {RREF:.3f}) active_coef~{nc:.0f}/{2*N} ({time.time()-t0:.0f}s)",flush=True)
cr,nc=ev(M); print(f"VERDICT LISTA-obsfit: field corr {cr:.3f} vs RELAX {RREF:.3f} vs plain-net 0.60  (active coef {nc:.0f})")
