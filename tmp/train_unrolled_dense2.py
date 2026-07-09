"""Deep-unrolled ridge-LS, FIELD loss, DENSE obs (M=1200) — determined regime.
Demonstrates a learned physics-informed solver produces structured wave fields when the problem is determined."""
import os, sys, numpy as np, torch, torch.nn as nn, time
sys.path.insert(0,'/home/vexit/wavesplat/wavesplat'); os.chdir('/home/vexit/wavesplat/wavesplat'); sys.path.insert(0,'scripts')
from d51_F1_continuous_density import disp_grid
from d52_F4_neural_operator import DEVICE
from wavesim.wave_generator import jonswap_components
from d40_learned_acquisition import pool_xy
G=9.81; DEPTH=50.0; K_TP=2.2; M_OBS=1200
gx=np.linspace(10,250,80); gy=np.linspace(-100,100,60); Xg,Yg=np.meshgrid(gx,gy); grid=np.column_stack([Xg.ravel(),Yg.ravel()])
KX,KY,OM=disp_grid(16,kmax=0.6); N=len(KX)
KXt=torch.tensor(KX,dtype=torch.float32,device=DEVICE); KYt=torch.tensor(KY,dtype=torch.float32,device=DEVICE); OMt=torch.tensor(OM,dtype=torch.float32,device=DEVICE)
def Bd(P):
    psi=P[...,0:1]*KXt+P[...,1:2]*KYt-P[...,2:3]*OMt
    return torch.cat([torch.cos(psi),-torch.sin(psi)],-1)
class Unrolled(nn.Module):
    def __init__(s,K=15):
        super().__init__(); s.K=K
        s.log_alpha=nn.Parameter(torch.full((K,),np.log(3e-4)))
        s.log_lam=nn.Parameter(torch.tensor(np.log(2.0)))
        s.log_pre=nn.Parameter(torch.zeros(2*N))
    def forward(s,obs):
        Bo=Bd(obs[...,:3]); z=obs[...,3]
        c=torch.zeros(obs.shape[0],2*N,device=obs.device); lam=s.log_lam.exp(); pre=torch.exp(s.log_pre)
        for k in range(s.K):
            r=(Bo*c[:,None,:]).sum(-1)-z
            g=torch.einsum('bm,bmn->bn',r,Bo)+lam*c
            c=c-s.log_alpha[k].exp()*pre[None,:]*g
        return c
def comps(Hs,Tp,seed):
    c=jonswap_components(Hs=Hs,Tp=Tp,n_components=64,depth=DEPTH,seed=seed)
    return (np.array([x.A for x in c]),np.array([x.omega for x in c]),np.array([x.kx for x in c]),np.array([x.ky for x in c]),np.array([x.phi for x in c]))
def make(Hs,Tp,seed,M,Q=0,fc=6.0):
    A,om,kx,ky,ph=comps(Hs,Tp,seed); Tobs=K_TP*Tp; pool=pool_xy(); rng=np.random.default_rng(seed)
    oi=rng.integers(0,len(pool),M); ox,oy=pool[oi,0],pool[oi,1]; ot=rng.uniform(0,Tobs,M)
    oz=(A[None]*np.cos(kx[None]*ox[:,None]+ky[None]*oy[:,None]-om[None]*ot[:,None]+ph[None])).sum(1)+rng.normal(0,0.02,M)
    obs=np.column_stack([ox,oy,ot,oz]).astype(np.float32)
    gt=(A[None]*np.cos(kx[None]*Xg.ravel()[:,None]+ky[None]*Yg.ravel()[:,None]-om[None]*Tobs+ph[None])).sum(1).reshape(Xg.shape)
    if Q:
        qx=rng.uniform(10,250,Q);qy=rng.uniform(-100,100,Q);qt=rng.uniform(0.5*Tobs,Tobs+fc,Q)
        qz=(A[None]*np.cos(kx[None]*qx[:,None]+ky[None]*qy[:,None]-om[None]*qt[:,None]+ph[None])).sum(1)
        return obs,np.column_stack([qx,qy,qt]).astype(np.float32),qz.astype(np.float32),Tobs,gt
    return obs,Tobs,gt
def gen(B,Q=400):
    obs=np.zeros((B,M_OBS,4),np.float32); qpt=np.zeros((B,Q,3),np.float32); qz=np.zeros((B,Q),np.float32); Hsb=np.zeros(B,np.float32)
    for b in range(B):
        Hs=np.random.uniform(0.6,2.0); Tp=np.random.uniform(5.5,9.5); sd=int(np.random.randint(1e9))
        o,q,z,Tobs,_=make(Hs,Tp,sd,M_OBS,Q); obs[b]=o; qpt[b]=q; qz[b]=z; Hsb[b]=Hs
    t=lambda a: torch.tensor(a,device=DEVICE); return t(obs),t(qpt),t(qz),t(Hsb)
EVS=[make(0.8,6.0,1,M_OBS),make(1.2,7.0,2,M_OBS),make(2.0,9.0,4,M_OBS)]
def field_eval(M):
    o=[]
    for obs,Tobs,gt in EVS:
        with torch.no_grad(): coef=M(torch.tensor(obs[None],device=DEVICE))[0]
        q=torch.tensor(np.column_stack([grid,np.full(len(grid),Tobs)]),dtype=torch.float32,device=DEVICE)
        fld=(Bd(q)*coef[None,:]).sum(-1).cpu().numpy().reshape(Xg.shape)
        o.append((np.sqrt(np.mean((fld-gt)**2))/(4*gt.std()),np.corrcoef(fld.ravel(),gt.ravel())[0,1]))
    return np.mean([x[0] for x in o]),np.mean([x[1] for x in o])
M=Unrolled(K=15).to(DEVICE); opt=torch.optim.AdamW(M.parameters(),lr=1e-2)
STEPS=2500; sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,STEPS); t0=time.time()
print(f"basis N={N}, DENSE obs M={M_OBS}, unrolled K=15, FIELD loss")
for s in range(STEPS):
    obs,q,qz,Hs=gen(16); coef=M(obs); predq=(Bd(q)*coef[:,None,:]).sum(-1)
    loss=(((predq-qz)**2).mean(1)/(Hs**2+1e-6)).mean()
    opt.zero_grad(); loss.backward(); opt.step(); sch.step()
    if s%400==0 or s==STEPS-1:
        nr,cr=field_eval(M); print(f"  step {s:4d} loss {loss.item():.4f} | field nRMSE {nr:.3f} corr {cr:.3f} lam {M.log_lam.exp().item():.2f} ({time.time()-t0:.0f}s)",flush=True)
torch.save(M.state_dict(),'results/direction3/unrolled_dense.pt'); print("saved unrolled_dense.pt")
