"""Deep-unrolled ridge-LS solver on the dispersion basis (LISTA-style, physics-informed).
c_{k+1} = c_k - alpha_k * Precond ⊙ (Bobs^T(Bobs c_k - z) + lam c_k). Learned: alpha_k, lam, Precond.
Structure is guaranteed (unrolled GD on obs-fit) -> cannot mean-collapse. Trained on obs-fit."""
import os, sys, numpy as np, torch, torch.nn as nn, time
sys.path.insert(0,'/home/vexit/wavesplat/wavesplat'); os.chdir('/home/vexit/wavesplat/wavesplat'); sys.path.insert(0,'scripts')
from d51_F1_continuous_density import disp_grid
from d52_F4_neural_operator import DEVICE
from wavesim.wave_generator import jonswap_components
from d40_learned_acquisition import pool_xy
G=9.81; DEPTH=50.0; K_TP=2.2
gx=np.linspace(10,250,80); gy=np.linspace(-100,100,60); Xg,Yg=np.meshgrid(gx,gy); grid=np.column_stack([Xg.ravel(),Yg.ravel()])
KX,KY,OM=disp_grid(16,kmax=0.6); N=len(KX)
KXt=torch.tensor(KX,dtype=torch.float32,device=DEVICE); KYt=torch.tensor(KY,dtype=torch.float32,device=DEVICE); OMt=torch.tensor(OM,dtype=torch.float32,device=DEVICE)
def Bd(P):
    psi=P[...,0:1]*KXt+P[...,1:2]*KYt-P[...,2:3]*OMt
    return torch.cat([torch.cos(psi),-torch.sin(psi)],-1)   # (...,2N)

class Unrolled(nn.Module):
    def __init__(s,K=15):
        super().__init__(); s.K=K
        s.log_alpha=nn.Parameter(torch.full((K,),np.log(1e-3)))   # per-step size
        s.log_lam=nn.Parameter(torch.tensor(np.log(0.5)))
        s.log_pre=nn.Parameter(torch.zeros(2*N))                  # learned diagonal spectral preconditioner
    def forward(s,obs):   # obs (B,M,4)
        Bo=Bd(obs[...,:3]); z=obs[...,3]                          # (B,M,2N),(B,M)
        c=torch.zeros(obs.shape[0],2*N,device=obs.device)
        lam=s.log_lam.exp(); pre=torch.exp(s.log_pre)
        for k in range(s.K):
            r=(Bo*c[:,None,:]).sum(-1)-z                          # (B,M)
            g=torch.einsum('bm,bmn->bn',r,Bo)+lam*c               # (B,2N)
            c=c-s.log_alpha[k].exp()*pre[None,:]*g
        return c

def gen(B,ncomp=64):
    pool=pool_xy(); obs=np.zeros((B,200,4),np.float32); Hsb=np.zeros(B,np.float32)
    for b in range(B):
        Hs=np.random.uniform(0.6,2.0); Tp=np.random.uniform(5.5,9.5)
        c=jonswap_components(Hs=Hs,Tp=Tp,n_components=ncomp,depth=DEPTH,seed=int(np.random.randint(1e9)))
        A=np.array([x.A for x in c]);om=np.array([x.omega for x in c]);kx=np.array([x.kx for x in c]);ky=np.array([x.ky for x in c]);ph=np.array([x.phi for x in c])
        Tobs=K_TP*Tp; oi=np.random.randint(0,len(pool),200); ox,oy=pool[oi,0],pool[oi,1]; ot=np.random.uniform(0,Tobs,200)
        oz=(A[None]*np.cos(kx[None]*ox[:,None]+ky[None]*oy[:,None]-om[None]*ot[:,None]+ph[None])).sum(1)+np.random.normal(0,0.02,200)
        obs[b]=np.column_stack([ox,oy,ot,oz]); Hsb[b]=Hs
    t=lambda a: torch.tensor(a,device=DEVICE); return t(obs),t(Hsb)
def epi(Hs,Tp,seed):
    c=jonswap_components(Hs=Hs,Tp=Tp,n_components=64,depth=DEPTH,seed=seed)
    A=np.array([x.A for x in c]);om=np.array([x.omega for x in c]);kx=np.array([x.kx for x in c]);ky=np.array([x.ky for x in c]);ph=np.array([x.phi for x in c])
    Tobs=K_TP*Tp;pool=pool_xy();rng=np.random.default_rng(seed);oi=rng.integers(0,len(pool),200);ox,oy=pool[oi,0],pool[oi,1];ot=rng.uniform(0,Tobs,200)
    oz=(A[None]*np.cos(kx[None]*ox[:,None]+ky[None]*oy[:,None]-om[None]*ot[:,None]+ph[None])).sum(1)+rng.normal(0,0.02,200)
    gt=(A[None]*np.cos(kx[None]*Xg.ravel()[:,None]+ky[None]*Yg.ravel()[:,None]-om[None]*Tobs+ph[None])).sum(1).reshape(Xg.shape)
    return np.column_stack([ox,oy,ot,oz]).astype(np.float32),Tobs,gt
EVS=[epi(0.8,6.0,1),epi(1.2,7.0,2),epi(2.0,9.0,4)]
def field_eval(M):
    out=[]
    for obs,Tobs,gt in EVS:
        with torch.no_grad(): coef=M(torch.tensor(obs[None],device=DEVICE))[0]
        q=torch.tensor(np.column_stack([grid,np.full(len(grid),Tobs)]),dtype=torch.float32,device=DEVICE)
        fld=(Bd(q)*coef[None,:]).sum(-1).cpu().numpy().reshape(Xg.shape)
        out.append((np.sqrt(np.mean((fld-gt)**2))/(4*gt.std()), np.corrcoef(fld.ravel(),gt.ravel())[0,1], fld.std()/gt.std()))
    return out

M=Unrolled(K=15).to(DEVICE); opt=torch.optim.AdamW(M.parameters(),lr=1e-2)
STEPS=4000; sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,STEPS); t0=time.time()
print(f"basis N={N}, unrolled K=15")
for s in range(STEPS):
    obs,Hs=gen(32); coef=M(obs)
    predobs=(Bd(obs[...,:3])*coef[:,None,:]).sum(-1)
    loss=(((predobs-obs[...,3])**2).mean(1)/(Hs**2+1e-6)).mean()
    opt.zero_grad(); loss.backward(); opt.step(); sch.step()
    if s%500==0 or s==STEPS-1:
        with torch.no_grad(): ofr=((predobs-obs[...,3])**2).mean().sqrt().item()
        ev=field_eval(M); mnr=np.mean([e[0] for e in ev]); mcr=np.mean([e[1] for e in ev]); mvr=np.mean([e[2] for e in ev])
        print(f"  step {s:4d} obsfit {ofr:.3f}/{obs[...,3].std().item():.3f} | field nRMSE {mnr:.3f} corr {mcr:.3f} std-ratio {mvr:.2f} ({time.time()-t0:.0f}s)",flush=True)
torch.save(M.state_dict(),'results/direction3/unrolled_solver.pt'); print("saved unrolled_solver.pt")
print("(RELAX~0.28/0.7, mean 0.25/0)")
