"""Amortized dispersion-basis coefficient regression, trained on OBS-FIT (like RELAX, not GT-field).
net(obs)->coef on fixed dispersion basis; render=Bd@coef. Structure forced by physics basis + obs-fit."""
import os, sys, numpy as np, torch, torch.nn as nn, time
sys.path.insert(0,'/home/vexit/wavesplat/wavesplat'); os.chdir('/home/vexit/wavesplat/wavesplat'); sys.path.insert(0,'scripts')
from d51_F1_continuous_density import disp_grid
from d52_F4_neural_operator import DEVICE
from wavesim.wave_generator import jonswap_components
from d40_learned_acquisition import pool_xy
G=9.81; DEPTH=50.0; K_TP=2.2
gx=np.linspace(10,250,80); gy=np.linspace(-100,100,60); Xg,Yg=np.meshgrid(gx,gy); grid=np.column_stack([Xg.ravel(),Yg.ravel()])
KX,KY,OM=disp_grid(16,kmax=0.6); N=len(KX); print(f"basis N={N}")
KXt=torch.tensor(KX,dtype=torch.float32,device=DEVICE); KYt=torch.tensor(KY,dtype=torch.float32,device=DEVICE); OMt=torch.tensor(OM,dtype=torch.float32,device=DEVICE)

def Bd(P):  # P (...,3) -> (...,2N) design [cos, -sin]
    psi=P[...,0:1]*KXt+P[...,1:2]*KYt-P[...,2:3]*OMt
    return torch.cat([torch.cos(psi),-torch.sin(psi)],-1)

class FF(nn.Module):
    def __init__(s,n=32):
        super().__init__(); g=torch.Generator().manual_seed(0)
        s.register_buffer('B',torch.randn(3,n,generator=g)*torch.tensor([0.3,0.3,1.0])[:,None])
    def forward(s,x): p=x@s.B; return torch.cat([torch.sin(2*np.pi*p),torch.cos(2*np.pi*p)],-1)

class AmortSpectral(nn.Module):
    def __init__(s,d=192,heads=4,enc=4,nff=32):
        super().__init__(); s.ff=FF(nff)
        s.oenc=nn.Sequential(nn.Linear(2*nff+1,d),nn.GELU(),nn.Linear(d,d))
        el=nn.TransformerEncoderLayer(d,heads,4*d,batch_first=True,activation='gelu',norm_first=True)
        s.enc=nn.TransformerEncoder(el,enc)
        s.head=nn.Sequential(nn.Linear(d,d),nn.GELU(),nn.Linear(d,2*N))
    def forward(s,obs):
        h=s.enc(s.oenc(torch.cat([s.ff(obs[...,:3]),obs[...,3:4]],-1)))
        return s.head(h.mean(1))    # (B,2N) coefficients

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

M=AmortSpectral().to(DEVICE); opt=torch.optim.AdamW(M.parameters(),lr=3e-4,weight_decay=1e-4)
STEPS=6000; sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,STEPS); REG=1e-3; t0=time.time()
for s in range(STEPS):
    obs,Hs=gen(24); coef=M(obs)                       # (B,2N)
    predobs=(Bd(obs[...,:3])*coef[:,None,:]).sum(-1)   # (B,200) fit obs
    loss=(((predobs-obs[...,3])**2).mean(1)/(Hs**2+1e-6)).mean() + REG*(coef**2).mean()
    opt.zero_grad(); loss.backward(); opt.step(); sch.step()
    if s%1000==0 or s==STEPS-1:
        with torch.no_grad(): obsrmse=((predobs-obs[...,3])**2).mean().sqrt().item()
        print(f"  step {s} loss {loss.item():.4f} obs-fit-rmse {obsrmse:.3f} (obs std {obs[...,3].std().item():.3f}) ({time.time()-t0:.0f}s)",flush=True)
torch.save(M.state_dict(),'results/direction3/amortized_spectral.pt'); print("saved amortized_spectral.pt")

# eval: field vs GT on fixed episodes
M.eval()
def epi(Hs,Tp,seed):
    c=jonswap_components(Hs=Hs,Tp=Tp,n_components=64,depth=DEPTH,seed=seed)
    A=np.array([x.A for x in c]);om=np.array([x.omega for x in c]);kx=np.array([x.kx for x in c]);ky=np.array([x.ky for x in c]);ph=np.array([x.phi for x in c])
    Tobs=K_TP*Tp;pool=pool_xy();rng=np.random.default_rng(seed);oi=rng.integers(0,len(pool),200);ox,oy=pool[oi,0],pool[oi,1];ot=rng.uniform(0,Tobs,200)
    oz=(A[None]*np.cos(kx[None]*ox[:,None]+ky[None]*oy[:,None]-om[None]*ot[:,None]+ph[None])).sum(1)+rng.normal(0,0.02,200)
    gt=(A[None]*np.cos(kx[None]*Xg.ravel()[:,None]+ky[None]*Yg.ravel()[:,None]-om[None]*Tobs+ph[None])).sum(1).reshape(Xg.shape)
    return np.column_stack([ox,oy,ot,oz]).astype(np.float32),Tobs,gt
print("=== amortized-spectral field vs GT (nowcast t*) ===")
for Hs,Tp,seed in [(0.8,6.0,1),(1.2,7.0,2),(2.0,9.0,4)]:
    obs,Tobs,gt=epi(Hs,Tp,seed); ob=torch.tensor(obs[None],device=DEVICE)
    with torch.no_grad(): coef=M(ob)[0]
    q=torch.tensor(np.column_stack([grid,np.full(len(grid),Tobs)]),dtype=torch.float32,device=DEVICE)
    fld=(Bd(q)*coef[None,:]).sum(-1).cpu().numpy().reshape(Xg.shape)
    nr=np.sqrt(np.mean((fld-gt)**2))/(4*gt.std()); cr=np.corrcoef(fld.ravel(),gt.ravel())[0,1]
    print(f"  Hs{Hs} Tp{Tp}: field std {fld.std():.3f} (GT {gt.std():.3f}) nRMSE {nr:.3f} corr {cr:.3f}  (RELAX~0.28/0.7, mean 0.25/0)")
