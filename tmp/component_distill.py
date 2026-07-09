"""Learn RELAX's core strength = OFF-GRID continuous component identification.
Net(obs) -> M_out continuous (kx,ky); Chamfer loss to TRUE components (known in sim, amplitude-weighted).
At eval: predicted (kx,ky) -> LS fit A,phi on obs -> render. This is supervised component identification
(RELAX's skill) with off-grid outputs -> can reach RELAX quality (grid methods capped ~0.3)."""
import os, sys, numpy as np, torch, torch.nn as nn, time, argparse
sys.path.insert(0,'/home/vexit/wavesplat/wavesplat'); os.chdir('/home/vexit/wavesplat/wavesplat'); sys.path.insert(0,'scripts')
from d52_F4_neural_operator import DEVICE
from amortized_ff import FourierFeatures
from wavesim.wave_generator import jonswap_components
from d40_learned_acquisition import pool_xy
from wavesplat.model.tnse import fit_amplitudes_phases, render_wave_field
G=9.81; DEPTH=50.0; K_TP=2.2; KMAX=0.6
gx=np.linspace(10,250,80); gy=np.linspace(-100,100,60); Xg,Yg=np.meshgrid(gx,gy); grid=np.column_stack([Xg.ravel(),Yg.ravel()])
ap=argparse.ArgumentParser(); ap.add_argument('--ncomp',type=int,default=4); ap.add_argument('--nobs',type=int,default=600); ap.add_argument('--steps',type=int,default=12000); ap.add_argument('--mout',type=int,default=10); a=ap.parse_args()
NC,MO,STEPS,MOUT=a.ncomp,a.nobs,a.steps,a.mout
class Net(nn.Module):
    """obs -> encoder -> MOUT query tokens -> each predicts a continuous (kx,ky) in [-KMAX,KMAX]."""
    def __init__(s,d=192,heads=6,enc=4):
        super().__init__(); s.ff=FourierFeatures(48,learnable=False); fd=2*48
        s.oenc=nn.Sequential(nn.Linear(fd+1,d),nn.GELU(),nn.Linear(d,d))
        el=nn.TransformerEncoderLayer(d,heads,4*d,batch_first=True,activation='gelu',norm_first=True); s.enc=nn.TransformerEncoder(el,enc)
        s.qtok=nn.Parameter(torch.randn(1,MOUT,d)*0.02); s.attn=nn.MultiheadAttention(d,heads,batch_first=True)
        s.head=nn.Sequential(nn.Linear(d,d),nn.GELU(),nn.Linear(d,2))
    def forward(s,obs):
        h=s.enc(s.oenc(torch.cat([s.ff(obs[...,:3]),obs[...,3:4]],-1)))
        q=s.qtok.expand(obs.shape[0],-1,-1); r,_=s.attn(q,h,h)
        return torch.tanh(s.head(r))*KMAX          # (B,MOUT,2) continuous (kx,ky)
def comps(Hs,Tp,seed):
    c=jonswap_components(Hs=Hs,Tp=Tp,n_components=NC,depth=DEPTH,seed=seed)
    return (np.array([x.A for x in c]),np.array([x.omega for x in c]),np.array([x.kx for x in c]),np.array([x.ky for x in c]),np.array([x.phi for x in c]))
def make(Hs,Tp,seed):
    A,om,kx,ky,ph=comps(Hs,Tp,seed); Tobs=K_TP*Tp; pool=pool_xy(); rng=np.random.default_rng(seed)
    oi=rng.integers(0,len(pool),MO); ox,oy=pool[oi,0],pool[oi,1]; ot=rng.uniform(0,Tobs,MO)
    oz=(A[None]*np.cos(kx[None]*ox[:,None]+ky[None]*oy[:,None]-om[None]*ot[:,None]+ph[None])).sum(1)+rng.normal(0,0.02,MO)
    gt=(A[None]*np.cos(kx[None]*Xg.ravel()[:,None]+ky[None]*Yg.ravel()[:,None]-om[None]*Tobs+ph[None])).sum(1).reshape(Xg.shape)
    return np.column_stack([ox,oy,ot,oz]).astype(np.float32),Tobs,gt,np.column_stack([kx,ky]).astype(np.float32),A.astype(np.float32)
def gen(B):
    obs=np.zeros((B,MO,4),np.float32); tk=np.zeros((B,NC,2),np.float32); ta=np.zeros((B,NC),np.float32)
    for b in range(B):
        Hs=np.random.uniform(0.6,2.0);Tp=np.random.uniform(5.5,9.5);o,_,_,k,A=make(Hs,Tp,int(np.random.randint(1e9)));obs[b]=o;tk[b]=k;ta[b]=A
    t=lambda x: torch.tensor(x,device=DEVICE); return t(obs),t(tk),t(ta)
def chamfer(pred,true,tw):   # pred (B,MOUT,2), true (B,NC,2), tw (B,NC) amplitude weights
    d=((pred[:,:,None,:]-true[:,None,:,:])**2).sum(-1)               # (B,MOUT,NC)
    t2p=d.min(1).values                                              # each true -> nearest pred (B,NC)
    p2t=d.min(2).values                                              # each pred -> nearest true (B,MOUT)
    return ((t2p*tw).sum(1)/(tw.sum(1)+1e-9)).mean() + 0.2*p2t.mean()
EVS=[make(0.8,6.0,1),make(1.2,7.0,2),make(2.0,9.0,4)]
def ev(M):
    o=[]
    for obs,Tobs,gt,tk,ta in EVS:
        with torch.no_grad(): pk=M(torch.tensor(obs[None],device=DEVICE))[0].cpu().numpy()
        A,phi=fit_amplitudes_phases(pk[:,0],pk[:,1],obs.astype(np.float32),DEPTH,lambda_reg=0.5)
        keep=A>0.05*A.max()
        fld=render_wave_field(pk[keep,0],pk[keep,1],A[keep],phi[keep],DEPTH,gx,gy,float(Tobs))
        o.append(np.corrcoef(fld.ravel(),gt.ravel())[0,1])
    return np.mean(o)
M=Net().to(DEVICE); opt=torch.optim.AdamW(M.parameters(),lr=3e-4,weight_decay=1e-4); sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,STEPS); t0=time.time()
print(f"[component-distill MOUT={MOUT} ncomp={NC} nobs={MO}] target=TRUE components, RELAX~0.98, plain-net 0.60")
for s in range(STEPS):
    obs,tk,ta=gen(24); pk=M(obs); loss=chamfer(pk,tk,ta)
    opt.zero_grad(); loss.backward(); opt.step(); sch.step()
    if s%1500==0 or s==STEPS-1: print(f"  step {s:5d} chamfer {loss.item():.4f} | net field corr {ev(M):.3f} (RELAX 0.98) ({time.time()-t0:.0f}s)",flush=True)
print(f"VERDICT component-distill: net corr {ev(M):.3f} vs RELAX 0.98 vs plain-net 0.60")
torch.save(M.state_dict(),'results/direction3/component_distill.pt')
