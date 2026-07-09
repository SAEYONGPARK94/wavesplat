"""Test FF-scale-bug hypothesis: single-time spatial interpolation with FF freq MATCHED to wave scale.
Original FF scale [0.3,0.3] -> effective wavelength ~3m << obs spacing ~9m -> aliasing. Fix: x,y scale ~0.02
(wavelength ~30-300m, matched to waves). If net now learns (corr high) -> FF-scale bug confirmed+fixed."""
import os, sys, numpy as np, torch, torch.nn as nn, time, argparse
sys.path.insert(0,'/home/vexit/wavesplat/wavesplat'); os.chdir('/home/vexit/wavesplat/wavesplat'); sys.path.insert(0,'scripts')
from d52_F4_neural_operator import DEVICE
from wavesim.wave_generator import jonswap_components
from d40_learned_acquisition import pool_xy
G=9.81; DEPTH=50.0; K_TP=2.2
gx=np.linspace(10,250,80); gy=np.linspace(-100,100,60); Xg,Yg=np.meshgrid(gx,gy); grid=np.column_stack([Xg.ravel(),Yg.ravel()])
ap=argparse.ArgumentParser(); ap.add_argument('--ncomp',type=int,default=4); ap.add_argument('--nobs',type=int,default=600); ap.add_argument('--steps',type=int,default=8000); ap.add_argument('--ffscale',type=float,default=0.02); a=ap.parse_args()
NC,MO,STEPS,FFS=a.ncomp,a.nobs,a.steps,a.ffscale
class FF(nn.Module):
    def __init__(s,n=64,sx=0.02,st=0.3):
        super().__init__(); g=torch.Generator().manual_seed(0)
        # log-spaced-ish spatial freqs matched to wave wavenumbers; t freqs to omega
        s.register_buffer('B',torch.randn(3,n,generator=g)*torch.tensor([sx,sx,st])[:,None])
    def forward(s,x): p=x@s.B; return torch.cat([torch.sin(2*np.pi*p),torch.cos(2*np.pi*p)],-1)
class Net(nn.Module):
    def __init__(s,d=192,heads=4,enc=4,dec=3,nff=64,sx=0.02):
        super().__init__(); s.ff=FF(nff,sx=sx); fd=2*nff
        s.oenc=nn.Sequential(nn.Linear(fd+1,d),nn.GELU(),nn.Linear(d,d))
        el=nn.TransformerEncoderLayer(d,heads,4*d,batch_first=True,activation='gelu',norm_first=True); s.enc=nn.TransformerEncoder(el,enc)
        s.qenc=nn.Sequential(nn.Linear(fd,d),nn.GELU(),nn.Linear(d,d))
        dl=nn.TransformerDecoderLayer(d,heads,4*d,batch_first=True,activation='gelu',norm_first=True); s.dec=nn.TransformerDecoder(dl,dec)
        s.head=nn.Sequential(nn.Linear(d,d),nn.GELU(),nn.Linear(d,1))
    def forward(s,obs,q):
        ctx=s.enc(s.oenc(torch.cat([s.ff(obs[...,:3]),obs[...,3:4]],-1)))
        return s.dec(s.qenc(s.ff(q)),ctx).mean(0)[None] if False else s.head(s.dec(s.qenc(s.ff(q)),ctx))[...,0]
def comps(Hs,Tp,seed):
    c=jonswap_components(Hs=Hs,Tp=Tp,n_components=NC,depth=DEPTH,seed=seed)
    return (np.array([x.A for x in c]),np.array([x.omega for x in c]),np.array([x.kx for x in c]),np.array([x.ky for x in c]),np.array([x.phi for x in c]))
def make(Hs,Tp,seed,Q=400):
    A,om,kx,ky,ph=comps(Hs,Tp,seed); Tobs=K_TP*Tp; pool=pool_xy(); rng=np.random.default_rng(seed)
    oi=rng.integers(0,len(pool),MO); ox,oy=pool[oi,0],pool[oi,1]; ot=np.full(MO,Tobs)          # SINGLE TIME
    oz=(A[None]*np.cos(kx[None]*ox[:,None]+ky[None]*oy[:,None]-om[None]*ot[:,None]+ph[None])).sum(1)+rng.normal(0,0.02,MO)
    obs=np.column_stack([ox,oy,ot,oz]).astype(np.float32)
    gt=(A[None]*np.cos(kx[None]*Xg.ravel()[:,None]+ky[None]*Yg.ravel()[:,None]-om[None]*Tobs+ph[None])).sum(1).reshape(Xg.shape)
    qx=rng.uniform(10,250,Q);qy=rng.uniform(-100,100,Q);qt=np.full(Q,Tobs)                       # SINGLE TIME
    qz=(A[None]*np.cos(kx[None]*qx[:,None]+ky[None]*qy[:,None]-om[None]*qt[:,None]+ph[None])).sum(1)
    return obs,np.column_stack([qx,qy,qt]).astype(np.float32),qz.astype(np.float32),Tobs,gt
def gen(B):
    obs=np.zeros((B,MO,4),np.float32);Q=400;qp=np.zeros((B,Q,3),np.float32);qz=np.zeros((B,Q),np.float32);Hsb=np.zeros(B,np.float32)
    for b in range(B):
        Hs=np.random.uniform(0.6,2.0);Tp=np.random.uniform(5.5,9.5);o,q,z,Tobs,_=make(Hs,Tp,int(np.random.randint(1e9)))
        obs[b]=o;qp[b]=q;qz[b]=z;Hsb[b]=Hs
    t=lambda x: torch.tensor(x,device=DEVICE); return t(obs),t(qp),t(qz),t(Hsb)
EVS=[make(0.8,6.0,1),make(1.2,7.0,2),make(2.0,9.0,4)]
def ev(M):
    o=[]
    for obs,q,qz,Tobs,gt in EVS:
        with torch.no_grad(): fld=M(torch.tensor(obs[None],device=DEVICE),torch.tensor(np.column_stack([grid,np.full(len(grid),Tobs)])[None],dtype=torch.float32,device=DEVICE))[0].cpu().numpy().reshape(Xg.shape)
        o.append(np.corrcoef(fld.ravel(),gt.ravel())[0,1])
    return np.mean(o)
M=Net(sx=FFS).to(DEVICE); opt=torch.optim.AdamW(M.parameters(),lr=3e-4,weight_decay=1e-4); sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,STEPS); t0=time.time()
print(f"[SINGLE-TIME ncomp={NC} nobs={MO} FFscale={FFS}] (orig scale was 0.3=aliased)")
for s in range(STEPS):
    obs,q,qz,Hs=gen(24); pred=M(obs,q); loss=(((pred-qz)**2).mean(1)/(Hs**2+1e-6)).mean()
    opt.zero_grad(); loss.backward(); opt.step(); sch.step()
    if s%1000==0 or s==STEPS-1: print(f"  step {s:5d} loss {loss.item():.4f} | NET corr {ev(M):.3f} ({time.time()-t0:.0f}s)",flush=True)
print(f"VERDICT FFscale={FFS}: net corr {ev(M):.3f}  (>0.7 => interpolation LEARNS => FF-scale bug confirmed+fixed)")
