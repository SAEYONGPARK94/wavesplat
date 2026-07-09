"""Extend: train component-net on MIXED regimes (comp 4-12, obs 300-600, MOUT=16) for generalization.
Padded variable true-comp count (zero-amplitude pad -> zero Chamfer weight)."""
import os, sys, numpy as np, torch, torch.nn as nn, time
sys.path.insert(0,'/home/vexit/wavesplat/wavesplat'); os.chdir('/home/vexit/wavesplat/wavesplat'); sys.path.insert(0,'scripts')
from d52_F4_neural_operator import DEVICE
from amortized_ff import FourierFeatures
from wavesim.wave_generator import jonswap_components
from d40_learned_acquisition import pool_xy
G=9.81; DEPTH=50.0; K_TP=2.2; KMAX=0.6; MOUT=16; MAXNC=12; STEPS=14000
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
def epi(Hs,Tp,NC,MOB,seed):
    c=jonswap_components(Hs=Hs,Tp=Tp,n_components=NC,depth=DEPTH,seed=seed)
    A=np.array([x.A for x in c]);om=np.array([x.omega for x in c]);kx=np.array([x.kx for x in c]);ky=np.array([x.ky for x in c]);ph=np.array([x.phi for x in c])
    Tobs=K_TP*Tp;pool=pool_xy();rng=np.random.default_rng(seed);oi=rng.integers(0,len(pool),MOB);ox,oy=pool[oi,0],pool[oi,1];ot=rng.uniform(0,Tobs,MOB)
    oz=(A[None]*np.cos(kx[None]*ox[:,None]+ky[None]*oy[:,None]-om[None]*ot[:,None]+ph[None])).sum(1)+rng.normal(0,0.02,MOB)
    return np.column_stack([ox,oy,ot,oz]).astype(np.float32),Tobs,kx.astype(np.float32),ky.astype(np.float32),A.astype(np.float32)
def gen(B):
    MOB=np.random.choice([300,450,600]); obs=np.zeros((B,MOB,4),np.float32); tk=np.zeros((B,MAXNC,2),np.float32); ta=np.zeros((B,MAXNC),np.float32)
    for b in range(B):
        Hs=np.random.uniform(0.6,2.0);Tp=np.random.uniform(5.5,9.5);NC=int(np.random.choice([4,6,8,12]))
        o,_,kx,ky,A=epi(Hs,Tp,NC,MOB,int(np.random.randint(1e9))); obs[b]=o
        n=min(NC,MAXNC); tk[b,:n,0]=kx[:n]; tk[b,:n,1]=ky[:n]; ta[b,:n]=A[:n]
    t=lambda x: torch.tensor(x,device=DEVICE); return t(obs),t(tk),t(ta)
def chamfer(pred,true,tw):
    d=((pred[:,:,None,:]-true[:,None,:,:])**2).sum(-1)
    t2p=d.min(1).values; p2t=d.min(2).values
    return ((t2p*tw).sum(1)/(tw.sum(1)+1e-9)).mean()+0.1*p2t.mean()
M=Net().to(DEVICE); opt=torch.optim.AdamW(M.parameters(),lr=3e-4,weight_decay=1e-4); sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,STEPS); t0=time.time()
print(f"[mixed component-distill MOUT={MOUT}] comp{{4,6,8,12}} obs{{300,450,600}}")
for s in range(STEPS):
    obs,tk,ta=gen(24); pk=M(obs); loss=chamfer(pk,tk,ta)
    opt.zero_grad(); loss.backward(); opt.step(); sch.step()
    if s%2000==0 or s==STEPS-1: print(f"  step {s} chamfer {loss.item():.4f} ({time.time()-t0:.0f}s)",flush=True)
torch.save(M.state_dict(),'results/direction3/component_distill_mixed.pt'); print("saved component_distill_mixed.pt")
