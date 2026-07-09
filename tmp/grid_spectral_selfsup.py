"""Approach 1 FIXED (physics as GRID): net(obs)->sparse spectrum (a,b) over a FIXED dispersion (kx,ky) grid.
obs-fit is LINEAR/convex in (a,b) (fixed frequencies) -> no non-convex frequency search (which stuck the
continuous version). Self-supervised: predict held-out obs. L1 -> sparse. = learned amortized sparse spectral
estimator (RELAX-family), label-free, real-applicable. Train LONG. Eval field corr vs GT (sim validation)."""
import os, sys, numpy as np, torch, torch.nn as nn, time, argparse
sys.path.insert(0,'/home/vexit/wavesplat/wavesplat'); os.chdir('/home/vexit/wavesplat/wavesplat'); sys.path.insert(0,'scripts')
from d52_F4_neural_operator import DEVICE
from amortized_ff import FourierFeatures
from wavesim.wave_generator import jonswap_components
from d40_learned_acquisition import pool_xy
from estimator_improved import relax_cv, estimate_klim
from d17_recursive_estimator import refine_params
from wavesplat.model.tnse import fit_amplitudes_phases, render_wave_field
G=9.81; DEPTH=50.0; K_TP=2.2
gx=np.linspace(10,250,80); gy=np.linspace(-100,100,60); Xg,Yg=np.meshgrid(gx,gy)
ap=argparse.ArgumentParser(); ap.add_argument('--ncomp',type=int,default=6); ap.add_argument('--nobs',type=int,default=600)
ap.add_argument('--ng',type=int,default=21); ap.add_argument('--steps',type=int,default=40000); ap.add_argument('--l1',type=float,default=0.01); a=ap.parse_args()
NC,MO,STEPS,L1=a.ncomp,a.nobs,a.steps,a.l1; NCTX=int(0.65*MO)
# fixed dispersion grid (kx,ky) covering wave band + all directions
kax=np.linspace(-0.22,0.22,a.ng); KX,KY=np.meshgrid(kax,kax); GKX=KX.ravel().astype(np.float32); GKY=KY.ravel().astype(np.float32)
kmag=np.sqrt(GKX**2+GKY**2); keep=(kmag>0.02)&(kmag<0.22); GKX,GKY=GKX[keep],GKY[keep]; NG=len(GKX)
GW=np.sqrt(G*np.sqrt(GKX**2+GKY**2)*np.tanh(np.sqrt(GKX**2+GKY**2)*DEPTH)).astype(np.float32)
tGKX=torch.tensor(GKX,device=DEVICE);tGKY=torch.tensor(GKY,device=DEVICE);tGW=torch.tensor(GW,device=DEVICE)
def comps(Hs,Tp,seed):
    c=jonswap_components(Hs=Hs,Tp=Tp,n_components=NC,depth=DEPTH,seed=seed)
    return (np.array([x.A for x in c]),np.array([x.omega for x in c]),np.array([x.kx for x in c]),np.array([x.ky for x in c]),np.array([x.phi for x in c]))
def make(Hs,Tp,seed):
    A,om,kx,ky,ph=comps(Hs,Tp,seed); Tobs=K_TP*Tp; pool=pool_xy(); rng=np.random.default_rng(seed)
    oi=rng.integers(0,len(pool),MO); ox,oy=pool[oi,0],pool[oi,1]; ot=rng.uniform(0,Tobs,MO)
    oz=(A[None]*np.cos(kx[None]*ox[:,None]+ky[None]*oy[:,None]-om[None]*ot[:,None]+ph[None])).sum(1)+rng.normal(0,0.02,MO)
    gt=(A[None]*np.cos(kx[None]*Xg.ravel()[:,None]+ky[None]*Yg.ravel()[:,None]-om[None]*Tobs+ph[None])).sum(1).reshape(Xg.shape)
    return np.column_stack([ox,oy,ot,oz]).astype(np.float32),Tobs,gt
def dgrid(P):  # P [B,N,3] -> design [B,N,2NG] (cos|sin) using fixed grid freqs
    ph=tGKX[None,None]*P[...,0:1]+tGKY[None,None]*P[...,1:2]-tGW[None,None]*P[...,2:3]
    return torch.cat([torch.cos(ph),torch.sin(ph)],-1)
class Net(nn.Module):  # obs -> spectrum coef (a,b) over grid
    def __init__(s,d=256,heads=8,enc=4):
        super().__init__(); s.ff=FourierFeatures(48,learnable=False)
        s.oenc=nn.Sequential(nn.Linear(2*48+1,d),nn.GELU(),nn.Linear(d,d))
        el=nn.TransformerEncoderLayer(d,heads,4*d,batch_first=True,activation='gelu',norm_first=True); s.enc=nn.TransformerEncoder(el,enc)
        s.head=nn.Sequential(nn.Linear(d,d),nn.GELU(),nn.Linear(d,2*NG))
    def forward(s,obs):
        h=s.enc(s.oenc(torch.cat([s.ff(obs[...,:3]),obs[...,3:4]],-1))); return s.head(h.mean(1))  # [B,2NG]
def gen(B):
    ctx=np.zeros((B,NCTX,4),np.float32);tq=np.zeros((B,MO-NCTX,4),np.float32);Hsb=np.zeros(B,np.float32)
    for b in range(B):
        Hs=np.random.uniform(0.6,2.0);Tp=np.random.uniform(5.5,9.5);o,_,_=make(Hs,Tp,int(np.random.randint(1e9)))
        p=np.random.permutation(MO);ctx[b]=o[p[:NCTX]];tq[b]=o[p[NCTX:]];Hsb[b]=Hs
    tt=lambda x: torch.tensor(x,device=DEVICE);return tt(ctx),tt(tq),tt(Hsb)
EVS=[make(0.8,6.0,1),make(1.2,7.0,2),make(2.0,9.0,4)]
rc=[]
for obs,Tobs,gt in EVS:
    P=obs[:,:3].astype(np.float32);z=obs[:,3].astype(np.float32);Hs=4*gt.std();tgt={'gx':gx,'gy':gy,'t':float(Tobs),'gt':gt,'Hs':Hs}
    kl=estimate_klim(P,z,DEPTH);kx,ky=relax_cv(P,z,DEPTH,kl);kx,ky,f=refine_params(kx,ky,P,z,DEPTH,tgt,iters=400);rc.append(np.corrcoef(f.ravel(),gt.ravel())[0,1])
RREF=np.mean(rc)
def render_grid(coef,Tobs):  # coef [2NG] numpy -> field on grid
    a,b=coef[:NG],coef[NG:]; ph=GKX[:,None]*Xg.ravel()[None]+GKY[:,None]*Yg.ravel()[None]-GW[:,None]*Tobs
    f=(a[:,None]*np.cos(ph)+b[:,None]*np.sin(ph)).sum(0); return f.reshape(Xg.shape)
def ev(M):
    o=[]
    for obs,Tobs,gt in EVS:
        with torch.no_grad(): coef=M(torch.tensor(obs[None],device=DEVICE))[0].cpu().numpy()
        f=render_grid(coef,Tobs); o.append(np.corrcoef(f.ravel(),gt.ravel())[0,1])
    return np.mean(o)
M=Net().to(DEVICE); opt=torch.optim.AdamW(M.parameters(),lr=3e-4,weight_decay=1e-5); sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,STEPS); t0=time.time()
print(f"[GRID-spectral self-sup, NG={NG} ncomp={NC} nobs={MO} l1={L1} steps={STEPS}] RELAX ref {RREF:.3f}")
for s in range(STEPS):
    ctx,tq,Hs=gen(24); coef=M(ctx)  # [B,2NG]
    predc=(dgrid(ctx[...,:3])*coef[:,None,:]).sum(-1); predt=(dgrid(tq[...,:3])*coef[:,None,:]).sum(-1)
    amp=torch.sqrt(coef[:,:NG]**2+coef[:,NG:]**2+1e-9)
    fit=((predc-ctx[...,3])**2).mean(1)/(Hs**2+1e-6)  # also fit context (well-posed)
    heldout=((predt-tq[...,3])**2).mean(1)/(Hs**2+1e-6)
    loss=(0.5*fit+heldout).mean()+L1*(amp.sum(1)/(Hs+1e-6)).mean()
    opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(M.parameters(),1.0); opt.step(); sch.step()
    if s%2500==0 or s==STEPS-1:
        with torch.no_grad(): hr=((predt-tq[...,3])**2).mean().sqrt().item()
        print(f"  step {s:5d} heldout-rmse {hr:.3f}/{tq[...,3].std().item():.3f} | field corr(GT) {ev(M):.3f} (RELAX {RREF:.3f}) ({time.time()-t0:.0f}s)",flush=True)
print(f"VERDICT GRID-spectral self-sup: field corr {ev(M):.3f} vs RELAX {RREF:.3f} (label-free, dispersion grid, convex)")
torch.save(M.state_dict(),'results/direction3/grid_spectral_selfsup.pt'); print("saved")
