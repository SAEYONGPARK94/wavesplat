"""Lever #1: DEEP-UNROLLED matching pursuit (learned ISTA, SELF-SUPERVISED obs-fit). Iterative sparse recovery
on fixed dispersion grid with learned per-layer step/threshold -> mimics RELAX greedy (residual-driven).
Tests if unrolled beats 1-shot A1 (detection bottleneck) toward RELAX. label-free. eval peaks+refine vs RELAX."""
import os, sys, numpy as np, torch, torch.nn as nn, time, argparse
sys.path.insert(0,'/home/vexit/wavesplat/wavesplat'); os.chdir('/home/vexit/wavesplat/wavesplat'); sys.path.insert(0,'scripts')
from d52_F4_neural_operator import DEVICE
from wavesim.wave_generator import jonswap_components
from d40_learned_acquisition import pool_xy
from estimator_improved import relax_cv, estimate_klim
from d17_recursive_estimator import refine_params
G=9.81; DEPTH=50.0; K_TP=2.2
gx=np.linspace(10,250,80); gy=np.linspace(-100,100,60); Xg,Yg=np.meshgrid(gx,gy)
kax=np.linspace(-0.22,0.22,27); KX,KY=np.meshgrid(kax,kax); GKX=KX.ravel().astype(np.float32); GKY=KY.ravel().astype(np.float32)
km=np.sqrt(GKX**2+GKY**2); keep=(km>0.02)&(km<0.22); GKX,GKY=GKX[keep],GKY[keep]; NG=len(GKX)
GW=np.sqrt(G*np.sqrt(GKX**2+GKY**2)*np.tanh(np.sqrt(GKX**2+GKY**2)*DEPTH)).astype(np.float32)
tGKX=torch.tensor(GKX,device=DEVICE);tGKY=torch.tensor(GKY,device=DEVICE);tGW=torch.tensor(GW,device=DEVICE)
ap=argparse.ArgumentParser(); ap.add_argument('--T',type=int,default=15); ap.add_argument('--steps',type=int,default=25000); a=ap.parse_args()
T=a.T; MO=500; NCTX=int(0.65*MO)
def episode(Hs,Tp,seed,NC,nobs=None):
    nb=MO if nobs is None else nobs
    c=jonswap_components(Hs=Hs,Tp=Tp,n_components=NC,depth=DEPTH,seed=seed)
    A=np.array([x.A for x in c]);om=np.array([x.omega for x in c]);kx=np.array([x.kx for x in c]);ky=np.array([x.ky for x in c]);ph=np.array([x.phi for x in c])
    Tobs=K_TP*Tp; pool=pool_xy(); rng=np.random.default_rng(seed)
    oi=rng.integers(0,len(pool),nb); ox,oy=pool[oi,0],pool[oi,1]; ot=rng.uniform(0,Tobs,nb)
    oz=(A[None]*np.cos(kx[None]*ox[:,None]+ky[None]*oy[:,None]-om[None]*ot[:,None]+ph[None])).sum(1)+rng.normal(0,0.02,nb)
    gt=(A[None]*np.cos(kx[None]*Xg.ravel()[:,None]+ky[None]*Yg.ravel()[:,None]-om[None]*Tobs+ph[None])).sum(1).reshape(Xg.shape)
    return np.column_stack([ox,oy,ot,oz]).astype(np.float32),Tobs,gt
def dmat(P):  # P [B,N,3] -> D [B,N,2NG]
    ph=tGKX[None,None]*P[...,0:1]+tGKY[None,None]*P[...,1:2]-tGW[None,None]*P[...,2:3]
    return torch.cat([torch.cos(ph),torch.sin(ph)],-1)
class LISTA(nn.Module):
    def __init__(s,T):
        super().__init__(); s.T=T
        s.la=nn.Parameter(torch.full((T,),1.0)); s.ll=nn.Parameter(torch.full((T,),-5.0))  # step=softplus(la)/L~1.31/L, thresh~0.007 (live)
    def lip(s,D,Nobs,it=4):  # power-iter max eig of DtD/N per episode
        v=torch.randn(D.shape[0],D.shape[-1],device=D.device); v=v/(v.norm(dim=1,keepdim=True)+1e-9)
        for _ in range(it):
            Dv=(D*v[:,None,:]).sum(-1); DtDv=(D*Dv[:,:,None]).sum(1)/Nobs; v=DtDv/(DtDv.norm(dim=1,keepdim=True)+1e-9)
        Dv=(D*v[:,None,:]).sum(-1); return (((D*Dv[:,:,None]).sum(1)/Nobs)*v).sum(1).clamp(min=1e-2)  # [B]
    def forward(s,D,z,Nobs):  # D [B,N,2NG], z [B,N] -> x [B,2NG]
        B=D.shape[0]; x=torch.zeros(B,2*NG,device=D.device); L=s.lip(D,Nobs).detach()[:,None]
        for t in range(s.T):
            al=torch.nn.functional.softplus(s.la[t])/L; lam=torch.nn.functional.softplus(s.ll[t])
            resid=(D*x[:,None,:]).sum(-1)-z; grad=torch.nan_to_num((D*resid[:,:,None]).sum(1)/Nobs)  # Dt(Dx-z)/N
            x=torch.clamp(x-al*grad,-20,20)
            a,b=x[:,:NG],x[:,NG:]; m=torch.sqrt(a*a+b*b+1e-9); sc=torch.clamp(1-lam/m,min=0.0)
            x=torch.cat([a*sc,b*sc],-1)
        return x
def gen(B):
    ctxP=np.zeros((B,NCTX,3),np.float32);ctxz=np.zeros((B,NCTX),np.float32);tqP=np.zeros((B,MO-NCTX,3),np.float32);tqz=np.zeros((B,MO-NCTX),np.float32);Hsb=np.zeros(B,np.float32)
    for b in range(B):
        Hs=np.random.uniform(0.6,2.0);Tp=np.random.uniform(5.5,9.5);NC=int(np.random.choice([4,5,6,8,10,12]))
        o,_,gt=episode(Hs,Tp,int(np.random.randint(1e9)),NC); p=np.random.permutation(MO)
        c=o[p[:NCTX]];tq=o[p[NCTX:]]; ctxP[b]=c[:,:3];ctxz[b]=c[:,3];tqP[b]=tq[:,:3];tqz[b]=tq[:,3];Hsb[b]=4*gt.std()
    tt=lambda x: torch.tensor(x,device=DEVICE); return tt(ctxP),tt(ctxz),tt(tqP),tt(tqz),tt(Hsb)
def peaks(x,K,min_sep=0.03):
    a,b=x[:NG],x[NG:]; amp=np.sqrt(a**2+b**2); order=np.argsort(-amp); pk=[]
    for idx in order:
        if all((GKX[idx]-GKX[p])**2+(GKY[idx]-GKY[p])**2>min_sep**2 for p in pk): pk.append(idx)
        if len(pk)>=K: break
    return GKX[pk],GKY[pk]
def corr(f,gt): return float(np.corrcoef(f.ravel(),gt.ravel())[0,1])
M=LISTA(T).to(DEVICE); opt=torch.optim.AdamW(M.parameters(),lr=5e-3); sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,a.steps); t0=time.time()
print(f"[LISTA unrolled self-sup, T={T} NG={NG} steps={a.steps}]")
for s in range(a.steps):
    ctxP,ctxz,tqP,tqz,Hs=gen(20); x=M(dmat(ctxP),ctxz,NCTX)
    predt=(dmat(tqP)*x[:,None,:]).sum(-1); loss=(((predt-tqz)**2).mean(1)/(Hs**2+1e-6)).mean()
    opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(M.parameters(),1.0); opt.step(); sch.step()
    if s%2500==0 or s==a.steps-1:
        with torch.no_grad(): hr=((predt-tqz)**2).mean().sqrt().item()
        print(f"  step {s:5d} heldout-rmse {hr:.3f}/{tqz.std().item():.3f} ({time.time()-t0:.0f}s)",flush=True)
M.eval(); torch.save(M.state_dict(),'results/direction3/lista_selfsup.pt'); print("saved lista")
print(f"\n{'NC':>3} {'MO':>4} | {'LISTA+refine':>12} {'A1(prev)':>8} {'RELAX':>7}")
A1prev={(6,600):0.92,(8,300):0.64,(12,300):0.65}
for NC,MO2 in [(6,600),(8,300),(12,300)]:
    li,rl=[],[]
    for sd in [1,2,3,4,5]:
        Hs=np.random.default_rng(sd).uniform(0.8,1.8);Tp=np.random.default_rng(sd+9).uniform(6,9)
        # sub-sample MO2 obs from a fresh episode
        o,Tobs,gt=episode(Hs,Tp,sd,NC,nobs=MO2)
        P=o[:,:3].astype(np.float32);z=o[:,3].astype(np.float32);Hs4=4*gt.std();tgt={'gx':gx,'gy':gy,'t':float(Tobs),'gt':gt,'Hs':Hs4}
        with torch.no_grad(): x=M(dmat(torch.tensor(P[None],device=DEVICE)),torch.tensor(z[None],device=DEVICE),len(z))[0].cpu().numpy()
        pkx,pky=peaks(x,K=min(NC+4,14)); rkx,rky,f=refine_params(pkx.astype(np.float32),pky.astype(np.float32),P,z,DEPTH,tgt,iters=400); li.append(corr(f,gt))
        kl=estimate_klim(P,z,DEPTH);Rkx,Rky=relax_cv(P,z,DEPTH,kl);Rkx,Rky,fr=refine_params(Rkx,Rky,P,z,DEPTH,tgt,iters=400); rl.append(corr(fr,gt))
    print(f"{NC:>3} {MO2:>4} | {np.mean(li):>12.3f} {A1prev.get((NC,MO2),0):>8.2f} {np.mean(rl):>7.3f}",flush=True)
