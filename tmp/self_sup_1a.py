"""Approach 1a: SELF-SUPERVISED held-out obs prediction (NO wavenumber/field labels — works on real points).
Net(context points) -> predict TARGET points' z. Train on obs split only. Eval field corr vs GT (sim, validation)."""
import os, sys, numpy as np, torch, time, argparse
sys.path.insert(0,'/home/vexit/wavesplat/wavesplat'); os.chdir('/home/vexit/wavesplat/wavesplat'); sys.path.insert(0,'scripts')
from amortized_ff import AmortizedFF
from d52_F4_neural_operator import DEVICE
from wavesim.wave_generator import jonswap_components
from d40_learned_acquisition import pool_xy
from estimator_improved import relax_cv, estimate_klim
from d17_recursive_estimator import refine_params
from wavesplat.evaluation.metrics import pearson_correlation
G=9.81; DEPTH=50.0; K_TP=2.2
gx=np.linspace(10,250,80); gy=np.linspace(-100,100,60); Xg,Yg=np.meshgrid(gx,gy); grid=np.column_stack([Xg.ravel(),Yg.ravel()])
ap=argparse.ArgumentParser(); ap.add_argument('--ncomp',type=int,default=4); ap.add_argument('--nobs',type=int,default=600); ap.add_argument('--steps',type=int,default=12000); a=ap.parse_args()
NC,MO,STEPS=a.ncomp,a.nobs,a.steps; NCTX=int(0.65*MO)
def comps(Hs,Tp,seed):
    c=jonswap_components(Hs=Hs,Tp=Tp,n_components=NC,depth=DEPTH,seed=seed)
    return (np.array([x.A for x in c]),np.array([x.omega for x in c]),np.array([x.kx for x in c]),np.array([x.ky for x in c]),np.array([x.phi for x in c]))
def make(Hs,Tp,seed):
    A,om,kx,ky,ph=comps(Hs,Tp,seed); Tobs=K_TP*Tp; pool=pool_xy(); rng=np.random.default_rng(seed)
    oi=rng.integers(0,len(pool),MO); ox,oy=pool[oi,0],pool[oi,1]; ot=rng.uniform(0,Tobs,MO)
    oz=(A[None]*np.cos(kx[None]*ox[:,None]+ky[None]*oy[:,None]-om[None]*ot[:,None]+ph[None])).sum(1)+rng.normal(0,0.02,MO)
    gt=(A[None]*np.cos(kx[None]*Xg.ravel()[:,None]+ky[None]*Yg.ravel()[:,None]-om[None]*Tobs+ph[None])).sum(1).reshape(Xg.shape)
    return np.column_stack([ox,oy,ot,oz]).astype(np.float32),Tobs,gt
def gen(B):
    ctx=np.zeros((B,NCTX,4),np.float32); tq=np.zeros((B,MO-NCTX,3),np.float32); tz=np.zeros((B,MO-NCTX),np.float32); Hsb=np.zeros(B,np.float32)
    for b in range(B):
        Hs=np.random.uniform(0.6,2.0);Tp=np.random.uniform(5.5,9.5);o,_,_=make(Hs,Tp,int(np.random.randint(1e9)))
        p=np.random.permutation(MO); c=o[p[:NCTX]]; t=o[p[NCTX:]]
        ctx[b]=c; tq[b]=t[:,:3]; tz[b]=t[:,3]; Hsb[b]=Hs
    tt=lambda x: torch.tensor(x,device=DEVICE); return tt(ctx),tt(tq),tt(tz),tt(Hsb)
EVS=[make(0.8,6.0,1),make(1.2,7.0,2),make(2.0,9.0,4)]
# GT-supervised reference (plain net, field-MSE) is ~0.60; RELAX ref:
rc=[]
for obs,Tobs,gt in EVS:
    P=obs[:,:3].astype(np.float32);z=obs[:,3].astype(np.float32);Hs=4*gt.std();tgt={'gx':gx,'gy':gy,'t':float(Tobs),'gt':gt,'Hs':Hs}
    klim=estimate_klim(P,z,DEPTH);kx,ky=relax_cv(P,z,DEPTH,klim);kx,ky,f=refine_params(kx,ky,P,z,DEPTH,tgt,iters=400); rc.append(np.corrcoef(f.ravel(),gt.ravel())[0,1])
RREF=np.mean(rc)
def ev(M):
    o=[]
    for obs,Tobs,gt in EVS:
        with torch.no_grad(): fld=M(torch.tensor(obs[None],device=DEVICE),torch.tensor(np.column_stack([grid,np.full(len(grid),Tobs)])[None],dtype=torch.float32,device=DEVICE))[0].cpu().numpy().reshape(Xg.shape)
        o.append(np.corrcoef(fld.ravel(),gt.ravel())[0,1])
    return np.mean(o)
M=AmortizedFF(uq=False).to(DEVICE); opt=torch.optim.AdamW(M.parameters(),lr=3e-4,weight_decay=1e-4); sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,STEPS); t0=time.time()
print(f"[1a self-supervised held-out, ncomp={NC} nobs={MO} ctx={NCTX}] RELAX ref {RREF:.3f}, GT-sup plain-net ~0.60")
for s in range(STEPS):
    ctx,tq,tz,Hs=gen(24); pred=M(ctx,tq); loss=(((pred-tz)**2).mean(1)/(Hs**2+1e-6)).mean()
    opt.zero_grad(); loss.backward(); opt.step(); sch.step()
    if s%1500==0 or s==STEPS-1:
        with torch.no_grad(): hr=((pred-tz)**2).mean().sqrt().item()
        print(f"  step {s:5d} heldout-rmse {hr:.3f}/{tz.std().item():.3f} | field corr(vs GT) {ev(M):.3f} (RELAX {RREF:.3f}) ({time.time()-t0:.0f}s)",flush=True)
print(f"VERDICT 1a self-supervised: field corr {ev(M):.3f} vs RELAX {RREF:.3f} vs GT-sup 0.60")
