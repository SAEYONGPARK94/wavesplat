"""Decisive careful test: does a supervised FF net RECONSTRUCT on a DETERMINED regime (few comp, enough obs)?
If yes -> learning works when determined; sparse collapse = info-limit. If no -> arch/opt bug (fixable).
Maps the boundary over (n_comp, n_obs). Long training (20k), monitors field corr vs RELAX reference."""
import os, sys, numpy as np, torch, time, argparse
sys.path.insert(0,'/home/vexit/wavesplat/wavesplat'); os.chdir('/home/vexit/wavesplat/wavesplat'); sys.path.insert(0,'scripts')
from amortized_ff import AmortizedFF
from d52_F4_neural_operator import DEVICE
from wavesim.wave_generator import jonswap_components
from d40_learned_acquisition import pool_xy
from estimator_improved import relax_cv, estimate_klim
from d17_recursive_estimator import refine_params
G=9.81; DEPTH=50.0; K_TP=2.2
gx=np.linspace(10,250,80); gy=np.linspace(-100,100,60); Xg,Yg=np.meshgrid(gx,gy); grid=np.column_stack([Xg.ravel(),Yg.ravel()])
ap=argparse.ArgumentParser(); ap.add_argument('--ncomp',type=int,default=4); ap.add_argument('--nobs',type=int,default=600); ap.add_argument('--steps',type=int,default=20000); a=ap.parse_args()
NC,MO,STEPS=a.ncomp,a.nobs,a.steps
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
    obs=np.zeros((B,MO,4),np.float32);Q=400;qpt=np.zeros((B,Q,3),np.float32);qz=np.zeros((B,Q),np.float32);Hsb=np.zeros(B,np.float32)
    for b in range(B):
        Hs=np.random.uniform(0.6,2.0);Tp=np.random.uniform(5.5,9.5);o,q,z,Tobs,_=make(Hs,Tp,int(np.random.randint(1e9)))
        obs[b]=o;qpt[b]=q;qz[b]=z;Hsb[b]=Hs
    t=lambda x: torch.tensor(x,device=DEVICE); return t(obs),t(qpt),t(qz),t(Hsb)
EVS=[make(0.8,6.0,1),make(1.2,7.0,2),make(2.0,9.0,4)]
def net_eval(M):
    o=[]
    for obs,q,qz,Tobs,gt in EVS:
        with torch.no_grad(): fld=M(torch.tensor(obs[None],device=DEVICE),torch.tensor(np.column_stack([grid,np.full(len(grid),Tobs)])[None],dtype=torch.float32,device=DEVICE))[0].cpu().numpy().reshape(Xg.shape)
        o.append(np.corrcoef(fld.ravel(),gt.ravel())[0,1])
    return np.mean(o)
# RELAX reference on the same eval episodes
relax_cr=[]
for obs,q,qz,Tobs,gt in EVS:
    P=obs[:,:3].astype(np.float32); z=obs[:,3].astype(np.float32); Hs=4*gt.std()
    target={'gx':gx,'gy':gy,'t':float(Tobs),'gt':gt,'Hs':Hs}
    klim=estimate_klim(P,z,DEPTH); kx,ky=relax_cv(P,z,DEPTH,klim); kx,ky,fld=refine_params(kx,ky,P,z,DEPTH,target,iters=400)
    relax_cr.append(np.corrcoef(fld.ravel(),gt.ravel())[0,1])
print(f"[ncomp={NC} nobs={MO}] RELAX ref corr {np.mean(relax_cr):.3f}  (determined-ness proxy)")
M=AmortizedFF(uq=False).to(DEVICE); opt=torch.optim.AdamW(M.parameters(),lr=3e-4,weight_decay=1e-4)
sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,STEPS); t0=time.time()
for s in range(STEPS):
    obs,q,qz,Hs=gen(24); pred=M(obs,q); loss=(((pred-qz)**2).mean(1)/(Hs**2+1e-6)).mean()
    opt.zero_grad(); loss.backward(); opt.step(); sch.step()
    if s%2000==0 or s==STEPS-1:
        print(f"  step {s:5d} loss {loss.item():.4f} | NET field corr {net_eval(M):.3f} (RELAX {np.mean(relax_cr):.3f}) ({time.time()-t0:.0f}s)",flush=True)
print(f"VERDICT [ncomp={NC} nobs={MO}]: net corr {net_eval(M):.3f} vs RELAX {np.mean(relax_cr):.3f}")
