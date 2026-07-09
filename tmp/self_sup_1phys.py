"""Approach 1 (PHYSICS-constrained SELF-SUPERVISED): net->(kx,ky); dispersion forces w; (A,phi) via ridge-LS on
CONTEXT obs; loss = held-out TARGET obs reconstruction (NO labels). Physics gives the FORM, self-sup fits it.
= amortized RELAX, label-free, trainable on real points. Eval field corr vs GT (sim, validation only)."""
import os, sys, numpy as np, torch, torch.nn as nn, time, argparse
sys.path.insert(0,'/home/vexit/wavesplat/wavesplat'); os.chdir('/home/vexit/wavesplat/wavesplat'); sys.path.insert(0,'scripts')
from d52_F4_neural_operator import DEVICE
from amortized_ff import FourierFeatures
from wavesim.wave_generator import jonswap_components
from d40_learned_acquisition import pool_xy
from estimator_improved import relax_cv, estimate_klim
from d17_recursive_estimator import refine_params
from wavesplat.model.tnse import fit_amplitudes_phases, render_wave_field
G=9.81; DEPTH=50.0; K_TP=2.2; KMAX=0.6
gx=np.linspace(10,250,80); gy=np.linspace(-100,100,60); Xg,Yg=np.meshgrid(gx,gy); grid=np.column_stack([Xg.ravel(),Yg.ravel()])
ap=argparse.ArgumentParser(); ap.add_argument('--ncomp',type=int,default=6); ap.add_argument('--nobs',type=int,default=600)
ap.add_argument('--mout',type=int,default=12); ap.add_argument('--steps',type=int,default=40000); ap.add_argument('--l1',type=float,default=0.02); a=ap.parse_args()
NC,MO,MOUT,STEPS,L1=a.ncomp,a.nobs,a.mout,a.steps,a.l1; NCTX=int(0.65*MO)
def disp(kx,ky): k=torch.sqrt(kx**2+ky**2)+1e-9; return torch.sqrt(G*k*torch.tanh(k*DEPTH))
def comps(Hs,Tp,seed):
    c=jonswap_components(Hs=Hs,Tp=Tp,n_components=NC,depth=DEPTH,seed=seed)
    return (np.array([x.A for x in c]),np.array([x.omega for x in c]),np.array([x.kx for x in c]),np.array([x.ky for x in c]),np.array([x.phi for x in c]))
def make(Hs,Tp,seed):
    A,om,kx,ky,ph=comps(Hs,Tp,seed); Tobs=K_TP*Tp; pool=pool_xy(); rng=np.random.default_rng(seed)
    oi=rng.integers(0,len(pool),MO); ox,oy=pool[oi,0],pool[oi,1]; ot=rng.uniform(0,Tobs,MO)
    oz=(A[None]*np.cos(kx[None]*ox[:,None]+ky[None]*oy[:,None]-om[None]*ot[:,None]+ph[None])).sum(1)+rng.normal(0,0.02,MO)
    gt=(A[None]*np.cos(kx[None]*Xg.ravel()[:,None]+ky[None]*Yg.ravel()[:,None]-om[None]*Tobs+ph[None])).sum(1).reshape(Xg.shape)
    return np.column_stack([ox,oy,ot,oz]).astype(np.float32),Tobs,gt
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
def design(kx,ky,P):  # kx,ky [B,M]; P [B,N,3] -> D [B,N,2M]
    w=disp(kx,ky); ph=kx[:,None,:]*P[...,0:1]+ky[:,None,:]*P[...,1:2]-w[:,None,:]*P[...,2:3]
    return torch.cat([torch.cos(ph),torch.sin(ph)],-1)
def ridge_ls(D,z,eps=1e-2):  # D [B,N,2M], z [B,N] -> coef [B,2M]
    Dt=D.transpose(1,2); A=Dt@D+eps*torch.eye(D.shape[-1],device=D.device)[None]; b=(Dt@z[...,None])
    return torch.linalg.solve(A,b)[...,0]
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
def ev(M):  # net->(kx,ky); LS-fit A,phi on ALL obs; render field; corr vs GT
    o=[]
    for obs,Tobs,gt in EVS:
        ob=torch.tensor(obs[None],device=DEVICE)
        with torch.no_grad(): pk=M(ob)[0].cpu().numpy()
        A,phi=fit_amplitudes_phases(pk[:,0],pk[:,1],obs.astype(np.float32),DEPTH,lambda_reg=0.5)
        f=render_wave_field(pk[:,0],pk[:,1],A,phi,DEPTH,gx,gy,float(Tobs)); o.append(np.corrcoef(f.ravel(),gt.ravel())[0,1])
    return np.mean(o)
M=Net().to(DEVICE); opt=torch.optim.AdamW(M.parameters(),lr=3e-4,weight_decay=1e-4); sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,STEPS); t0=time.time()
print(f"[1-PHYS self-sup, ncomp={NC} nobs={MO} MOUT={MOUT} l1={L1} steps={STEPS}] RELAX ref {RREF:.3f}")
for s in range(STEPS):
    ctx,tq,Hs=gen(24); kk=M(ctx); kx,ky=kk[...,0],kk[...,1]
    coef=ridge_ls(design(kx,ky,ctx[...,:3]),ctx[...,3])
    pred=(design(kx,ky,tq[...,:3])@coef[...,None])[...,0]
    amp=torch.sqrt(coef[:,:MOUT]**2+coef[:,MOUT:]**2+1e-9)
    loss=(((pred-tq[...,3])**2).mean(1)/(Hs**2+1e-6)).mean()+L1*(amp.mean(1)/(Hs+1e-6)).mean()
    opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(M.parameters(),1.0); opt.step(); sch.step()
    if s%2500==0 or s==STEPS-1:
        with torch.no_grad(): hr=((pred-tq[...,3])**2).mean().sqrt().item()
        print(f"  step {s:5d} heldout-rmse {hr:.3f}/{tq[...,3].std().item():.3f} | field corr(GT) {ev(M):.3f} (RELAX {RREF:.3f}) ({time.time()-t0:.0f}s)",flush=True)
print(f"VERDICT 1-PHYS self-sup: field corr {ev(M):.3f} vs RELAX {RREF:.3f} (label-free, dispersion form)")
torch.save(M.state_dict(),'results/direction3/self_sup_1phys.pt'); print("saved")
