"""A1: SELF-SUPERVISED neural-guided RELAX. Trained grid-spectral net -> extract top-K peaks (NMS) ->
physics REFINE (obs-fit, label-free) -> breaks grid limit. vs grid-only(0.49) vs RELAX. sim determined+hard."""
import os, sys, numpy as np, torch, torch.nn as nn, argparse
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
# SAME grid as training
kax=np.linspace(-0.22,0.22,27); KX,KY=np.meshgrid(kax,kax); GKX=KX.ravel().astype(np.float32); GKY=KY.ravel().astype(np.float32)
km=np.sqrt(GKX**2+GKY**2); keep=(km>0.02)&(km<0.22); GKX,GKY=GKX[keep],GKY[keep]; NG=len(GKX)
GW=np.sqrt(G*np.sqrt(GKX**2+GKY**2)*np.tanh(np.sqrt(GKX**2+GKY**2)*DEPTH)).astype(np.float32)
class Net(nn.Module):
    def __init__(s,d=256,heads=8,enc=4):
        super().__init__(); s.ff=FourierFeatures(48,learnable=False)
        s.oenc=nn.Sequential(nn.Linear(2*48+1,d),nn.GELU(),nn.Linear(d,d))
        el=nn.TransformerEncoderLayer(d,heads,4*d,batch_first=True,activation='gelu',norm_first=True); s.enc=nn.TransformerEncoder(el,enc)
        s.head=nn.Sequential(nn.Linear(d,d),nn.GELU(),nn.Linear(d,2*NG))
    def forward(s,obs): h=s.enc(s.oenc(torch.cat([s.ff(obs[...,:3]),obs[...,3:4]],-1))); return s.head(h.mean(1))
M=Net().to(DEVICE); M.load_state_dict(torch.load('results/direction3/grid_spectral_v2.pt',map_location=DEVICE)); M.eval()
def make(Hs,Tp,seed,NC):
    c=jonswap_components(Hs=Hs,Tp=Tp,n_components=NC,depth=DEPTH,seed=seed)
    A=np.array([x.A for x in c]);om=np.array([x.omega for x in c]);kx=np.array([x.kx for x in c]);ky=np.array([x.ky for x in c]);ph=np.array([x.phi for x in c])
    Tobs=K_TP*Tp; pool=pool_xy(); rng=np.random.default_rng(seed); MO=len(c)*0  # set below
    return A,om,kx,ky,ph,Tobs
def episode(Hs,Tp,seed,NC,MO):
    A,om,kx,ky,ph,Tobs=make(Hs,Tp,seed,NC); pool=pool_xy(); rng=np.random.default_rng(seed)
    oi=rng.integers(0,len(pool),MO); ox,oy=pool[oi,0],pool[oi,1]; ot=rng.uniform(0,Tobs,MO)
    oz=(A[None]*np.cos(kx[None]*ox[:,None]+ky[None]*oy[:,None]-om[None]*ot[:,None]+ph[None])).sum(1)+rng.normal(0,0.02,MO)
    gt=(A[None]*np.cos(kx[None]*Xg.ravel()[:,None]+ky[None]*Yg.ravel()[:,None]-om[None]*Tobs+ph[None])).sum(1).reshape(Xg.shape)
    return np.column_stack([ox,oy,ot,oz]).astype(np.float32),Tobs,gt
def peaks(coef,K,min_sep=0.035):
    a,b=coef[:NG],coef[NG:]; amp=np.sqrt(a**2+b**2); order=np.argsort(-amp); pk=[]
    for idx in order:
        if all((GKX[idx]-GKX[p])**2+(GKY[idx]-GKY[p])**2>min_sep**2 for p in pk): pk.append(idx)
        if len(pk)>=K: break
    return GKX[pk],GKY[pk]
def render_grid(coef,Tobs):
    a,b=coef[:NG],coef[NG:]; ph=GKX[:,None]*Xg.ravel()[None]+GKY[:,None]*Yg.ravel()[None]-GW[:,None]*Tobs
    return (a[:,None]*np.cos(ph)+b[:,None]*np.sin(ph)).sum(0).reshape(Xg.shape)
def corr(f,gt): return float(np.corrcoef(f.ravel(),gt.ravel())[0,1])
print(f"grid NG={NG} | net=grid_spectral_v2.pt")
print(f"{'NC':>3} {'MO':>4} | {'grid-only':>9} {'A1(net+refine)':>14} {'RELAX':>7}  (3-seed corr)")
for NC,MO in [(4,600),(6,600),(6,300),(8,300),(12,300),(12,150)]:
    go,a1,rl=[],[],[]
    for sd in [1,2,3]:
        obs,Tobs,gt=episode(np.random.default_rng(sd).uniform(0.8,1.8),np.random.default_rng(sd+9).uniform(6,9),sd,NC,MO)
        P=obs[:,:3].astype(np.float32); z=obs[:,3].astype(np.float32); Hs=4*gt.std(); tgt={'gx':gx,'gy':gy,'t':float(Tobs),'gt':gt,'Hs':Hs}
        with torch.no_grad(): coef=M(torch.tensor(obs[None],device=DEVICE))[0].cpu().numpy()
        go.append(corr(render_grid(coef,Tobs),gt))
        pkx,pky=peaks(coef,K=min(NC+4,14)); rkx,rky,f=refine_params(pkx.astype(np.float32),pky.astype(np.float32),P,z,DEPTH,tgt,iters=400); a1.append(corr(f,gt))
        kl=estimate_klim(P,z,DEPTH); Rkx,Rky=relax_cv(P,z,DEPTH,kl); Rkx,Rky,fr=refine_params(Rkx,Rky,P,z,DEPTH,tgt,iters=400); rl.append(corr(fr,gt))
    print(f"{NC:>3} {MO:>4} | {np.mean(go):>9.3f} {np.mean(a1):>14.3f} {np.mean(rl):>7.3f}",flush=True)
