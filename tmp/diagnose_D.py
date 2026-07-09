"""Is D's blank output a bug or a real NLL-collapse? Numerical checks + MSE-operator control."""
import os, sys, numpy as np, torch, time
sys.path.insert(0, '/home/vexit/wavesplat/wavesplat'); os.chdir('/home/vexit/wavesplat/wavesplat'); sys.path.insert(0, 'scripts')
from d56_amortized_posterior import PosteriorNet
from d52_F4_neural_operator import NeuralOperator, DEVICE
from wavesim.wave_generator import jonswap_components
from d40_learned_acquisition import pool_xy
DEPTH=50.0; K_TP=2.2
gx=np.linspace(10,250,80); gy=np.linspace(-100,100,60); Xg,Yg=np.meshgrid(gx,gy)
grid=np.column_stack([Xg.ravel(),Yg.ravel()])

def ep(Hs,Tp,seed):
    c=jonswap_components(Hs=Hs,Tp=Tp,n_components=64,depth=DEPTH,seed=seed)
    A=np.array([x.A for x in c]);om=np.array([x.omega for x in c]);kx=np.array([x.kx for x in c]);ky=np.array([x.ky for x in c]);ph=np.array([x.phi for x in c])
    Tobs=K_TP*Tp;pool=pool_xy();rng=np.random.default_rng(seed)
    oi=rng.integers(0,len(pool),200);ox,oy=pool[oi,0],pool[oi,1];ot=rng.uniform(0,Tobs,200)
    oz=(A[None]*np.cos(kx[None]*ox[:,None]+ky[None]*oy[:,None]-om[None]*ot[:,None]+ph[None])).sum(1)+rng.normal(0,0.02,200)
    P=np.column_stack([ox,oy,ot]).astype(np.float32);z=oz.astype(np.float32)
    gt=(A[None]*np.cos(kx[None]*Xg.ravel()[:,None]+ky[None]*Yg.ravel()[:,None]-om[None]*Tobs+ph[None])).sum(1).reshape(Xg.shape)
    return P,z,Tobs,gt

# ---- load D (NLL posterior) ----
D=PosteriorNet().to(DEVICE);D.load_state_dict(torch.load('results/direction3/amortized_posterior_fc.pt',map_location=DEVICE));D.eval()
P,z,Tobs,gt=ep(1.2,7.0,2)
ob=torch.tensor(np.column_stack([P[:,0],P[:,1],P[:,2],z])[None],dtype=torch.float32,device=DEVICE)
# (a) D at the query grid (t*)
q=torch.tensor(np.column_stack([grid,np.full(len(grid),float(Tobs))])[None],dtype=torch.float32,device=DEVICE)
with torch.no_grad(): m,lv=D(ob,q)
dfield=m[0].cpu().numpy()
print("=== D output check ===")
print(f"GT field std {gt.std():.3f}  |  D field std {dfield.std():.4f}  mean {dfield.mean():.4f}  range [{dfield.min():.3f},{dfield.max():.3f}]")
print(f"D sigma (exp .5logv) mean {(0.5*lv[0]).exp().mean().item():.3f}  (Hs/4 ~ {gt.std():.3f})")
# (b) D AT THE OBSERVED points/times — can it even fit training obs?
qo=torch.tensor(np.column_stack([P[:,0],P[:,1],P[:,2]])[None],dtype=torch.float32,device=DEVICE)
with torch.no_grad(): mo,_=D(ob,qo)
mo=mo[0].cpu().numpy()
print(f"D at OBS points: pred std {mo.std():.4f} vs obs z std {z.std():.3f}  |  rmse(pred,z) {np.sqrt(np.mean((mo-z)**2)):.3f}  corr {np.corrcoef(mo,z)[0,1]:.3f}")
print("  -> if D can't fit its own obs, it collapsed to ~0 (prior).")

