"""Control: same architecture, pure MSE (no variance head). Does it reconstruct (fit obs + structure)?
If yes -> NLL objective caused D's collapse (fixable). If no -> deeper bug."""
import os, sys, numpy as np, torch, time
sys.path.insert(0, '/home/vexit/wavesplat/wavesplat'); os.chdir('/home/vexit/wavesplat/wavesplat'); sys.path.insert(0, 'scripts')
from d52_F4_neural_operator import NeuralOperator, DEVICE
from wavesim.wave_generator import jonswap_components
from d40_learned_acquisition import pool_xy
DEPTH=50.0; K_TP=2.2
gx=np.linspace(10,250,80); gy=np.linspace(-100,100,60); Xg,Yg=np.meshgrid(gx,gy); grid=np.column_stack([Xg.ravel(),Yg.ravel()])

def gen(B, fc=6.0):
    pool=pool_xy(); obs=np.zeros((B,200,4),np.float32); qpt=np.zeros((B,400,3),np.float32); qz=np.zeros((B,400),np.float32); Hsb=np.zeros(B,np.float32)
    for b in range(B):
        Hs=np.random.uniform(0.6,2.0); Tp=np.random.uniform(5.5,9.5)
        c=jonswap_components(Hs=Hs,Tp=Tp,n_components=64,depth=DEPTH,seed=int(np.random.randint(1e9)))
        A=np.array([x.A for x in c]);om=np.array([x.omega for x in c]);kx=np.array([x.kx for x in c]);ky=np.array([x.ky for x in c]);ph=np.array([x.phi for x in c])
        Tobs=K_TP*Tp; oi=np.random.randint(0,len(pool),200); ox,oy=pool[oi,0],pool[oi,1]; ot=np.random.uniform(0,Tobs,200)
        oz=(A[None]*np.cos(kx[None]*ox[:,None]+ky[None]*oy[:,None]-om[None]*ot[:,None]+ph[None])).sum(1)+np.random.normal(0,0.02,200)
        obs[b]=np.column_stack([ox,oy,ot,oz]); qx=np.random.uniform(10,250,400);qy=np.random.uniform(-100,100,400);qt=np.random.uniform(0.5*Tobs,Tobs+fc,400)
        qz[b]=(A[None]*np.cos(kx[None]*qx[:,None]+ky[None]*qy[:,None]-om[None]*qt[:,None]+ph[None])).sum(1); qpt[b]=np.column_stack([qx,qy,qt]); Hsb[b]=Hs
    t=lambda a: torch.tensor(a,device=DEVICE); return t(obs),t(qpt),t(qz),t(Hsb)

M=NeuralOperator().to(DEVICE); opt=torch.optim.AdamW(M.parameters(),lr=3e-4,weight_decay=1e-4)
sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,4000); t0=time.time()
for s in range(4000):
    obs,q,qz,Hs=gen(24); pred=M(obs,q); loss=(((pred-qz)**2).mean(1)/(Hs**2+1e-6)).mean()
    opt.zero_grad(); loss.backward(); opt.step(); sch.step()
    if s%800==0 or s==3999:
        with torch.no_grad(): rmse=((pred-qz)**2).mean().sqrt().item()
        print(f"  step {s} loss {loss.item():.4f} rmse {rmse:.3f} ({time.time()-t0:.0f}s)",flush=True)

# eval on a fixed episode (same as diagnose)
def epi(Hs,Tp,seed):
    c=jonswap_components(Hs=Hs,Tp=Tp,n_components=64,depth=DEPTH,seed=seed)
    A=np.array([x.A for x in c]);om=np.array([x.omega for x in c]);kx=np.array([x.kx for x in c]);ky=np.array([x.ky for x in c]);ph=np.array([x.phi for x in c])
    Tobs=K_TP*Tp;pool=pool_xy();rng=np.random.default_rng(seed);oi=rng.integers(0,len(pool),200);ox,oy=pool[oi,0],pool[oi,1];ot=rng.uniform(0,Tobs,200)
    oz=(A[None]*np.cos(kx[None]*ox[:,None]+ky[None]*oy[:,None]-om[None]*ot[:,None]+ph[None])).sum(1)+rng.normal(0,0.02,200)
    gt=(A[None]*np.cos(kx[None]*Xg.ravel()[:,None]+ky[None]*Yg.ravel()[:,None]-om[None]*Tobs+ph[None])).sum(1).reshape(Xg.shape)
    return np.column_stack([ox,oy,ot]).astype(np.float32),oz.astype(np.float32),Tobs,gt
M.eval(); P,z,Tobs,gt=epi(1.2,7.0,2)
ob=torch.tensor(np.column_stack([P[:,0],P[:,1],P[:,2],z])[None],dtype=torch.float32,device=DEVICE)
with torch.no_grad():
    fld=M(ob,torch.tensor(np.column_stack([grid,np.full(len(grid),float(Tobs))])[None],dtype=torch.float32,device=DEVICE))[0].cpu().numpy().reshape(Xg.shape)
    ofit=M(ob,torch.tensor(np.column_stack([P[:,0],P[:,1],P[:,2]])[None],dtype=torch.float32,device=DEVICE))[0].cpu().numpy()
print(f"\nMSE-operator: field std {fld.std():.3f} (GT {gt.std():.3f})  nRMSE {np.sqrt(np.mean((fld-gt)**2))/(4*gt.std()):.3f}  corr {np.corrcoef(fld.ravel(),gt.ravel())[0,1]:.3f}")
print(f"MSE-operator at OBS: rmse(pred,z) {np.sqrt(np.mean((ofit-z)**2)):.3f}  corr {np.corrcoef(ofit,z)[0,1]:.3f}")
torch.save(M.state_dict(),'results/direction3/mse_operator_fc.pt'); print("saved mse_operator_fc.pt")
