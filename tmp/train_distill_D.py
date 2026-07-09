"""Distill RELAX into the FF amortized net: obs -> RELAX-rendered field (structure-preserving, non-blank).
Target is a deterministic fn of obs (RELAX output) so the net can learn it (no mean-collapse)."""
import os, sys, numpy as np, torch, time
sys.path.insert(0,'/home/vexit/wavesplat/wavesplat'); os.chdir('/home/vexit/wavesplat/wavesplat'); sys.path.insert(0,'scripts')
from amortized_ff import AmortizedFF
from d52_F4_neural_operator import DEVICE
G=9.81; DEPTH=50.0
cache=list(np.load('results/direction3/_relax_distill_cache.npz',allow_pickle=True)['rows'])
print(f"loaded {len(cache)} RELAX episodes")

def render(kx,ky,A,phi,qx,qy,qt):
    k=np.sqrt(kx**2+ky**2+1e-12); om=np.sqrt(G*k*np.tanh(k*DEPTH)+1e-12)
    return (A[None]*np.cos(kx[None]*qx[:,None]+ky[None]*qy[:,None]-om[None]*qt[:,None]+phi[None])).sum(1)

def batch(B=24, Q=400, fc=6.0):
    obs=np.zeros((B,200,4),np.float32); qpt=np.zeros((B,Q,3),np.float32); qz=np.zeros((B,Q),np.float32); Hsb=np.zeros(B,np.float32)
    idx=np.random.randint(0,len(cache),B)
    for b,j in enumerate(idx):
        e=cache[j]; P=e['P']; z=e['z']; Tobs=e['Tobs']
        obs[b]=np.column_stack([P[:,0],P[:,1],P[:,2],z])
        qx=np.random.uniform(10,250,Q); qy=np.random.uniform(-100,100,Q); qt=np.random.uniform(0.5*Tobs,Tobs+fc,Q)
        qz[b]=render(e['kx'],e['ky'],e['A'],e['phi'],qx,qy,qt)   # RELAX field (target)
        qpt[b]=np.column_stack([qx,qy,qt]); Hsb[b]=e['Hs']
    t=lambda a: torch.tensor(a,device=DEVICE); return t(obs),t(qpt),t(qz),t(Hsb)

M=AmortizedFF(uq=False).to(DEVICE); opt=torch.optim.AdamW(M.parameters(),lr=3e-4,weight_decay=1e-4)
STEPS=7000; sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,STEPS); t0=time.time()
for s in range(STEPS):
    obs,q,qz,Hs=batch(); pred=M(obs,q); loss=(((pred-qz)**2).mean(1)/(Hs**2+1e-6)).mean()
    opt.zero_grad(); loss.backward(); opt.step(); sch.step()
    if s%1000==0 or s==STEPS-1:
        with torch.no_grad(): rmse=((pred-qz)**2).mean().sqrt().item()
        print(f"  step {s} loss {loss.item():.4f} rmse-vs-RELAX {rmse:.3f} (sig std {qz.std().item():.3f}) ({time.time()-t0:.0f}s)",flush=True)
torch.save(M.state_dict(),'results/direction3/amortized_distill.pt'); print("saved amortized_distill.pt")

# how well does distilled-D match RELAX AND how does it do vs GT? (eval on held-out fresh episodes needs GT)
M.eval()
gx=np.linspace(10,250,80); gy=np.linspace(-100,100,60); Xg,Yg=np.meshgrid(gx,gy); grid=np.column_stack([Xg.ravel(),Yg.ravel()])
print("=== distilled-D field std check (should be NON-zero = structure) ===")
for j in [0,1,2]:
    e=cache[j]; P=e['P']; z=e['z']; Tobs=e['Tobs']
    ob=torch.tensor(np.column_stack([P[:,0],P[:,1],P[:,2],z])[None],dtype=torch.float32,device=DEVICE)
    with torch.no_grad(): fld=M(ob,torch.tensor(np.column_stack([grid,np.full(len(grid),Tobs)])[None],dtype=torch.float32,device=DEVICE))[0].cpu().numpy().reshape(Xg.shape)
    relaxf=render(e['kx'],e['ky'],e['A'],e['phi'],Xg.ravel(),Yg.ravel(),np.full(len(grid),Tobs)).reshape(Xg.shape)
    cr=np.corrcoef(fld.ravel(),relaxf.ravel())[0,1]
    print(f"  ep{j}: distilled-D std {fld.std():.3f} (RELAX std {relaxf.std():.3f})  corr(D,RELAX) {cr:.3f}")
