"""Train the best learned net (plain wave-band-FF AmortizedFF, field-MSE) on determined episodes,
then GIF: [GT | learned-net | RELAX | diff-net | diff-RELAX] over nowcast->forecast. Honest net(~0.6) vs RELAX(~0.98)."""
import os, sys, numpy as np, torch, time
sys.path.insert(0,'/home/vexit/wavesplat/wavesplat'); os.chdir('/home/vexit/wavesplat/wavesplat'); sys.path.insert(0,'scripts')
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from amortized_ff import AmortizedFF
from d52_F4_neural_operator import DEVICE
from wavesim.wave_generator import jonswap_components
from d40_learned_acquisition import pool_xy
from estimator_improved import relax_cv, estimate_klim
from d17_recursive_estimator import refine_params
from wavesplat.model.tnse import fit_amplitudes_phases, render_wave_field
G=9.81; DEPTH=50.0; K_TP=2.2; NC=4; MO=600
gx=np.linspace(10,250,80); gy=np.linspace(-100,100,60); Xg,Yg=np.meshgrid(gx,gy); grid=np.column_stack([Xg.ravel(),Yg.ravel()])
def comps(Hs,Tp,seed):
    c=jonswap_components(Hs=Hs,Tp=Tp,n_components=NC,depth=DEPTH,seed=seed)
    return (np.array([x.A for x in c]),np.array([x.omega for x in c]),np.array([x.kx for x in c]),np.array([x.ky for x in c]),np.array([x.phi for x in c]))
def episode(Hs,Tp,seed):
    A,om,kx,ky,ph=comps(Hs,Tp,seed); Tobs=K_TP*Tp; pool=pool_xy(); rng=np.random.default_rng(seed)
    oi=rng.integers(0,len(pool),MO); ox,oy=pool[oi,0],pool[oi,1]; ot=rng.uniform(0,Tobs,MO)
    oz=(A[None]*np.cos(kx[None]*ox[:,None]+ky[None]*oy[:,None]-om[None]*ot[:,None]+ph[None])).sum(1)+rng.normal(0,0.02,MO)
    obs=np.column_stack([ox,oy,ot,oz]).astype(np.float32)
    def gt_at(t): return (A[None]*np.cos(kx[None]*Xg.ravel()[:,None]+ky[None]*Yg.ravel()[:,None]-om[None]*t+ph[None])).sum(1).reshape(Xg.shape)
    return obs,Tobs,gt_at
def train_batch(B):
    obs=np.zeros((B,MO,4),np.float32);Q=400;qp=np.zeros((B,Q,3),np.float32);qz=np.zeros((B,Q),np.float32);Hsb=np.zeros(B,np.float32)
    pool=pool_xy()
    for b in range(B):
        Hs=np.random.uniform(0.6,2.0);Tp=np.random.uniform(5.5,9.5); A,om,kx,ky,ph=comps(Hs,Tp,int(np.random.randint(1e9)))
        Tobs=K_TP*Tp; oi=np.random.randint(0,len(pool),MO); ox,oy=pool[oi,0],pool[oi,1]; ot=np.random.uniform(0,Tobs,MO)
        oz=(A[None]*np.cos(kx[None]*ox[:,None]+ky[None]*oy[:,None]-om[None]*ot[:,None]+ph[None])).sum(1)+np.random.normal(0,0.02,MO)
        obs[b]=np.column_stack([ox,oy,ot,oz]); qx=np.random.uniform(10,250,Q);qy=np.random.uniform(-100,100,Q);qt=np.random.uniform(0.5*Tobs,Tobs+6,Q)
        qz[b]=(A[None]*np.cos(kx[None]*qx[:,None]+ky[None]*qy[:,None]-om[None]*qt[:,None]+ph[None])).sum(1); qp[b]=np.column_stack([qx,qy,qt]); Hsb[b]=Hs
    t=lambda x: torch.tensor(x,device=DEVICE); return t(obs),t(qp),t(qz),t(Hsb)
# --- train ---
M=AmortizedFF(uq=False).to(DEVICE); opt=torch.optim.AdamW(M.parameters(),lr=3e-4,weight_decay=1e-4)
STEPS=13000; sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,STEPS); t0=time.time()
print("training plain wave-band-FF net...")
for s in range(STEPS):
    obs,q,qz,Hs=train_batch(24); pred=M(obs,q); loss=(((pred-qz)**2).mean(1)/(Hs**2+1e-6)).mean()
    opt.zero_grad(); loss.backward(); opt.step(); sch.step()
    if s%3000==0 or s==STEPS-1: print(f"  step {s} loss {loss.item():.4f} ({time.time()-t0:.0f}s)",flush=True)
M.eval(); torch.save(M.state_dict(),'results/direction3/net_vs_relax.pt')
def net_at(obs,t):
    ob=torch.tensor(obs[None],device=DEVICE); q=torch.tensor(np.column_stack([grid,np.full(len(grid),float(t))])[None],dtype=torch.float32,device=DEVICE)
    with torch.no_grad(): return M(ob,q)[0].cpu().numpy().reshape(Xg.shape)
# --- GIF ---
os.makedirs('results/gifs',exist_ok=True)
for ei,(Hs,Tp,seed) in enumerate([(0.8,6.0,1),(1.2,7.0,2),(2.0,9.0,4)]):
    obs,Tobs,gt_at=episode(Hs,Tp,seed); P=obs[:,:3].astype(np.float32); z=obs[:,3].astype(np.float32); Hsv=4*gt_at(Tobs).std()
    target={'gx':gx,'gy':gy,'t':float(Tobs),'gt':gt_at(Tobs),'Hs':Hsv}
    klim=estimate_klim(P,z,DEPTH); rkx,rky=relax_cv(P,z,DEPTH,klim); rkx,rky,_=refine_params(rkx,rky,P,z,DEPTH,target,iters=400)
    rA,rphi=fit_amplitudes_phases(rkx,rky,obs.astype(np.float32),DEPTH,lambda_reg=0.5)
    leads=np.arange(0,6.01,0.5); times=Tobs+leads; vmax=2*gt_at(Tobs).std()
    fig,axes=plt.subplots(1,5,figsize=(17,3.4))
    def draw(k):
        t=times[k]; g=gt_at(t); nf=net_at(obs,t); rf=render_wave_field(rkx,rky,rA,rphi,DEPTH,gx,gy,float(t))
        cn=np.corrcoef(nf.ravel(),g.ravel())[0,1]; cr=np.corrcoef(rf.ravel(),g.ravel())[0,1]
        for ax in axes: ax.clear(); ax.set_xticks([]); ax.set_yticks([])
        axes[0].imshow(g,vmin=-vmax,vmax=vmax,cmap='RdBu_r',origin='lower',aspect='auto'); axes[0].set_title('GT',fontsize=10)
        axes[1].imshow(nf,vmin=-vmax,vmax=vmax,cmap='RdBu_r',origin='lower',aspect='auto'); axes[1].set_title(f'learned net  r={cn:.2f}',fontsize=10)
        axes[2].imshow(rf,vmin=-vmax,vmax=vmax,cmap='RdBu_r',origin='lower',aspect='auto'); axes[2].set_title(f'RELAX  r={cr:.2f}',fontsize=10)
        axes[3].imshow(nf-g,vmin=-vmax,vmax=vmax,cmap='PuOr',origin='lower',aspect='auto'); axes[3].set_title('diff net',fontsize=10)
        axes[4].imshow(rf-g,vmin=-vmax,vmax=vmax,cmap='PuOr',origin='lower',aspect='auto'); axes[4].set_title('diff RELAX',fontsize=10)
        lead=leads[k]; fig.suptitle(f'Ep{ei+1} Hs{Hs} Tp{Tp}s | {"NOWCAST" if lead==0 else f"FORECAST +{lead:.1f}s"}  (learned net vs RELAX)',fontsize=12)
    anim=FuncAnimation(fig,draw,frames=len(times),interval=350); out=f'results/gifs/net_vs_RELAX_ep{ei+1}.gif'
    anim.save(out,writer=PillowWriter(fps=3)); plt.close(fig); print(f"saved {out}",flush=True)
print("DONE")
