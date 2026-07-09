"""5-panel GIF with DISTILLED D (structure-preserving). [GT | D distilled | B physics | diffD | diffB]."""
import os, sys, numpy as np, torch
sys.path.insert(0,'/home/vexit/wavesplat/wavesplat'); os.chdir('/home/vexit/wavesplat/wavesplat'); sys.path.insert(0,'scripts')
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from amortized_ff import AmortizedFF
from d52_F4_neural_operator import DEVICE
from estimator_improved import relax_cv, estimate_klim
from d17_recursive_estimator import refine_params
from wavesplat.model.tnse import fit_amplitudes_phases, render_wave_field
from wavesim.wave_generator import jonswap_components
from d40_learned_acquisition import pool_xy
G=9.81; DEPTH=50.0; K_TP=2.2
gx=np.linspace(10,250,80); gy=np.linspace(-100,100,60); Xg,Yg=np.meshgrid(gx,gy); grid=np.column_stack([Xg.ravel(),Yg.ravel()])
os.makedirs('results/gifs',exist_ok=True)
D=AmortizedFF(uq=False).to(DEVICE); D.load_state_dict(torch.load('results/direction3/amortized_distill.pt',map_location=DEVICE)); D.eval()

def episode(Hs,Tp,seed):
    c=jonswap_components(Hs=Hs,Tp=Tp,n_components=64,depth=DEPTH,seed=seed)
    A=np.array([x.A for x in c]);om=np.array([x.omega for x in c]);kx=np.array([x.kx for x in c]);ky=np.array([x.ky for x in c]);ph=np.array([x.phi for x in c])
    Tobs=K_TP*Tp;pool=pool_xy();rng=np.random.default_rng(seed)
    oi=rng.integers(0,len(pool),200);ox,oy=pool[oi,0],pool[oi,1];ot=rng.uniform(0,Tobs,200)
    oz=(A[None]*np.cos(kx[None]*ox[:,None]+ky[None]*oy[:,None]-om[None]*ot[:,None]+ph[None])).sum(1)+rng.normal(0,0.02,200)
    P=np.column_stack([ox,oy,ot]).astype(np.float32);z=oz.astype(np.float32)
    def gt_at(t): return (A[None]*np.cos(kx[None]*Xg.ravel()[:,None]+ky[None]*Yg.ravel()[:,None]-om[None]*t+ph[None])).sum(1).reshape(Xg.shape)
    return P,z,Tobs,gt_at

def B_components(P,z,Tobs,gt_star,Hs):
    target={'gx':gx,'gy':gy,'t':float(Tobs),'gt':gt_star,'Hs':Hs}
    klim=estimate_klim(P,z,DEPTH); kx,ky=relax_cv(P,z,DEPTH,klim)
    kx,ky,_=refine_params(kx,ky,P,z,DEPTH,target,iters=400)
    A,phi=fit_amplitudes_phases(kx,ky,np.column_stack([P,z]).astype(np.float32),DEPTH,lambda_reg=0.5)
    return kx,ky,A,phi
def D_at(P,z,t):
    ob=torch.tensor(np.column_stack([P[:,0],P[:,1],P[:,2],z])[None],dtype=torch.float32,device=DEVICE)
    q=torch.tensor(np.column_stack([grid,np.full(len(grid),float(t))])[None],dtype=torch.float32,device=DEVICE)
    with torch.no_grad(): m=D(ob,q)
    return m[0].cpu().numpy().reshape(Xg.shape)

EPS=[(0.8,6.0,1),(1.2,7.0,2),(1.6,8.0,3),(2.0,9.0,4)]
for ei,(Hs,Tp,seed) in enumerate(EPS):
    P,z,Tobs,gt_at=episode(Hs,Tp,seed); kx,ky,A,phi=B_components(P,z,Tobs,gt_at(Tobs),Hs)
    leads=np.arange(0,6.01,0.5); times=Tobs+leads; vmax=2.0*float(gt_at(Tobs).std())
    fig,axes=plt.subplots(1,5,figsize=(17,3.4))
    def draw(k):
        t=times[k]; g=gt_at(t); d=D_at(P,z,t); b=render_wave_field(kx,ky,A,phi,DEPTH,gx,gy,float(t))
        nrd=np.sqrt(np.mean((d-g)**2))/(4*g.std()); nrb=np.sqrt(np.mean((b-g)**2))/(4*g.std())
        for ax in axes: ax.clear(); ax.set_xticks([]); ax.set_yticks([])
        axes[0].imshow(g,vmin=-vmax,vmax=vmax,cmap='RdBu_r',origin='lower',aspect='auto'); axes[0].set_title('GT (original)',fontsize=10)
        axes[1].imshow(d,vmin=-vmax,vmax=vmax,cmap='RdBu_r',origin='lower',aspect='auto'); axes[1].set_title(f'D distilled  nRMSE {nrd:.2f}',fontsize=10)
        axes[2].imshow(b,vmin=-vmax,vmax=vmax,cmap='RdBu_r',origin='lower',aspect='auto'); axes[2].set_title(f'B physics  nRMSE {nrb:.2f}',fontsize=10)
        axes[3].imshow(d-g,vmin=-vmax,vmax=vmax,cmap='PuOr',origin='lower',aspect='auto'); axes[3].set_title('diff D',fontsize=10)
        axes[4].imshow(b-g,vmin=-vmax,vmax=vmax,cmap='PuOr',origin='lower',aspect='auto'); axes[4].set_title('diff B',fontsize=10)
        lead=leads[k]; tag='NOWCAST' if lead==0 else f'FORECAST +{lead:.1f}s'
        fig.suptitle(f'Ep{ei+1}: Hs={Hs} Tp={Tp}s  |  {tag}  (D=RELAX-distilled amortized)',fontsize=12)
    anim=FuncAnimation(fig,draw,frames=len(times),interval=350)
    out=f'results/gifs/DvsB_distill_ep{ei+1}_Hs{Hs}_Tp{Tp}.gif'
    anim.save(out,writer=PillowWriter(fps=3)); plt.close(fig); print(f'saved {out}',flush=True)
print('DONE')
