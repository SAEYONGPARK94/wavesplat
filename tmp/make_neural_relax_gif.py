"""GIF: [GT | neural-guided RELAX (net+refine) | RELAX | diff | diff] over nowcast->forecast.
Shows the learned component-net + physics refine matches RELAX."""
import os, sys, numpy as np, torch, torch.nn as nn
sys.path.insert(0,'/home/vexit/wavesplat/wavesplat'); os.chdir('/home/vexit/wavesplat/wavesplat'); sys.path.insert(0,'scripts')
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from d52_F4_neural_operator import DEVICE
from amortized_ff import FourierFeatures
from wavesim.wave_generator import jonswap_components
from d40_learned_acquisition import pool_xy
from wavesplat.model.tnse import fit_amplitudes_phases, render_wave_field
from estimator_improved import relax_cv, estimate_klim
from d17_recursive_estimator import refine_params
G=9.81; DEPTH=50.0; K_TP=2.2; KMAX=0.6; NC=4; MO=600; MOUT=10
gx=np.linspace(10,250,80); gy=np.linspace(-100,100,60); Xg,Yg=np.meshgrid(gx,gy); grid=np.column_stack([Xg.ravel(),Yg.ravel()])
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
M=Net().to(DEVICE); M.load_state_dict(torch.load('results/direction3/component_distill.pt',map_location=DEVICE)); M.eval()
def comps(Hs,Tp,seed):
    c=jonswap_components(Hs=Hs,Tp=Tp,n_components=NC,depth=DEPTH,seed=seed)
    return (np.array([x.A for x in c]),np.array([x.omega for x in c]),np.array([x.kx for x in c]),np.array([x.ky for x in c]),np.array([x.phi for x in c]))
def make(Hs,Tp,seed):
    A,om,kx,ky,ph=comps(Hs,Tp,seed); Tobs=K_TP*Tp; pool=pool_xy(); rng=np.random.default_rng(seed)
    oi=rng.integers(0,len(pool),MO); ox,oy=pool[oi,0],pool[oi,1]; ot=rng.uniform(0,Tobs,MO)
    oz=(A[None]*np.cos(kx[None]*ox[:,None]+ky[None]*oy[:,None]-om[None]*ot[:,None]+ph[None])).sum(1)+rng.normal(0,0.02,MO)
    def gt_at(t): return (A[None]*np.cos(kx[None]*Xg.ravel()[:,None]+ky[None]*Yg.ravel()[:,None]-om[None]*t+ph[None])).sum(1).reshape(Xg.shape)
    return np.column_stack([ox,oy,ot,oz]).astype(np.float32),Tobs,gt_at
os.makedirs('results/gifs',exist_ok=True)
for ei,(Hs,Tp,seed) in enumerate([(1.2,7.0,2),(2.0,9.0,4),(1.0,6.5,9)]):
    obs,Tobs,gt_at=make(Hs,Tp,seed); P=obs[:,:3].astype(np.float32); z=obs[:,3].astype(np.float32); Hs4=4*gt_at(Tobs).std()
    target={'gx':gx,'gy':gy,'t':float(Tobs),'gt':gt_at(Tobs),'Hs':Hs4}
    with torch.no_grad(): pk=M(torch.tensor(obs[None],device=DEVICE))[0].cpu().numpy()
    nkx,nky,_=refine_params(pk[:,0].astype(np.float32),pk[:,1].astype(np.float32),P,z,DEPTH,target,iters=400)
    nA,nphi=fit_amplitudes_phases(nkx,nky,obs.astype(np.float32),DEPTH,lambda_reg=0.5)
    klim=estimate_klim(P,z,DEPTH); rkx,rky=relax_cv(P,z,DEPTH,klim); rkx,rky,_=refine_params(rkx,rky,P,z,DEPTH,target,iters=400)
    rA,rphi=fit_amplitudes_phases(rkx,rky,obs.astype(np.float32),DEPTH,lambda_reg=0.5)
    leads=np.arange(0,6.01,0.5); times=Tobs+leads; vmax=2*gt_at(Tobs).std()
    fig,axes=plt.subplots(1,5,figsize=(17,3.4))
    def draw(k):
        t=times[k]; g=gt_at(t); nf=render_wave_field(nkx,nky,nA,nphi,DEPTH,gx,gy,float(t)); rf=render_wave_field(rkx,rky,rA,rphi,DEPTH,gx,gy,float(t))
        cn=np.corrcoef(nf.ravel(),g.ravel())[0,1]; cr=np.corrcoef(rf.ravel(),g.ravel())[0,1]
        for ax in axes: ax.clear(); ax.set_xticks([]); ax.set_yticks([])
        axes[0].imshow(g,vmin=-vmax,vmax=vmax,cmap='RdBu_r',origin='lower',aspect='auto'); axes[0].set_title('GT',fontsize=10)
        axes[1].imshow(nf,vmin=-vmax,vmax=vmax,cmap='RdBu_r',origin='lower',aspect='auto'); axes[1].set_title(f'neural-guided (net+refine) r={cn:.2f}',fontsize=9)
        axes[2].imshow(rf,vmin=-vmax,vmax=vmax,cmap='RdBu_r',origin='lower',aspect='auto'); axes[2].set_title(f'RELAX r={cr:.2f}',fontsize=10)
        axes[3].imshow(nf-g,vmin=-vmax,vmax=vmax,cmap='PuOr',origin='lower',aspect='auto'); axes[3].set_title('diff neural',fontsize=10)
        axes[4].imshow(rf-g,vmin=-vmax,vmax=vmax,cmap='PuOr',origin='lower',aspect='auto'); axes[4].set_title('diff RELAX',fontsize=10)
        fig.suptitle(f'Ep{ei+1} Hs{Hs} Tp{Tp}s | {"NOWCAST" if leads[k]==0 else f"FORECAST +{leads[k]:.1f}s"}  neural-guided RELAX vs RELAX',fontsize=12)
    anim=FuncAnimation(fig,draw,frames=len(times),interval=350); out=f'results/gifs/neuralRELAX_vs_RELAX_ep{ei+1}.gif'
    anim.save(out,writer=PillowWriter(fps=3)); plt.close(fig); print(f"saved {out}",flush=True)
print("DONE")
