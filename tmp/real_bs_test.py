"""Real-data test: neural-guided RELAX on BS_2011 (WASS stereo). Train component-net on BS-DOMAIN synthetic
(14m patch, Tp2-4, kp~0.45, KMAX 1.5, sparse obs) -> apply net+refine to REAL BS obs -> vs RELAX. sim2real."""
import os, sys, numpy as np, torch, torch.nn as nn, time
sys.path.insert(0,'/home/vexit/wavesplat/wavesplat'); os.chdir('/home/vexit/wavesplat/wavesplat'); sys.path.insert(0,'scripts')
from d52_F4_neural_operator import DEVICE
from scripts.realdata_analysis import load_surface_nc, peak_period
from wavesim.wave_generator import jonswap_components
from wavesplat.model.tnse import fit_amplitudes_phases, render_wave_field
from estimator_improved import relax_cv, estimate_klim
from d17_recursive_estimator import refine_params
G=9.81; K_TP=2.2; KMAX=1.5; MOUT=16; DEPTH=30.0; NOBS=200
# --- load real BS, build eval grid + obs windows ---
F='data/real_bergamasco/BS_2011/Surfaces_20111004_113800_short.nc'
Z,gxf,gyf,t,meta=load_surface_nc(F,tmax_frames=400); Tf,Ny,Nx=Z.shape; dt=float(np.median(np.diff(t)))
# coarse eval grid over the BS patch
gx=np.linspace(gxf.min(),gxf.max(),64); gy=np.linspace(gyf.min(),gyf.max(),64); Xg,Yg=np.meshgrid(gx,gy)
Tp_bs,_=peak_period(Z,t); Tp_bs=max(Tp_bs,2.0); n_back=min(int(4*Tp_bs/dt),Tf-2)
XSc=gxf.max()-gxf.min()
class FF(nn.Module):
    def __init__(s,n=48):
        super().__init__(); g=torch.Generator().manual_seed(0)
        mag=torch.exp(torch.linspace(np.log(0.1/(2*np.pi)),np.log(1.4/(2*np.pi)),n)); ang=torch.rand(n,generator=g)*2*np.pi
        Bt=torch.exp(torch.linspace(np.log(1.0/(2*np.pi)),np.log(6.0/(2*np.pi)),n))
        s.register_buffer('B',torch.stack([mag*torch.cos(ang),mag*torch.sin(ang),Bt]))
    def forward(s,x): p=x@s.B; return torch.cat([torch.sin(2*np.pi*p),torch.cos(2*np.pi*p)],-1)
class Net(nn.Module):
    def __init__(s,d=192,heads=6,enc=4):
        super().__init__(); s.ff=FF(48); fd=2*48
        s.oenc=nn.Sequential(nn.Linear(fd+1,d),nn.GELU(),nn.Linear(d,d))
        el=nn.TransformerEncoderLayer(d,heads,4*d,batch_first=True,activation='gelu',norm_first=True); s.enc=nn.TransformerEncoder(el,enc)
        s.qtok=nn.Parameter(torch.randn(1,MOUT,d)*0.02); s.attn=nn.MultiheadAttention(d,heads,batch_first=True)
        s.head=nn.Sequential(nn.Linear(d,d),nn.GELU(),nn.Linear(d,2))
    def forward(s,obs):
        h=s.enc(s.oenc(torch.cat([s.ff(obs[...,:3]),obs[...,3:4]],-1)))
        q=s.qtok.expand(obs.shape[0],-1,-1); r,_=s.attn(q,h,h); return torch.tanh(s.head(r))*KMAX
# --- BS-domain synthetic (patch-matched) for training ---
def syn(seed):
    rng=np.random.default_rng(seed); Hs=rng.uniform(0.15,0.6); Tp=rng.uniform(2.0,4.5); NC=int(rng.choice([3,4,6,8]))
    c=jonswap_components(Hs=Hs,Tp=Tp,n_components=NC,depth=DEPTH,seed=seed)
    A=np.array([x.A for x in c]);om=np.array([x.omega for x in c]);kx=np.array([x.kx for x in c]);ky=np.array([x.ky for x in c]);ph=np.array([x.phi for x in c])
    Tobs=K_TP*Tp; ox=rng.uniform(gx.min(),gx.max(),NOBS); oy=rng.uniform(gy.min(),gy.max(),NOBS); ot=rng.uniform(0,Tobs,NOBS)
    oz=(A[None]*np.cos(kx[None]*ox[:,None]+ky[None]*oy[:,None]-om[None]*ot[:,None]+ph[None])).sum(1)+rng.normal(0,0.01,NOBS)
    tk=np.zeros((MOUT,2),np.float32); ta=np.zeros(MOUT,np.float32); n=min(NC,MOUT); tk[:n,0]=kx[:n];tk[:n,1]=ky[:n];ta[:n]=A[:n]
    return np.column_stack([ox,oy,ot,oz]).astype(np.float32),tk,ta
def gen(B):
    obs=np.zeros((B,NOBS,4),np.float32);tk=np.zeros((B,MOUT,2),np.float32);ta=np.zeros((B,MOUT),np.float32)
    for b in range(B): o,k,A=syn(int(np.random.randint(1e9))); obs[b]=o;tk[b]=k;ta[b]=A
    t=lambda x: torch.tensor(x,device=DEVICE); return t(obs),t(tk),t(ta)
def chamfer(pred,true,tw):
    d=((pred[:,:,None,:]-true[:,None,:,:])**2).sum(-1); return ((d.min(1).values*tw).sum(1)/(tw.sum(1)+1e-9)).mean()+0.1*d.min(2).values.mean()
# --- train ---
M=Net().to(DEVICE); opt=torch.optim.AdamW(M.parameters(),lr=3e-4,weight_decay=1e-4); STEPS=10000; sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,STEPS); t0=time.time()
print(f"BS-domain train: patch {XSc:.0f}m Tp~{Tp_bs:.1f} KMAX{KMAX} obs{NOBS}")
for s in range(STEPS):
    obs,tk,ta=gen(24); loss=chamfer(M(obs),tk,ta); opt.zero_grad(); loss.backward(); opt.step(); sch.step()
    if s%2500==0 or s==STEPS-1: print(f"  step {s} chamfer {loss.item():.4f} ({time.time()-t0:.0f}s)",flush=True)
M.eval()
# --- eval on REAL BS ---
def score(kx,ky,obs,Tobs,gt,depth):
    A,phi=fit_amplitudes_phases(kx,ky,obs.astype(np.float32),depth,lambda_reg=0.5)
    f=render_wave_field(kx,ky,A,phi,depth,gx,gy,float(Tobs)); v=~np.isnan(gt); return np.corrcoef(f[v],gt[v])[0,1]
print("\n=== REAL BS_2011: neural-guided vs RELAX (3 frames, sparse obs from stereo field) ===")
rng=np.random.default_rng(0); frames=np.linspace(n_back+2,Tf-2,3).astype(int)
Xf,Yf=np.meshgrid(gxf,gyf); Xfr,Yfr=Xf.ravel(),Yf.ravel()
for i_star in frames:
    gt=Z[i_star]; Hs=4*np.nanstd(gt); Ps,zs=[],[]
    for i in range(i_star-n_back,i_star+1):
        fr=Z[i].ravel(); idx=rng.choice(len(fr),NOBS//(n_back+1)+1,replace=False)
        Ps.append(np.column_stack([Xfr[idx],Yfr[idx],np.full(len(idx),i*dt)])); zs.append(fr[idx])
    P=np.vstack(Ps).astype(np.float32); z=np.concatenate(zs).astype(np.float32)
    m=~np.isnan(z); P,z=P[m],z[m]; obs=np.column_stack([P,z]).astype(np.float32)
    tgt={'gx':gx,'gy':gy,'t':float(i_star*dt),'gt':np.nan_to_num(gt),'Hs':Hs}
    with torch.no_grad(): pk=M(torch.tensor(obs[None],device=DEVICE))[0].cpu().numpy()
    nkx,nky,_=refine_params(pk[:,0].astype(np.float32),pk[:,1].astype(np.float32),P,z,DEPTH,tgt,iters=400); c_net=score(nkx,nky,obs,i_star*dt,gt,DEPTH)
    klim=estimate_klim(P,z,DEPTH); rkx,rky=relax_cv(P,z,DEPTH,klim); rkx,rky,_=refine_params(rkx,rky,P,z,DEPTH,tgt,iters=400); c_rel=score(rkx,rky,obs,i_star*dt,gt,DEPTH)
    print(f"  frame{i_star} nobs={len(z)}: neural-guided corr {c_net:.3f} | RELAX corr {c_rel:.3f}",flush=True)
