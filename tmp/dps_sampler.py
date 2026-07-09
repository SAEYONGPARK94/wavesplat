"""CREATIVE step 2: DPS. 2-channel (eta,eta_t) diffusion prior + physics forward (FFT 2-branch dispersion
propagation, ij-consistent) -> posterior sampling of wave state given sparse space-time obs. Output posterior
MEAN (point est) + STD (UQ). vs RELAX on hard/sparse regimes. Forward operator verified corr~0.92."""
import os, sys, numpy as np, torch, torch.nn as nn, torch.nn.functional as F, time, math, argparse
sys.path.insert(0,'/home/vexit/wavesplat/wavesplat'); os.chdir('/home/vexit/wavesplat/wavesplat'); sys.path.insert(0,'scripts')
from d52_F4_neural_operator import DEVICE
from wavesim.wave_generator import jonswap_components
from estimator_improved import relax_cv, estimate_klim
from d17_recursive_estimator import refine_params
from d40_learned_acquisition import pool_xy
POOL=pool_xy()
G=9.81; DEPTH=50.0; N=64
gx=np.linspace(10,250,N); gy=np.linspace(-100,100,N); Xg,Yg=np.meshgrid(gx,gy,indexing='ij')  # axis0=x
x0c=gx[0]; y0c=gy[0]; dx=gx[1]-gx[0]; dy=gy[1]-gy[0]
kxf=2*np.pi*np.fft.fftfreq(N,d=dx); kyf=2*np.pi*np.fft.fftfreq(N,d=dy)
KXf,KYf=np.meshgrid(kxf,kyf,indexing='ij'); KK=np.sqrt(KXf**2+KYf**2); WW=np.sqrt(G*KK*np.tanh(KK*DEPTH)); WW[0,0]=1e9
tKX=torch.tensor(KXf,device=DEVICE,dtype=torch.float32); tKY=torch.tensor(KYf,device=DEVICE,dtype=torch.float32); tW=torch.tensor(WW,device=DEVICE,dtype=torch.float32)
# --- UNet + schedule from prior script ---
exec(open('/home/vexit/wavesplat/tmp/diffusion_prior.py').read().split('# --- DDPM')[0].replace('a=ap.parse_args()','import argparse;a=argparse.Namespace(steps=0,T=1000)'))
ck=torch.load('results/direction3/diffusion_prior.pt',map_location=DEVICE); T=ck['T']; betas=ck['betas'].to(DEVICE)
alphas=1-betas; abar=torch.cumprod(alphas,0); M=UNet().to(DEVICE); M.load_state_dict(ck['model']); M.eval()
def fwd2(eta,eta_t,ox,oy,ot,tstar):  # 2-ch state@t* -> obs pred (2-branch dispersion propagation). differentiable
    P=torch.fft.fft2(eta); Q=torch.fft.fft2(eta_t); hp=(P+1j*Q/tW)/2; hm=(P-1j*Q/tW)/2
    tp=(ot-tstar)[:,None,None]; sp=tKX[None]*(ox-x0c)[:,None,None]+tKY[None]*(oy-y0c)[:,None,None]
    fld=hp[None]*torch.exp(1j*(sp-tW[None]*tp))+hm[None]*torch.exp(1j*(sp+tW[None]*tp))
    return fld.real.sum((1,2))/(N*N)
def episode(Hs,Tp,seed,NC,nobs):
    c=jonswap_components(Hs=Hs,Tp=Tp,n_components=NC,depth=DEPTH,seed=seed)
    A=np.array([x.A for x in c]);om=np.array([x.omega for x in c]);kx=np.array([x.kx for x in c]);ky=np.array([x.ky for x in c]);ph=np.array([x.phi for x in c])
    Tobs=2.2*Tp; rng=np.random.default_rng(seed)
    oi=rng.integers(0,len(POOL),nobs); ox=POOL[oi,0].copy(); oy=POOL[oi,1].copy(); ot=rng.uniform(0,Tobs,nobs)
    oz=(A[None]*np.cos(kx[None]*ox[:,None]+ky[None]*oy[:,None]-om[None]*ot[:,None]+ph[None])).sum(1)+rng.normal(0,0.02,nobs)
    gt=(A[None]*np.cos(kx[None]*Xg.ravel()[:,None]+ky[None]*Yg.ravel()[:,None]-om[None]*Tobs+ph[None])).sum(1).reshape(N,N)  # ij (axis0=x)
    return ox,oy,ot,oz,Tobs,gt
def dps(ox,oy,ot,oz,tstar,zeta=5.0,nsamp=8,stride=8,eta_ddim=0.5):
    ox=torch.tensor(ox,device=DEVICE,dtype=torch.float32);oy=torch.tensor(oy,device=DEVICE,dtype=torch.float32)
    ot=torch.tensor(ot,device=DEVICE,dtype=torch.float32);oz=torch.tensor(oz,device=DEVICE,dtype=torch.float32)
    ts=list(range(T-1,-1,-stride)); states=[]
    for _ in range(nsamp):
        x=torch.randn(1,2,N,N,device=DEVICE)
        for i,t in enumerate(ts):
            tt=torch.full((1,),t,device=DEVICE); ab=abar[t]
            xr=x.detach().requires_grad_(True); eps=M(xr,tt); x0=(xr-torch.sqrt(1-ab)*eps)/torch.sqrt(ab)
            pred=fwd2(x0[0,0],x0[0,1],ox,oy,ot,tstar); s=(pred*oz).sum()/((pred*pred).sum()+1e-6)
            res=((s*pred-oz)**2).mean(); g=torch.autograd.grad(res,xr)[0]
            with torch.no_grad():
                x0d=x0.detach(); tp=ts[i+1] if i+1<len(ts) else 0; abp=abar[tp]
                sig=eta_ddim*torch.sqrt((1-abp)/(1-ab))*torch.sqrt(1-ab/abp)
                x=torch.sqrt(abp)*x0d+torch.sqrt(torch.clamp(1-abp-sig**2,min=0))*eps.detach()+(sig*torch.randn_like(x) if tp>0 else 0)
                x=x-zeta/(torch.sqrt(res)+1e-3)*g
        with torch.no_grad():
            pred=fwd2(x[0,0],x[0,1],ox,oy,ot,tstar); s=(pred*oz).sum()/((pred*pred).sum()+1e-6)
            states.append((x[0]*s).cpu().numpy())  # [2,N,N]
    S=np.stack(states); sm=S.mean(0)  # [2,N,N]
    with torch.no_grad():
        pm=fwd2(torch.tensor(sm[0],device=DEVICE),torch.tensor(sm[1],device=DEVICE),ox,oy,ot,tstar).cpu().numpy()
    ofc=float(np.corrcoef(pm,oz.cpu().numpy())[0,1])
    return sm[0], S[:,0].std(0), ofc
def corr(f,g): return float(np.corrcoef(f.ravel(),g.ravel())[0,1])
if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('--config',default='8,300'); ap.add_argument('--nsamp',type=int,default=8)
    ap.add_argument('--zeta',type=float,default=5.0); ap.add_argument('--stride',type=int,default=8); ap.add_argument('--seeds',default='1,2,3'); ap.add_argument('--eta',type=float,default=0.5); A=ap.parse_args()
    NC,MO=map(int,A.config.split(',')); seeds=[int(s) for s in A.seeds.split(',')]
    print(f"DPS config NC={NC} MO={MO} nsamp={A.nsamp} zeta={A.zeta} stride={A.stride} ({len(list(range(T-1,-1,-A.stride)))} steps)")
    print(f"{'seed':>4} | {'DPS-mean':>9} {'obs-fit':>7} {'RELAX':>7} | {'UQ':>6}")
    dc,rc,uq=[],[],[]
    for sd in seeds:
        Hs=np.random.default_rng(sd).uniform(0.8,1.8);Tp=np.random.default_rng(sd+9).uniform(6,9)
        ox,oy,ot,oz,Tobs,gt=episode(Hs,Tp,sd,NC,MO)
        if sd==seeds[0]:
            # fwd sanity with a gridded true field (2-ch)
            c=jonswap_components(Hs=Hs,Tp=Tp,n_components=NC,depth=DEPTH,seed=sd)
            Aa=np.array([x.A for x in c]);om=np.array([x.omega for x in c]);kxx=np.array([x.kx for x in c]);kyy=np.array([x.ky for x in c]);phh=np.array([x.phi for x in c])
            ang=kxx[None]*Xg.ravel()[:,None]+kyy[None]*Yg.ravel()[:,None]-om[None]*Tobs+phh[None]
            e=(Aa[None]*np.cos(ang)).sum(1).reshape(N,N); et=(Aa[None]*om[None]*np.sin(ang)).sum(1).reshape(N,N)
            tt=lambda v:torch.tensor(v,device=DEVICE,dtype=torch.float32)
            pr=fwd2(tt(e),tt(et),tt(ox),tt(oy),tt(ot),Tobs).cpu().numpy()
            print(f"  [fwd sanity corr(A(true),obs)={corr(pr,oz):.3f}]")
        fmean,fstd,ofc=dps(ox,oy,ot,oz,Tobs,zeta=A.zeta,nsamp=A.nsamp,stride=A.stride,eta_ddim=A.eta); cd=corr(fmean,gt)
        P=np.column_stack([ox,oy,ot]).astype(np.float32);z=oz.astype(np.float32);tgt={'gx':gx,'gy':gy,'t':float(Tobs),'gt':gt.T,'Hs':4*gt.std()}
        kl=estimate_klim(P,z,DEPTH);rkx,rky=relax_cv(P,z,DEPTH,kl);rkx,rky,fr=refine_params(rkx,rky,P,z,DEPTH,tgt,iters=400);cr=corr(fr,gt.T)
        err=np.abs(fmean-gt); uqc=corr(fstd,err); dc.append(cd);rc.append(cr);uq.append(uqc)
        print(f"{sd:>4} | {cd:>9.3f} {ofc:>7.3f} {cr:>7.3f} | {uqc:>6.3f}",flush=True)
    print(f"MEAN | DPS {np.mean(dc):.3f}  RELAX {np.mean(rc):.3f}  UQ {np.mean(uq):.3f}")
