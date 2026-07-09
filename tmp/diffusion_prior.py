"""CREATIVE direction step 1: DIFFUSION PRIOR over wave fields (generative, not discriminative).
Train a compact U-Net DDPM on simulated reference-time wave-field snapshots (64x64, unit-std normalized).
Later: DPS posterior sampling with dispersion+obs likelihood -> beat RELAX in sparse + UQ. Prior needs NO obs labels."""
import os, sys, numpy as np, torch, torch.nn as nn, torch.nn.functional as F, time, argparse, math
sys.path.insert(0,'/home/vexit/wavesplat/wavesplat'); os.chdir('/home/vexit/wavesplat/wavesplat'); sys.path.insert(0,'scripts')
from d52_F4_neural_operator import DEVICE
from wavesim.wave_generator import jonswap_components
G=9.81; DEPTH=50.0; N=64
gx=np.linspace(10,250,N); gy=np.linspace(-100,100,N); Xg,Yg=np.meshgrid(gx,gy,indexing='ij')  # axis0=x,axis1=y
ap=argparse.ArgumentParser(); ap.add_argument('--steps',type=int,default=30000); ap.add_argument('--T',type=int,default=1000); a=ap.parse_args()
def sample_field():  # 2-channel (eta, eta_t) reference-time state; normalized by std(eta) (needed for propagation)
    Hs=np.random.uniform(0.6,2.5);Tp=np.random.uniform(5.0,10.0);NC=int(np.random.choice([3,4,5,6,8,10,12,16]))
    c=jonswap_components(Hs=Hs,Tp=Tp,n_components=NC,depth=DEPTH,seed=int(np.random.randint(1e9)))
    A=np.array([x.A for x in c]);om=np.array([x.omega for x in c]);kx=np.array([x.kx for x in c]);ky=np.array([x.ky for x in c]);ph=np.array([x.phi for x in c])
    t0=np.random.uniform(0,50); ang=kx[None]*Xg.ravel()[:,None]+ky[None]*Yg.ravel()[:,None]-om[None]*t0+ph[None]
    eta=(A[None]*np.cos(ang)).sum(1).reshape(N,N); eta_t=(A[None]*om[None]*np.sin(ang)).sum(1).reshape(N,N)
    s=eta.std()+1e-6; return np.stack([eta/s,eta_t/s]).astype(np.float32)  # [2,N,N]
def batch(B): return torch.tensor(np.stack([sample_field() for _ in range(B)]),device=DEVICE)  # [B,2,N,N]
# --- compact time-conditioned U-Net ---
def timeemb(t,d=128):
    half=d//2; freqs=torch.exp(-math.log(10000)*torch.arange(half,device=t.device)/half); ar=t[:,None].float()*freqs[None]
    return torch.cat([torch.sin(ar),torch.cos(ar)],-1)
class Blk(nn.Module):
    def __init__(s,ci,co,td=128):
        super().__init__(); s.c1=nn.Conv2d(ci,co,3,padding=1); s.c2=nn.Conv2d(co,co,3,padding=1)
        s.n1=nn.GroupNorm(8,co); s.n2=nn.GroupNorm(8,co); s.temb=nn.Linear(td,co); s.sk=nn.Conv2d(ci,co,1) if ci!=co else nn.Identity()
    def forward(s,x,te):
        h=F.silu(s.n1(s.c1(x))); h=h+s.temb(te)[:,:,None,None]; h=F.silu(s.n2(s.c2(h))); return h+s.sk(x)
class UNet(nn.Module):
    def __init__(s,ch=(48,96,192)):
        super().__init__(); s.tproj=nn.Sequential(nn.Linear(128,128),nn.SiLU(),nn.Linear(128,128))
        s.inc=nn.Conv2d(2,ch[0],3,padding=1)
        s.d1=Blk(ch[0],ch[0]); s.d2=Blk(ch[0],ch[1]); s.d3=Blk(ch[1],ch[2])
        s.mid=Blk(ch[2],ch[2])
        s.u3=Blk(ch[2]+ch[2],ch[1]); s.u2=Blk(ch[1]+ch[1],ch[0]); s.u1=Blk(ch[0]+ch[0],ch[0])
        s.outc=nn.Conv2d(ch[0],2,3,padding=1); s.pool=nn.AvgPool2d(2)
    def forward(s,x,t):
        te=s.tproj(timeemb(t))
        x0=s.inc(x); e1=s.d1(x0,te); e2=s.d2(s.pool(e1),te); e3=s.d3(s.pool(e2),te)
        m=s.mid(e3,te)
        u3=s.u3(torch.cat([m,e3],1),te); u3=F.interpolate(u3,scale_factor=2,mode='nearest')
        u2=s.u2(torch.cat([u3,e2],1),te); u2=F.interpolate(u2,scale_factor=2,mode='nearest')
        u1=s.u1(torch.cat([u2,e1],1),te); return s.outc(u1)
# --- DDPM schedule ---
T=a.T; betas=torch.linspace(1e-4,0.02,T,device=DEVICE); alphas=1-betas; abar=torch.cumprod(alphas,0)
M=UNet().to(DEVICE); opt=torch.optim.AdamW(M.parameters(),lr=2e-4,weight_decay=1e-5)
sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,a.steps); t0=time.time()
np_=sum(p.numel() for p in M.parameters()); print(f"[Diffusion prior, {N}x{N}, T={T}, params={np_/1e6:.2f}M, steps={a.steps}]")
for s in range(a.steps):
    x0=batch(32); tt=torch.randint(0,T,(x0.shape[0],),device=DEVICE); noise=torch.randn_like(x0)
    xt=torch.sqrt(abar[tt])[:,None,None,None]*x0+torch.sqrt(1-abar[tt])[:,None,None,None]*noise
    pred=M(xt,tt); loss=F.mse_loss(pred,noise)
    opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(M.parameters(),1.0); opt.step(); sch.step()
    if s%2000==0 or s==a.steps-1: print(f"  step {s:5d} noise-mse {loss.item():.4f} ({time.time()-t0:.0f}s)",flush=True)
torch.save({'model':M.state_dict(),'T':T,'betas':betas.cpu(),'N':N},'results/direction3/diffusion_prior.pt'); print("saved diffusion prior")
