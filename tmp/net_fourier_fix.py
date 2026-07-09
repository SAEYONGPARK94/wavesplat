"""Hypothesis: raw linear coord embedding -> spectral bias -> can't represent oscillatory waves -> mean collapse.
Fix: Fourier features on (x,y,t). Test overfit + easy(3) + full(64)."""
import os, sys, numpy as np, torch, torch.nn as nn, time
sys.path.insert(0, '/home/vexit/wavesplat/wavesplat'); os.chdir('/home/vexit/wavesplat/wavesplat'); sys.path.insert(0, 'scripts')
from d52_F4_neural_operator import DEVICE
from wavesim.wave_generator import jonswap_components
from d40_learned_acquisition import pool_xy
DEPTH=50.0; K_TP=2.2
gx=np.linspace(10,250,80); gy=np.linspace(-100,100,60); Xg,Yg=np.meshgrid(gx,gy); grid=np.column_stack([Xg.ravel(),Yg.ravel()])
XS,YS,TS=120.0,100.0,10.0

class FF(nn.Module):
    """Fourier features on normalized coords: wave k up to ~0.6 rad/m -> need freqs spanning that."""
    def __init__(self, n=16):
        super().__init__()
        # frequencies in physical rad/m,rad/s style: sample log-spaced spatial + temporal
        self.register_buffer('B', torch.randn(3, n)*torch.tensor([0.3,0.3,1.0])[:,None])  # scale per (x,y,t) in physical units
    def forward(self, xyt):   # xyt physical (…,3)
        proj = xyt @ self.B     # (...,n)
        return torch.cat([torch.sin(2*np.pi*proj), torch.cos(2*np.pi*proj)], -1)  # (...,2n)

class OpFF(nn.Module):
    def __init__(self, d=160, heads=4, enc=4, dec=3, nff=32):
        super().__init__()
        self.ff=FF(nff); fdim=2*nff
        self.oenc=nn.Sequential(nn.Linear(fdim+1,d),nn.GELU(),nn.Linear(d,d))   # +1 for z
        el=nn.TransformerEncoderLayer(d,heads,4*d,batch_first=True,activation='gelu',norm_first=True)
        self.enc=nn.TransformerEncoder(el,enc)
        self.qenc=nn.Sequential(nn.Linear(fdim,d),nn.GELU(),nn.Linear(d,d))
        dl=nn.TransformerDecoderLayer(d,heads,4*d,batch_first=True,activation='gelu',norm_first=True)
        self.dec=nn.TransformerDecoder(dl,dec)
        self.head=nn.Sequential(nn.Linear(d,d),nn.GELU(),nn.Linear(d,1))
    def forward(self, obs, q):
        of=self.ff(obs[...,:3]); oin=torch.cat([of,obs[...,3:4]],-1)
        ctx=self.enc(self.oenc(oin))
        h=self.dec(self.qenc(self.ff(q)),ctx)
        return self.head(h)[...,0]

def gen(B, ncomp=64, fc=6.0):
    pool=pool_xy(); obs=np.zeros((B,200,4),np.float32); qpt=np.zeros((B,400,3),np.float32); qz=np.zeros((B,400),np.float32); Hsb=np.zeros(B,np.float32)
    for b in range(B):
        Hs=np.random.uniform(0.6,2.0); Tp=np.random.uniform(5.5,9.5)
        c=jonswap_components(Hs=Hs,Tp=Tp,n_components=ncomp,depth=DEPTH,seed=int(np.random.randint(1e9)))
        A=np.array([x.A for x in c]);om=np.array([x.omega for x in c]);kx=np.array([x.kx for x in c]);ky=np.array([x.ky for x in c]);ph=np.array([x.phi for x in c])
        Tobs=K_TP*Tp; oi=np.random.randint(0,len(pool),200); ox,oy=pool[oi,0],pool[oi,1]; ot=np.random.uniform(0,Tobs,200)
        oz=(A[None]*np.cos(kx[None]*ox[:,None]+ky[None]*oy[:,None]-om[None]*ot[:,None]+ph[None])).sum(1)+np.random.normal(0,0.02,200)
        obs[b]=np.column_stack([ox,oy,ot,oz]); qx=np.random.uniform(10,250,400);qy=np.random.uniform(-100,100,400);qt=np.random.uniform(0.5*Tobs,Tobs+fc,400)
        qz[b]=(A[None]*np.cos(kx[None]*qx[:,None]+ky[None]*qy[:,None]-om[None]*qt[:,None]+ph[None])).sum(1); qpt[b]=np.column_stack([qx,qy,qt]); Hsb[b]=Hs
    t=lambda a: torch.tensor(a,device=DEVICE); return t(obs),t(qpt),t(qz),t(Hsb)

print("=== (a) overfit ONE batch WITH Fourier features ===")
M=OpFF().to(DEVICE); opt=torch.optim.AdamW(M.parameters(),lr=5e-4)
ob,q,qz,Hs=gen(8,64)
for s in range(2500):
    pred=M(ob,q); loss=((pred-qz)**2).mean(); opt.zero_grad(); loss.backward(); opt.step()
    if s%500==0 or s==2499: print(f"  step {s} train-rmse {loss.sqrt().item():.4f} (signal std {qz.std().item():.3f})",flush=True)
