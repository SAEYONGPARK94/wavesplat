"""Retrain set-diffusion prior BIGGER + LONGER (d=256, layers=8, 600 epochs) for the 'train sufficiently' review."""
import os, sys, time, numpy as np, torch
sys.path.insert(0,'/home/vexit/wavesplat/wavesplat'); os.chdir('/home/vexit/wavesplat/wavesplat'); sys.path.insert(0,'scripts')
from d3_setdiff_prior import SetDenoiser, cosine_betas, T, M, OUT
import json
DEVICE=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
d=np.load(f'{OUT}/atoms_dataset.npz'); Z=d['Z']; stats=json.loads(str(d['stats']))
print(f'dataset {Z.shape}')
X=torch.tensor(Z,device=DEVICE)
betas=cosine_betas(T).to(DEVICE); abar=torch.cumprod(1-betas,0)
ARCH=dict(d=256,heads=8,layers=8)
model=SetDenoiser(**ARCH).to(DEVICE)
opt=torch.optim.AdamW(model.parameters(),lr=2e-4,weight_decay=1e-4)
EPOCHS=600; sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,EPOCHS)
n=X.shape[0]; nparams=sum(p.numel() for p in model.parameters()); bs=256
print(f'model {nparams/1e6:.2f}M params (d256 l8), {EPOCHS} epochs'); t0=time.time()
for ep in range(EPOCHS):
    perm=torch.randperm(n,device=DEVICE); tot=0.0
    for i in range(0,n,bs):
        xb=X[perm[i:i+bs]]; tt=torch.randint(0,T,(xb.shape[0],),device=DEVICE)
        ab=abar[tt][:,None,None]; eps=torch.randn_like(xb)
        xt=ab.sqrt()*xb+(1-ab).sqrt()*eps
        loss=((model(xt,tt)-eps)**2).mean()
        opt.zero_grad(); loss.backward(); opt.step(); tot+=loss.item()*xb.shape[0]
    sched.step()
    if (ep+1)%50==0 or ep==0: print(f'  ep {ep+1} loss {tot/n:.4f} ({time.time()-t0:.0f}s)',flush=True)
torch.save(dict(model=model.state_dict(),stats=stats,betas=betas.cpu(),abar=abar.cpu(),arch=ARCH),f'{OUT}/setdiff_prior_long.pt')
print(f'saved setdiff_prior_long.pt ({time.time()-t0:.0f}s)')
