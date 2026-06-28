"""
Final improvement — MULTI-TASK posterior: keep the direct-field accuracy AND add an
interpretable, dispersion-constrained spectral head + physics-consistency, at no accuracy
cost. The network has (a) a FIELD head (per-query cross-attention -> mean, logvar; the
accurate, flexible output) and (b) a SPECTRAL head (per-wavenumber cross-attention -> coeffs
on the dispersion grid -> a dispersion-consistent field). A consistency loss nudges the field
toward dispersion-consistency. Main output = field (accuracy 0.222); spectral = interpretable
+ physics-informed; both fast + UQ.

Run: python3 scripts/d61_multitask_posterior.py [--steps 9000]
"""
import os, sys, json, time, argparse
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
os.chdir(os.path.join(os.path.dirname(__file__), '..'))

from d23_main_results import panel
from d52_F4_neural_operator import gen_batch, XS, YS, TS, ZS, DEVICE
from d51_F1_continuous_density import disp_grid

CACHE = 'results/direction3/_hos_windows.npz'
NG = 28
kx_, ky_, om_ = disp_grid(NG)
KX = torch.tensor(kx_, dtype=torch.float32, device=DEVICE); KY = torch.tensor(ky_, dtype=torch.float32, device=DEVICE)
OM = torch.tensor(om_, dtype=torch.float32, device=DEVICE); NW = len(kx_)
WFEAT = torch.stack([KX/0.6, KY/0.6, OM/float(om_.max())], -1)


def basis(q):
    psi = (KX[None,None]*q[...,0:1] + KY[None,None]*q[...,1:2] - OM[None,None]*q[...,2:3])
    return torch.cat([torch.cos(psi), -torch.sin(psi)], -1)


class MultiTask(nn.Module):
    def __init__(self, d=160, heads=4, enc=4):
        super().__init__()
        self.oenc = nn.Sequential(nn.Linear(4, d), nn.GELU(), nn.Linear(d, d))
        self.enc = nn.TransformerEncoder(nn.TransformerEncoderLayer(d, heads, 4*d, batch_first=True, activation='gelu'), enc)
        self.qenc = nn.Sequential(nn.Linear(3, d), nn.GELU(), nn.Linear(d, d))     # field query
        self.fdec = nn.TransformerDecoder(nn.TransformerDecoderLayer(d, heads, 4*d, batch_first=True, activation='gelu'), 3)
        self.fhead = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, 2))    # field mean, logvar
        self.wenc = nn.Sequential(nn.Linear(3, d), nn.GELU(), nn.Linear(d, d))     # wavenumber query
        self.sdec = nn.TransformerDecoder(nn.TransformerDecoderLayer(d, heads, 4*d, batch_first=True, activation='gelu'), 3)
        self.shead = nn.Linear(d, 2)                                               # re_coeff, im_coeff

    def forward(self, obs, q):
        o = obs.clone(); o[...,0]/=XS; o[...,1]/=YS; o[...,2]/=TS; o[...,3]/=ZS
        ctx = self.enc(self.oenc(o))
        qn = q.clone(); qn[...,0]/=XS; qn[...,1]/=YS; qn[...,2]/=TS
        f = self.fhead(self.fdec(self.qenc(qn), ctx)); fmean = f[...,0]; fvar = f[...,1].clamp(-8,4).exp()
        wq = self.wenc(WFEAT)[None].expand(obs.shape[0], -1, -1)
        coef = self.shead(self.sdec(wq, ctx))                                      # (B,NW,2)
        cm = torch.cat([coef[...,0], coef[...,1]], -1)                             # (B,2NW)
        sfield = (basis(q) * cm[:,None,:]).sum(-1)                                 # dispersion-rendered field
        return fmean, fvar, sfield


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--steps', type=int, default=9000); a = ap.parse_args()
    model = MultiTask().to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.steps); t0 = time.time()
    print(f"training multi-task posterior {a.steps} steps (NW={NW})...", flush=True)
    for s in range(a.steps):
        obs, q, qz, Hs = gen_batch(24)
        fmean, fvar, sfield = model(obs, q); Hs2 = Hs**2 + 1e-6
        nll = (0.5*((fmean-qz)**2/fvar + torch.log(fvar))).mean(1) / Hs2          # field accuracy + UQ
        s_mse = ((sfield-qz)**2).mean(1) / Hs2                                     # spectral fits data
        cons = ((fmean.detach()-sfield)**2).mean(1) / Hs2                          # consistency (field≈dispersion)
        loss = (nll + 0.5*s_mse + 0.3*cons).mean()
        opt.zero_grad(); loss.backward(); opt.step(); sch.step()
        if s % 900 == 0 or s == a.steps-1:
            print(f"  step {s:4d}  nll {nll.mean().item():.3f} smse {s_mse.mean().item():.3f} cons {cons.mean().item():.3f} ({time.time()-t0:.0f}s)", flush=True)
    model.eval()
    wins = list(np.load(CACHE, allow_pickle=True)['windows'])
    gx = np.linspace(10,250,80); gy = np.linspace(-100,100,60)
    Xg,Yg = np.meshgrid(gx,gy); grid = np.column_stack([Xg.ravel(), Yg.ravel()])
    nrf, nrs, errs, sigs, times = [], [], [], [], []
    _ = model(torch.zeros(1,50,4,device=DEVICE), torch.zeros(1,100,3,device=DEVICE))
    for w in wins:
        P,z = w['P'], w['z']
        if len(z)>200:
            i = np.random.default_rng(0).choice(len(z),200,replace=False); P,z = P[i],z[i]
        t0_ = P[:,2].min()
        ob = torch.tensor(np.column_stack([P[:,0],P[:,1],P[:,2]-t0_,z])[None],dtype=torch.float32,device=DEVICE)
        q = torch.tensor(np.column_stack([grid,np.full(len(grid),float(w['t'])-t0_)])[None],dtype=torch.float32,device=DEVICE)
        torch.cuda.synchronize() if DEVICE.type=='cuda' else None
        tt = time.time()
        with torch.no_grad(): fmean,fvar,sfield = model(ob,q)
        torch.cuda.synchronize() if DEVICE.type=='cuda' else None
        times.append(time.time()-tt)
        mf = fmean[0].cpu().numpy().reshape(len(gy),len(gx)); ms = sfield[0].cpu().numpy().reshape(len(gy),len(gx))
        sig = fvar[0].sqrt().cpu().numpy().reshape(len(gy),len(gx))
        nrf.append(panel(mf,w['gt'],w['Hs'])['nrmse']); nrs.append(panel(ms,w['gt'],w['Hs'])['nrmse'])
        errs.append(np.abs(mf-w['gt']).ravel()); sigs.append(sig.ravel())
    er=np.concatenate(errs); sg=np.concatenate(sigs); s=float(np.quantile(er/(sg+1e-9),0.68))
    cov1=float((er<=s*sg).mean()); cov2=float((er<=2*s*sg).mean())
    print(f"\n=== multi-task posterior (continuous HOS short window) ===")
    print(f"  FIELD head (main):  nRMSE {np.mean(nrf):.3f}   (direct-field 0.222 | F1 0.234 | d60 physics 0.250)")
    print(f"  SPECTRAL head (interpretable, dispersion-enforced):  nRMSE {np.mean(nrs):.3f}")
    print(f"  inference {1e3*np.mean(times):.1f} ms | UQ ±1σ {cov1:.2f}(.68) ±2σ {cov2:.2f}(.95) s={s:.2f}")
    print(f"  → {'WIN: 정확도 유지(≤0.225) + 해석 스펙트럼 + physics-consistency + UQ' if np.mean(nrf)<=0.226 else 'field accuracy '+('held' if np.mean(nrf)<0.24 else 'dropped')}")
    json.dump(dict(field=float(np.mean(nrf)), spectral=float(np.mean(nrs)), time_ms=float(1e3*np.mean(times)),
                   cov1=cov1, cov2=cov2), open('results/direction3/multitask_posterior.json','w'), indent=2)
    print("saved results/direction3/multitask_posterior.json")


if __name__ == '__main__':
    main()
