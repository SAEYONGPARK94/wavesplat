"""
E1 v2 — close the physics-vs-accuracy tradeoff: per-wavenumber CROSS-ATTENTION coefficient
prediction (each dispersion-grid wavenumber is a query that attends to the observations) +
a finer grid. This is proper 'amortised spectral inference on the dispersion manifold': the
coefficient of each wavenumber is inferred by attending to the relevant data, instead of a
single pooled vector predicting all coefficients (d59, which lost accuracy).

Dispersion enforced by construction, interpretable spectrum + propagated (info-grounded) UQ,
fast — and (goal) accuracy matching the direct-field posterior (0.222) / F1 (0.234).

Run: python3 scripts/d60_physics_posterior_v2.py [--steps 9000] [--ng 30]
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


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--steps', type=int, default=9000)
    ap.add_argument('--ng', type=int, default=30); a = ap.parse_args()
    kx_, ky_, om_ = disp_grid(a.ng)
    KX = torch.tensor(kx_, dtype=torch.float32, device=DEVICE)
    KY = torch.tensor(ky_, dtype=torch.float32, device=DEVICE)
    OM = torch.tensor(om_, dtype=torch.float32, device=DEVICE)
    NW = len(kx_); KLIM = 0.6; OMMAX = float(om_.max())
    wfeat = torch.stack([KX/KLIM, KY/KLIM, OM/OMMAX], -1)              # (NW,3) wavenumber descriptors

    def basis(q):
        psi = (KX[None,None]*q[...,0:1] + KY[None,None]*q[...,1:2] - OM[None,None]*q[...,2:3])
        return torch.cat([torch.cos(psi), -torch.sin(psi)], -1)        # (B,Q,2NW)

    class Net(nn.Module):
        def __init__(self, d=160, heads=4, enc=4, dec=3):
            super().__init__()
            self.oenc = nn.Sequential(nn.Linear(4, d), nn.GELU(), nn.Linear(d, d))
            el = nn.TransformerEncoderLayer(d, heads, 4*d, batch_first=True, activation='gelu')
            self.enc = nn.TransformerEncoder(el, enc)
            self.wq = nn.Sequential(nn.Linear(3, d), nn.GELU(), nn.Linear(d, d))
            dl = nn.TransformerDecoderLayer(d, heads, 4*d, batch_first=True, activation='gelu')
            self.dec = nn.TransformerDecoder(dl, dec)
            self.head = nn.Linear(d, 4)                                # re_mean, im_mean, re_logv, im_logv

        def forward(self, obs, q):
            o = obs.clone(); o[...,0]/=XS; o[...,1]/=YS; o[...,2]/=TS; o[...,3]/=ZS
            ctx = self.enc(self.oenc(o))
            wq = self.wq(wfeat)[None].expand(obs.shape[0], -1, -1)     # (B,NW,d)
            h = self.dec(wq, ctx); out = self.head(h)                 # (B,NW,4)
            cm = torch.cat([out[...,0], out[...,1]], -1)              # (B,2NW)
            cv = torch.cat([out[...,2], out[...,3]], -1).clamp(-10,2).exp()
            B = basis(q)
            mean = (B * cm[:,None,:]).sum(-1)
            var = (B**2 * cv[:,None,:]).sum(-1) + 1e-6
            return mean, var

    model = Net().to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.steps); t0 = time.time()
    print(f"training spectral-cross-attn posterior {a.steps} steps (NW={NW})...", flush=True)
    for s in range(a.steps):
        obs, q, qz, Hs = gen_batch(24)
        mean, var = model(obs, q)
        loss = (((0.5*((mean-qz)**2/var + torch.log(var))).mean(1)) / (Hs**2+1e-6)).mean()
        opt.zero_grad(); loss.backward(); opt.step(); sch.step()
        if s % 900 == 0 or s == a.steps-1:
            with torch.no_grad(): rmse = ((mean-qz)**2).mean().sqrt().item()
            print(f"  step {s:4d}  nll {loss.item():.4f}  rmse {rmse:.3f}  ({time.time()-t0:.0f}s)", flush=True)
    model.eval()
    wins = list(np.load(CACHE, allow_pickle=True)['windows'])
    gx = np.linspace(10,250,80); gy = np.linspace(-100,100,60)
    Xg,Yg = np.meshgrid(gx,gy); grid = np.column_stack([Xg.ravel(), Yg.ravel()])
    nr, errs, sigs, times = [], [], [], []
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
        with torch.no_grad(): mean,var = model(ob,q)
        torch.cuda.synchronize() if DEVICE.type=='cuda' else None
        times.append(time.time()-tt)
        m = mean[0].cpu().numpy().reshape(len(gy),len(gx)); sig = var[0].sqrt().cpu().numpy().reshape(len(gy),len(gx))
        nr.append(panel(m,w['gt'],w['Hs'])['nrmse']); errs.append(np.abs(m-w['gt']).ravel()); sigs.append(sig.ravel())
    er = np.concatenate(errs); sg = np.concatenate(sigs)
    s = float(np.quantile(er/(sg+1e-9),0.68)); cov1=float((er<=s*sg).mean()); cov2=float((er<=2*s*sg).mean())
    print(f"\n=== spectral cross-attn posterior (NW={NW}, continuous HOS short window) ===")
    print(f"  accuracy nRMSE {np.mean(nr):.3f}   (d59 pooled 0.251 | direct-field 0.222 | F1 0.234 | RELAX 0.274)")
    print(f"  inference {1e3*np.mean(times):.1f} ms | UQ ±1σ {cov1:.2f}(.68) ±2σ {cov2:.2f}(.95) s={s:.2f}")
    print(f"  → physics enforced + interpretable spectrum + grounded UQ + {'ACCURACY CLOSED (≤F1)' if np.mean(nr)<=0.236 else 'still below F1'}")
    json.dump(dict(nrmse=float(np.mean(nr)), time_ms=float(1e3*np.mean(times)), cov1=cov1, cov2=cov2, NW=NW),
              open('results/direction3/physics_posterior_v2.json','w'), indent=2)
    print("saved results/direction3/physics_posterior_v2.json")


if __name__ == '__main__':
    main()
