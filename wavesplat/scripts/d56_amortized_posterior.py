"""
Deployment-ideal — a FAST amortised neural posterior with calibrated UQ.

The whole investigation converges here: real-time control needs short window + ms inference
+ UQ, a package no existing method gives (F1 fast+accurate but no UQ; DPS accurate+UQ but
9.8 s; RELAX slow + long-window). This is a conditional neural field: obs -> per-query
(mean, sigma) in ONE forward pass, trained by Gaussian NLL (mean = accuracy ~ info-limit;
sigma = calibrated uncertainty). Evaluated on cached HOS windows for accuracy, UQ calibration
(coverage), and inference time.

Run: python3 scripts/d56_amortized_posterior.py [--steps 8000]
"""
import os, sys, json, time, argparse
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
os.chdir(os.path.join(os.path.dirname(__file__), '..'))

from d23_main_results import panel
import d52_F4_neural_operator as F4
from d52_F4_neural_operator import gen_batch, XS, YS, TS, ZS, DEVICE

CACHE = 'results/direction3/_hos_windows.npz'


class PosteriorNet(nn.Module):
    def __init__(self, d=160, heads=4, enc=4, dec=3):
        super().__init__()
        self.oenc = nn.Sequential(nn.Linear(4, d), nn.GELU(), nn.Linear(d, d))
        el = nn.TransformerEncoderLayer(d, heads, 4*d, batch_first=True, activation='gelu')
        self.enc = nn.TransformerEncoder(el, enc)
        self.qenc = nn.Sequential(nn.Linear(3, d), nn.GELU(), nn.Linear(d, d))
        dl = nn.TransformerDecoderLayer(d, heads, 4*d, batch_first=True, activation='gelu')
        self.dec = nn.TransformerDecoder(dl, dec)
        self.head = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, 2))  # mean, log_var

    def forward(self, obs, q):
        o = obs.clone(); o[..., 0] /= XS; o[..., 1] /= YS; o[..., 2] /= TS; o[..., 3] /= ZS
        ctx = self.enc(self.oenc(o))
        qn = q.clone(); qn[..., 0] /= XS; qn[..., 1] /= YS; qn[..., 2] /= TS
        h = self.dec(self.qenc(qn), ctx); out = self.head(h)
        return out[..., 0], out[..., 1]                      # mean, log_var


def train(steps, B=24, lr=3e-4):
    model = PosteriorNet().to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps); t0 = time.time()
    for s in range(steps):
        obs, q, qz, Hs = gen_batch(B)
        mean, logv = model(obs, q)
        logv = logv.clamp(-8, 4)
        # Gaussian NLL (Hs-normalised): accuracy (mean) + calibration (var)
        nll = 0.5 * ((mean - qz)**2 / logv.exp() + logv)
        loss = (nll.mean(1) / (Hs**2 + 1e-6)).mean()
        opt.zero_grad(); loss.backward(); opt.step(); sch.step()
        if s % 800 == 0 or s == steps-1:
            with torch.no_grad():
                rmse = ((mean - qz)**2).mean().sqrt().item()
            print(f"  step {s:4d}  nll-loss {loss.item():.4f}  rmse {rmse:.3f}  ({time.time()-t0:.0f}s)", flush=True)
    return model


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--steps', type=int, default=8000); a = ap.parse_args()
    print(f"training amortised posterior {a.steps} steps...", flush=True)
    model = train(a.steps); model.eval()
    wins = list(np.load(CACHE, allow_pickle=True)['windows'])
    gx = np.linspace(10, 250, 80); gy = np.linspace(-100, 100, 60)
    Xg, Yg = np.meshgrid(gx, gy); grid = np.column_stack([Xg.ravel(), Yg.ravel()])
    nr, errs, sigs, times = [], [], [], []
    # warmup
    _ = model(torch.zeros(1, 50, 4, device=DEVICE), torch.zeros(1, 100, 3, device=DEVICE))
    for w in wins:
        P, z = w['P'], w['z']
        if len(z) > 200:
            i = np.random.default_rng(0).choice(len(z), 200, replace=False); P, z = P[i], z[i]
        ob = torch.tensor(np.column_stack([P[:,0], P[:,1], P[:,2]-P[:,2].min(), z])[None], dtype=torch.float32, device=DEVICE)
        q = torch.tensor(np.column_stack([grid, np.full(len(grid), float(w['t'])-P[:,2].min())])[None], dtype=torch.float32, device=DEVICE)
        torch.cuda.synchronize() if DEVICE.type == 'cuda' else None
        t0 = time.time()
        with torch.no_grad():
            mean, logv = model(ob, q)
        torch.cuda.synchronize() if DEVICE.type == 'cuda' else None
        times.append(time.time()-t0)
        m = mean[0].cpu().numpy().reshape(len(gy), len(gx)); sig = (0.5*logv[0]).exp().cpu().numpy().reshape(len(gy), len(gx))
        nr.append(panel(m, w['gt'], w['Hs'])['nrmse'])
        errs.append(np.abs(m - w['gt']).ravel()); sigs.append(sig.ravel())
    er = np.concatenate(errs); sg = np.concatenate(sigs)
    s_cal = float(np.quantile(er/(sg+1e-9), 0.68))           # 1-param temperature recalibration
    cov1 = float((er <= s_cal*sg).mean()); cov2 = float((er <= 2*s_cal*sg).mean())
    corr = float(np.corrcoef(np.log(sg+1e-9), np.log(er+1e-9))[0,1])
    print(f"\n=== amortised posterior on continuous HOS (short window) ===")
    print(f"  accuracy nRMSE {np.mean(nr):.3f}   (F1 0.235 | RELAX 0.279 | DPS 0.249)")
    print(f"  inference {1e3*np.mean(times):.1f} ms/window   (F1 404ms | RELAX 1054ms | DPS 9782ms)")
    print(f"  UQ: ±1σ coverage {cov1:.2f} (ideal .68) ±2σ {cov2:.2f} (ideal .95) after temp s={s_cal:.2f}; "
          f"corr(logσ,log|err|)={corr:.2f}")
    json.dump(dict(nrmse=float(np.mean(nr)), time_ms=float(1e3*np.mean(times)), cov1=cov1, cov2=cov2,
                   s_cal=s_cal, sig_err_corr=corr), open('results/direction3/amortized_posterior.json', 'w'), indent=2)
    print("saved results/direction3/amortized_posterior.json")


if __name__ == '__main__':
    main()
