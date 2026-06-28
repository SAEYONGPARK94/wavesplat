"""
E1 — physics-constrained amortised SPECTRAL posterior (ML-novelty boost).

Ties the dispersion physics into the learned posterior: the network outputs spectral
coefficients (real/imag amplitudes) AND their log-variances on a FIXED dispersion-constrained
wavenumber grid; the field MEAN and VARIANCE are rendered via the physics (a linear map). So
dispersion is enforced by construction, the output is an interpretable spectrum, and the UQ is
the PROPAGATED spectral uncertainty (information-grounded — it knows which wavenumbers are
under-determined). 'Amortised inference on the dispersion manifold.' One forward pass.

Compare accuracy / UQ / speed to d56 (direct-field posterior 0.222) and F1 (0.234) on the
cached HOS windows.

Run: python3 scripts/d59_physics_posterior.py [--steps 8000]
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
NG = 22
_kx, _ky, _om = disp_grid(NG)                              # fixed dispersion-constrained grid
KX = torch.tensor(_kx, dtype=torch.float32, device=DEVICE)
KY = torch.tensor(_ky, dtype=torch.float32, device=DEVICE)
OM = torch.tensor(_om, dtype=torch.float32, device=DEVICE)
NW = len(_kx)


def basis(q):
    """(B,Q,2*NW): [cos(psi), -sin(psi)] per grid wavenumber. q=(B,Q,3)[x,y,t]."""
    psi = (KX[None, None] * q[..., 0:1] + KY[None, None] * q[..., 1:2] - OM[None, None] * q[..., 2:3])
    return torch.cat([torch.cos(psi), -torch.sin(psi)], dim=-1)   # (B,Q,2NW)


class SpectralPosterior(nn.Module):
    def __init__(self, d=160, heads=4, enc=4):
        super().__init__()
        self.oenc = nn.Sequential(nn.Linear(4, d), nn.GELU(), nn.Linear(d, d))
        el = nn.TransformerEncoderLayer(d, heads, 4*d, batch_first=True, activation='gelu')
        self.enc = nn.TransformerEncoder(el, enc)
        self.pool = nn.Parameter(torch.randn(1, 1, d) * 0.02)
        dl = nn.TransformerDecoderLayer(d, heads, 4*d, batch_first=True, activation='gelu')
        self.dec = nn.TransformerDecoder(dl, 2)
        self.cmean = nn.Linear(d, 2*NW)                   # spectral coeff means (re,im per wavenumber)
        self.clogv = nn.Linear(d, 2*NW)                   # spectral coeff log-variances

    def forward(self, obs, q):
        o = obs.clone(); o[..., 0] /= XS; o[..., 1] /= YS; o[..., 2] /= TS; o[..., 3] /= ZS
        ctx = self.enc(self.oenc(o))
        h = self.dec(self.pool.expand(obs.shape[0], -1, -1), ctx)[:, 0]   # (B,d) pooled
        cm = self.cmean(h); cv = self.clogv(h).clamp(-10, 2).exp()        # (B,2NW)
        B = basis(q)                                                      # (B,Q,2NW)
        mean = (B * cm[:, None, :]).sum(-1)                               # physics render of mean
        var = (B**2 * cv[:, None, :]).sum(-1) + 1e-6                      # propagated spectral variance
        return mean, var


def train(steps, Bsz=24, lr=3e-4):
    model = SpectralPosterior().to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps); t0 = time.time()
    for s in range(steps):
        obs, q, qz, Hs = gen_batch(Bsz)
        mean, var = model(obs, q)
        nll = 0.5 * ((mean - qz)**2 / var + torch.log(var))
        loss = (nll.mean(1) / (Hs**2 + 1e-6)).mean()
        opt.zero_grad(); loss.backward(); opt.step(); sch.step()
        if s % 800 == 0 or s == steps-1:
            with torch.no_grad():
                rmse = ((mean - qz)**2).mean().sqrt().item()
            print(f"  step {s:4d}  nll {loss.item():.4f}  rmse {rmse:.3f}  ({time.time()-t0:.0f}s)", flush=True)
    return model


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--steps', type=int, default=8000); a = ap.parse_args()
    print(f"training physics-constrained spectral posterior {a.steps} steps (NW={NW} dispersion wavenumbers)...", flush=True)
    model = train(a.steps); model.eval()
    wins = list(np.load(CACHE, allow_pickle=True)['windows'])
    gx = np.linspace(10, 250, 80); gy = np.linspace(-100, 100, 60)
    Xg, Yg = np.meshgrid(gx, gy); grid = np.column_stack([Xg.ravel(), Yg.ravel()])
    nr, errs, sigs, times = [], [], [], []
    _ = model(torch.zeros(1, 50, 4, device=DEVICE), torch.zeros(1, 100, 3, device=DEVICE))
    for w in wins:
        P, z = w['P'], w['z']
        if len(z) > 200:
            i = np.random.default_rng(0).choice(len(z), 200, replace=False); P, z = P[i], z[i]
        t0_ = P[:, 2].min()
        ob = torch.tensor(np.column_stack([P[:,0], P[:,1], P[:,2]-t0_, z])[None], dtype=torch.float32, device=DEVICE)
        q = torch.tensor(np.column_stack([grid, np.full(len(grid), float(w['t'])-t0_)])[None], dtype=torch.float32, device=DEVICE)
        torch.cuda.synchronize() if DEVICE.type == 'cuda' else None
        t0 = time.time()
        with torch.no_grad():
            mean, var = model(ob, q)
        torch.cuda.synchronize() if DEVICE.type == 'cuda' else None
        times.append(time.time()-t0)
        m = mean[0].cpu().numpy().reshape(len(gy), len(gx)); sig = var[0].sqrt().cpu().numpy().reshape(len(gy), len(gx))
        nr.append(panel(m, w['gt'], w['Hs'])['nrmse']); errs.append(np.abs(m-w['gt']).ravel()); sigs.append(sig.ravel())
    er = np.concatenate(errs); sg = np.concatenate(sigs)
    s = float(np.quantile(er/(sg+1e-9), 0.68)); cov1 = float((er <= s*sg).mean()); cov2 = float((er <= 2*s*sg).mean())
    print(f"\n=== physics-constrained spectral posterior (continuous HOS, short window) ===")
    print(f"  accuracy nRMSE {np.mean(nr):.3f}   (direct-field posterior 0.222 | F1 0.234 | RELAX 0.274)")
    print(f"  inference {1e3*np.mean(times):.1f} ms/window")
    print(f"  UQ: ±1σ {cov1:.2f} (.68) ±2σ {cov2:.2f} (.95) after s={s:.2f}; dispersion ENFORCED + interpretable spectrum")
    json.dump(dict(nrmse=float(np.mean(nr)), time_ms=float(1e3*np.mean(times)), cov1=cov1, cov2=cov2, NW=NW),
              open('results/direction3/physics_posterior.json', 'w'), indent=2)
    print("saved results/direction3/physics_posterior.json")


if __name__ == '__main__':
    main()
