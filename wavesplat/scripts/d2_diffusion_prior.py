"""
Direction II — Step 2b: train a DDPM diffusion prior over directional spectra S(kx,ky).

Compact U-Net + standard DDPM (cosine schedule). Learns p_theta(S) from the
8000 simulator-sampled spectra. The trained score is used in Step 3 for
physics-constrained posterior sampling p(S|z) ∝ p(z|S)·p_theta(S).
"""
import os, sys, math, time, argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
os.chdir(os.path.join(os.path.dirname(__file__), '..'))

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DATA = 'results/direction2/spectra_dataset.npz'
CKPT = 'results/direction2/diffusion_prior.pt'
T_STEPS = 500
EPS = 1e-4   # log-transform floor


# ── transform: S (nonneg, peaky) → x (~[-1,1]) ────────────────────────────────
def load_transform():
    d = np.load(DATA)
    S = d['S'].astype(np.float32)
    x = np.log(S + EPS)
    lo, hi = float(x.min()), float(np.percentile(x, 99.9))
    return S, x, lo, hi

def to_x(S, lo, hi):
    x = np.log(S + EPS)
    return np.clip(2*(x - lo)/(hi - lo) - 1, -1, 1)

def from_x(x, lo, hi):
    xx = (np.clip(x, -1, 1)+1)/2 * (hi - lo) + lo
    return np.maximum(np.exp(xx) - EPS, 0.0)


# ── U-Net ─────────────────────────────────────────────────────────────────────
def timestep_emb(t, dim):
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
    a = t[:, None].float() * freqs[None]
    return torch.cat([torch.sin(a), torch.cos(a)], dim=-1)

class ResBlock(nn.Module):
    def __init__(self, cin, cout, tdim):
        super().__init__()
        self.n1 = nn.GroupNorm(8, cin); self.c1 = nn.Conv2d(cin, cout, 3, padding=1)
        self.temb = nn.Linear(tdim, cout)
        self.n2 = nn.GroupNorm(8, cout); self.c2 = nn.Conv2d(cout, cout, 3, padding=1)
        self.skip = nn.Conv2d(cin, cout, 1) if cin != cout else nn.Identity()
    def forward(self, x, t):
        h = self.c1(F.silu(self.n1(x)))
        h = h + self.temb(t)[:, :, None, None]
        h = self.c2(F.silu(self.n2(h)))
        return h + self.skip(x)

class UNet(nn.Module):
    def __init__(self, base=64, tdim=128):
        super().__init__()
        self.tdim = tdim
        self.tmlp = nn.Sequential(nn.Linear(tdim, tdim), nn.SiLU(), nn.Linear(tdim, tdim))
        self.inc = nn.Conv2d(1, base, 3, padding=1)
        self.d1 = ResBlock(base, base, tdim)
        self.d2 = ResBlock(base, base*2, tdim)
        self.d3 = ResBlock(base*2, base*2, tdim)
        self.pool = nn.AvgPool2d(2)
        self.mid = ResBlock(base*2, base*2, tdim)
        self.u3 = ResBlock(base*4, base*2, tdim)
        self.u2 = ResBlock(base*4, base, tdim)
        self.u1 = ResBlock(base*2, base, tdim)
        self.outc = nn.Sequential(nn.GroupNorm(8, base), nn.SiLU(), nn.Conv2d(base, 1, 3, padding=1))
    def forward(self, x, t):
        te = self.tmlp(timestep_emb(t, self.tdim))
        x1 = self.inc(x)                       # 64
        x2 = self.d1(x1, te)                    # 64
        x3 = self.d2(self.pool(x2), te)         # 32
        x4 = self.d3(self.pool(x3), te)         # 16
        m = self.mid(x4, te)                    # 16
        u = self.u3(torch.cat([m, x4], 1), te)
        u = F.interpolate(u, scale_factor=2, mode='nearest')   # 32
        u = self.u2(torch.cat([u, x3], 1), te)
        u = F.interpolate(u, scale_factor=2, mode='nearest')   # 64
        u = self.u1(torch.cat([u, x2], 1), te)
        return self.outc(u)


# ── DDPM schedule (cosine) ────────────────────────────────────────────────────
def cosine_alpha_bar(T):
    s = 0.008
    t = torch.linspace(0, T, T+1)
    f = torch.cos((t/T + s)/(1+s) * math.pi/2)**2
    ab = f / f[0]
    return ab.clamp(1e-5, 0.9999)

AB = cosine_alpha_bar(T_STEPS).to(DEVICE)   # (T+1,)


def train(epochs=120, bs=64, lr=2e-4):
    S, _, lo, hi = load_transform()
    x = to_x(S, lo, hi).astype(np.float32)[:, None]   # (N,1,64,64)
    print(f'dataset {x.shape}, x-range[{x.min():.2f},{x.max():.2f}], transform lo={lo:.3f} hi={hi:.3f}')
    dl = DataLoader(TensorDataset(torch.from_numpy(x)), batch_size=bs, shuffle=True, drop_last=True)
    model = UNet().to(DEVICE)
    print('params %.2fM' % (sum(p.numel() for p in model.parameters())/1e6))
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    for ep in range(1, epochs+1):
        model.train(); tot=0; nb=0; t0=time.time()
        for (xb,) in dl:
            xb = xb.to(DEVICE)
            t = torch.randint(1, T_STEPS+1, (xb.size(0),), device=DEVICE)
            ab = AB[t][:, None, None, None]
            noise = torch.randn_like(xb)
            xt = ab.sqrt()*xb + (1-ab).sqrt()*noise
            pred = model(xt, t)
            loss = F.mse_loss(pred, noise)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item(); nb += 1
        sched.step()
        if ep % 10 == 0 or ep == 1:
            print(f'  ep {ep:3d}  loss {tot/nb:.4f}  {time.time()-t0:.1f}s')
    torch.save({'model': model.state_dict(), 'lo': lo, 'hi': hi, 'T': T_STEPS}, CKPT)
    print('saved', CKPT)


@torch.no_grad()
def sample(n=8):
    ck = torch.load(CKPT, map_location=DEVICE)
    model = UNet().to(DEVICE); model.load_state_dict(ck['model']); model.eval()
    x = torch.randn(n, 1, 64, 64, device=DEVICE)
    for t in range(T_STEPS, 0, -1):
        tt = torch.full((n,), t, device=DEVICE)
        ab = AB[t]; ab_prev = AB[t-1]
        eps = model(x, tt)
        x0 = (x - (1-ab).sqrt()*eps) / ab.sqrt()
        x0 = x0.clamp(-1, 1)
        if t > 1:
            beta = 1 - ab/ab_prev
            mean = ab_prev.sqrt()*beta/(1-ab)*x0 + (ab/ab_prev).sqrt()*(1-ab_prev)/(1-ab)*x
            x = mean + (beta*(1-ab_prev)/(1-ab)).sqrt()*torch.randn_like(x)
        else:
            x = x0
    S = from_x(x.squeeze(1).cpu().numpy(), ck['lo'], ck['hi'])
    return S


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', default='train', choices=['train', 'sample'])
    ap.add_argument('--epochs', type=int, default=120)
    a = ap.parse_args()
    if a.mode == 'train':
        train(epochs=a.epochs)
    else:
        S = sample(12)
        np.save('results/direction2/prior_samples.npy', S)
        print('prior samples saved', S.shape)
