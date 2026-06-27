"""
Direction III — grid-free spectral prior: permutation-invariant SET diffusion.

A spectrum is M=32 continuous atoms (kx, ky, log-energy). We learn a DDPM
ε-prediction prior over such atom sets with a small permutation-equivariant set
Transformer (self-attention is permutation-equivariant). Training data are atom
sets from the SAME generator the episodes are drawn from
(wavesim.wave_generator.jonswap_components) — no grid, no train/test mismatch.

Run:  python3 scripts/d3_setdiff_prior.py            # generate data + train + sanity
"""
import os, sys, math, json, time
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
os.chdir(os.path.join(os.path.dirname(__file__), '..'))

from wavesim.wave_generator import jonswap_components

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
OUT = 'results/direction3'
os.makedirs(OUT, exist_ok=True)
M = 32                       # atoms per spectrum
K_LIM = 1.1                  # k-space half-extent (matches Dir-II grid)
N_SAMPLES = 16000
T = 500                      # diffusion steps


# ── data ────────────────────────────────────────────────────────────────────
def gen_dataset(seed=0):
    """(N, M, 3) atoms [kx, ky, logE]. Always M real atoms (no padding) — each
    spectrum is M continuous samples of the JONSWAP directional density, so the
    prior has a unimodal energy channel and no origin-cluster artifact."""
    rng = np.random.default_rng(seed)
    A = np.zeros((N_SAMPLES, M, 3), np.float32)
    for i in range(N_SAMPLES):
        Hs = rng.uniform(0.8, 3.2); Tp = rng.uniform(5.0, 11.0)
        dird = rng.uniform(-50, 50)
        spread = rng.uniform(12, 40); gamma = rng.uniform(1.5, 4.0)
        comps = jonswap_components(Hs, Tp, n_components=M, depth=50.0, gamma=gamma,
                                   direction_deg=dird, direction_spread_deg=spread,
                                   seed=int(rng.integers(1e9)))
        kx = np.array([c.kx for c in comps], np.float32)[:M]
        ky = np.array([c.ky for c in comps], np.float32)[:M]
        amp = np.array([c.A for c in comps], np.float32)[:M]
        n = len(amp)
        A[i, :n, 0] = kx; A[i, :n, 1] = ky
        A[i, :n, 2] = np.log(amp ** 2 + 1e-12)
        if n < M:                                   # rare: replicate last atom
            A[i, n:] = A[i, n-1]
        if (i + 1) % 4000 == 0:
            print(f'  gen {i+1}/{N_SAMPLES}')
    return A


def normalize(A):
    """to [-1,1]^3. kx,ky by K_LIM; logE by its actual [min,max] range."""
    lo_e, hi_e = float(A[..., 2].min()), float(A[..., 2].max())
    Z = A.copy()
    Z[..., 0] /= K_LIM; Z[..., 1] /= K_LIM
    Z[..., 2] = 2 * (A[..., 2] - lo_e) / (hi_e - lo_e) - 1
    return Z.astype(np.float32), dict(lo_e=lo_e, hi_e=hi_e, K_LIM=K_LIM, M=M)


def denorm_atoms(Z, stats):
    """normalized (…,3) → (kx, ky, energy a). torch in/out."""
    kx = Z[..., 0] * stats['K_LIM']; ky = Z[..., 1] * stats['K_LIM']
    le = (Z[..., 2] + 1) / 2 * (stats['hi_e'] - stats['lo_e']) + stats['lo_e']
    a = torch.exp(le)                                # energy = A²
    return kx, ky, a


# ── DDPM schedule ───────────────────────────────────────────────────────────
def cosine_betas(T, s=0.008):
    t = torch.linspace(0, T, T + 1)
    f = torch.cos(((t / T + s) / (1 + s)) * math.pi / 2) ** 2
    ac = f / f[0]
    betas = (1 - ac[1:] / ac[:-1]).clamp(1e-5, 0.999)
    return betas


# ── set-transformer score net ──────────────────────────────────────────────
def timestep_emb(t, dim):
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
    a = t[:, None].float() * freqs[None]
    return torch.cat([torch.sin(a), torch.cos(a)], -1)


class SetDenoiser(nn.Module):
    def __init__(self, d=128, heads=4, layers=4):
        super().__init__()
        self.d = d
        self.inp = nn.Linear(3, d)
        self.temb = nn.Sequential(nn.Linear(d, d), nn.SiLU(), nn.Linear(d, d))
        self.blocks = nn.ModuleList([
            nn.ModuleDict(dict(
                ln1=nn.LayerNorm(d),
                attn=nn.MultiheadAttention(d, heads, batch_first=True),
                ln2=nn.LayerNorm(d),
                ff=nn.Sequential(nn.Linear(d, 2 * d), nn.SiLU(), nn.Linear(2 * d, d)),
            )) for _ in range(layers)])
        self.out = nn.Linear(d, 3)

    def forward(self, x, t):
        h = self.inp(x) + self.temb(timestep_emb(t, self.d))[:, None, :]
        for b in self.blocks:
            z = b['ln1'](h); a, _ = b['attn'](z, z, z); h = h + a
            h = h + b['ff'](b['ln2'](h))
        return self.out(h)


# ── train ───────────────────────────────────────────────────────────────────
def train(epochs=260, bs=256, lr=2e-4):
    cache = f'{OUT}/atoms_dataset.npz'
    if os.path.exists(cache):
        d = np.load(cache); Z, stats = d['Z'], json.loads(str(d['stats']))
        print(f'loaded {Z.shape} from cache')
    else:
        print('generating dataset...')
        A = gen_dataset(); Z, stats = normalize(A)
        np.savez_compressed(cache, Z=Z, stats=json.dumps(stats))
        print(f'saved dataset {Z.shape}')
    X = torch.tensor(Z, device=DEVICE)
    betas = cosine_betas(T).to(DEVICE)
    alphas = (1 - betas); abar = torch.cumprod(alphas, 0)
    model = SetDenoiser().to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    n = X.shape[0]; nparams = sum(p.numel() for p in model.parameters())
    print(f'model {nparams/1e6:.2f}M params, train {n} samples, {epochs} epochs')
    t0 = time.time()
    for ep in range(epochs):
        perm = torch.randperm(n, device=DEVICE); tot = 0.0
        for i in range(0, n, bs):
            xb = X[perm[i:i+bs]]
            tt = torch.randint(0, T, (xb.shape[0],), device=DEVICE)
            ab = abar[tt][:, None, None]
            eps = torch.randn_like(xb)
            xt = ab.sqrt() * xb + (1 - ab).sqrt() * eps
            pred = model(xt, tt)
            loss = ((pred - eps) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item() * xb.shape[0]
        sched.step()
        if (ep + 1) % 20 == 0 or ep == 0:
            print(f'  ep {ep+1:3d}  loss {tot/n:.4f}  ({time.time()-t0:.0f}s)')
    torch.save(dict(model=model.state_dict(), stats=stats,
                    betas=betas.cpu(), abar=abar.cpu()), f'{OUT}/setdiff_prior.pt')
    print(f'saved {OUT}/setdiff_prior.pt')
    return model, stats, abar, betas


# ── unconditional sampling (sanity) ─────────────────────────────────────────
@torch.no_grad()
def sample(model, abar, betas, n=8):
    alphas = 1 - betas
    x = torch.randn(n, M, 3, device=DEVICE)
    for t in reversed(range(T)):
        tt = torch.full((n,), t, device=DEVICE)
        eps = model(x, tt)
        ab = abar[t]; a = alphas[t]
        mean = (x - betas[t] / (1 - ab).sqrt() * eps) / a.sqrt()
        x = mean + (betas[t].sqrt() * torch.randn_like(x) if t > 0 else 0)
    return x


def sanity(model, stats, abar, betas):
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    Zs = sample(model, abar.to(DEVICE), betas.to(DEVICE), n=6)
    kx, ky, a = denorm_atoms(Zs, stats)
    kx, ky, a = kx.cpu().numpy(), ky.cpu().numpy(), a.cpu().numpy()
    # reference real spectra
    d = np.load(f'{OUT}/atoms_dataset.npz'); Z = d['Z'][:6]
    rkx = Z[..., 0] * stats['K_LIM']; rky = Z[..., 1] * stats['K_LIM']
    fig, axes = plt.subplots(2, 6, figsize=(20, 7))
    for j in range(6):
        amax = a[j].max() + 1e-12
        sz = 200 * a[j] / amax
        axes[0, j].scatter(kx[j], ky[j], s=sz, c='tab:red', alpha=0.6)
        axes[0, j].set_title(f'sample {j}'); axes[0, j].set_xlim(-0.6, 0.6); axes[0, j].set_ylim(-0.6, 0.6)
        axes[1, j].scatter(rkx[j], rky[j], s=30, c='tab:blue', alpha=0.6)
        axes[1, j].set_title(f'real {j}'); axes[1, j].set_xlim(-0.6, 0.6); axes[1, j].set_ylim(-0.6, 0.6)
        for ax in (axes[0, j], axes[1, j]):
            ax.set_aspect('equal'); ax.grid(alpha=0.2)
    axes[0, 0].set_ylabel('PRIOR samples (ky vs kx)'); axes[1, 0].set_ylabel('REAL JONSWAP')
    fig.suptitle('Direction III — grid-free set-diffusion prior: unconditional samples vs real spectra')
    fig.tight_layout(); fig.savefig(f'{OUT}/fig_prior_samples.png', dpi=130); plt.close(fig)
    # energy-weighted |k| comparison (the physically meaningful one)
    gen_kmag = np.sqrt(kx**2 + ky**2); real_kmag = np.sqrt(rkx**2 + rky**2)
    _, _, ra = denorm_atoms(torch.tensor(Z), stats); ra = ra.numpy()
    def ew(km, en):  # energy-weighted mean over all atoms
        return float((km * en).sum() / (en.sum() + 1e-12))
    print(f'  energy-weighted |k|: prior={ew(gen_kmag, a):.3f}  real={ew(real_kmag, ra):.3f}')
    print(f'  energy-weighted direction[deg]: prior='
          f'{np.degrees(np.arctan2((ky*a).sum(), (kx*a).sum())):.1f}  '
          f'real={np.degrees(np.arctan2((rky*ra).sum(), (rkx*ra).sum())):.1f}')
    print(f'Saved {OUT}/fig_prior_samples.png')


if __name__ == '__main__':
    model, stats, abar, betas = train()
    sanity(model, stats, abar, betas)
