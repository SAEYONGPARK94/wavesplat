"""Mixed-cardinality set-diffusion prior — fixes the OOD simple-sea gap.

The _long prior was trained ONLY on n_components=32 JONSWAP spectra (effective
cardinality ~9, never below 7.4), so sparse (~5-component) seas were out-of-distribution.
Here every spectrum has a RANDOM number of real discrete components n_real∈[3,32]
(so sparse seas are well represented), the remaining M−n_real atoms are near-zero-energy
"null" atoms placed in the SAME k-region as real ones (the energy channel carries
presence). Broad sea states. Large N. Same M=32 / T / SetDenoiser as d3_setdiff_prior,
so the decoder (denorm_atoms) and the DPS posterior work unchanged with --ckpt.

Run: python3 scripts/d3_setdiff_prior_mixed.py [--n 48000] [--epochs 500]
"""
import os, sys, math, json, time, argparse
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
os.chdir(os.path.join(os.path.dirname(__file__), '..'))

from wavesim.wave_generator import jonswap_components
from d3_setdiff_prior import SetDenoiser, cosine_betas, T, DEVICE, OUT

M = 32
K_LIM = 1.4                 # contains the 30-comp high-freq tail (max |k|~1.36); stats-carried
LOG_FLOOR = -14.0            # null-atom log-energy (fixed normalization floor)
LOG_MAX = 0.0               # a=1 upper bound
N_MIN, N_MAX = 3, 32        # real-component count range (INCLUDES sparse)
ARCH = dict(d=256, heads=8, layers=8)


def gen_dataset(n_samples, seed=0):
    """(N, M, 3) atoms [kx, ky, logE]. n_real∈[N_MIN,N_MAX] real JONSWAP components,
    M−n_real null atoms (tiny energy, k in the real region)."""
    rng = np.random.default_rng(seed)
    A = np.zeros((n_samples, M, 3), np.float32)
    ncounts = np.zeros(n_samples, np.int32)
    for i in range(n_samples):
        Hs = rng.uniform(0.5, 4.0); Tp = rng.uniform(5.0, 12.0)   # Tp≥5 keeps |k|<K_LIM (as _long)
        dird = rng.uniform(-60, 60); spread = rng.uniform(8, 45); gamma = rng.uniform(1.0, 5.0)
        n_real = int(rng.integers(N_MIN, N_MAX + 1))
        comps = jonswap_components(Hs, Tp, n_components=n_real, depth=50.0, gamma=gamma,
                                   direction_deg=dird, direction_spread_deg=spread,
                                   seed=int(rng.integers(1e9)))
        kx = np.array([c.kx for c in comps], np.float32)
        ky = np.array([c.ky for c in comps], np.float32)
        amp = np.array([c.A for c in comps], np.float32)
        n = min(len(amp), M); ncounts[i] = n
        A[i, :n, 0] = kx[:n]; A[i, :n, 1] = ky[:n]
        A[i, :n, 2] = np.clip(np.log(amp[:n] ** 2 + 1e-12), LOG_FLOOR + 0.5, LOG_MAX)
        if n < M:                                   # null atoms: real-region k, floor energy
            kmag = rng.uniform(0.02, 0.35, M - n); ang = rng.uniform(0, 2 * np.pi, M - n)
            A[i, n:, 0] = kmag * np.cos(ang); A[i, n:, 1] = kmag * np.sin(ang)
            A[i, n:, 2] = LOG_FLOOR
        if (i + 1) % 8000 == 0:
            print(f'  gen {i+1}/{n_samples}  (last n_real={n})')
    return A, ncounts


def normalize(A):
    """to [-1,1]^3 with FIXED energy range [LOG_FLOOR, LOG_MAX] (reproducible)."""
    Z = A.copy()
    Z[..., 0] /= K_LIM; Z[..., 1] /= K_LIM
    Z[..., 2] = 2 * (A[..., 2] - LOG_FLOOR) / (LOG_MAX - LOG_FLOOR) - 1
    return Z.astype(np.float32), dict(lo_e=LOG_FLOOR, hi_e=LOG_MAX, K_LIM=K_LIM, M=M)


def train(n_samples, epochs, bs=256, lr=2e-4, tag='', arch=None):
    arch = arch or ARCH
    cache = f'{OUT}/atoms_dataset_mixed{tag}.npz'
    if os.path.exists(cache):
        d = np.load(cache); Z = d['Z']; stats = json.loads(str(d['stats']))
        print(f'loaded {Z.shape} from cache (eff-card range reflects mixed n_real)')
    else:
        print(f'generating mixed dataset n={n_samples}...')
        A, nc = gen_dataset(n_samples); Z, stats = normalize(A)
        np.savez_compressed(cache, Z=Z, stats=json.dumps(stats), ncounts=nc)
        pr = ((np.exp(A[..., 2]).sum(1)) ** 2) / ((np.exp(A[..., 2]) ** 2).sum(1))
        print(f'saved {Z.shape}. n_real∈[{nc.min()},{nc.max()}] mean {nc.mean():.1f}; '
              f'participation ratio mean {pr.mean():.1f} min {pr.min():.1f} (sparse now present)')
    X = torch.tensor(Z, device=DEVICE)
    betas = cosine_betas(T).to(DEVICE); abar = torch.cumprod(1 - betas, 0)
    model = SetDenoiser(**arch).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    n = X.shape[0]; nparams = sum(p.numel() for p in model.parameters())
    print(f'model {nparams/1e6:.2f}M, arch={arch}, train {n} samples, {epochs} epochs')
    t0 = time.time()
    for ep in range(epochs):
        perm = torch.randperm(n, device=DEVICE); tot = 0.0
        for i in range(0, n, bs):
            xb = X[perm[i:i+bs]]
            tt = torch.randint(0, T, (xb.shape[0],), device=DEVICE)
            ab = abar[tt][:, None, None]
            eps = torch.randn_like(xb)
            xt = ab.sqrt() * xb + (1 - ab).sqrt() * eps
            loss = ((model(xt, tt) - eps) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item() * xb.shape[0]
        sched.step()
        if (ep + 1) % 20 == 0 or ep == 0:
            print(f'  ep {ep+1:3d}  loss {tot/n:.4f}  ({time.time()-t0:.0f}s)', flush=True)
    torch.save(dict(model=model.state_dict(), stats=stats, arch=arch,
                    betas=betas.cpu(), abar=abar.cpu()), f'{OUT}/setdiff_prior_mixed{tag}.pt')
    print(f'saved {OUT}/setdiff_prior_mixed{tag}.pt  ({time.time()-t0:.0f}s)')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=48000)
    ap.add_argument('--epochs', type=int, default=500)
    ap.add_argument('--tag', default='', help='suffix for cache+ckpt (e.g. _big)')
    ap.add_argument('--d', type=int, default=ARCH['d'])
    ap.add_argument('--heads', type=int, default=ARCH['heads'])
    ap.add_argument('--layers', type=int, default=ARCH['layers'])
    a = ap.parse_args()
    train(a.n, a.epochs, tag=a.tag, arch=dict(d=a.d, heads=a.heads, layers=a.layers))
