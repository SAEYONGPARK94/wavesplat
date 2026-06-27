"""
Direction III — 본격 학습: scaled-up set-diffusion spectral prior (v2).

Upgrades over d3_setdiff_prior (v1: 0.56M / 16k / 260ep / 88s):
  - bigger model (~2M: d=192, 6 heads, 5 attn blocks)
  - 40k samples + held-out val split
  - 600 epochs, EMA(0.9995), min-SNR-γ loss weighting, cosine LR
Saves results/direction3/setdiff_prior_v2.pt (EMA weights + arch + stats + schedule).

Run: python3 scripts/d5_train_prior_v2.py
"""
import os, sys, math, json, time, copy
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
os.chdir(os.path.join(os.path.dirname(__file__), '..'))

import d3_setdiff_prior as P1
from d3_setdiff_prior import (SetDenoiser, cosine_betas, normalize, denorm_atoms,
                              gen_dataset, M, T, OUT)
from wavesim.wave_generator import jonswap_components

DEVICE = P1.DEVICE
ARCH = dict(d=192, heads=6, layers=5)
N_SAMPLES = 40000
N_VAL = 2000
EPOCHS = 600
BS = 512
LR = 3e-4
EMA_DECAY = 0.9995
MIN_SNR_GAMMA = 5.0
DATA = f'{OUT}/atoms_dataset_v2.npz'
CKPT = f'{OUT}/setdiff_prior_v2.pt'


def gen_big(seed=1):
    P1.N_SAMPLES = N_SAMPLES               # reuse generator with larger N
    return gen_dataset(seed=seed)


class EMA:
    def __init__(self, model, decay):
        self.decay = decay
        self.shadow = copy.deepcopy(model.state_dict())

    @torch.no_grad()
    def update(self, model):
        for k, v in model.state_dict().items():
            if v.dtype.is_floating_point:
                self.shadow[k].mul_(self.decay).add_(v, alpha=1 - self.decay)
            else:
                self.shadow[k].copy_(v)


def main():
    if os.path.exists(DATA):
        d = np.load(DATA); Z, stats = d['Z'], json.loads(str(d['stats']))
        print(f'loaded {Z.shape} from cache')
    else:
        print('generating 40k dataset...')
        A = gen_big(); Z, stats = normalize(A)
        np.savez_compressed(DATA, Z=Z, stats=json.dumps(stats))
        print(f'saved {Z.shape}')
    X = torch.tensor(Z, device=DEVICE)
    Xtr, Xval = X[:-N_VAL], X[-N_VAL:]

    betas = cosine_betas(T).to(DEVICE)
    abar = torch.cumprod(1 - betas, 0)
    snr = abar / (1 - abar)
    w_snr = torch.clamp(snr, max=MIN_SNR_GAMMA) / snr     # ε-pred min-SNR weight

    model = SetDenoiser(**ARCH).to(DEVICE)
    ema = EMA(model, EMA_DECAY)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EPOCHS)
    nparams = sum(p.numel() for p in model.parameters())
    n = Xtr.shape[0]
    print(f'model {nparams/1e6:.2f}M params  train {n}  val {N_VAL}  {EPOCHS} epochs')

    def loss_on(xb, weighted=True):
        tt = torch.randint(0, T, (xb.shape[0],), device=DEVICE)
        ab = abar[tt][:, None, None]
        eps = torch.randn_like(xb)
        xt = ab.sqrt() * xb + (1 - ab).sqrt() * eps
        pred = model(xt, tt)
        se = ((pred - eps) ** 2).mean(dim=(1, 2))
        return (w_snr[tt] * se).mean() if weighted else se.mean()

    t0 = time.time(); best_val = 1e9
    for ep in range(EPOCHS):
        model.train(); perm = torch.randperm(n, device=DEVICE); tot = 0.0
        for i in range(0, n, BS):
            xb = Xtr[perm[i:i+BS]]
            loss = loss_on(xb)
            opt.zero_grad(); loss.backward(); opt.step(); ema.update(model)
            tot += loss.item() * xb.shape[0]
        sched.step()
        if (ep + 1) % 40 == 0 or ep == 0:
            model.eval()
            with torch.no_grad():
                vl = loss_on(Xval, weighted=False).item()
            best_val = min(best_val, vl)
            print(f'  ep {ep+1:3d}  train {tot/n:.4f}  val(unweighted) {vl:.4f}  ({time.time()-t0:.0f}s)')
    torch.save(dict(model=ema.shadow, arch=ARCH, stats=stats,
                    betas=betas.cpu(), abar=abar.cpu()), CKPT)
    print(f'saved EMA prior → {CKPT}  (best val {best_val:.4f})')

    # sanity: energy-weighted |k| over 64 samples
    model.load_state_dict(ema.shadow); model.eval()
    from d3_setdiff_prior import sample as sample_fn
    Zs = sample_fn(model, abar, betas, n=64)
    kx, ky, a = denorm_atoms(Zs, stats)
    kx, ky, a = kx.cpu().numpy(), ky.cpu().numpy(), a.cpu().numpy()
    km = np.sqrt(kx**2 + ky**2)
    ewk = float((km * a).sum() / (a.sum() + 1e-12))
    # real reference
    rkx = Z[:64, :, 0] * stats['K_LIM']; rky = Z[:64, :, 1] * stats['K_LIM']
    _, _, ra = denorm_atoms(torch.tensor(Z[:64]), stats); ra = ra.numpy()
    rkm = np.sqrt(rkx**2 + rky**2)
    ewr = float((rkm * ra).sum() / (ra.sum() + 1e-12))
    print(f'  energy-weighted |k|: prior(EMA)={ewk:.3f}  real={ewr:.3f}')


if __name__ == '__main__':
    main()
