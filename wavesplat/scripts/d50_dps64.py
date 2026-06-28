"""
D2 — richer continuous representation: retrain the set-diffusion prior with 64 atoms
(finer continuous-density samples of JONSWAP) and test whether DPS improves on the
continuous-spectrum (HOS) regime beyond the 32-atom −16%. d49 ruled out the observation
budget (D1); this tests the representation lever (D2) directly.

Trains a 64-atom prior, then evaluates DPS-64 vs RELAX vs DPS-32 on the cached HOS windows.

Run: python3 scripts/d50_dps64.py [--steps_epochs 400]
"""
import os, sys, json, time, copy, argparse
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
os.chdir(os.path.join(os.path.dirname(__file__), '..'))

import d3_setdiff_prior as P1
P1.M = 64                                                  # <-- richer representation
from d3_setdiff_prior import SetDenoiser, cosine_betas, normalize, denorm_atoms, gen_dataset, T, OUT
import d3_setdiff_posterior as DP
from d23_main_results import panel
from d26_relax_iterative import iterative_relax

DEVICE = P1.DEVICE
ARCH = dict(d=192, heads=6, layers=5)
N_SAMPLES = 24000
N_VAL = 1500
BS = 512
LR = 3e-4
EMA_DECAY = 0.9995
MIN_SNR = 5.0
DATA = f'{OUT}/atoms_dataset_m64.npz'
CKPT = f'{OUT}/setdiff_prior_m64.pt'
CACHE = 'results/direction3/_hos_windows.npz'
DEPTH = 50.0


class EMA:
    def __init__(self, model, decay):
        self.decay = decay; self.shadow = copy.deepcopy(model.state_dict())
    @torch.no_grad()
    def update(self, model):
        for k, v in model.state_dict().items():
            if v.dtype.is_floating_point:
                self.shadow[k].mul_(self.decay).add_(v, alpha=1 - self.decay)
            else:
                self.shadow[k].copy_(v)


def train_prior(epochs):
    if os.path.exists(DATA):
        d = np.load(DATA); Z, stats = d['Z'], json.loads(str(d['stats']))
        print(f'loaded {Z.shape}', flush=True)
    else:
        print('generating 64-atom dataset...', flush=True)
        P1.N_SAMPLES = N_SAMPLES; A = gen_dataset(seed=1); Z, stats = normalize(A)
        np.savez_compressed(DATA, Z=Z, stats=json.dumps(stats)); print(f'saved {Z.shape}', flush=True)
    X = torch.tensor(Z, device=DEVICE); Xtr = X[:-N_VAL]
    betas = cosine_betas(T).to(DEVICE); abar = torch.cumprod(1 - betas, 0)
    snr = abar / (1 - abar); w_snr = torch.clamp(snr, max=MIN_SNR) / snr
    model = SetDenoiser(**ARCH).to(DEVICE); ema = EMA(model, EMA_DECAY)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    n = Xtr.shape[0]; t0 = time.time()
    print(f'train 64-atom prior: {n} samples, {epochs} epochs', flush=True)
    for ep in range(epochs):
        model.train(); perm = torch.randperm(n, device=DEVICE); tot = 0.0
        for i in range(0, n, BS):
            xb = Xtr[perm[i:i+BS]]; tt = torch.randint(0, T, (xb.shape[0],), device=DEVICE)
            ab = abar[tt][:, None, None]; eps = torch.randn_like(xb)
            xt = ab.sqrt() * xb + (1 - ab).sqrt() * eps
            se = ((model(xt, tt) - eps) ** 2).mean(dim=(1, 2)); loss = (w_snr[tt] * se).mean()
            opt.zero_grad(); loss.backward(); opt.step(); ema.update(model); tot += loss.item() * xb.shape[0]
        sch.step()
        if (ep + 1) % 80 == 0 or ep == 0:
            print(f'  ep {ep+1}  train {tot/n:.4f}  ({time.time()-t0:.0f}s)', flush=True)
    torch.save(dict(model=ema.shadow, arch=ARCH, stats=stats, betas=betas.cpu(), abar=abar.cpu()), CKPT)
    print(f'saved {CKPT}', flush=True)


def evaluate():
    wins = list(np.load(CACHE, allow_pickle=True)['windows'])
    ck = torch.load(CKPT, map_location=DEVICE); stats = ck['stats']
    model = SetDenoiser(**ck['arch']).to(DEVICE); model.load_state_dict(ck['model']); model.eval()
    abar = ck['abar'].to(DEVICE); betas = ck['betas'].to(DEVICE)
    gx = np.linspace(10, 250, 80); gy = np.linspace(-100, 100, 60)
    d64, rel = [], []
    for w in wins:
        target = {'gx': gx, 'gy': gy, 't': float(w['t']), 'gt': w['gt'], 'Hs': float(w['Hs'])}
        P, z = w['P'], w['z']
        rel.append(panel(iterative_relax(P, z, DEPTH, target, 3)[0], w['gt'], w['Hs'])['nrmse'])
        idx = np.random.default_rng(0).choice(len(z), min(200, len(z)), replace=False)
        Xd = DP.dps_sample(model, stats, abar, betas, P[idx], z[idx], DEPTH, guidance=0.1, n=8, seed=0)
        flds = [DP.atoms_to_field(Xd[b], stats, P[idx], z[idx], DEPTH, target)[0] for b in range(8)]
        flds = [f for f in flds if f is not None]
        d64.append(panel(np.mean(flds, axis=0) if flds else np.zeros_like(w['gt']), w['gt'], w['Hs'])['nrmse'])
        print(f"[{w['tag']}] DPS-64 {d64[-1]:.3f} | RELAX {rel[-1]:.3f}", flush=True)
    md, mr = float(np.mean(d64)), float(np.mean(rel))
    print(f"\nOVERALL (HOS continuous)  DPS-64 {md:.3f} | RELAX {mr:.3f} | (DPS-32 was 0.234)")
    print(f"  → DPS-64 vs DPS-32(0.234): {md-0.234:+.3f}  | vs RELAX: {'DPS wins −%d%%'%round(100*(mr-md)/mr) if md<mr else 'RELAX wins'}")
    json.dump(dict(dps64=md, relax=mr, dps32=0.234), open('results/direction3/dps64_hos.json', 'w'), indent=2)
    print("saved results/direction3/dps64_hos.json")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--epochs', type=int, default=400)
    ap.add_argument('--eval_only', action='store_true'); a = ap.parse_args()
    if not a.eval_only:
        train_prior(a.epochs)
    print("\n=== eval DPS-64 on continuous HOS ===", flush=True)
    evaluate()


if __name__ == '__main__':
    main()
