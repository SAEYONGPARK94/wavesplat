"""B1 — beamform-conditioned set-diffusion (amortized posterior sampler).

Fuses the two winning ideas: the beamform power map (which made DETECTION well-posed) becomes
the CONDITION of the set-diffusion denoiser (which gives grid-free atoms + UQ). Instead of an
unconditional prior steered by weak DPS likelihood-guidance (B2), the denoiser directly ingests
the observation via its beamform map through cross-attention → an amortized posterior sampler:
one conditioned reverse diffusion emits posterior spectral atoms, no per-step likelihood grad.

Atoms use the SAME M=32 / normalization as the frozen prior (K_LIM, logE range reused from
setdiff_prior.pt) so the field back-end (atoms_to_field) and B2's polish drop in unchanged.

CAVEAT (from prior sessions): amortized nets trained on synthetic-clean data have failed to
transfer to real broadband+noisy data. This is a sim-domain study; real deployment needs a
realistic retrain / fine-tune.

Run: python3 scripts/d3_beamform_cond_diffusion.py [--steps 15000] [--eval_only ckpt.pt]
"""
import os, sys, json, time, argparse, math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
os.chdir(os.path.join(os.path.dirname(__file__), '..'))

from wavesim.wave_generator import jonswap_components
from wavesplat.data_loader import EpisodeDataLoader
from d6_integrated_eval import tag_to_file
from d15_sequential_fusion import gather_frames
from d23_main_results import K_TP, panel
from d40_learned_acquisition import pool_xy
from d3_beamform_net import beamform, KLIM, NK, M_OBS, DEPTH
from d3_setdiff_prior import cosine_betas, timestep_emb, M, T
from d3_setdiff_posterior import atoms_to_field, loglik_atoms
from d3_setdiff_posterior_b2 import polish_atoms
from nonlinear_forward import elevation

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
PRIOR_CKPT = 'results/direction3/setdiff_prior.pt'


# ── beamform-map encoder → conditioning tokens ──────────────────────────────
class CondEncoder(nn.Module):
    def __init__(self, d=128):
        super().__init__()
        def blk(i, o, s=1):
            return nn.Sequential(nn.Conv2d(i, o, 3, s, 1), nn.GroupNorm(8, o), nn.GELU())
        self.net = nn.Sequential(blk(1, d), blk(d, d, 2),   # 64->32
                                 blk(d, d), blk(d, d, 2),   # 32->16
                                 blk(d, d))
        self.pos = nn.Parameter(torch.randn(1, 16 * 16, d) * 0.02)

    def forward(self, B):
        f = self.net(B[:, None]).flatten(2).transpose(1, 2)   # (B,256,d)
        return f + self.pos


# ── conditional set denoiser (self-attn over atoms + cross-attn to beamform) ─
class CondSetDenoiser(nn.Module):
    def __init__(self, d=128, heads=4, layers=4):
        super().__init__()
        self.d = d
        self.enc = CondEncoder(d)
        self.inp = nn.Linear(3, d)
        self.temb = nn.Sequential(nn.Linear(d, d), nn.SiLU(), nn.Linear(d, d))
        self.blocks = nn.ModuleList([nn.ModuleDict(dict(
            ln1=nn.LayerNorm(d), attn=nn.MultiheadAttention(d, heads, batch_first=True),
            lnc=nn.LayerNorm(d), cross=nn.MultiheadAttention(d, heads, batch_first=True),
            ln2=nn.LayerNorm(d),
            ff=nn.Sequential(nn.Linear(d, 2 * d), nn.SiLU(), nn.Linear(2 * d, d)),
        )) for _ in range(layers)])
        self.out = nn.Linear(d, 3)

    def forward(self, x, t, Bmap=None, cond=None):
        if cond is None:
            cond = self.enc(Bmap)                              # (B,256,d)
        h = self.inp(x) + self.temb(timestep_emb(t, self.d))[:, None, :]
        for b in self.blocks:
            hn = b['ln1'](h); h = h + b['attn'](hn, hn, hn)[0]
            hc = b['lnc'](h); h = h + b['cross'](hc, cond, cond)[0]
            h = h + b['ff'](b['ln2'](h))
        return self.out(h)


def make_batch(Bsz, pool, stats, hos_frac=0.0):
    """32-component JONSWAP spectra (prior representation) + their beamform maps.
    hos_frac: fraction of samples whose obs use the 2nd-order Stokes nonlinear forward."""
    K_LIM = stats['K_LIM']; lo_e, hi_e = stats['lo_e'], stats['hi_e']
    atoms = np.zeros((Bsz, M, 3), np.float32)
    P = np.zeros((Bsz, M_OBS, 3), np.float32); Z = np.zeros((Bsz, M_OBS), np.float32)
    for b in range(Bsz):
        Hs = np.random.uniform(0.8, 3.2); Tp = np.random.uniform(5.0, 11.0)
        dird = np.random.uniform(-50, 50); spread = np.random.uniform(12, 40)
        gamma = np.random.uniform(1.5, 4.0)
        c = jonswap_components(Hs, Tp, n_components=M, depth=DEPTH, gamma=gamma,
                               direction_deg=dird, direction_spread_deg=spread,
                               seed=int(np.random.randint(1e9)))
        kx = np.array([x.kx for x in c], np.float32); ky = np.array([x.ky for x in c], np.float32)
        amp = np.array([x.A for x in c], np.float32); om = np.array([x.omega for x in c])
        ph = np.array([x.phi for x in c])
        atoms[b, :, 0] = kx / K_LIM; atoms[b, :, 1] = ky / K_LIM
        atoms[b, :, 2] = 2 * (np.log(amp ** 2 + 1e-12) - lo_e) / (hi_e - lo_e) - 1
        Tobs = K_TP * Tp
        idx = np.random.choice(len(pool), M_OBS, replace=len(pool) < M_OBS)
        ox, oy = pool[idx, 0], pool[idx, 1]; ot = np.random.uniform(0, Tobs, M_OBS)
        nl = np.random.rand() < hos_frac
        Z[b] = elevation(kx, ky, amp, om, ph, ox, oy, ot, DEPTH, nonlinear=nl) \
            + np.random.normal(0, 0.02, M_OBS)
        P[b] = np.column_stack([ox, oy, ot])
    Bmap = beamform(torch.tensor(P, device=DEVICE), torch.tensor(Z, device=DEVICE))
    return torch.tensor(atoms, device=DEVICE), Bmap


def build_from_comps(comps, P, z, stats):
    """Fixed-dataset comps (kx,ky,A) + obs → (normalized atom tensor, beamform map)."""
    K_LIM = stats['K_LIM']; lo_e, hi_e = stats['lo_e'], stats['hi_e']
    B = len(comps)
    atoms = np.zeros((B, M, 3), np.float32)
    for b, c in enumerate(comps):
        atoms[b, :, 0] = c[:, 0] / K_LIM; atoms[b, :, 1] = c[:, 1] / K_LIM
        atoms[b, :, 2] = 2 * (np.log(c[:, 2] ** 2 + 1e-12) - lo_e) / (hi_e - lo_e) - 1
    Bmap = beamform(torch.tensor(P, device=DEVICE), torch.tensor(z, device=DEVICE))
    return torch.tensor(atoms, device=DEVICE), Bmap


def train(steps, stats, abar, Bsz=32, lr=2e-4, dataset=None, linear_only=False, log=print):
    pool = pool_xy().astype(np.float32)
    sampler = None
    if dataset:
        from fixed_dataset import FixedSampler
        sampler = FixedSampler(dataset, aug_noise=0.015, sub_obs=320,
                               nl_filter=(0 if linear_only else None))
        log(f"  training from fixed dataset {dataset} (N={sampler.N}, linear_only={linear_only})")
    model = CondSetDenoiser().to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    t0 = time.time()
    for s in range(steps):
        if sampler is not None:
            comps, P, z = sampler.batch(Bsz)
            x0, Bmap = build_from_comps(comps, P, z, stats)
        else:
            x0, Bmap = make_batch(Bsz, pool, stats)
        t = torch.randint(0, T, (Bsz,), device=DEVICE)
        ab = abar[t][:, None, None]
        eps = torch.randn_like(x0)
        xt = ab.sqrt() * x0 + (1 - ab).sqrt() * eps
        pred = model(xt, t, Bmap)
        loss = F.mse_loss(pred, eps)
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        if s % 250 == 0 or s == steps - 1:
            log(f"  step {s:5d}  loss {loss.item():.5f}  ({time.time()-t0:.0f}s)")
    return model


@torch.no_grad()
def sample(model, Bmap, abar, betas, n=4, seed=0):
    """Conditioned ancestral DDPM sampling → (n, M, 3) normalized atoms."""
    torch.manual_seed(seed)
    alphas = 1 - betas
    cond = model.enc(Bmap).expand(n, -1, -1)
    x = torch.randn(n, M, 3, device=DEVICE)
    for t in reversed(range(T)):
        tt = torch.full((n,), t, device=DEVICE)
        eps = model(x, tt, cond=cond)
        ab = abar[t]
        mean = (x - betas[t] / (1 - ab).sqrt() * eps) / alphas[t].sqrt()
        x = mean + (betas[t].sqrt() * torch.randn_like(x) if t > 0 else 0)
    return x.clamp(-1, 1)


@torch.no_grad()
def _bmap_from_obs(Ps, z):
    return beamform(torch.tensor(Ps[None], device=DEVICE, dtype=torch.float32),
                    torch.tensor(z[None], device=DEVICE, dtype=torch.float32))


def evaluate(model, stats, abar, betas, episodes, nsamp=4, polish=True, log=print):
    model.eval(); out = {}
    for tag in episodes:
        L = EpisodeDataLoader('data/' + tag_to_file(tag)); depth = L.depth; Hs = L.Hs_true
        gx = L.gt_grid_x.numpy(); gy = L.gt_grid_y.numpy(); nb = int(round(K_TP * L.Tp_true / 0.1))
        i0 = nb + 3; i1 = len(L) - 2; sel = np.linspace(i0, i1, 3).astype(int)
        nr_raw, nr_pol = [], []
        for i_star in sel:
            fstar = L[int(i_star)]; P, z = gather_frames(L, int(i_star), nb)
            if len(z) > M_OBS:
                idx = np.random.default_rng(0).choice(len(z), M_OBS, replace=False); P, z = P[idx], z[idx]
            t0 = P[:, 2].min(); Ps = P.copy(); Ps[:, 2] -= t0
            Bmap = _bmap_from_obs(Ps, z)
            X = sample(model, Bmap, abar, betas, n=nsamp)
            target = {'gx': gx, 'gy': gy, 't': float(fstar.t) - t0, 'gt': fstar.gt_field.numpy(), 'Hs': Hs}
            Pt = torch.tensor(Ps, device=DEVICE, dtype=torch.float32)
            zt = torch.tensor(z, device=DEVICE, dtype=torch.float32)
            fr, fp = [], []
            for b in range(nsamp):
                f, r = atoms_to_field(X[b], stats, Ps, z, depth, target); fr.append((f, r))
                if polish:
                    xb = polish_atoms(X[b], stats, Pt, zt, depth, iters=80)
                    f2, r2 = atoms_to_field(xb, stats, Ps, z, depth, target); fp.append((f2, r2))
            bma_raw = np.mean([f for f, _ in fr if f is not None], axis=0) if any(f is not None for f, _ in fr) else None
            nr_raw.append(float(np.sqrt(np.mean((bma_raw - target['gt']) ** 2)) / max(Hs, 1e-6)) if bma_raw is not None else 1.0)
            if polish:
                bma_pol = np.mean([f for f, _ in fp if f is not None], axis=0) if any(f is not None for f, _ in fp) else None
                nr_pol.append(float(np.sqrt(np.mean((bma_pol - target['gt']) ** 2)) / max(Hs, 1e-6)) if bma_pol is not None else 1.0)
        g = float(np.mean(nr_raw)); p = float(np.mean(nr_pol)) if polish else g
        out[tag] = dict(cond_bma=g, cond_polish_bma=p)
        log(f"  {tag:12s} cond-BMA {g:.3f}  cond+polish-BMA {p:.3f}")
    return out


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--steps', type=int, default=15000)
    ap.add_argument('--ckpt', default='results/direction3/beamform_cond_diffusion.pt')
    ap.add_argument('--eval_only', default=None)
    ap.add_argument('--nsamp', type=int, default=4)
    ap.add_argument('--dataset', default=None)
    ap.add_argument('--linear_only', action='store_true')
    a = ap.parse_args()
    stats = torch.load(PRIOR_CKPT, map_location=DEVICE)['stats']
    betas = cosine_betas(T).to(DEVICE); abar = torch.cumprod(1 - betas, 0)
    eps = ['BF4_5c_s0', 'BF4_15c_s0', 'BF3_5c_s1']
    epfiles = {'BF4_5c_s0': 'episode_BF4_5comp_seed0_uniform.h5',
               'BF4_15c_s0': 'episode_BF4_15comp_seed0_uniform.h5',
               'BF3_5c_s1': 'episode_BF3_5comp_seed1_uniform.h5'}
    if a.eval_only:
        model = CondSetDenoiser().to(DEVICE); model.load_state_dict(torch.load(a.eval_only, map_location=DEVICE))
    else:
        print(f"training {a.steps} steps on {DEVICE} (beamform-conditioned diffusion) "
              f"dataset={a.dataset}...", flush=True)
        model = train(a.steps, stats, abar, dataset=a.dataset,
                      linear_only=a.linear_only, log=lambda m: print(m, flush=True))
        torch.save(model.state_dict(), a.ckpt); print(f"saved {a.ckpt}", flush=True)
    print("\n=== B1 eval (amortized conditioned posterior) ===", flush=True)
    # map tags to episode files for eval loader (reuse tag_to_file where possible)
    res = evaluate(model, stats, abar, betas, list(epfiles.keys()), nsamp=a.nsamp, log=lambda m: print(m, flush=True))
    mg = float(np.mean([v['cond_bma'] for v in res.values()]))
    mp = float(np.mean([v['cond_polish_bma'] for v in res.values()]))
    print(f"\nMEAN  cond-BMA {mg:.3f}  cond+polish-BMA {mp:.3f}", flush=True)
    print("ref: B2 dps+polish BMA 0.193 / baseline DPS 0.240 / WaveSplat fused 0.205", flush=True)
    json.dump(dict(per_ep=res, mean_cond=mg, mean_cond_polish=mp),
              open('results/direction3/beamform_cond_diffusion_eval.json', 'w'), indent=2)
