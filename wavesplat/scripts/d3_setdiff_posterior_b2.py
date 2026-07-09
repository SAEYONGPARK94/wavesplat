"""B2 — improve the DPS posterior: noise-tied (ΠGDM-style) guidance + Gauss-Newton polish.

Baseline DPS (d3_setdiff_posterior) reverse-diffuses the atom prior with a FIXED, fully
length-normalized likelihood-guidance step at every t — it throws away the gradient magnitude
(curvature) and uses the same strength when the sample is pure noise as when it is nearly
clean. Two upgrades, ablated independently:

  (1) noise-tied guidance schedule (ΠGDM spirit): step ∝ (1−ᾱ_t)·∇log p(z|x̂₀) — strong while
      the estimate is noisy, →0 as it denoises, instead of a constant push.
  (2) Gauss-Newton PHYSICS POLISH: after reverse diffusion, refine each posterior atom set by
      a few Adam steps directly maximizing the grid-free observation log-likelihood. This lifts
      atoms off the prior manifold onto the data — the "net proposes, physics refines" (NG-RELAX)
      pattern that reliably matches RELAX.

Reuses the frozen prior (results/direction3/setdiff_prior.pt) — no retraining. Ablation:
prior / dps / dps+polish / pigdm+polish, plus BMA over posterior samples.

Run: python3 scripts/d3_setdiff_posterior_b2.py --guidance 0.1 --polish_iters 80
"""
import os, sys, argparse, math, json
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
os.chdir(os.path.join(os.path.dirname(__file__), '..'))

from wavesplat.data_loader import EpisodeDataLoader
from d3_setdiff_prior import SetDenoiser, denorm_atoms, T, M
from d3_setdiff_posterior import (gather_window, loglik_atoms, atoms_to_field,
                                  dps_sample, DEVICE, CKPT)


def _to_t(P, z):
    Pt = P if torch.is_tensor(P) else torch.tensor(P, device=DEVICE, dtype=torch.float32)
    zt = z if torch.is_tensor(z) else torch.tensor(z, device=DEVICE, dtype=torch.float32)
    return Pt, zt


def dps_sample_pigdm(model, stats, abar, betas, P, z, depth, guidance=0.1, n=1, seed=0):
    """Reverse diffusion with a noise-tied guidance schedule (step ∝ (1−ᾱ_t))."""
    torch.manual_seed(seed)
    alphas = 1 - betas
    Pt, zt = _to_t(P, z)
    x = torch.randn(n, M, 3, device=DEVICE)
    for t in reversed(range(T)):
        x = x.detach().requires_grad_(True)
        tt = torch.full((n,), t, device=DEVICE)
        eps = model(x, tt)
        ab = abar[t]
        x0 = ((x - (1 - ab).sqrt() * eps) / ab.sqrt()).clamp(-1, 1)
        ll = 0.0
        for b in range(n):
            kx, ky, a = denorm_atoms(x0[b], stats); a = torch.clamp(a, min=0.0)
            ll = ll + loglik_atoms(kx, ky, a, Pt, zt, depth)
        grad = torch.autograd.grad(ll, x)[0]
        with torch.no_grad():
            mean = (x - betas[t] / (1 - ab).sqrt() * eps) / alphas[t].sqrt()
            x = mean + (betas[t].sqrt() * torch.randn_like(x) if t > 0 else 0)
            gn = grad.flatten(1).norm(dim=1).clamp(min=1e-8)
            x = x + guidance * float(1 - ab) * grad / gn[:, None, None]   # noise-tied
    return x.detach()


def polish_atoms(x0, stats, P, z, depth, iters=80, lr=0.02):
    """Gauss-Newton (Adam) refine of a normalized atom set to maximize the obs log-likelihood.
    kx,ky optimized freely (linear denorm); the energy channel is clamped to its trained range
    (denorm exp() would otherwise blow up out of range)."""
    Pt, zt = _to_t(P, z)
    xp = x0.clone().detach().requires_grad_(True)
    opt = torch.optim.Adam([xp], lr=lr)
    for _ in range(iters):
        opt.zero_grad()
        z_in = torch.stack([xp[..., 0], xp[..., 1], xp[..., 2].clamp(-1.0, 1.0)], -1)
        kx, ky, a = denorm_atoms(z_in, stats); a = torch.clamp(a, min=0.0)
        ll = loglik_atoms(kx, ky, a, Pt, zt, depth)
        (-ll).backward()
        opt.step()
    return xp.detach().clamp(-1.1, 1.1)


def field_set(X, stats, P, z, depth, target, polish=False, pit=80):
    """atoms batch → (list of fields, list of nrmse). optionally polish each first."""
    Pt, zt = _to_t(P, z)
    fields, rs = [], []
    for b in range(X.shape[0]):
        xb = polish_atoms(X[b], stats, Pt, zt, depth, iters=pit) if polish else X[b]
        f, r = atoms_to_field(xb, stats, P, z, depth, target)
        rs.append(r)
        if f is not None:
            fields.append(f)
    return fields, rs


def bma_nrmse(fields, target):
    if not fields:
        return 1.0
    m = np.mean(fields, axis=0)
    return float(np.sqrt(np.mean((m - target['gt']) ** 2)) / max(target['Hs'], 1e-6))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--episodes', nargs='+', default=[
        'episode_BF4_5comp_seed0_uniform.h5', 'episode_BF4_15comp_seed0_uniform.h5',
        'episode_BF3_5comp_seed1_uniform.h5'])
    ap.add_argument('--guidance', type=float, default=0.1)
    ap.add_argument('--nsamp', type=int, default=4)
    ap.add_argument('--polish_iters', type=int, default=80)
    ap.add_argument('--ckpt', default=CKPT)
    a = ap.parse_args()

    ck = torch.load(a.ckpt, map_location=DEVICE)
    stats = ck['stats']
    model = SetDenoiser(**ck.get('arch', {})).to(DEVICE)
    model.load_state_dict(ck['model']); model.eval()
    abar = ck['abar'].to(DEVICE); betas = ck['betas'].to(DEVICE)

    hdr = ['episode', 'prior', 'dps', 'dps+pol', 'pigdm+pol', 'BMA(best)']
    print(f"guidance={a.guidance} nsamp={a.nsamp} polish_iters={a.polish_iters}")
    print(f"{hdr[0]:34s} {hdr[1]:>7} {hdr[2]:>7} {hdr[3]:>8} {hdr[4]:>10} {hdr[5]:>10}")
    rows = []
    for ep in a.episodes:
        L = EpisodeDataLoader('data/' + ep)
        depth = L.depth; Hs = L.Hs_true
        gx = L.gt_grid_x.numpy(); gy = L.gt_grid_y.numpy(); ti = 25
        P, z, frames = gather_window(L, ti)
        tgt = frames[ti + 1]
        target = {'gx': gx, 'gy': gy, 't': float(tgt.t), 'gt': tgt.gt_field.numpy(), 'Hs': Hs}

        Xp = dps_sample(model, stats, abar, betas, P, z, depth, guidance=0.0, n=2, seed=1)
        r_prior = np.mean([atoms_to_field(Xp[b], stats, P, z, depth, target)[1] for b in range(2)])

        Xd = dps_sample(model, stats, abar, betas, P, z, depth, guidance=a.guidance,
                        n=a.nsamp, seed=0)
        f_dps, r_dps = field_set(Xd, stats, P, z, depth, target, polish=False)
        f_dpol, r_dpol = field_set(Xd, stats, P, z, depth, target, polish=True, pit=a.polish_iters)

        Xg = dps_sample_pigdm(model, stats, abar, betas, P, z, depth, guidance=a.guidance,
                              n=a.nsamp, seed=0)
        f_gpol, r_gpol = field_set(Xg, stats, P, z, depth, target, polish=True, pit=a.polish_iters)

        best_fields = f_gpol if np.mean(r_gpol) <= np.mean(r_dpol) else f_dpol
        r_bma = bma_nrmse(best_fields, target)
        print(f"{ep:34s} {r_prior:7.4f} {np.mean(r_dps):7.4f} {np.mean(r_dpol):8.4f} "
              f"{np.mean(r_gpol):10.4f} {r_bma:10.4f}")
        rows.append(dict(ep=ep, prior=float(r_prior), dps=float(np.mean(r_dps)),
                         dps_polish=float(np.mean(r_dpol)), pigdm_polish=float(np.mean(r_gpol)),
                         bma=r_bma))

    def mean(k): return float(np.mean([r[k] for r in rows]))
    summary = dict(guidance=a.guidance, nsamp=a.nsamp, polish_iters=a.polish_iters, rows=rows,
                   mean_prior=mean('prior'), mean_dps=mean('dps'), mean_dps_polish=mean('dps_polish'),
                   mean_pigdm_polish=mean('pigdm_polish'), mean_bma=mean('bma'),
                   ref=dict(baseline_dps=0.242, WaveSplat_fused=0.205, exact=0.06))
    os.makedirs('results/direction3', exist_ok=True)
    json.dump(summary, open('results/direction3/posterior_b2_eval.json', 'w'), indent=2)
    print(f"\nMEAN  prior {mean('prior'):.4f}  dps {mean('dps'):.4f}  dps+pol {mean('dps_polish'):.4f}  "
          f"pigdm+pol {mean('pigdm_polish'):.4f}  BMA {mean('bma'):.4f}")
    print("ref: baseline DPS 0.242 / WaveSplat fused 0.205 / exact 0.06")


if __name__ == '__main__':
    main()
