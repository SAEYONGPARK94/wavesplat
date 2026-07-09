"""
Direction III — grid-free posterior: DPS with the set-diffusion spectral prior.

Reverse-diffuse the atom prior (d3_setdiff_prior) from noise; at each step add the
likelihood guidance ∇_{x_t} log p(z | atoms(x̂₀)) (Chung et al. 2023 DPS), with the
phase-marginalized GP-marginal likelihood evaluated GRID-FREE on continuous atoms:
    K = Σᵢ aᵢ·cos(kxᵢ·Δx + kyᵢ·Δy − ω(kᵢ)·Δt),   log p = −½ zᵀ(K+σ²I)⁻¹z − ½log|·|.
Posterior atoms → (kx,ky) → fit A,φ by LS → render field → nRMSE.

Run: python3 scripts/d3_setdiff_posterior.py --guidance 0.3
"""
import os, sys, argparse, math, json
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
os.chdir(os.path.join(os.path.dirname(__file__), '..'))

from wavesplat.data_loader import EpisodeDataLoader
from wavesplat.model.tnse import fit_amplitudes_phases, render_wave_field
from d3_setdiff_prior import SetDenoiser, denorm_atoms, T, M

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
G = 9.81
SIGMA = 0.02
W = 10
M_CAP = 150
CKPT = 'results/direction3/setdiff_prior.pt'


def gather_window(L, ti):
    frames = L.get_tracking_frames(2.0)
    P, z = [], []
    for f in frames[ti-W+1:ti+1]:
        x = f.x_obs.numpy(); y = f.y_obs.numpy(); zz = f.z_obs.numpy(); t = float(f.t)
        P.append(np.column_stack([x, y, np.full_like(x, t)])); z.append(zz)
    P = np.vstack(P); z = np.concatenate(z)
    if len(z) > M_CAP:
        idx = np.random.default_rng(0).choice(len(z), M_CAP, replace=False)
        P, z = P[idx], z[idx]
    return P, z, frames


def loglik_atoms(kx, ky, a, P, z, depth, eye=None):
    """Differentiable grid-free GP marginal log-likelihood in EXACT rank-2M form.
    K_ij = Σ_m a_m cos(φ_i^m − φ_j^m) = Φ Φᵀ with Φ ∈ R^{m×2M} and
    φ_i^m = k_m·r_i − ω_m t_i, so (K+σ²I) is low-rank-plus-diagonal and the marginal
    likelihood costs O(m·M²) via Woodbury / matrix-determinant lemma instead of the
    O(m³) dense Cholesky. Mathematically identical to the dense form (same σ²=jit).
    kx,ky,a: (M,); P: (m,3) [x,y,t]; z: (m,). `eye` kept for signature compat."""
    k = torch.sqrt(kx**2 + ky**2 + 1e-12)
    om = torch.sqrt(G * k * torch.tanh(k * depth) + 1e-12)
    phi = (kx[:, None] * P[None, :, 0] + ky[:, None] * P[None, :, 1]
           - om[:, None] * P[None, :, 2])                        # (M, m)
    sa = torch.sqrt(torch.clamp(a, min=0.0) + 1e-12)             # (M,)
    Phi = torch.cat([sa[:, None] * torch.cos(phi),
                     sa[:, None] * torch.sin(phi)], 0).transpose(0, 1)   # (m, 2M)
    s2 = SIGMA**2 + 1e-4                                         # noise + diagonal jitter
    r = Phi.shape[1]
    # Inner 2M×2M solve in float64: I+ΦᵀΦ/s2 is theoretically SPD but the 1/s2≈2e3
    # scaling can push a float32 Cholesky pivot negative for high-energy atom sets.
    Phid = Phi.double(); eyed = torch.eye(r, device=Phi.device, dtype=torch.float64)
    G_in = eyed + (Phid.transpose(0, 1) @ Phid) / s2
    Ptz = Phid.transpose(0, 1) @ z.double()                     # (2M,)
    jit = 0.0
    for _ in range(6):                                          # jitter escalation (rare)
        try:
            Lc = torch.linalg.cholesky(G_in + jit * eyed); break
        except Exception:
            jit = 1e-9 if jit == 0 else jit * 10
    v = torch.cholesky_solve(Ptz[:, None], Lc)[:, 0]           # G_in⁻¹ Ptz
    zd = z.double()
    quad = (zd @ zd) / s2 - (Ptz @ v) / s2**2                  # zᵀ(K+σ²I)⁻¹z
    logdet = z.shape[0] * math.log(s2) + 2.0 * torch.log(torch.diag(Lc)).sum()  # log|K+σ²I|
    return (-0.5 * quad - 0.5 * logdet).float()


def dps_sample(model, stats, abar, betas, P, z, depth, guidance=0.3, n=1, seed=0):
    """DPS reverse diffusion with grid-free likelihood guidance."""
    torch.manual_seed(seed)
    alphas = 1 - betas
    Pt = torch.tensor(P, device=DEVICE, dtype=torch.float32)
    zt = torch.tensor(z, device=DEVICE, dtype=torch.float32)
    eye = torch.eye(len(z), device=DEVICE)
    x = torch.randn(n, M, 3, device=DEVICE)
    for t in reversed(range(T)):
        x = x.detach().requires_grad_(True)
        tt = torch.full((n,), t, device=DEVICE)
        eps = model(x, tt)
        ab = abar[t]
        x0 = ((x - (1 - ab).sqrt() * eps) / ab.sqrt()).clamp(-1, 1)  # bound decode
        # likelihood over batch (sum); guidance is per-sample grad
        ll = 0.0
        for b in range(n):
            kx, ky, a = denorm_atoms(x0[b], stats)
            a = torch.clamp(a, min=0.0)
            ll = ll + loglik_atoms(kx, ky, a, Pt, zt, depth, eye)
        grad = torch.autograd.grad(ll, x)[0]
        with torch.no_grad():
            mean = (x - betas[t] / (1 - ab).sqrt() * eps) / alphas[t].sqrt()
            x = mean + (betas[t].sqrt() * torch.randn_like(x) if t > 0 else 0)
            # DPS guidance: normalized per-sample step
            gnorm = grad.flatten(1).norm(dim=1).clamp(min=1e-8)
            x = x + guidance * grad / gnorm[:, None, None]
    return x.detach()


def atoms_to_field(x0, stats, P, z, depth, target, rel_thresh=0.03, max_comp=30):
    """Posterior atoms → rendered field at target. Returns (pred_field, nrmse)."""
    kx, ky, a = denorm_atoms(x0, stats)
    kx, ky, a = kx.cpu().numpy(), ky.cpu().numpy(), a.cpu().numpy()
    if a.max() <= 0:
        return None, 1.0
    keep = a > rel_thresh * a.max()
    kxi, kyi, ai = kx[keep], ky[keep], a[keep]
    if len(kxi) == 0:
        return None, 1.0
    if len(kxi) > max_comp:
        order = np.argsort(-ai)[:max_comp]
        kxi, kyi = kxi[order], kyi[order]
    obs = np.column_stack([P[:, 0], P[:, 1], P[:, 2], z]).astype(np.float32)
    A, phi = fit_amplitudes_phases(kxi, kyi, obs, depth, lambda_reg=0.5)
    pred = render_wave_field(kxi, kyi, A, phi, depth, target['gx'], target['gy'], target['t'])
    nr = float(np.sqrt(np.mean((pred - target['gt'])**2)) / max(target['Hs'], 1e-6))
    return pred, nr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--episodes', nargs='+', default=[
        'episode_BF4_5comp_seed0_uniform.h5', 'episode_BF4_15comp_seed0_uniform.h5',
        'episode_BF3_5comp_seed1_uniform.h5'])
    ap.add_argument('--guidance', type=float, default=0.3)
    ap.add_argument('--nsamp', type=int, default=4)   # posterior samples (UQ)
    ap.add_argument('--ckpt', default=CKPT)
    a = ap.parse_args()

    ck = torch.load(a.ckpt, map_location=DEVICE)
    stats = ck['stats']
    model = SetDenoiser(**ck.get('arch', {})).to(DEVICE)
    model.load_state_dict(ck['model']); model.eval()
    abar = ck['abar'].to(DEVICE); betas = ck['betas'].to(DEVICE)

    print(f"guidance={a.guidance}  nsamp={a.nsamp}")
    print(f"{'episode':34s} {'prior':>7} {'best':>7} {'persamp':>8} {'BMA':>7} {'spread':>7}")
    rows = []
    for ep in a.episodes:
        L = EpisodeDataLoader('data/' + ep)
        depth = L.depth; Hs = L.Hs_true
        gx = L.gt_grid_x.numpy(); gy = L.gt_grid_y.numpy()
        ti = 25
        P, z, frames = gather_window(L, ti)
        tgt = frames[ti + 1]
        target = {'gx': gx, 'gy': gy, 't': float(tgt.t), 'gt': tgt.gt_field.numpy(), 'Hs': Hs}
        # unconditional prior (no guidance) baseline
        Xp = dps_sample(model, stats, abar, betas, P, z, depth, guidance=0.0, n=2, seed=1)
        r_prior = np.mean([atoms_to_field(Xp[b], stats, P, z, depth, target)[1] for b in range(2)])
        # DPS posterior samples
        Xd = dps_sample(model, stats, abar, betas, P, z, depth, guidance=a.guidance,
                        n=a.nsamp, seed=0)
        fields, rs = [], []
        for b in range(a.nsamp):
            f, r = atoms_to_field(Xd[b], stats, P, z, depth, target)
            rs.append(r)
            if f is not None:
                fields.append(f)
        # Bayesian model averaging: nRMSE of the posterior-mean field
        bma = np.mean(fields, axis=0) if fields else None
        r_bma = (float(np.sqrt(np.mean((bma - target['gt'])**2)) / max(Hs, 1e-6))
                 if bma is not None else 1.0)
        print(f"{ep:34s} {r_prior:7.4f} {min(rs):7.4f} {np.mean(rs):8.4f} "
              f"{r_bma:7.4f} {np.std(rs):7.4f}")
        rows.append(dict(ep=ep, prior=float(r_prior), dps_best=float(min(rs)),
                         dps_mean=float(np.mean(rs)), dps_bma=r_bma, dps_std=float(np.std(rs)),
                         all=rs))

    summary = dict(guidance=a.guidance, nsamp=a.nsamp, rows=rows,
                   mean_dps_best=float(np.mean([r['dps_best'] for r in rows])),
                   mean_dps_mean=float(np.mean([r['dps_mean'] for r in rows])),
                   mean_dps_bma=float(np.mean([r['dps_bma'] for r in rows])),
                   ref=dict(grid_diffusion=0.294, grid_oracle=0.282, exact=0.06,
                            WaveSplat_fused=0.205))
    json.dump(summary, open('results/direction3/posterior_eval.json', 'w'), indent=2)
    print(f"\nMEAN  best={summary['mean_dps_best']:.4f}  per-sample={summary['mean_dps_mean']:.4f}  "
          f"BMA={summary['mean_dps_bma']:.4f}")
    print("ref: grid-diffusion 0.294 / grid-oracle 0.282 / exact 0.06 / WaveSplat fused 0.205")


if __name__ == '__main__':
    main()
