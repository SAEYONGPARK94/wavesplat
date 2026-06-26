"""
Direction II — Step 3: physics-constrained posterior for S(kx,ky).

Plug-and-play: alternate (a) GP-marginal-likelihood gradient ascent on S, and
(b) diffusion-prior projection (one-step denoise). Compares direction/field
recovery WITH prior (PnP) vs WITHOUT prior (MLE) vs oracle vs WaveSplat.

Likelihood (phases marginalised): log N(z; 0, K_S + σ²I),
  K_S(X,X) = Σ_grid S(k)·cos(k·Δx − ω(k)·Δt).
"""
import os, sys, argparse, time
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
os.chdir(os.path.join(os.path.dirname(__file__), '..'))

from wavesplat.data_loader import EpisodeDataLoader
from wavesplat.model.spectral_field import (KX_GRID, KY_GRID, omega_grid, N_GRID,
                                            spectrum_to_components)
from wavesplat.model.tnse import fit_amplitudes_phases, render_wave_field
import scripts.d2_diffusion_prior as DP

DEVICE = DP.DEVICE
SIGMA = 0.02
W = 10
M_CAP = 150


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


def build_cos_basis(P, depth):
    """C[g,i,j] = cos(kx_g Δx + ky_g Δy − ω_g Δt). (G, M, M) on device."""
    kx = torch.tensor(KX_GRID.ravel(), device=DEVICE, dtype=torch.float32)
    ky = torch.tensor(KY_GRID.ravel(), device=DEVICE, dtype=torch.float32)
    om = torch.tensor(omega_grid(depth).ravel(), device=DEVICE, dtype=torch.float32)
    Pt = torch.tensor(P, device=DEVICE, dtype=torch.float32)
    dx = Pt[:, 0:1] - Pt[None, :, 0]
    dy = Pt[:, 1:2] - Pt[None, :, 1]
    dt = Pt[:, 2:3] - Pt[None, :, 2]
    phase = (kx[:, None, None]*dx[None] + ky[:, None, None]*dy[None] - om[:, None, None]*dt[None])
    return torch.cos(phase)   # (G,M,M)


def loglik(S_flat, C, z, eye):
    """Differentiable GP marginal log-likelihood. S_flat:(G,) >=0."""
    K = torch.einsum('g,gij->ij', S_flat, C)
    A = K + (SIGMA**2 + 1e-5) * eye
    L = torch.linalg.cholesky(A)
    alpha = torch.cholesky_solve(z[:, None], L)[:, 0]
    quad = -0.5 * (z @ alpha)
    logdet = -torch.log(torch.diag(L)).sum()
    return quad + logdet


@torch.no_grad()
def prior_project(model, x_img, t_d):
    """One-step denoise toward prior manifold at noise level t_d."""
    ab = DP.AB[t_d]
    xt = ab.sqrt()*x_img + (1-ab).sqrt()*torch.randn_like(x_img)
    eps = model(xt, torch.full((1,), t_d, device=DEVICE))
    x0 = (xt - (1-ab).sqrt()*eps) / ab.sqrt()
    return x0.clamp(-1, 1)


def estimate_S(P, z, depth, model, lo, hi, use_prior, n_iter=120, lr=0.15, t_d=120):
    C = build_cos_basis(P, depth)
    zt = torch.tensor(z, device=DEVICE, dtype=torch.float32)
    eye = torch.eye(len(z), device=DEVICE)
    # work in x-space (transformed); start near "empty"
    x = torch.full((1, 1, N_GRID, N_GRID), -1.0, device=DEVICE, requires_grad=True)
    opt = torch.optim.Adam([x], lr=lr)
    for it in range(n_iter):
        opt.zero_grad()
        S = torch_from_x(x[0, 0], lo, hi).reshape(-1)
        ll = loglik(S, C, zt, eye)
        (-ll).backward()
        opt.step()
        if use_prior and (it+1) % 15 == 0:
            with torch.no_grad():
                xp = prior_project(model, x.detach(), t_d)
                x.data = xp
    S = torch_from_x(x[0, 0].detach(), lo, hi).cpu().numpy()
    return np.maximum(S, 0.0)


def torch_from_x(x, lo, hi):
    xx = (x.clamp(-1, 1)+1)/2 * (hi - lo) + lo
    return torch.clamp(torch.exp(xx) - DP.EPS, min=0.0)


def eval_field(S, P, z, depth, target_frame):
    comps = spectrum_to_components(S, max_comp=30, rel_thresh=0.03)
    if len(comps) == 0:
        return 1.0
    kx, ky = comps[:, 2], comps[:, 3]
    obs = np.column_stack([P[:, 0], P[:, 1], P[:, 2], z]).astype(np.float32)
    A, phi = fit_amplitudes_phases(kx, ky, obs, depth, lambda_reg=0.5)
    gx = target_frame['gx']; gy = target_frame['gy']; t = target_frame['t']
    pred = render_wave_field(kx, ky, A, phi, depth, gx, gy, t)
    gt = target_frame['gt']
    return float(np.sqrt(np.mean((pred-gt)**2)) / max(target_frame['Hs'], 1e-6))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--episodes', nargs='+', default=[
        'episode_BF4_5comp_seed0_uniform.h5', 'episode_BF4_15comp_seed0_uniform.h5',
        'episode_BF3_5comp_seed1_uniform.h5'])
    a = ap.parse_args()
    ck = torch.load(DP.CKPT, map_location=DEVICE)
    model = DP.UNet().to(DEVICE); model.load_state_dict(ck['model']); model.eval()
    lo, hi = ck['lo'], ck['hi']

    print(f"{'episode':32s} {'MLE':>8} {'PnP':>8} {'oracle':>8}")
    rows = []
    for ep in a.episodes:
        L = EpisodeDataLoader('data/'+ep)
        depth = L.depth; Hs = L.Hs_true
        gx = L.gt_grid_x.numpy(); gy = L.gt_grid_y.numpy()
        ti = 25
        P, z, frames = gather_window(L, ti)
        tgt = frames[ti+1]
        target = {'gx': gx, 'gy': gy, 't': float(tgt.t), 'gt': tgt.gt_field.numpy(), 'Hs': Hs}

        S_mle = estimate_S(P, z, depth, model, lo, hi, use_prior=False)
        S_pnp = estimate_S(P, z, depth, model, lo, hi, use_prior=True)
        from wavesplat.model.spectral_field import rasterize_components
        S_or = rasterize_components(np.asarray(L.wave_components, np.float32))

        r_mle = eval_field(S_mle, P, z, depth, target)
        r_pnp = eval_field(S_pnp, P, z, depth, target)
        r_or  = eval_field(S_or,  P, z, depth, target)
        print(f"{ep:32s} {r_mle:8.4f} {r_pnp:8.4f} {r_or:8.4f}")
        rows.append((ep, r_mle, r_pnp, r_or))

    import json
    json.dump({'rows': rows, 'note': 'field nRMSE: MLE(no prior) vs PnP(diffusion prior) vs oracle-S',
               'ref': {'WaveSplat_perframe': 0.271, 'WaveSplat_fused': 0.205, 'oracle_GP': 0.070}},
              open('results/direction2/posterior_eval.json', 'w'), indent=2)
    arr = np.array([[r[1], r[2], r[3]] for r in rows])
    print(f"\nMEAN  MLE={arr[:,0].mean():.4f}  PnP={arr[:,1].mean():.4f}  oracle-S={arr[:,2].mean():.4f}")
    print("ref: WaveSplat perframe 0.271 / fused 0.205 / oracle-GP 0.070")


if __name__ == '__main__':
    main()
