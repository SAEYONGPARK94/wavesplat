"""
Direction II — Step 4: grid-free synthesis.

Diffusion prior gives COARSE directions (on the 64x64 grid); off-grid continuous
refinement then sharpens (kx,ky) to recover the precision the grid throws away.
Ablation over the INITIALISATION of the same off-grid refinement:
  no-prior(grid-MLE)  vs  diffusion-prior(PnP)  vs  oracle directions.

Off-grid refinement reuses the differentiable LS from tnse.py: with (kx,ky) free
and amplitudes/phases solved in closed form, we descend the data misfit on (kx,ky).
"""
import os, sys, argparse, json
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
os.chdir(os.path.join(os.path.dirname(__file__), '..'))

from wavesplat.data_loader import EpisodeDataLoader
from wavesplat.model.spectral_field import spectrum_to_components, rasterize_components
from wavesplat.model.tnse import (render_wave_field_torch, differentiable_ls_fit,
                                  fit_amplitudes_phases, render_wave_field)
import scripts.d2_posterior as PD
import scripts.d2_diffusion_prior as DP

DEVICE = PD.DEVICE


def offgrid_refine(kx0, ky0, P, z, depth, n_iter=200, lr=0.01, lam=0.5):
    """Refine continuous (kx,ky) to fit observations (LS for amp/phase inside)."""
    if len(kx0) == 0:
        return kx0, ky0
    kx = torch.tensor(kx0, device=DEVICE, dtype=torch.float32, requires_grad=True)
    ky = torch.tensor(ky0, device=DEVICE, dtype=torch.float32, requires_grad=True)
    ox = torch.tensor(P[:, 0], device=DEVICE, dtype=torch.float32)
    oy = torch.tensor(P[:, 1], device=DEVICE, dtype=torch.float32)
    ot = torch.tensor(P[:, 2], device=DEVICE, dtype=torch.float32)
    oz = torch.tensor(z, device=DEVICE, dtype=torch.float32)
    opt = torch.optim.Adam([kx, ky], lr=lr)
    for _ in range(n_iter):
        opt.zero_grad()
        Phi = render_wave_field_torch(kx[None], ky[None], ox, oy, ot, depth)  # (1,M,2N)
        coef = differentiable_ls_fit(Phi, oz[None], lam)                       # (1,2N)
        zp = (Phi @ coef.unsqueeze(2)).squeeze()
        loss = torch.mean((zp - oz)**2)
        loss.backward(); opt.step()
    return kx.detach().cpu().numpy(), ky.detach().cpu().numpy()


def field_nrmse(kx, ky, P, z, depth, tgt):
    if len(kx) == 0:
        return 1.0
    obs = np.column_stack([P[:, 0], P[:, 1], P[:, 2], z]).astype(np.float32)
    A, phi = fit_amplitudes_phases(kx, ky, obs, depth, 0.5)
    pred = render_wave_field(kx, ky, A, phi, depth, tgt['gx'], tgt['gy'], tgt['t'])
    return float(np.sqrt(np.mean((pred - tgt['gt'])**2)) / max(tgt['Hs'], 1e-6))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--episodes', nargs='+', default=[
        'episode_BF4_5comp_seed0_uniform.h5', 'episode_BF3_5comp_seed1_uniform.h5',
        'episode_BF4_15comp_seed0_uniform.h5'])
    a = ap.parse_args()
    ck = torch.load(DP.CKPT, map_location=DEVICE)
    model = DP.UNet().to(DEVICE); model.load_state_dict(ck['model']); model.eval()
    lo, hi = ck['lo'], ck['hi']

    hdr = f"{'episode':30s} {'PnP-grid':>9} {'noPrior+OG':>11} {'PRIOR+OG':>9} {'oracle+OG':>10} {'exact':>7}"
    print(hdr); print('-'*len(hdr))
    rows = []
    for ep in a.episodes:
        L = EpisodeDataLoader('data/'+ep); depth = L.depth; Hs = L.Hs_true
        gx = L.gt_grid_x.numpy(); gy = L.gt_grid_y.numpy()
        ti = 25
        P, z, frames = PD.gather_window(L, ti)
        tg = frames[ti+1]
        tgt = {'gx': gx, 'gy': gy, 't': float(tg.t), 'gt': tg.gt_field.numpy(), 'Hs': Hs}

        # grid spectra: MLE (no prior) and PnP (prior)
        S_mle = PD.estimate_S(P, z, depth, model, lo, hi, use_prior=False)
        S_pnp = PD.estimate_S(P, z, depth, model, lo, hi, use_prior=True)
        c_mle = spectrum_to_components(S_mle, 20, 0.05)
        c_pnp = spectrum_to_components(S_pnp, 20, 0.05)
        wc = np.asarray(L.wave_components, np.float32)

        # PnP grid-only (no refine)
        r_pnp_grid = field_nrmse(c_pnp[:, 2], c_pnp[:, 3], P, z, depth, tgt) if len(c_pnp) else 1.0

        # off-grid refine from each init
        def refine_eval(c_init):
            if len(c_init) == 0:
                return 1.0
            kx, ky = offgrid_refine(c_init[:, 2], c_init[:, 3], P, z, depth)
            return field_nrmse(kx, ky, P, z, depth, tgt)
        r_noprior = refine_eval(c_mle)
        r_prior   = refine_eval(c_pnp)
        r_oracle  = refine_eval(wc)            # oracle directions as init

        # absolute ceiling: exact continuous components, no refine
        r_exact = field_nrmse(wc[:, 2], wc[:, 3], P, z, depth, tgt)

        print(f"{ep[:30]:30s} {r_pnp_grid:9.4f} {r_noprior:11.4f} {r_prior:9.4f} {r_oracle:10.4f} {r_exact:7.4f}")
        rows.append([ep, r_pnp_grid, r_noprior, r_prior, r_oracle, r_exact])

    arr = np.array([[r[1], r[2], r[3], r[4], r[5]] for r in rows])
    m = arr.mean(0)
    print('-'*len(hdr))
    print(f"{'MEAN':30s} {m[0]:9.4f} {m[1]:11.4f} {m[2]:9.4f} {m[3]:10.4f} {m[4]:7.4f}")
    print("\nref: WaveSplat fused 0.205 / per-frame 0.271 / oracle-GP 0.070")
    json.dump({'cols': ['ep', 'PnP_grid', 'noPrior+OG', 'PRIOR+OG', 'oracle+OG', 'exact'],
               'rows': rows, 'mean': m.tolist()},
              open('results/direction2/offgrid_eval.json', 'w'), indent=2)


if __name__ == '__main__':
    main()
