"""A1 step 1 — the learned set-proposal (cond-diffusion + polish) on the PAPER's
finemesh episodes, scored with the PAPER's normalizer (nRMSE = RMSE/Hs, metrics.py:32).

The old-mesh eval reported cond+polish 0.110 (RMSE/Hs) — same normalizer as the paper's
WaveSplat 0.199 (finemesh). This re-runs the learned proposal on the FINEMESH episodes so
the data version matches, and spans sparse (5-comp) → dense (30-comp) so the sparse regime
— where a discrete set-proposal must prove itself — is visible.

Reuses d3_beamform_cond_diffusion's own loop (sample → polish → atoms_to_field), only the
data path is redirected to data_finemesh_2026-07-23/. Nowcast at 3 frames per episode,
best-of-4 sample average, same as the shipped eval.
Caveat: this is nowcast (single-frame reconstruction), NOT the paper's sequential tracker;
a fully matched WS-vs-proposal run at identical frames is the follow-up.
"""
import argparse
import json

import numpy as np
import torch

import d3_beamform_cond_diffusion as cd
from wavesplat.data_loader import EpisodeDataLoader

FINE = 'data_finemesh_2026-07-23/'
EPISODES = [
    'episode_BF3_5comp_seed0_uniform.h5',
    'episode_BF3_5comp_seed1_uniform.h5',
    'episode_BF4_5comp_seed0_uniform.h5',
    'episode_BF4_15comp_seed0_uniform.h5',
    'episode_BF4_30comp_seed0_uniform.h5',
    'episode_BF5_15comp_seed0_uniform.h5',
    'episode_BF5_30comp_seed1_uniform.h5',
]


def eval_episode(model, stats, abar, betas, path, nsamp=4):
    L = EpisodeDataLoader(path)
    depth, Hs = L.depth, L.Hs_true
    gx, gy = L.gt_grid_x.numpy(), L.gt_grid_y.numpy()
    nb = int(round(cd.K_TP * L.Tp_true / 0.1))
    i0, i1 = nb + 3, len(L) - 2
    sel = np.linspace(i0, i1, 3).astype(int)
    nr_pol = []
    for i_star in sel:
        fstar = L[int(i_star)]
        P, z = cd.gather_frames(L, int(i_star), nb)
        if len(z) > cd.M_OBS:
            idx = np.random.default_rng(0).choice(len(z), cd.M_OBS, replace=False)
            P, z = P[idx], z[idx]
        t0 = P[:, 2].min()
        Ps = P.copy(); Ps[:, 2] -= t0
        Bmap = cd._bmap_from_obs(Ps, z)
        X = cd.sample(model, Bmap, abar, betas, n=nsamp)
        target = {'gx': gx, 'gy': gy, 't': float(fstar.t) - t0,
                  'gt': fstar.gt_field.numpy(), 'Hs': Hs}
        Pt = torch.tensor(Ps, device=cd.DEVICE, dtype=torch.float32)
        zt = torch.tensor(z, device=cd.DEVICE, dtype=torch.float32)
        fp = []
        for b in range(nsamp):
            xb = cd.polish_atoms(X[b], stats, Pt, zt, depth, iters=80)
            f2, _ = cd.atoms_to_field(xb, stats, Ps, z, depth, target)
            fp.append(f2)
        bma = np.mean([f for f in fp if f is not None], axis=0) \
            if any(f is not None for f in fp) else None
        nr_pol.append(float(np.sqrt(np.mean((bma - target['gt']) ** 2)) / max(Hs, 1e-6))
                      if bma is not None else 1.0)
    return float(np.mean(nr_pol))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', default='results/direction3/beamform_cond_diffusion.pt')
    a = ap.parse_args()
    stats = torch.load(cd.PRIOR_CKPT, map_location=cd.DEVICE)['stats']
    betas = cd.cosine_betas(cd.T).to(cd.DEVICE)
    abar = torch.cumprod(1 - betas, 0)
    model = cd.CondSetDenoiser().to(cd.DEVICE)
    model.load_state_dict(torch.load(a.ckpt, map_location=cd.DEVICE))
    model.eval()

    out = {}
    print(f"{'episode':40s} {'ncomp':>5s} {'cond+polish nRMSE/Hs':>20s}")
    for ep in EPISODES:
        nc = int(ep.split('comp')[0].split('_')[-1])
        val = eval_episode(model, stats, abar, betas, FINE + ep)
        out[ep] = {'ncomp': nc, 'cond_polish_nrmse_Hs': val}
        print(f"{ep:40s} {nc:5d} {val:20.3f}", flush=True)

    with open('tmp/G1_waveprior_power/a1_cond_finemesh.json', 'w') as f:
        json.dump(out, f, indent=1)
    vals = [v['cond_polish_nrmse_Hs'] for v in out.values()]
    print(f"\nMEAN cond+polish (finemesh, /Hs): {np.mean(vals):.3f}")
    print("paper WaveSplat (finemesh, /Hs): ~0.199 (100ep, tracked)  / 0.182 (12ep median)")
    print("by ncomp (sparse→dense):")
    for ep, v in out.items():
        print(f"  {v['ncomp']:2d}c  {v['cond_polish_nrmse_Hs']:.3f}  {ep}")


if __name__ == '__main__':
    main()
