"""
RELAX-style recursive sequential estimator — stable component add/prune + warm-start.

Fixes the d16/d17 instability: instead of grabbing a fixed 30 grid peaks, grow the
component set by matching pursuit on the residual (RELAX), prune weak components, and
warm-start across the growing window. Information then accumulates and the model order
self-adapts ⇒ near-monotonic and closer to oracle.

Per window T (warm-start from previous T):
  1. LS-fit (cos,sin) coeffs for current components → residual.
  2. while residual still improves & N<max: beamform residual → add strongest peak.
  3. off-grid refine all (kx,ky)+coeffs on the window.
  4. prune components with amplitude < frac of the max.
  5. reconstruct at t*.

Run: python3 scripts/d18_relax_estimator.py
"""
import os, sys, json
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
os.chdir(os.path.join(os.path.dirname(__file__), '..'))

from wavesplat.data_loader import EpisodeDataLoader
from wavesplat.model.tnse import fit_amplitudes_phases, render_wave_field
from wavesplat.evaluation.metrics import variance_ratio, pearson_correlation
from d6_integrated_eval import tag_to_file
from d15_sequential_fusion import gather_frames
from d17_recursive_estimator import refine_params

G = 9.81
DEV = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MCAP = 3000


def omega(k, d): return np.sqrt(G * k * np.tanh(k * d) + 1e-12)


def ls_fit(kx, ky, P, z, depth, lam=1e-2):
    """LS fit of (cos,sin) coeffs; return coeffs, residual, amplitudes."""
    k = np.sqrt(kx**2 + ky**2 + 1e-12); om = omega(k, depth)
    psi = (kx[None] * P[:, 0:1] + ky[None] * P[:, 1:2] - om[None] * P[:, 2:3])
    D = np.concatenate([np.cos(psi), np.sin(psi)], axis=1)        # (M, 2N)
    N = len(kx)
    A = D.T @ D + lam * np.eye(2 * N); b = D.T @ z
    coef = np.linalg.solve(A, b)
    resid = z - D @ coef
    amp = np.sqrt(coef[:N]**2 + coef[N:]**2)
    return coef, resid, amp


def beamform_top(P, r, depth, klim=0.6, n=80):
    """Strongest dispersion-constrained peak of residual r."""
    x = torch.tensor(P[:, 0], dtype=torch.float32, device=DEV)
    y = torch.tensor(P[:, 1], dtype=torch.float32, device=DEV)
    t = torch.tensor(P[:, 2], dtype=torch.float32, device=DEV)
    rt = torch.tensor(r, dtype=torch.float32, device=DEV)
    ax = torch.linspace(-klim, klim, n, device=DEV)
    KX, KY = torch.meshgrid(ax, ax, indexing='xy'); kx = KX.reshape(-1); ky = KY.reshape(-1)
    k = torch.sqrt(kx**2 + ky**2 + 1e-9); om = torch.sqrt(G * k * torch.tanh(k * depth) + 1e-9)
    pw = torch.empty(kx.shape[0], device=DEV); B = 2048
    for i in range(0, kx.shape[0], B):
        ps = kx[i:i+B, None]*x[None] + ky[i:i+B, None]*y[None] - om[i:i+B, None]*t[None]
        pw[i:i+B] = (rt[None]*torch.cos(ps)).sum(1)**2 + (rt[None]*torch.sin(ps)).sum(1)**2
    j = int(torch.argmax(pw)); return float(kx[j]), float(ky[j])


def relax_step(kx0, ky0, P, z, depth, max_comp=32, prune_frac=0.05):
    """Grow/refine/prune components on the window; warm-start from (kx0,ky0)."""
    if len(z) > MCAP:
        ii = np.random.default_rng(0).choice(len(z), MCAP, replace=False); P, z = P[ii], z[ii]
    kx = list(kx0); ky = list(ky0)
    if len(kx) == 0:
        bx, by = beamform_top(P, z, depth); kx, ky = [bx], [by]
    prev_rms = 1e9
    for _ in range(max_comp):
        _, resid, _ = ls_fit(np.array(kx), np.array(ky), P, z, depth)
        rms = np.sqrt(np.mean(resid**2))
        if rms > 0.985 * prev_rms or len(kx) >= max_comp:   # diminishing returns
            break
        prev_rms = rms
        bx, by = beamform_top(P, resid, depth); kx.append(bx); ky.append(by)
    kx, ky = np.array(kx), np.array(ky)
    # prune by amplitude
    _, _, amp = ls_fit(kx, ky, P, z, depth)
    keep = amp > prune_frac * amp.max()
    return kx[keep], ky[keep]


def met(f, target):
    return (float(np.sqrt(np.mean((f - target['gt'])**2)) / max(target['Hs'], 1e-6)),
            pearson_correlation(f, target['gt']), variance_ratio(f, target['gt']))


def main():
    tags = ['BF4_15c_s0', 'BF4_30c_s0', 'BF5_30c_s1']
    i_star = 250; T_list = [3.0, 6.0, 10.0, 15.0]
    print("RELAX recursive estimator — nRMSE (Pearson, VR, Ncomp) vs T:")
    rows = []
    for tag in tags:
        L = EpisodeDataLoader('data/' + tag_to_file(tag)); depth = L.depth; Hs = L.Hs_true
        wc = np.array(L.wave_components); tkx, tky = wc[:, 2], wc[:, 3]
        fstar = L[i_star]
        target = {'gx': L.gt_grid_x.numpy(), 'gy': L.gt_grid_y.numpy(),
                  't': float(fstar.t), 'gt': fstar.gt_field.numpy(), 'Hs': Hs}
        kx_s = np.zeros(0); ky_s = np.zeros(0); res = []
        for T in T_list:
            P, z = gather_frames(L, i_star, int(round(T / 0.1)))
            kx_s, ky_s = relax_step(kx_s, ky_s, P, z, depth)         # add/prune (warm-start)
            kx_s, ky_s, fld = refine_params(kx_s, ky_s, P, z, depth, target)  # off-grid refine
            nr, pe, vr = met(fld, target)
            # oracle for reference
            obs = np.column_stack([P[:, 0], P[:, 1], P[:, 2], z]).astype(np.float32)
            if len(obs) > MCAP: obs = obs[np.random.default_rng(1).choice(len(obs), MCAP, replace=False)]
            A, phi = fit_amplitudes_phases(tkx, tky, obs, depth, lambda_reg=0.5)
            fo = render_wave_field(tkx, tky, A, phi, depth, target['gx'], target['gy'], target['t'])
            onr = float(np.sqrt(np.mean((fo - target['gt'])**2)) / Hs)
            res.append(dict(T=T, nrmse=nr, P=pe, VR=float(vr), N=len(kx_s), oracle=onr))
            print(f"  {tag:12s} T={T:4.0f}  nRMSE {nr:.3f}  P {pe:.2f}  VR {vr:.2f}  N {len(kx_s):2d}  | oracle {onr:.3f}")
        rows.append(dict(tag=tag, res=res))
    json.dump(rows, open('results/direction3/relax_estimator.json', 'w'), indent=2)
    print("\nExpect ~monotonic nRMSE down + Pearson up + N self-adapting toward true order.")


if __name__ == '__main__':
    main()
