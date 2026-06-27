"""
Why does the deployable estimator's RMSE rise with T sometimes? — and the fix.

Diagnosis: d16 re-estimates wavenumbers FROM SCRATCH at each T (beamform peaks +
non-convex off-grid refine). That extraction is unstable across windows (different
peaks / local optima / model order), so RMSE is non-monotonic even though the
information (oracle CRB) grows monotonically.

Fix: a truly RECURSIVE estimator — estimate once on the first window, then as the
window grows only WARM-START refine the existing components on the larger set.
Information then only accumulates ⇒ monotonic improvement. We compare
'from-scratch' (d16) vs 'recursive warm-start' vs oracle.

Run: python3 scripts/d17_recursive_estimator.py
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
from d16_sequential_estimator import estimate_wavenumbers, MCAP

G = 9.81
DEV = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def refine_params(kx0, ky0, P, z, depth, target, iters=200, lr=0.01):
    """Off-grid refine (kx,ky)+(cos,sin) on (P,z); return refined kx,ky and field."""
    if len(z) > MCAP:
        ii = np.random.default_rng(2).choice(len(z), MCAP, replace=False); P, z = P[ii], z[ii]
    kx = torch.tensor(kx0, dtype=torch.float32, device=DEV, requires_grad=True)
    ky = torch.tensor(ky0, dtype=torch.float32, device=DEV, requires_grad=True)
    N = len(kx0)
    c = torch.zeros(N, device=DEV, requires_grad=True)
    s = torch.zeros(N, device=DEV, requires_grad=True)
    Pt = torch.tensor(P, dtype=torch.float32, device=DEV); zt = torch.tensor(z, dtype=torch.float32, device=DEV)
    opt = torch.optim.Adam([{'params': [kx, ky], 'lr': lr}, {'params': [c, s], 'lr': 0.05}])
    x, y, t = Pt[:, 0], Pt[:, 1], Pt[:, 2]
    for _ in range(iters):
        k = torch.sqrt(kx**2 + ky**2 + 1e-9); om = torch.sqrt(G*k*torch.tanh(k*depth)+1e-9)
        psi = kx[None]*x[:, None] + ky[None]*y[:, None] - om[None]*t[:, None]
        pred = (c[None]*torch.cos(psi) + s[None]*torch.sin(psi)).sum(1)
        loss = ((pred-zt)**2).mean() + 1e-3*(c**2+s**2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        gx = torch.tensor(target['gx'], dtype=torch.float32, device=DEV)
        gy = torch.tensor(target['gy'], dtype=torch.float32, device=DEV)
        Xg, Yg = torch.meshgrid(gx, gy, indexing='xy'); xf, yf = Xg.reshape(-1), Yg.reshape(-1)
        k = torch.sqrt(kx**2+ky**2+1e-9); om = torch.sqrt(G*k*torch.tanh(k*depth)+1e-9)
        psi = kx[None]*xf[:, None] + ky[None]*yf[:, None] - om[None]*float(target['t'])
        fld = (c[None]*torch.cos(psi)+s[None]*torch.sin(psi)).sum(1).reshape(Xg.shape).cpu().numpy()
    return kx.detach().cpu().numpy(), ky.detach().cpu().numpy(), fld


def met(f, target):
    nr = float(np.sqrt(np.mean((f-target['gt'])**2))/max(target['Hs'],1e-6))
    return nr, pearson_correlation(f, target['gt'])


def main():
    tags = ['BF4_15c_s0', 'BF4_30c_s0', 'BF5_30c_s1']
    i_star = 250; T_list = [3.0, 6.0, 10.0, 15.0]
    print("recursive (warm-start) vs from-scratch vs oracle — nRMSE (Pearson):")
    print(f"{'episode':12s} {'metric':>10}  " + "  ".join(f"T={t:.0f}" for t in T_list))
    rows = []
    for tag in tags:
        L = EpisodeDataLoader('data/'+tag_to_file(tag)); depth = L.depth; Hs = L.Hs_true
        wc = np.array(L.wave_components); tkx, tky = wc[:, 2], wc[:, 3]
        fstar = L[i_star]
        target = {'gx': L.gt_grid_x.numpy(), 'gy': L.gt_grid_y.numpy(),
                  't': float(fstar.t), 'gt': fstar.gt_field.numpy(), 'Hs': Hs}
        rec, scr, ora, recP = [], [], [], []
        kx_state = ky_state = None
        for T in T_list:
            P, z = gather_frames(L, i_star, int(round(T/0.1)))
            # recursive: init from previous state (warm-start); seed once at first T
            if kx_state is None:
                est = estimate_wavenumbers(P, z, depth); kx_state, ky_state = est[:, 0], est[:, 1]
            kx_state, ky_state, fr = refine_params(kx_state, ky_state, P, z, depth, target)
            nr, pe = met(fr, target); rec.append(nr); recP.append(pe)
            # from-scratch (re-estimate each T) for contrast
            est2 = estimate_wavenumbers(P, z, depth)
            _, _, fs = refine_params(est2[:, 0], est2[:, 1], P, z, depth, target)
            scr.append(met(fs, target)[0])
            # oracle
            obs = np.column_stack([P[:, 0], P[:, 1], P[:, 2], z]).astype(np.float32)
            if len(obs) > MCAP: obs = obs[np.random.default_rng(1).choice(len(obs), MCAP, replace=False)]
            A, phi = fit_amplitudes_phases(tkx, tky, obs, depth, lambda_reg=0.5)
            fo = render_wave_field(tkx, tky, A, phi, depth, target['gx'], target['gy'], target['t'])
            ora.append(met(fo, target)[0])
        print(f"{tag:12s} {'recursive':>10}  " + "  ".join(f"{v:.3f}" for v in rec))
        print(f"{tag:12s} {'(Pearson)':>10}  " + "  ".join(f"{v:.3f}" for v in recP))
        print(f"{tag:12s} {'scratch':>10}  " + "  ".join(f"{v:.3f}" for v in scr))
        print(f"{tag:12s} {'oracle':>10}  " + "  ".join(f"{v:.3f}" for v in ora))
        rows.append(dict(tag=tag, T=T_list, recursive=rec, recursive_P=recP, scratch=scr, oracle=ora))
    json.dump(rows, open('results/direction3/recursive_estimator.json', 'w'), indent=2)
    print("\nrecursive should be ~monotonic (info only accumulates); scratch is the noisy one.")


if __name__ == '__main__':
    main()
