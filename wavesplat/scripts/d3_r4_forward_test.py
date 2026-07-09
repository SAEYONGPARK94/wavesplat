"""R4 — is the HOS nonlinear gap a TRAINING-DATA problem or a FORWARD-MODEL problem?

Fable's key diagnostic. We recover free-wave components on real HOS windows (via the frozen
DPS prior + likelihood polish — a training-free-ish estimator) and RENDER the field two ways:
  (1) LINEAR:  η = Σ Aᵢ cos ψᵢ            (the estimator's current forward)
  (2) STOKES:  η = Σ Aᵢ cos ψᵢ + Σ cᵢ Aᵢ² cos 2ψᵢ   (2nd-order bound harmonics added)
Same recovered components, only the rendering forward differs. If STOKES ≪ LINEAR, the residual
is a FORWARD-MODEL limitation (bound waves unrepresentable by a linear render) → HOS training
data is the WRONG lever; fixing the forward is cheaper. If STOKES ≈ LINEAR, the gap is component
recovery / training distribution → HOS-augmented training is justified.

Run: python3 scripts/d3_r4_forward_test.py [--nsamp 4]
"""
import os, sys, json, time, argparse, math
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
os.chdir(os.path.join(os.path.dirname(__file__), '..'))

from d23_main_results import panel
from wavesplat.model.tnse import fit_amplitudes_phases, render_wave_field
import d44_hos_nonlinear as H
import d3_setdiff_posterior as DP
from d3_setdiff_prior import SetDenoiser, denorm_atoms
from d3_setdiff_posterior_b2 import polish_atoms
import nonlinear_forward as NF

DEVICE = DP.DEVICE; G = 9.81
SEAS = [('BF3', 0.6, 5.5), ('BF4', 1.2, 7.0), ('BF5', 2.0, 9.0)]
SEEDS = [0, 1, 2]
GX = np.linspace(10, 250, 80); GY = np.linspace(-100, 100, 60)
PRIOR = 'results/direction3/setdiff_prior.pt'


def components_from_atoms(xb, stats, Ps, zs, depth, rel_thresh=0.03, max_comp=30):
    kx, ky, a = denorm_atoms(xb, stats)
    kx, ky, a = kx.cpu().numpy(), ky.cpu().numpy(), a.cpu().numpy()
    if a.max() <= 0:
        return None
    keep = a > rel_thresh * a.max(); kx, ky, a = kx[keep], ky[keep], a[keep]
    if len(kx) == 0:
        return None
    if len(kx) > max_comp:
        o = np.argsort(-a)[:max_comp]; kx, ky = kx[o], ky[o]
    obs = np.column_stack([Ps[:, 0], Ps[:, 1], Ps[:, 2], zs]).astype(np.float32)
    A, phi = fit_amplitudes_phases(kx, ky, obs, depth, lambda_reg=0.5)
    return kx.astype(np.float32), ky.astype(np.float32), A.astype(np.float32), phi.astype(np.float32)


def render_stokes(kx, ky, A, phi, depth, gx, gy, t):
    k = np.sqrt(kx ** 2 + ky ** 2) + 1e-12
    om = np.sqrt(G * k * np.tanh(k * depth))
    GXm, GYm = np.meshgrid(gx, gy)                        # (ny,nx) — matches build_window gt
    ox, oy = GXm.ravel(), GYm.ravel(); ot = np.full(ox.shape, t)
    z = NF.elevation(kx, ky, A, om, phi, ox, oy, ot, depth, nonlinear=True)
    return z.reshape(GXm.shape)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--nsamp', type=int, default=4)
    ap.add_argument('--polish_iters', type=int, default=80)
    a = ap.parse_args()
    ck = torch.load(PRIOR, map_location=DEVICE); stats = ck['stats']
    prior = SetDenoiser(**ck.get('arch', {})).to(DEVICE); prior.load_state_dict(ck['model']); prior.eval()
    abar = ck['abar'].to(DEVICE); betas = ck['betas'].to(DEVICE)

    print(f"R4 forward-model test  nsamp={a.nsamp}", flush=True)
    print(f"{'window':10s} {'linear':>8} {'stokes':>8} {'Δ(lin-stk)':>11}", flush=True)
    rows = []; t0 = time.time()
    for tag, Hs, Tp in SEAS:
        n_back = int(round(2.2 * Tp / 0.1))
        for seed in SEEDS:
            fr = H.build_window(Hs, Tp, seed, n_back, GX, GY, M=3)
            i_star = len(fr) - 2
            gt = fr[i_star]['gt']; Hs_t = float(gt.std() * 4); t = fr[i_star]['t']; depth = H.DEPTH
            P = np.vstack([f['P'] for f in fr[:i_star + 1]]); z = np.concatenate([f['z'] for f in fr[:i_star + 1]])
            idx = np.random.default_rng(seed).choice(len(z), min(400, len(z)), replace=False)
            Ps, zs = P[idx].astype(np.float32), z[idx].astype(np.float32)
            Pt = torch.tensor(Ps, device=DEVICE); zt = torch.tensor(zs, device=DEVICE)
            Xd = DP.dps_sample(prior, stats, abar, betas, Ps, zs, depth, guidance=0.1, n=a.nsamp, seed=0)
            lin_f, stk_f = [], []
            for b in range(a.nsamp):
                xb = polish_atoms(Xd[b], stats, Pt, zt, depth, iters=a.polish_iters)
                comp = components_from_atoms(xb, stats, Ps, zs, depth)
                if comp is None:
                    continue
                kx, ky, A, phi = comp
                lin_f.append(render_wave_field(kx, ky, A, phi, depth, GX, GY, t))
                stk_f.append(render_stokes(kx, ky, A, phi, depth, GX, GY, t))
            if not lin_f:
                continue
            lin_bma = np.mean(lin_f, axis=0); stk_bma = np.mean(stk_f, axis=0)
            nr_lin = panel(lin_bma, gt, Hs_t)['nrmse']; nr_stk = panel(stk_bma, gt, Hs_t)['nrmse']
            rows.append((nr_lin, nr_stk))
            print(f"{tag+' s'+str(seed):10s} {nr_lin:8.3f} {nr_stk:8.3f} {nr_lin-nr_stk:+11.3f} "
                  f"({time.time()-t0:.0f}s)", flush=True)
    L = np.array([r[0] for r in rows]); S = np.array([r[1] for r in rows])
    print(f"\nMEAN  linear {L.mean():.3f}  stokes {S.mean():.3f}  Δ {L.mean()-S.mean():+.3f}", flush=True)
    print("verdict:", "FORWARD-MODEL limited (Stokes render closes gap → fix forward, not data)"
          if L.mean() - S.mean() > 0.02 else
          "component/training limited (Stokes render ~no help → HOS data justified)", flush=True)
    json.dump(dict(mean_linear=float(L.mean()), mean_stokes=float(S.mean()),
                   rows=[[float(x), float(y)] for x, y in rows]),
              open('results/direction3/r4_forward_test.json', 'w'), indent=2)


if __name__ == '__main__':
    main()
