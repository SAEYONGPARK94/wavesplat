"""
Direction III — improvement experiment: fix the over-smoothing (no GT, deployable).

BMA (posterior mean) over-smooths (VR 0.17-0.33). Two deployable sharpeners:
  (1) MAX-LIK sample: pick the posterior sample with highest DATA likelihood
      (phase-marginalized GP-marginal) — preserves amplitude, no GT used.
  (2) OFF-GRID refine: Gauss-Newton/Adam refine (kx,ky)+(cos,sin coeffs) of the
      max-lik sample's atoms on the observation window, then render.
Compare nRMSE / variance-ratio / Pearson against BMA and ST-DF-GP.

Run: python3 scripts/d8_sharpen.py
"""
import os, sys, json
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
os.chdir(os.path.join(os.path.dirname(__file__), '..'))

from wavesplat.data_loader import EpisodeDataLoader
from wavesplat.evaluation.metrics import variance_ratio, pearson_correlation, skill_score
import d3_setdiff_posterior as DP
from d3_setdiff_prior import SetDenoiser, denorm_atoms
from d6_integrated_eval import gather_W, W, TI

DEVICE = DP.DEVICE
G = 9.81
CKPT = 'results/direction3/setdiff_prior_v2.pt'
NSAMP = 12


def offgrid_refine(kx0, ky0, P, z, depth, target, iters=150, lr=0.02):
    """Refine (kx,ky)+linear coeffs on the observation window; render at target."""
    kx = torch.tensor(kx0, dtype=torch.float32, device=DEVICE, requires_grad=True)
    ky = torch.tensor(ky0, dtype=torch.float32, device=DEVICE, requires_grad=True)
    Pt = torch.tensor(P, dtype=torch.float32, device=DEVICE)
    zt = torch.tensor(z, dtype=torch.float32, device=DEVICE)
    N = len(kx0)
    c = torch.zeros(N, device=DEVICE, requires_grad=True)
    s = torch.zeros(N, device=DEVICE, requires_grad=True)
    opt = torch.optim.Adam([{'params': [kx, ky], 'lr': lr},
                            {'params': [c, s], 'lr': 0.05}])
    x, y, t = Pt[:, 0], Pt[:, 1], Pt[:, 2]
    for _ in range(iters):
        k = torch.sqrt(kx**2 + ky**2 + 1e-9)
        om = torch.sqrt(G * k * torch.tanh(k * depth) + 1e-9)
        psi = kx[None, :]*x[:, None] + ky[None, :]*y[:, None] - om[None, :]*t[:, None]
        pred = (c[None, :]*torch.cos(psi) + s[None, :]*torch.sin(psi)).sum(1)
        loss = ((pred - zt)**2).mean() + 1e-3*(c**2 + s**2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    # render at target grid/time
    with torch.no_grad():
        gx = torch.tensor(target['gx'], dtype=torch.float32, device=DEVICE)
        gy = torch.tensor(target['gy'], dtype=torch.float32, device=DEVICE)
        Xg, Yg = torch.meshgrid(gx, gy, indexing='xy')
        xf, yf = Xg.reshape(-1), Yg.reshape(-1)
        k = torch.sqrt(kx**2 + ky**2 + 1e-9)
        om = torch.sqrt(G * k * torch.tanh(k * depth) + 1e-9)
        psi = kx[None, :]*xf[:, None] + ky[None, :]*yf[:, None] - om[None, :]*float(target['t'])
        fld = (c[None, :]*torch.cos(psi) + s[None, :]*torch.sin(psi)).sum(1)
        return fld.reshape(Xg.shape).cpu().numpy()


def metrics(pred, gt, Hs):
    nr = float(np.sqrt(np.mean((pred - gt)**2)) / Hs)
    return nr, variance_ratio(pred, gt), pearson_correlation(pred, gt), skill_score(pred, gt)


def main():
    ck = torch.load(CKPT, map_location=DEVICE)
    stats = ck['stats']
    model = SetDenoiser(**ck.get('arch', {})).to(DEVICE)
    model.load_state_dict(ck['model']); model.eval()
    abar = ck['abar'].to(DEVICE); betas = ck['betas'].to(DEVICE)

    eps = [('episode_BF4_5comp_seed0_uniform.h5', 0.073),
           ('episode_BF4_30comp_seed0_uniform.h5', 0.287),
           ('episode_BF5_30comp_seed1_uniform.h5', 0.265)]
    print(f"{'episode':32s}{'method':>10} {'nRMSE':>7} {'VR':>6} {'Pear':>6} {'Skill':>7}")
    rows = []
    for ep, gp_ref in eps:
        L = EpisodeDataLoader('data/' + ep); Hs = L.Hs_true; depth = L.depth
        gx = L.gt_grid_x.numpy(); gy = L.gt_grid_y.numpy()
        frames = L.get_tracking_frames(2.0)
        P, z, _ = gather_W(L, TI, W)
        tgt = frames[TI + 1]
        target = {'gx': gx, 'gy': gy, 't': float(tgt.t), 'gt': tgt.gt_field.numpy(), 'Hs': Hs}
        gt = target['gt']
        Xd = DP.dps_sample(model, stats, abar, betas, P, z, depth, guidance=0.1, n=NSAMP, seed=0)
        # likelihood of each sample + fields
        Pt = torch.tensor(P, dtype=torch.float32, device=DEVICE)
        zt = torch.tensor(z, dtype=torch.float32, device=DEVICE)
        eye = torch.eye(len(z), device=DEVICE)
        fields, lls, atomsets = [], [], []
        for b in range(NSAMP):
            kx, ky, a = denorm_atoms(Xd[b], stats)
            f, _ = DP.atoms_to_field(Xd[b], stats, P, z, depth, target)
            ll = float(DP.loglik_atoms(kx, ky, torch.clamp(a, min=0), Pt, zt, depth, eye))
            fields.append(f); lls.append(ll)
            keep = (a > 0.03*a.max()).cpu().numpy()
            atomsets.append((kx.cpu().numpy()[keep], ky.cpu().numpy()[keep]))
        fields = [f if f is not None else np.zeros_like(gt) for f in fields]
        F = np.stack(fields)
        bma = F.mean(0)
        imax = int(np.argmax(lls))
        maxlik = fields[imax]
        kxr, kyr = atomsets[imax]
        refined = offgrid_refine(kxr, kyr, P, z, depth, target) if len(kxr) > 0 else maxlik
        for name, pred in [('BMA', bma), ('max-lik', maxlik), ('offgrid', refined)]:
            nr, vr, pe, sk = metrics(pred, gt, Hs)
            print(f"{ep:32s}{name:>10} {nr:7.3f} {vr:6.2f} {pe:6.2f} {sk:7.2f}")
            rows.append(dict(ep=ep, method=name, nrmse=nr, VR=vr, Pearson=pe, Skill=sk))
        print(f"{ep:32s}{'ST-DF-GP':>10} {gp_ref:7.3f}    (ref)")
    json.dump(rows, open('results/direction3/sharpen.json', 'w'), indent=2)
    print("\nBMA over-smooths (VR low). max-lik/offgrid should raise VR+Pearson if they help.")


if __name__ == '__main__':
    main()
