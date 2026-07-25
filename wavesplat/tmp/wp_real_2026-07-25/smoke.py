"""STAGE-3b wiring smoke test (NOT the measurement). Validate:
  1. scene_real produces obs; kp within beamform k-range [-0.6,0.6].
  2. hand arm reproduces the doppler_real baseline (YS17 w0 = 0.741).
  3. learned arm (beamform-cond-diffusion + polish + atoms_to_field) runs on real obs,
     produces a nowcast field scored with the established /gt_std metric.
  4. Doppler-aware fit+render helper == atoms_to_field at U=0 (so the 2x2 row is
     internally consistent with the primary field back-end).
"""
import os, sys
os.environ.setdefault('OMP_NUM_THREADS', '1')
import warnings; warnings.filterwarnings('ignore')
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'scripts'))
import d3_compare_gifs as cg                 # chdir's to pkg dir
from d3_sa_nomp import mmse_bank_estimate
import d3_beamform_cond_diffusion as cd
from d3_setdiff_prior import denorm_atoms
from d3_beamform_net import beamform, KLIM, NK
from wavesplat.model.dispersion import G


def score_nowcast(sc, pred_now):
    gt = sc['maps_gt']
    sg = float(np.std(gt[np.isfinite(gt)]))
    g0 = gt[0]
    m = np.isfinite(g0) & np.isfinite(pred_now)
    return float(np.sqrt(np.mean((pred_now[m] - g0[m]) ** 2)) / sg)


def fit_render_doppler(kx, ky, Ps, z, depth, gx, gy, t_now, Uc=(0.0, 0.0),
                       lambda_reg=0.5):
    """LS-fit A,phi then render at (gx,gy,t_now), with omega = disp(k)+k.U.
    At Uc=(0,0) reproduces fit_amplitudes_phases + render_wave_field exactly."""
    kx = np.asarray(kx, float); ky = np.asarray(ky, float)
    N = len(kx)
    k = np.sqrt(kx ** 2 + ky ** 2 + 1e-12)
    om = np.sqrt(G * k * np.tanh(k * depth)) + kx * Uc[0] + ky * Uc[1]
    x, y, t = Ps[:, 0], Ps[:, 1], Ps[:, 2]
    psi = kx[None] * x[:, None] + ky[None] * y[:, None] - om[None] * t[:, None]
    Phi = np.concatenate([np.cos(psi), np.sin(psi)], axis=1)
    A_mat = Phi.T @ Phi + lambda_reg * np.eye(2 * N)
    coef = np.linalg.solve(A_mat, Phi.T @ z)
    a, b = coef[:N], coef[N:]
    A = np.sqrt(a ** 2 + b ** 2); phi = np.arctan2(-b, a)
    Xg, Yg = np.meshgrid(gx, gy)
    eta = np.zeros_like(Xg)
    for n in range(N):
        eta += A[n] * np.cos(kx[n] * Xg + ky[n] * Yg - om[n] * t_now + phi[n])
    return eta


def select_atoms(xb, stats, rel_thresh=0.03, max_comp=30):
    kx, ky, a = denorm_atoms(xb, stats)
    kx, ky, a = kx.cpu().numpy(), ky.cpu().numpy(), a.cpu().numpy()
    if a.max() <= 0:
        return None
    keep = a > rel_thresh * a.max()
    kxi, kyi, ai = kx[keep], ky[keep], a[keep]
    if len(kxi) > max_comp:
        order = np.argsort(-ai)[:max_comp]
        kxi, kyi = kxi[order], kyi[order]
    return kxi, kyi


stats = torch.load(cd.PRIOR_CKPT, map_location=cd.DEVICE)['stats']
betas = cd.cosine_betas(cd.T).to(cd.DEVICE); abar = torch.cumprod(1 - betas, 0)
model = cd.CondSetDenoiser().to(cd.DEVICE)
model.load_state_dict(torch.load('results/direction3/beamform_cond_diffusion.pt',
                                 map_location=cd.DEVICE))
model.eval()

for rec, win, ref in [('YS17', 0, 0.741), ('AA14', 0, 2.309)]:
    sc = cg.scene_real(rec, win=win)
    kp = sc['kp']
    print(f"\n=== {rec} w{win} ===  n_obs={len(sc['z'])}  Hs={sc['Hs']:.3f}  "
          f"kp={kp:.4f} (lambda_p={2*np.pi/max(kp,1e-6):.1f}m)  "
          f"npx={sc['npx']} nl={len(sc['leads'])}")
    # hand arm
    s0 = 0.05 * sc['Hs']
    fq = mmse_bank_estimate(sc['P'], sc['z'], sc['depth'], sc['Pq'], sigma0=s0,
                            kp_hint=kp, seed=sc['seed'])
    npx = sc['npx']; nl = len(sc['leads'])
    hand_now = score_nowcast(sc, fq[:npx * nl].reshape(nl, *sc['Xq'].shape)[0])
    print(f"  HAND nowcast nRMSE = {hand_now:.3f}  (doppler ref {ref})  "
          f"{'OK' if abs(hand_now-ref)<0.01 else 'MISMATCH'}")

    # learned arm
    P = sc['P'].astype(np.float32); z = sc['z'].astype(np.float32)
    if len(z) > cd.M_OBS:
        idx = np.random.default_rng(0).choice(len(z), cd.M_OBS, replace=False)
        P, z = P[idx], z[idx]
    t0 = P[:, 2].min(); Ps = P.copy(); Ps[:, 2] -= t0
    Bmap = cd._bmap_from_obs(Ps, z)
    # beamform peak diagnostic
    bm = Bmap[0].cpu().numpy()
    kax = np.linspace(-KLIM, KLIM, NK)
    ij = np.unravel_index(np.argmax(bm), bm.shape)
    print(f"  beamform map {bm.shape} peak at kx~{kax[ij[1]]:+.3f} ky~{kax[ij[0]]:+.3f} "
          f"(edge={'YES' if (ij[0] in (0,NK-1) or ij[1] in (0,NK-1)) else 'no'})")
    t_now = float(sc['Pq'][0, 2]) - t0
    target = {'gx': sc['gxq'], 'gy': sc['gyq'], 't': t_now,
              'gt': sc['maps_gt'][0], 'Hs': sc['Hs']}
    X = cd.sample(model, Bmap, abar, betas, n=4, seed=0)
    Pt = torch.tensor(Ps, device=cd.DEVICE, dtype=torch.float32)
    zt = torch.tensor(z, device=cd.DEVICE, dtype=torch.float32)
    preds, neffs = [], []
    diff_max = 0.0
    for b in range(4):
        xb = cd.polish_atoms(X[b], stats, Pt, zt, cd.DEPTH, iters=80)
        pred, _ = cd.atoms_to_field(xb, stats, Ps, z, sc['depth'], target)
        sel = select_atoms(xb, stats)
        if sel is not None:
            neffs.append(len(sel[0]))
            # doppler-helper U=0 vs atoms_to_field consistency
            eta0 = fit_render_doppler(sel[0], sel[1], Ps, z, sc['depth'],
                                      sc['gxq'], sc['gyq'], t_now, Uc=(0.0, 0.0))
            if pred is not None:
                diff_max = max(diff_max, float(np.nanmax(np.abs(eta0 - pred))))
        if pred is not None:
            preds.append(pred)
    bma = np.mean(preds, axis=0)
    learned_now = score_nowcast(sc, bma)
    print(f"  LEARNED nowcast nRMSE (BMA of {len(preds)}) = {learned_now:.3f}  "
          f"n_eff atoms ~ {np.mean(neffs):.1f}  "
          f"| doppler-helper(U=0) vs atoms_to_field max|diff|={diff_max:.2e}")
    print(f"  paired d(hand-learned) = {hand_now-learned_now:+.3f} "
          f"({'learned better' if learned_now<hand_now else 'hand better'})")
print("\nsmoke OK")
