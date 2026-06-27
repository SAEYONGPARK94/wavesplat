"""
Direction III×I pilot (b) — is the DPS posterior UQ calibrated, and does it match
the CRB predictable-zone?

The grid-free DPS gives a *distribution* of fields (one per posterior sample). We
ask two things:
 (1) Reliability — does the per-pixel posterior std σ_post predict the actual error
     |BMA − GT|? (binned calibration curve + ±1σ/±2σ coverage)
 (2) Physics agreement — does σ_post's spatial structure match the Direction-I CRB
     predictability map σ_z(x,y)/Hs computed from the SAME observation window with
     the true components? (map correlation)
If both hold, the DPS posterior is a *physics-consistent, calibrated* uncertainty —
the (C) contribution.

Run: python3 scripts/d4_uq_calibration.py
"""
import os, sys, json
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
os.chdir(os.path.join(os.path.dirname(__file__), '..'))

from wavesplat.data_loader import EpisodeDataLoader
import d3_setdiff_posterior as DP
from d3_setdiff_prior import SetDenoiser
import d1_crb as C1
from d1_predictable_zone import pointwise_field_crb

DEVICE = DP.DEVICE
EP = 'episode_BF4_5comp_seed0_uniform.h5'
NSAMP = 16
GUID = 0.1


def crb_map(L, P, target, depth):
    """Direction-I pointwise CRB σ_z/Hs over the grid at the target frame, from the
    same observation window, using the true components."""
    wc = np.array(L.wave_components)                       # [A,omega,kx,ky,phi,Q]
    theta = torch.tensor(np.stack([wc[:, 0], wc[:, 2], wc[:, 3], wc[:, 4]], 1),
                         dtype=C1.DT, device=C1.DEVICE)
    x = torch.tensor(P[:, 0], dtype=C1.DT, device=C1.DEVICE)
    y = torch.tensor(P[:, 1], dtype=C1.DT, device=C1.DEVICE)
    t = torch.tensor(P[:, 2], dtype=C1.DT, device=C1.DEVICE)
    F = C1.fisher(theta, x, y, t, depth, DP.SIGMA)
    n = F.shape[0]
    Finv = torch.linalg.inv(F + 1e-10 * torch.trace(F) / n * torch.eye(n, dtype=C1.DT, device=F.device))
    gx = torch.tensor(target['gx'], dtype=C1.DT, device=C1.DEVICE)
    gy = torch.tensor(target['gy'], dtype=C1.DT, device=C1.DEVICE)
    return pointwise_field_crb(theta, Finv, gx, gy, target['t'], depth, target['Hs'])


def main():
    ck = torch.load(DP.CKPT, map_location=DEVICE)
    stats = ck['stats']
    model = SetDenoiser().to(DEVICE); model.load_state_dict(ck['model']); model.eval()
    abar = ck['abar'].to(DEVICE); betas = ck['betas'].to(DEVICE)

    L = EpisodeDataLoader('data/' + EP)
    depth = L.depth; Hs = L.Hs_true
    gx = L.gt_grid_x.numpy(); gy = L.gt_grid_y.numpy()
    P, z, frames = DP.gather_window(L, 25)
    tgt = frames[26]
    target = {'gx': gx, 'gy': gy, 't': float(tgt.t), 'gt': tgt.gt_field.numpy(), 'Hs': Hs}
    gt = target['gt']

    # posterior ensemble of fields
    Xd = DP.dps_sample(model, stats, abar, betas, P, z, depth, guidance=GUID, n=NSAMP, seed=0)
    fields = []
    for b in range(NSAMP):
        f, _ = DP.atoms_to_field(Xd[b], stats, P, z, depth, target)
        if f is not None:
            fields.append(f)
    F = np.stack(fields)                                   # (S, Ny, Nx)
    bma = F.mean(0); sig_post = F.std(0)
    err = np.abs(bma - gt)
    bma_nrmse = float(np.sqrt(np.mean((bma - gt)**2)) / Hs)

    # (1) reliability: bin by sigma_post, RMS error per bin
    sp = sig_post.ravel() / Hs; er = err.ravel() / Hs
    q = np.quantile(sp, np.linspace(0, 1, 9))
    cx, cy = [], []
    for i in range(8):
        m = (sp >= q[i]) & (sp <= q[i+1])
        if m.sum() > 5:
            cx.append(sp[m].mean()); cy.append(np.sqrt((er[m]**2).mean()))
    corr_err = float(np.corrcoef(sp, er)[0, 1])
    cov1 = float((err <= sig_post).mean()); cov2 = float((err <= 2*sig_post).mean())
    # one-parameter recalibration: scale s so ±1σ coverage hits 0.68
    ratio = err / (sig_post + 1e-9)
    s_cal = float(np.quantile(ratio, 0.68))
    cov1_c = float((err <= s_cal * sig_post).mean())
    cov2_c = float((err <= 2 * s_cal * sig_post).mean())

    # (2) CRB predictable-zone map + spatial correlation with sigma_post
    crb = crb_map(L, P, target, depth)                     # (Ny,Nx) sigma_z/Hs
    crb_c = np.clip(crb, 0, 1)
    corr_map = float(np.corrcoef(np.log(sig_post.ravel()/Hs + 1e-6),
                                 np.log(crb_c.ravel() + 1e-6))[0, 1])

    print(f"BMA nRMSE={bma_nrmse:.4f}")
    print(f"(1) reliability: corr(sigma_post, |err|)={corr_err:.3f}  "
          f"coverage ±1σ={cov1:.2f} (ideal~0.68)  ±2σ={cov2:.2f} (ideal~0.95)")
    print(f"    after 1-param recalib (s={s_cal:.2f}): ±1σ={cov1_c:.2f}  ±2σ={cov2_c:.2f}")
    print(f"(2) CRB agreement: corr(log sigma_post, log CRB)={corr_map:.3f}")

    out = dict(ep=EP, nsamp=NSAMP, guidance=GUID, bma_nrmse=bma_nrmse,
               corr_err=corr_err, cov1=cov1, cov2=cov2, s_cal=s_cal,
               cov1_cal=cov1_c, cov2_cal=cov2_c, corr_map=corr_map,
               reliability=dict(x=[float(v) for v in cx], y=[float(v) for v in cy]))
    json.dump(out, open('results/direction3/uq_calibration.json', 'w'), indent=2)

    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    ext = [gx.min(), gx.max(), gy.min(), gy.max()]
    fig = plt.figure(figsize=(16, 8))
    foot = P[:, :2]
    def hm(ax, M, title, vmax=None, cmap='viridis'):
        im = ax.imshow(M, origin='lower', extent=ext, aspect='auto', cmap=cmap,
                       vmin=0, vmax=vmax); ax.set_title(title, fontsize=10)
        ax.scatter(foot[:, 0], foot[:, 1], s=1, c='w', alpha=0.4)
        fig.colorbar(im, ax=ax, fraction=0.025)
    ax = fig.add_subplot(2, 3, 1); hm(ax, bma, f'BMA field (nRMSE={bma_nrmse:.3f})', None, 'RdBu_r')
    ax = fig.add_subplot(2, 3, 2); hm(ax, sig_post/Hs, 'DPS posterior std σ_post/Hs', 0.4)
    ax = fig.add_subplot(2, 3, 3); hm(ax, err/Hs, 'actual error |BMA−GT|/Hs', 0.4)
    ax = fig.add_subplot(2, 3, 4); hm(ax, crb_c, 'Dir-I CRB σ_z/Hs (predictable-zone)', 0.4)
    ax = fig.add_subplot(2, 3, 5)
    ax.plot(cx, cy, 'o-', color='tab:purple'); mx = max(max(cx), max(cy)) if cx else 0.4
    ax.plot([0, mx], [0, mx], 'k--', alpha=0.5, label='ideal (calibrated)')
    ax.set_xlabel('posterior σ_post/Hs (binned)'); ax.set_ylabel('RMS actual error/Hs')
    ax.set_title(f'Reliability  corr={corr_err:.2f}, cov±1σ={cov1:.2f}'); ax.legend(fontsize=8); ax.grid(alpha=0.3)
    ax = fig.add_subplot(2, 3, 6)
    ax.scatter(crb_c.ravel(), sig_post.ravel()/Hs, s=2, alpha=0.2, c='teal')
    ax.set_xlabel('Dir-I CRB σ_z/Hs'); ax.set_ylabel('DPS σ_post/Hs')
    ax.set_title(f'DPS UQ vs CRB  (log-corr={corr_map:.2f})'); ax.grid(alpha=0.3)
    fig.suptitle('Pilot (b) — DPS posterior UQ: reliability + agreement with Dir-I CRB predictable-zone', fontsize=12)
    fig.tight_layout(); fig.savefig('results/direction3/fig_uq_calibration.png', dpi=140)
    print('saved results/direction3/uq_calibration.json + fig_uq_calibration.png')


if __name__ == '__main__':
    main()
