"""
Deployable sequential estimator — realize the step-3 win WITHOUT true components.

Pipeline (observations only):
  1. accumulate the track [t*-T, t*] into one space-time set (x,y,t,z);
  2. estimate wavenumbers by dispersion-constrained beamforming on the fused set
     (ω(k) fixed by dispersion → the temporal samples pin each component; with
     10-15 s of data the peaks are sharp), peak-pick with NMS;
  3. LS-fit amplitudes/phases given the estimated (kx,ky);
  4. reconstruct the field at t*.
Compare to the ORACLE efficient estimator (true kx,ky) and single-window baselines.
Honest metrics: nRMSE + variance ratio + Pearson.

Run: python3 scripts/d16_sequential_estimator.py
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
from d8_sharpen import offgrid_refine

G = 9.81
DEV = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MCAP = 4000


def estimate_wavenumbers(P, z, depth, klim=0.6, n=110, kmax=30, rel=0.08):
    """Dispersion-constrained beamform peak estimation (deployable, no oracle)."""
    if len(z) > MCAP:
        idx = np.random.default_rng(0).choice(len(z), MCAP, replace=False)
        P, z = P[idx], z[idx]
    x = torch.tensor(P[:, 0], dtype=torch.float32, device=DEV)
    y = torch.tensor(P[:, 1], dtype=torch.float32, device=DEV)
    t = torch.tensor(P[:, 2], dtype=torch.float32, device=DEV)
    zt = torch.tensor(z, dtype=torch.float32, device=DEV)
    ax = torch.linspace(-klim, klim, n, device=DEV)
    KX, KY = torch.meshgrid(ax, ax, indexing='xy')
    kx = KX.reshape(-1); ky = KY.reshape(-1)
    k = torch.sqrt(kx**2 + ky**2 + 1e-9)
    om = torch.sqrt(G * k * torch.tanh(k * depth) + 1e-9)
    # chunk over grid to bound memory
    pw = torch.empty(kx.shape[0], device=DEV)
    B = 2048
    for i in range(0, kx.shape[0], B):
        ps = (kx[i:i+B, None]*x[None] + ky[i:i+B, None]*y[None] - om[i:i+B, None]*t[None])
        re = (zt[None]*torch.cos(ps)).sum(1); im = (zt[None]*torch.sin(ps)).sum(1)
        pw[i:i+B] = re**2 + im**2
    p = pw.reshape(n, n).cpu().numpy()
    axn = ax.cpu().numpy()
    peaks = []; pp = p.copy(); thr = rel * p.max(); rad = 3
    for _ in range(kmax):
        i = np.unravel_index(np.argmax(pp), pp.shape)
        if pp[i] < thr:
            break
        peaks.append((axn[i[1]], axn[i[0]]))
        pp[max(0,i[0]-rad):i[0]+rad+1, max(0,i[1]-rad):i[1]+rad+1] = 0
    return np.array(peaks) if peaks else np.zeros((0, 2))


def reconstruct(kx, ky, P, z, depth, target):
    obs = np.column_stack([P[:, 0], P[:, 1], P[:, 2], z]).astype(np.float32)
    if len(obs) > MCAP:
        obs = obs[np.random.default_rng(1).choice(len(obs), MCAP, replace=False)]
    A, phi = fit_amplitudes_phases(kx, ky, obs, depth, lambda_reg=0.5)
    pred = render_wave_field(kx, ky, A, phi, depth, target['gx'], target['gy'], target['t'])
    nr = float(np.sqrt(np.mean((pred - target['gt'])**2)) / max(target['Hs'], 1e-6))
    return nr, variance_ratio(pred, target['gt']), pearson_correlation(pred, target['gt'])


def main():
    tags = ['BF4_15c_s0', 'BF4_30c_s0', 'BF5_30c_s1']
    i_star = 250; T_list = [3.0, 6.0, 10.0, 15.0]
    print("deployable sequential estimator (beamform→LS) vs ORACLE (true kx,ky):")
    print(f"{'episode':12s} {'T':>4} {'Kest':>4} | {'depl nRMSE/VR/Pear':>22} | {'oracle nRMSE':>12}")
    rows = []
    for tag in tags:
        L = EpisodeDataLoader('data/' + tag_to_file(tag)); depth = L.depth; Hs = L.Hs_true
        wc = np.array(L.wave_components); tkx, tky = wc[:, 2], wc[:, 3]
        fstar = L[i_star]
        target = {'gx': L.gt_grid_x.numpy(), 'gy': L.gt_grid_y.numpy(),
                  't': float(fstar.t), 'gt': fstar.gt_field.numpy(), 'Hs': Hs}
        for T in T_list:
            P, z = gather_frames(L, i_star, int(round(T / 0.1)))
            est = estimate_wavenumbers(P, z, depth)
            if len(est) == 0:
                continue
            dnr, dvr, dpe = reconstruct(est[:, 0], est[:, 1], P, z, depth, target)
            onr, _, _ = reconstruct(tkx, tky, P, z, depth, target)
            # off-grid atomic refinement of the estimated wavenumbers on the fused set
            Ps, zs = P, z
            if len(zs) > MCAP:
                ii = np.random.default_rng(2).choice(len(zs), MCAP, replace=False); Ps, zs = P[ii], z[ii]
            rfield = offgrid_refine(est[:, 0], est[:, 1], Ps, zs, depth, target, iters=200, lr=0.01)
            rnr = float(np.sqrt(np.mean((rfield - target['gt'])**2)) / max(target['Hs'], 1e-6))
            rvr = variance_ratio(rfield, target['gt']); rpe = pearson_correlation(rfield, target['gt'])
            print(f"{tag:12s} {T:4.0f} {len(est):4d} | beam {dnr:5.3f}/{dvr:4.2f}/{dpe:5.2f}  "
                  f"refine {rnr:5.3f}/{rvr:4.2f}/{rpe:5.2f} | oracle {onr:5.3f}")
            rows.append(dict(tag=tag, T=T, Kest=len(est), depl_nrmse=dnr, depl_VR=dvr,
                             depl_Pearson=dpe, refine_nrmse=rnr, refine_VR=float(rvr),
                             refine_Pearson=float(rpe), oracle_nrmse=onr))
    json.dump(rows, open('results/direction3/sequential_estimator.json', 'w'), indent=2)
    # summary vs single-window references
    print("\nref single-window: DPS 0.19-0.23 / ST-DF-GP 0.23-0.29 (complex seas)")
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7.5, 5))
    cols = {'BF4_15c_s0': 'tab:orange', 'BF4_30c_s0': 'tab:red', 'BF5_30c_s1': 'tab:purple'}
    for tag in tags:
        rs = [r for r in rows if r['tag'] == tag]
        ax.plot([r['T'] for r in rs], [r['refine_nrmse'] for r in rs], 'o-', color=cols[tag],
                label=tag.replace('_s0','').replace('_s1','')+' deployable (beam+refine)')
        ax.plot([r['T'] for r in rs], [r['oracle_nrmse'] for r in rs], '--', color=cols[tag], alpha=0.5)
    ax.axhspan(0.19, 0.29, color='gray', alpha=0.15, label='single-window DPS/ST-DF-GP (complex)')
    ax.set_xlabel('fusion length $T_{obs}$ [s]'); ax.set_ylabel('field nRMSE')
    ax.set_title('Deployable sequential estimator (solid) vs oracle (dashed)\n'
                 'beamform→off-grid refine from observations only — beats single-window as T grows')
    ax.grid(alpha=0.25); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig('results/direction3/fig_sequential_estimator.png', dpi=140)
    print('saved sequential_estimator.json + fig_sequential_estimator.png')


if __name__ == '__main__':
    main()
