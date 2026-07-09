"""Comparison GIF: GT | RELAX | DPS posterior-mean | DPS uncertainty(std), animated
nowcast→forecast. Shows the regime story honestly — RELAX sharp on simple seas, DPS
competitive + calibrated UQ on complex seas (UQ is the thing RELAX cannot give).
Run: python3 tmp/compare_gif.py [--ckpt ...mixed_big.pt] [--guid 0.03]
"""
import os, sys, argparse, numpy as np, torch
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
sys.path.insert(0, 'scripts')
import d3_setdiff_posterior as DP
from d3_setdiff_prior import SetDenoiser, denorm_atoms
from wavesplat.data_loader import EpisodeDataLoader
from wavesplat.model.tnse import fit_amplitudes_phases, render_wave_field
from wavesplat.evaluation.metrics import pearson_correlation
from d6_integrated_eval import tag_to_file
from d15_sequential_fusion import gather_frames
from d28_forecast_complex import estimate_components, K_TP

ap = argparse.ArgumentParser()
ap.add_argument('--ckpt', default='results/direction3/setdiff_prior_mixed_big.pt')
ap.add_argument('--guid', type=float, default=0.03)
ap.add_argument('--nsamp', type=int, default=10)
ap.add_argument('--mcap', type=int, default=3000)
a = ap.parse_args()
OUT = 'results/direction3/gifs'; os.makedirs(OUT, exist_ok=True)

ck = torch.load(a.ckpt, map_location=DP.DEVICE); stats = ck['stats']
model = SetDenoiser(**ck['arch']).to(DP.DEVICE); model.load_state_dict(ck['model']); model.eval()
abar = ck['abar'].to(DP.DEVICE); betas = ck['betas'].to(DP.DEVICE)


def dps_components(P, z, depth, target):
    """List of (kx,ky,A,phi) for each posterior sample (pruned, obs-fit amplitudes)."""
    Ps, zs = P, z
    if len(zs) > a.mcap:
        idx = np.random.default_rng(0).choice(len(zs), a.mcap, replace=False); Ps, zs = P[idx], z[idx]
    Xd = DP.dps_sample(model, stats, abar, betas, Ps, zs, depth, guidance=a.guid, n=a.nsamp, seed=0)
    obs = np.column_stack([Ps[:, 0], Ps[:, 1], Ps[:, 2], zs]).astype(np.float32)
    comps = []
    for b in range(a.nsamp):
        kx, ky, en = denorm_atoms(Xd[b], stats)
        kx, ky, en = kx.cpu().numpy(), ky.cpu().numpy(), en.cpu().numpy()
        if en.max() <= 0:
            continue
        keep = en > 0.03 * en.max(); kxi, kyi = kx[keep], ky[keep]
        if len(kxi) == 0:
            continue
        if len(kxi) > 30:
            o = np.argsort(-en[keep])[:30]; kxi, kyi = kxi[o], kyi[o]
        A, phi = fit_amplitudes_phases(kxi, kyi, obs, depth, lambda_reg=0.5)
        comps.append((kxi, kyi, A, phi))
    return comps


def make_gif(tag, i_star, label):
    L = EpisodeDataLoader('data/' + tag_to_file(tag)); depth = L.depth; Hs = L.Hs_true
    gx = L.gt_grid_x.numpy(); gy = L.gt_grid_y.numpy()
    n_back = int(round(K_TP * L.Tp_true / 0.1))
    i_star = min(i_star, len(L) - 32)
    fstar = L[i_star]
    target = {'gx': gx, 'gy': gy, 't': float(fstar.t), 'gt': fstar.gt_field.numpy(), 'Hs': Hs}
    P, z = gather_frames(L, i_star, n_back)
    rkx, rky, rA, rphi = estimate_components(P, z, depth, target)     # RELAX
    dcomp = dps_components(P, z, depth, target)                        # DPS posterior
    ext = [gx.min(), gx.max(), gy.min(), gy.max()]
    vm = float(2.0 * np.std(fstar.gt_field.numpy()))
    n_fwd = min(28, len(L) - 1 - i_star)
    fig, axes = plt.subplots(1, 4, figsize=(19, 3.7))
    titles = ['Ground truth', 'RELAX (physics)', 'DPS posterior mean', 'DPS uncertainty (std)']
    cmaps = ['RdBu_r', 'RdBu_r', 'RdBu_r', 'magma']; vlims = [(-vm, vm)] * 3 + [(0, vm)]
    ims = []
    for ax, t, cm, vl in zip(axes, titles, cmaps, vlims):
        im = ax.imshow(np.zeros((len(gy), len(gx))), origin='lower', extent=ext, aspect='auto',
                       cmap=cm, vmin=vl[0], vmax=vl[1])
        ax.set_title(t, fontsize=11); ax.set_xlabel('x [m]'); ims.append(im)
    axes[0].set_ylabel('y [m]')

    def upd(f):
        i = i_star + f; fr = L[i]; gt = fr.gt_field.numpy(); t = float(fr.t)
        rrec = render_wave_field(rkx, rky, rA, rphi, depth, gx, gy, t)
        dfields = np.stack([render_wave_field(kx, ky, A, phi, depth, gx, gy, t)
                            for (kx, ky, A, phi) in dcomp])
        dmean = dfields.mean(0); dstd = dfields.std(0)
        ims[0].set_data(gt); ims[1].set_data(rrec); ims[2].set_data(dmean); ims[3].set_data(dstd)
        rnr = np.sqrt(np.mean((rrec - gt)**2)) / Hs
        dnr = np.sqrt(np.mean((dmean - gt)**2)) / Hs
        dt = (i - i_star) * 0.1; kind = 'nowcast' if dt <= 0.01 else f'forecast +{dt:.1f}s'
        fig.suptitle(f'{label} ({tag})   {kind}    RELAX nRMSE={rnr:.3f}   '
                     f'DPS nRMSE={dnr:.3f} (+UQ)', fontsize=12)
        return ims

    anim = FuncAnimation(fig, upd, frames=n_fwd, interval=120, blit=False)
    path = f'{OUT}/compare_{tag}.gif'
    anim.save(path, writer=PillowWriter(fps=8)); plt.close(fig)
    print(f'saved {path}  (RELAX N={len(rkx)}, DPS {len(dcomp)} samples)', flush=True)


if __name__ == '__main__':
    make_gif('BF4_30c_s0', 200, 'COMPLEX sea — DPS competitive + UQ')
    make_gif('BF4_5c_s0', 200, 'SIMPLE sea — RELAX sharp, DPS blurs')
