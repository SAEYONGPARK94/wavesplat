"""Beamform-detection comparison GIF + full eval. Panels: GT | RELAX | beamform-detect
(field) | detection heatmap (the net's component-location belief = its UQ). nowcast→forecast.
Run: python3 tmp/beamform_gif.py
"""
import os, sys, json, numpy as np, torch
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
sys.path.insert(0, 'scripts')
from d3_beamform_net import DetectCNN, beamform, extract, DEVICE, M_OBS, KLIM, NK
from d17_recursive_estimator import refine_params
from wavesplat.model.tnse import fit_amplitudes_phases, render_wave_field
from wavesplat.evaluation.metrics import pearson_correlation
from d6_integrated_eval import tag_to_file
from d15_sequential_fusion import gather_frames
from d23_main_results import K_TP, panel
from d28_forecast_complex import estimate_components
from wavesplat.data_loader import EpisodeDataLoader

OUT = 'results/direction3/gifs'; os.makedirs(OUT, exist_ok=True)
m = DetectCNN().to(DEVICE); m.load_state_dict(torch.load('results/direction3/beamform_net.pt', map_location=DEVICE)); m.eval()


G = 9.81
def _om(kx, ky, d):
    k = np.sqrt(kx**2 + ky**2) + 1e-9; return np.sqrt(G * k * np.tanh(k * d))
def _render_obs(kx, ky, A, phi, P, d):
    o = _om(kx, ky, d); psi = kx[None]*P[:, 0][:, None] + ky[None]*P[:, 1][:, None] - o[None]*P[:, 2][:, None] + phi[None]
    return (A[None] * np.cos(psi)).sum(1)


def bf_components(P, z, depth, target, npass=2):
    """Iterative beamform-detect: detect → refine → subtract residual → re-detect (learned CLEAN)."""
    t0 = P[:, 2].min(); Ps = P.copy(); Ps[:, 2] -= t0
    tgt = dict(target); tgt['t'] = target['t'] - t0
    obs = np.column_stack([Ps[:, 0], Ps[:, 1], Ps[:, 2], z]).astype(np.float32)
    allkx, allky = np.zeros(0), np.zeros(0); zres = z.copy(); Hp0 = None
    A = phi = np.zeros(0)
    for p in range(npass):
        with torch.no_grad():
            B = beamform(torch.tensor(Ps[None], device=DEVICE, dtype=torch.float32),
                         torch.tensor(zres[None], device=DEVICE, dtype=torch.float32))
            Hp = m(B)[0].cpu().numpy()
        if Hp0 is None:
            Hp0 = Hp
        kx, ky = extract(torch.tensor(Hp))
        if len(kx):
            allkx = np.concatenate([allkx, kx]); allky = np.concatenate([allky, ky])
        if len(allkx) == 0:
            continue
        allkx, allky, _ = refine_params(allkx, allky, Ps, z, depth, tgt)
        A, phi = fit_amplitudes_phases(allkx, allky, obs, depth, lambda_reg=0.5)
        zres = z - _render_obs(allkx, allky, A, phi, Ps, depth)
    return allkx, allky, A, phi, Hp0, t0


def make_gif(tag, i_star, label):
    L = EpisodeDataLoader('data/' + tag_to_file(tag)); depth = L.depth; Hs = L.Hs_true
    gx = L.gt_grid_x.numpy(); gy = L.gt_grid_y.numpy(); nb = int(round(K_TP * L.Tp_true / 0.1))
    i_star = min(i_star, len(L) - 32); fstar = L[i_star]
    target = {'gx': gx, 'gy': gy, 't': float(fstar.t), 'gt': fstar.gt_field.numpy(), 'Hs': Hs}
    P, z = gather_frames(L, i_star, nb)
    if len(z) > M_OBS:
        idx = np.random.default_rng(0).choice(len(z), M_OBS, replace=False); P, z = P[idx], z[idx]
    rkx, rky, rA, rphi = estimate_components(P, z, depth, target)               # RELAX
    bkx, bky, bA, bphi, Hp, t0 = bf_components(P, z, depth, target)             # beamform-detect
    ext = [gx.min(), gx.max(), gy.min(), gy.max()]; vm = float(2 * np.std(fstar.gt_field.numpy()))
    n_fwd = min(28, len(L) - 1 - i_star)
    fig, ax = plt.subplots(1, 4, figsize=(19, 3.7))
    ims = []
    for a, t, cm, vl in zip(ax, ['Ground truth', 'RELAX (physics)', 'Beamform-detect net', 'detection heatmap'],
                            ['RdBu_r', 'RdBu_r', 'RdBu_r', 'magma'], [(-vm, vm)] * 3 + [(0, 1)]):
        e = ext if t != 'detection heatmap' else [-KLIM, KLIM, -KLIM, KLIM]
        im = a.imshow(np.zeros((NK, NK)) if t == 'detection heatmap' else np.zeros((len(gy), len(gx))),
                      origin='lower', extent=e, aspect='auto', cmap=cm, vmin=vl[0], vmax=vl[1])
        a.set_title(t, fontsize=11); ims.append(im)
    ims[3].set_data(Hp)                                                          # static detection map
    ax[3].set_xlabel('kx'); ax[3].set_ylabel('ky')

    def upd(f):
        i = i_star + f; fr = L[i]; gt = fr.gt_field.numpy(); t = float(fr.t)
        rrec = render_wave_field(rkx, rky, rA, rphi, depth, gx, gy, t) if len(rkx) else np.zeros_like(gt)
        brec = render_wave_field(bkx, bky, bA, bphi, depth, gx, gy, t - t0) if len(bkx) else np.zeros_like(gt)
        ims[0].set_data(gt); ims[1].set_data(rrec); ims[2].set_data(brec)
        rnr = np.sqrt(np.mean((rrec - gt) ** 2)) / Hs; bnr = np.sqrt(np.mean((brec - gt) ** 2)) / Hs
        dt = (i - i_star) * 0.1; kind = 'nowcast' if dt <= 0.01 else f'forecast +{dt:.1f}s'
        fig.suptitle(f'{label} ({tag})  {kind}   RELAX nRMSE={rnr:.3f}   Beamform-detect nRMSE={bnr:.3f}', fontsize=12)
        return ims

    anim = FuncAnimation(fig, upd, frames=n_fwd, interval=120, blit=False)
    path = f'{OUT}/beamform_{tag}.gif'; anim.save(path, writer=PillowWriter(fps=8)); plt.close(fig)
    print(f'saved {path}  (RELAX N={len(rkx)}, beamform-detect N={len(bkx)})', flush=True)


if __name__ == '__main__':
    # full eval (grid-lock + refine) on the benchmark
    eps = ['BF3_5c_s0', 'BF4_5c_s0', 'BF4_15c_s0', 'BF4_30c_s0', 'BF5_15c_s0', 'BF5_30c_s1']
    make_gif('BF3_5c_s0', 200, 'SIMPLE sea — beamform-detect breaks the collapse')
    make_gif('BF4_30c_s0', 200, 'COMPLEX sea')
    print('done')
