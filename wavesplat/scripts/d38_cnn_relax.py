"""
Integration — CNN-detection + classical off-grid refine ("learned-detection RELAX"),
evaluated on the real benchmark vs greedy RELAX. Tests whether the learned beamform
deconvolution (d37) actually helps reconstruction on real complex seas (closing oracle gap).

Pipeline (CNN-RELAX):
  beamform map of (residual) obs -> CNN clean -> extract off-grid peaks -> refine_params
  (classical, reaches precision) -> subtract -> repeat a few passes.
vs B = iterative RELAX (greedy beamform detection), and oracle.

Run: python3 scripts/d38_cnn_relax.py [--frames 3]
"""
import os, sys, json, argparse, time
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
os.chdir(os.path.join(os.path.dirname(__file__), '..'))

from wavesplat.data_loader import EpisodeDataLoader
from wavesplat.model.tnse import fit_amplitudes_phases, render_wave_field
from d6_integrated_eval import tag_to_file
from d15_sequential_fusion import gather_frames
from d23_main_results import K_TP, panel
from d17_recursive_estimator import refine_params
from d26_relax_iterative import iterative_relax
import d37_learned_detection as LD

EPISODES = ['BF4_15c_s0', 'BF4_30c_s0', 'BF5_15c_s0', 'BF5_30c_s1', 'BF3_5c_s0', 'BF4_5c_s0']


def extract_peaks(cm, thresh=0.15, max_n=34):
    """Off-grid local-maxima of a (K,K) map -> (kx,ky) arrays via 3x3 centroid."""
    K = cm.shape[0]; axn = np.linspace(-LD.KLIM, LD.KLIM, K)
    mx = cm.max()
    pk = []
    for iy in range(1, K - 1):
        for ix in range(1, K - 1):
            v = cm[iy, ix]
            if v < thresh * mx:
                continue
            w = cm[iy-1:iy+2, ix-1:ix+2]
            if v < w.max() - 1e-9:
                continue
            # 3x3 centroid for off-grid location
            wy, wx = np.mgrid[iy-1:iy+2, ix-1:ix+2]
            sw = w.sum() + 1e-9
            cy = (w * wy).sum() / sw; cx = (w * wx).sum() / sw
            kx = np.interp(cx, np.arange(K), axn); ky = np.interp(cy, np.arange(K), axn)
            pk.append((kx, ky, v))
    if not pk:
        return np.zeros(0), np.zeros(0)
    pk.sort(key=lambda p: -p[2]); pk = pk[:max_n]
    return np.array([p[0] for p in pk]), np.array([p[1] for p in pk])


@torch.no_grad()
def cnn_detect(P, r, model):
    ox = torch.tensor(P[:, 0], dtype=torch.float32, device=LD.DEVICE)[None]
    oy = torch.tensor(P[:, 1], dtype=torch.float32, device=LD.DEVICE)[None]
    ot = torch.tensor(P[:, 2] - P[:, 2].min(), dtype=torch.float32, device=LD.DEVICE)[None]
    rt = torch.tensor(r, dtype=torch.float32, device=LD.DEVICE)[None]
    Bm = LD.norm_map(LD.beamform_map(ox, oy, ot, rt))
    Cm = model(Bm[:, None])[0, 0].cpu().numpy()
    return extract_peaks(Cm)


def render_obs(kx, ky, P, depth, A, phi):
    k = np.sqrt(kx**2 + ky**2 + 1e-9); om = np.sqrt(9.81 * k * np.tanh(k * depth) + 1e-9)
    ps = kx[None] * P[:, 0:1] + ky[None] * P[:, 1:2] - om[None] * P[:, 2:3] + phi[None]
    return (A[None] * np.cos(ps)).sum(1)


def cnn_relax(P, z, depth, target, model, passes=3):
    kx, ky = np.zeros(0), np.zeros(0)
    fld = None
    for _ in range(passes):
        # residual at obs from current atoms
        if len(kx) > 0:
            obs = np.column_stack([P[:, 0], P[:, 1], P[:, 2], z]).astype(np.float32)
            A, phi = fit_amplitudes_phases(kx, ky, obs, depth, lambda_reg=0.5)
            r = z - render_obs(kx, ky, P, depth, A, phi)
        else:
            r = z
        nkx, nky = cnn_detect(P, r, model)
        if len(nkx) == 0 and len(kx) == 0:
            break
        kx = np.concatenate([kx, nkx]); ky = np.concatenate([ky, nky])
        kx, ky, fld = refine_params(kx, ky, P, z, depth, target)
        # prune low-amplitude duplicates
        obs = np.column_stack([P[:, 0], P[:, 1], P[:, 2], z]).astype(np.float32)
        A, _ = fit_amplitudes_phases(kx, ky, obs, depth, lambda_reg=0.5)
        keep = np.abs(A) > 0.05 * (np.abs(A).max() + 1e-9)
        if keep.sum() >= 1:
            kx, ky = kx[keep], ky[keep]
    return fld


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', default='results/direction3/learned_detection.pt')
    ap.add_argument('--frames', type=int, default=3)
    ap.add_argument('--episodes', nargs='+', default=EPISODES)
    a = ap.parse_args()
    model = LD.DetCNN().to(LD.DEVICE); model.load_state_dict(torch.load(a.ckpt, map_location=LD.DEVICE)); model.eval()
    relax_full = json.load(open('results/direction3/main_results.json'))
    relax_iter = json.load(open('results/direction3/main_results_iter.json'))
    out = {}; t0 = time.time()
    for tag in a.episodes:
        L = EpisodeDataLoader('data/' + tag_to_file(tag)); depth = L.depth; Hs = L.Hs_true
        gx = L.gt_grid_x.numpy(); gy = L.gt_grid_y.numpy()
        Tobs = K_TP * L.Tp_true; n_back = int(round(Tobs / 0.1)); nfr = len(L)
        i0 = n_back + 3; i1 = nfr - 2
        if i1 - i0 < 4:
            i0 = max(n_back + 1, nfr - 60)
        sel = np.linspace(i0, i1, a.frames).astype(int)
        cnn, base = [], []
        for i_star in sel:
            fstar = L[int(i_star)]
            target = {'gx': gx, 'gy': gy, 't': float(fstar.t), 'gt': fstar.gt_field.numpy(), 'Hs': Hs}
            P, z = gather_frames(L, int(i_star), n_back)
            fc = cnn_relax(P, z, depth, target, model)
            cnn.append(panel(fc, target['gt'], Hs)['nrmse'] if fc is not None else 1.0)
            fb, _ = iterative_relax(P, z, depth, target, 3)
            base.append(panel(fb, target['gt'], Hs)['nrmse'])
        out[tag] = dict(cnn_relax=float(np.mean(cnn)), relax=float(np.mean(base)),
                        relax3_full=relax_iter[tag]['deployable']['nrmse'][0],
                        oracle=relax_full[tag]['oracle']['nrmse'][0])
        d = out[tag]
        print(f"[{tag:12s}] CNN-RELAX {d['cnn_relax']:.3f} | RELAX {d['relax']:.3f} | "
              f"oracle {d['oracle']:.3f}  ({time.time()-t0:.0f}s)", flush=True)
    json.dump(out, open('results/direction3/cnn_relax_eval.json', 'w'), indent=2)
    m = lambda k: float(np.mean([out[t][k] for t in out]))
    print(f"\nOVERALL  CNN-RELAX {m('cnn_relax'):.3f} | RELAX {m('relax'):.3f} | oracle {m('oracle'):.3f}")
    print("saved results/direction3/cnn_relax_eval.json")


if __name__ == '__main__':
    main()
