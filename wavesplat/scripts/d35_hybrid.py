"""
Hybrid — learned coarse init + classical off-grid refinement. Tests whether the
feedforward atoms (good coarse placement, but too coarse for the field on their own) help
the iterative estimator reach / beat RELAX, closing the oracle gap on complex seas.

Variants (same obs window per frame):
  L  : learned atoms -> refine_params only            (pure learned-init refine)
  B  : iterative RELAX from beamform                   (baseline, headline estimator)
  H  : learned atoms warm-start relax_step -> refine   (learned init for RELAX)
  HU : union(RELAX atoms, learned atoms) -> refine     (learned proposes missed components)
vs oracle.

Run: python3 scripts/d35_hybrid.py [--ckpt ...] [--frames 3]
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
from d18_relax_estimator import relax_step, MCAP
from d26_relax_iterative import iterative_relax
import d34_learned_offgrid as LO

EPISODES = ['BF4_15c_s0', 'BF4_30c_s0', 'BF5_15c_s0', 'BF5_30c_s1', 'BF3_5c_s0', 'BF4_5c_s0']


@torch.no_grad()
def learned_atoms(model, P, z, m_in=600, pres_th=0.4, max_n=24):
    """Run the feedforward model on the window -> selected off-grid (kx,ky)."""
    if len(z) > m_in:
        idx = np.random.default_rng(1).choice(len(z), m_in, replace=False); P, z = P[idx], z[idx]
    t0 = P[:, 2].min()
    obs = np.column_stack([P[:, 0], P[:, 1], P[:, 2] - t0, z]).astype(np.float32)
    kx, ky, A, cs, sn, pres = model(torch.tensor(obs[None], device=LO.DEVICE))
    kx = kx[0].cpu().numpy(); ky = ky[0].cpu().numpy()
    w = (pres[0] * A[0]).cpu().numpy()
    keep = np.where(pres[0].cpu().numpy() > pres_th)[0]
    if len(keep) == 0:
        keep = np.argsort(-w)[:8]
    if len(keep) > max_n:
        keep = keep[np.argsort(-w[keep])[:max_n]]
    return kx[keep], ky[keep]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', default='results/direction3/learned_offgrid.pt')
    ap.add_argument('--frames', type=int, default=3)
    ap.add_argument('--episodes', nargs='+', default=EPISODES)
    a = ap.parse_args()

    model = LO.Model().to(LO.DEVICE)
    model.load_state_dict(torch.load(a.ckpt, map_location=LO.DEVICE)); model.eval()

    relax_full = json.load(open('results/direction3/main_results.json'))
    relax_iter = json.load(open('results/direction3/main_results_iter.json'))
    out = {}
    t0 = time.time()
    for tag in a.episodes:
        L = EpisodeDataLoader('data/' + tag_to_file(tag)); depth = L.depth; Hs = L.Hs_true
        gx = L.gt_grid_x.numpy(); gy = L.gt_grid_y.numpy()
        Tobs = K_TP * L.Tp_true; n_back = int(round(Tobs / 0.1)); nfr = len(L)
        i0 = n_back + 3; i1 = nfr - 2
        if i1 - i0 < 4:
            i0 = max(n_back + 1, nfr - 60)
        sel = np.linspace(i0, i1, a.frames).astype(int)
        rec = {k: [] for k in ['L', 'B', 'H', 'HU']}
        for i_star in sel:
            fstar = L[int(i_star)]
            target = {'gx': gx, 'gy': gy, 't': float(fstar.t), 'gt': fstar.gt_field.numpy(), 'Hs': Hs}
            P, z = gather_frames(L, int(i_star), n_back)
            lkx, lky = learned_atoms(model, P, z)
            # L: learned init -> refine only
            _, _, fL = refine_params(lkx, lky, P, z, depth, target)
            rec['L'].append(panel(fL, target['gt'], Hs)['nrmse'])
            # B: baseline iterative RELAX (beamform init)
            fB, _ = iterative_relax(P, z, depth, target, 3)
            rec['B'].append(panel(fB, target['gt'], Hs)['nrmse'])
            # H: learned warm-start RELAX add/prune -> refine (2 outer passes)
            kx, ky = lkx.copy(), lky.copy()
            fH = fB
            for _ in range(2):
                kx, ky = relax_step(kx, ky, P, z, depth)
                kx, ky, fH = refine_params(kx, ky, P, z, depth, target)
            rec['H'].append(panel(fH, target['gt'], Hs)['nrmse'])
            # HU: union(RELAX atoms, learned atoms) -> refine
            rkx, rky = relax_step(np.zeros(0), np.zeros(0), P, z, depth)
            ukx = np.concatenate([rkx, lkx]); uky = np.concatenate([rky, lky])
            _, _, fHU = refine_params(ukx, uky, P, z, depth, target)
            rec['HU'].append(panel(fHU, target['gt'], Hs)['nrmse'])
        agg = {k: float(np.mean(v)) for k, v in rec.items()}
        out[tag] = dict(agg, relax3_full=relax_iter[tag]['deployable']['nrmse'][0],
                        oracle=relax_full[tag]['oracle']['nrmse'][0])
        print(f"[{tag:12s}] L {agg['L']:.3f} | B(RELAX) {agg['B']:.3f} | "
              f"H(learn-warm) {agg['H']:.3f} | HU(union) {agg['HU']:.3f} | "
              f"oracle {out[tag]['oracle']:.3f}  ({time.time()-t0:.0f}s)", flush=True)
    json.dump(out, open('results/direction3/hybrid_eval.json', 'w'), indent=2)
    m = lambda k: float(np.mean([out[t][k] for t in out]))
    print(f"\nOVERALL  L {m('L'):.3f} | B(RELAX) {m('B'):.3f} | H(learn-warm) {m('H'):.3f} | "
          f"HU(union) {m('HU'):.3f} | oracle {m('oracle'):.3f}")
    print("saved results/direction3/hybrid_eval.json")


if __name__ == '__main__':
    main()
