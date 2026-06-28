"""
Feasibility gate for LEARNED active sensing: does a DEPLOYABLE active-sensing loop (no
oracle components) actually lower the achieved reconstruction error on complex seas?

Loop: seed with a small random subset -> coarse RELAX estimate -> use the coarse components
to pick the D-optimal K returns (greedy Fisher) -> RELAX on the selected K -> achieved nRMSE.
Compare to random-K and the oracle-D-optimal-K (true components) upper bound, all estimated
with the same RELAX. If coarse-D-opt beats random and approaches oracle, the lever is
deployable and a learned acquisition policy is worth building.

Run: python3 scripts/d39_active_deployable.py [--K 120] [--frames 2]
"""
import os, sys, json, argparse, time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
os.chdir(os.path.join(os.path.dirname(__file__), '..'))

import torch
from wavesplat.data_loader import EpisodeDataLoader
from wavesplat.model.tnse import fit_amplitudes_phases
from d6_integrated_eval import tag_to_file
from d15_sequential_fusion import gather_frames
from d23_main_results import K_TP, panel
from d18_relax_estimator import relax_step
from d17_recursive_estimator import refine_params
from d26_relax_iterative import iterative_relax
from d11_active_sensing import theta_of, greedy_dopt
import d1_crb as C1

EPISODES = ['BF4_15c_s0', 'BF4_30c_s0', 'BF5_15c_s0', 'BF5_30c_s1']


def coarse_theta(P, z, depth, seed_idx):
    """Coarse component estimate from a seed subset -> theta=[A,kx,ky,phi] (deployable)."""
    Ps, zs = P[seed_idx], z[seed_idx]
    kx, ky = relax_step(np.zeros(0), np.zeros(0), Ps, zs, depth)
    obs = np.column_stack([Ps[:, 0], Ps[:, 1], Ps[:, 2], zs]).astype(np.float32)
    A, phi = fit_amplitudes_phases(kx, ky, obs, depth, lambda_reg=0.5)
    th = torch.tensor(np.stack([A, kx, ky, phi], 1), dtype=C1.DT, device=C1.DEVICE)
    return th


def relax_field(P, z, depth, target):
    fld, _ = iterative_relax(P, z, depth, target, 3)
    return panel(fld, target['gt'], target['Hs'])['nrmse']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--K', type=int, default=120)
    ap.add_argument('--seedK', type=int, default=80)
    ap.add_argument('--pool', type=int, default=900)
    ap.add_argument('--frames', type=int, default=2)
    ap.add_argument('--episodes', nargs='+', default=EPISODES)
    a = ap.parse_args()
    rng = np.random.default_rng(0)
    out = {}; t0 = time.time()
    for tag in a.episodes:
        L = EpisodeDataLoader('data/' + tag_to_file(tag)); depth = L.depth; Hs = L.Hs_true
        gx = L.gt_grid_x.numpy(); gy = L.gt_grid_y.numpy()
        gxt = torch.tensor(gx, dtype=C1.DT, device=C1.DEVICE)
        gyt = torch.tensor(gy, dtype=C1.DT, device=C1.DEVICE)
        th_true = theta_of(L)
        Tobs = K_TP * L.Tp_true; n_back = int(round(Tobs / 0.1)); nfr = len(L)
        i0 = n_back + 3; i1 = nfr - 2
        if i1 - i0 < 4:
            i0 = max(n_back + 1, nfr - 60)
        sel = np.linspace(i0, i1, a.frames).astype(int)
        rec = {k: [] for k in ['rand', 'coarse_dopt', 'oracle_dopt']}
        for i_star in sel:
            fstar = L[int(i_star)]
            target = {'gx': gx, 'gy': gy, 't': float(fstar.t), 'gt': fstar.gt_field.numpy(), 'Hs': Hs}
            P, z = gather_frames(L, int(i_star), n_back)
            if len(z) > a.pool:
                ii = rng.choice(len(z), a.pool, replace=False); P, z = P[ii], z[ii]
            # random-K
            ridx = rng.choice(len(z), a.K, replace=False)
            rec['rand'].append(relax_field(P[ridx], z[ridx], depth, target))
            # coarse-D-optimal-K (deployable): seed -> coarse theta -> greedy dopt
            seed = rng.choice(len(z), min(a.seedK, len(z)), replace=False)
            th_c = coarse_theta(P, z, depth, seed)
            cidx = greedy_dopt(th_c, P, a.K, depth)
            rec['coarse_dopt'].append(relax_field(P[cidx], z[cidx], depth, target))
            # oracle-D-optimal-K (true theta) upper bound
            oidx = greedy_dopt(th_true, P, a.K, depth)
            rec['oracle_dopt'].append(relax_field(P[oidx], z[oidx], depth, target))
        agg = {k: float(np.mean(v)) for k, v in rec.items()}
        out[tag] = agg
        print(f"[{tag:12s}] random {agg['rand']:.3f} | coarse-Dopt {agg['coarse_dopt']:.3f} | "
              f"oracle-Dopt {agg['oracle_dopt']:.3f}   ({time.time()-t0:.0f}s)", flush=True)
    json.dump(out, open('results/direction3/active_deployable.json', 'w'), indent=2)
    m = lambda k: float(np.mean([out[t][k] for t in out]))
    print(f"\nOVERALL  random {m('rand'):.3f} | coarse-Dopt {m('coarse_dopt'):.3f} "
          f"(gain {100*(m('rand')-m('coarse_dopt'))/m('rand'):+.0f}%) | oracle-Dopt {m('oracle_dopt'):.3f}")
    print("saved results/direction3/active_deployable.json")


if __name__ == '__main__':
    main()
