"""
Close the oracle gap — iterative (multi-pass) RELAX.

The deployable estimator (0.130) sits above the oracle (0.037); the gap is
component-identification accuracy. d23 does ONE add-pass then ONE off-grid refine.
True RELAX interleaves: add → refine → re-beamform residual → add → refine ...,
so components masked in the raw residual surface after refinement. We test
N_outer = 1,2,3 passes on complex seas (period-scaled window, multi-frame) vs oracle.

Run: python3 scripts/d26_relax_iterative.py
"""
import os, sys, json
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
os.chdir(os.path.join(os.path.dirname(__file__), '..'))

from wavesplat.data_loader import EpisodeDataLoader
from wavesplat.model.tnse import fit_amplitudes_phases, render_wave_field
from wavesplat.evaluation.metrics import pearson_correlation, skill_score
from d6_integrated_eval import tag_to_file
from d15_sequential_fusion import gather_frames
from d17_recursive_estimator import refine_params
from d18_relax_estimator import relax_step, MCAP

K_TP = 2.2


def iterative_relax(P, z, depth, target, n_outer):
    kx, ky = np.zeros(0), np.zeros(0)
    fld = None
    for _ in range(n_outer):
        kx, ky = relax_step(kx, ky, P, z, depth)        # add/prune from residual (warm-start)
        kx, ky, fld = refine_params(kx, ky, P, z, depth, target)  # off-grid refine all
    return fld, len(kx)


def met(f, gt, Hs):
    return (float(np.sqrt(np.mean((f-gt)**2))/Hs), float(pearson_correlation(f, gt)),
            float(skill_score(f, gt)))


def main():
    tags = ['BF4_15c_s0', 'BF4_30c_s0', 'BF5_30c_s1']
    frames = [210, 240, 270]
    print("iterative RELAX — nRMSE (Pearson, N) vs #passes:")
    out = {}
    for tag in tags:
        L = EpisodeDataLoader('data/' + tag_to_file(tag)); depth = L.depth; Hs = L.Hs_true
        wc = np.array(L.wave_components); tkx, tky = wc[:, 2], wc[:, 3]
        gx = L.gt_grid_x.numpy(); gy = L.gt_grid_y.numpy()
        n_back = int(round(K_TP * L.Tp_true / 0.1))
        res = {1: [], 2: [], 3: [], 'oracle': []}
        for i_star in frames:
            fstar = L[i_star]
            target = {'gx': gx, 'gy': gy, 't': float(fstar.t), 'gt': fstar.gt_field.numpy(), 'Hs': Hs}
            P, z = gather_frames(L, i_star, n_back)
            for npass in (1, 2, 3):
                fld, N = iterative_relax(P, z, depth, target, npass)
                res[npass].append(met(fld, target['gt'], Hs) + (N,))
            obs = np.column_stack([P[:, 0], P[:, 1], P[:, 2], z]).astype(np.float32)
            if len(obs) > MCAP:
                obs = obs[np.random.default_rng(1).choice(len(obs), MCAP, replace=False)]
            A, phi = fit_amplitudes_phases(tkx, tky, obs, depth, lambda_reg=0.5)
            fo = render_wave_field(tkx, tky, A, phi, depth, gx, gy, target['t'])
            res['oracle'].append(met(fo, target['gt'], Hs))
        out[tag] = {str(k): [list(v) for v in res[k]] for k in res}
        line = f"  {tag:12s} "
        for npass in (1, 2, 3):
            nr = np.mean([r[0] for r in res[npass]]); P_ = np.mean([r[1] for r in res[npass]]); N = np.mean([r[3] for r in res[npass]])
            line += f"| {npass}p {nr:.3f}(P{P_:.2f},N{N:.0f}) "
        line += f"| oracle {np.mean([r[0] for r in res['oracle']]):.3f}"
        print(line, flush=True)
    json.dump(out, open('results/direction3/relax_iterative.json', 'w'), indent=2)
    print("saved results/direction3/relax_iterative.json")


if __name__ == '__main__':
    main()
