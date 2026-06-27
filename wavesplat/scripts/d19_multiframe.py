"""
Multi-frame hardening of the RELAX deployable sequential estimator.

Single-frame eval has ~±0.03 noise. Here we evaluate the deployable estimator
(beamform→RELAX add/prune→off-grid refine, observations only) over several target
frames per episode at fixed fusion lengths, and report mean ± std of the full metric
panel (nRMSE, Pearson, VR, skill) vs the oracle (true wavenumbers).

Run: python3 scripts/d19_multiframe.py
"""
import os, sys, json
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
os.chdir(os.path.join(os.path.dirname(__file__), '..'))

from wavesplat.data_loader import EpisodeDataLoader
from wavesplat.model.tnse import fit_amplitudes_phases, render_wave_field
from wavesplat.evaluation.metrics import variance_ratio, pearson_correlation, skill_score
from d6_integrated_eval import tag_to_file
from d15_sequential_fusion import gather_frames
from d17_recursive_estimator import refine_params
from d18_relax_estimator import relax_step, MCAP


def panel(f, gt, Hs):
    return dict(nrmse=float(np.sqrt(np.mean((f - gt)**2)) / max(Hs, 1e-6)),
                P=float(pearson_correlation(f, gt)), VR=float(variance_ratio(f, gt)),
                skill=float(skill_score(f, gt)))


def main():
    tags = ['BF4_5c_s0', 'BF4_15c_s0', 'BF4_30c_s0', 'BF5_30c_s1']
    frames = [160, 185, 210, 235, 260]      # target frames (need >=150 for T=15)
    T_list = [10.0, 15.0]
    out = {}
    for tag in tags:
        L = EpisodeDataLoader('data/' + tag_to_file(tag)); depth = L.depth; Hs = L.Hs_true
        wc = np.array(L.wave_components); tkx, tky = wc[:, 2], wc[:, 3]
        gx = L.gt_grid_x.numpy(); gy = L.gt_grid_y.numpy()
        out[tag] = {}
        for T in T_list:
            dep, ora = [], []
            for i_star in frames:
                fstar = L[i_star]
                target = {'gx': gx, 'gy': gy, 't': float(fstar.t), 'gt': fstar.gt_field.numpy(), 'Hs': Hs}
                P, z = gather_frames(L, i_star, int(round(T / 0.1)))
                kx, ky = relax_step(np.zeros(0), np.zeros(0), P, z, depth)
                _, _, fld = refine_params(kx, ky, P, z, depth, target)
                dep.append(panel(fld, target['gt'], Hs))
                obs = np.column_stack([P[:, 0], P[:, 1], P[:, 2], z]).astype(np.float32)
                if len(obs) > MCAP:
                    obs = obs[np.random.default_rng(1).choice(len(obs), MCAP, replace=False)]
                A, phi = fit_amplitudes_phases(tkx, tky, obs, depth, lambda_reg=0.5)
                fo = render_wave_field(tkx, tky, A, phi, depth, gx, gy, target['t'])
                ora.append(panel(fo, target['gt'], Hs))
            agg = lambda L_, k: (float(np.mean([d[k] for d in L_])), float(np.std([d[k] for d in L_])))
            out[tag][f'T{T:.0f}'] = dict(
                deployable={k: agg(dep, k) for k in ['nrmse', 'P', 'VR', 'skill']},
                oracle={k: agg(ora, k) for k in ['nrmse', 'P', 'skill']})
    json.dump(out, open('results/direction3/multiframe.json', 'w'), indent=2)

    print(f"multi-frame (n={len(frames)} frames) deployable RELAX — mean±std:")
    print(f"{'episode':12s} {'T':>3} | {'nRMSE':>13} {'Pearson':>13} {'VR':>11} {'skill':>13} | {'oracle nRMSE':>13}")
    for tag in tags:
        for T in T_list:
            d = out[tag][f'T{T:.0f}']['deployable']; o = out[tag][f'T{T:.0f}']['oracle']
            print(f"{tag:12s} {T:3.0f} | {d['nrmse'][0]:.3f}±{d['nrmse'][1]:.3f} "
                  f"{d['P'][0]:.2f}±{d['P'][1]:.2f}  {d['VR'][0]:.2f}±{d['VR'][1]:.2f} "
                  f"{d['skill'][0]:.2f}±{d['skill'][1]:.2f} | {o['nrmse'][0]:.3f}±{o['nrmse'][1]:.3f}")
    print("\nref single-window (their own eval): DPS 0.19-0.23 / ST-DF-GP simple~0.10 complex 0.23-0.29")


if __name__ == '__main__':
    main()
