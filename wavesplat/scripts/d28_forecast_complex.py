"""
Phase 2b — forecast skill on COMPLEX seas + predictable-zone, with the iterative
estimator. Does the multi-second forecast hold when reconstruction is harder?

Run: python3 scripts/d28_forecast_complex.py
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
from d18_relax_estimator import relax_step, MCAP
from d17_recursive_estimator import refine_params

K_TP = 2.2


def estimate_components(P, z, depth, target, n_outer=3):
    kx, ky = np.zeros(0), np.zeros(0)
    for _ in range(n_outer):
        kx, ky = relax_step(kx, ky, P, z, depth)
        kx, ky, _ = refine_params(kx, ky, P, z, depth, target)
    obs = np.column_stack([P[:, 0], P[:, 1], P[:, 2], z]).astype(np.float32)
    if len(obs) > MCAP:
        obs = obs[np.random.default_rng(1).choice(len(obs), MCAP, replace=False)]
    A, phi = fit_amplitudes_phases(kx, ky, obs, depth, lambda_reg=0.5)
    return kx, ky, A, phi


def main():
    tags = ['BF4_15c_s0', 'BF4_30c_s0', 'BF5_30c_s1']
    i_star = 200
    leads = [0.0, 1.0, 2.0, 3.0, 5.0, 8.0]
    out = {}
    for tag in tags:
        L = EpisodeDataLoader('data/' + tag_to_file(tag)); depth = L.depth; Hs = L.Hs_true
        gx = L.gt_grid_x.numpy(); gy = L.gt_grid_y.numpy()
        n_back = int(round(K_TP * L.Tp_true / 0.1))
        fstar = L[i_star]
        target = {'gx': gx, 'gy': gy, 't': float(fstar.t), 'gt': fstar.gt_field.numpy(), 'Hs': Hs}
        P, z = gather_frames(L, i_star, n_back)
        kx, ky, A, phi = estimate_components(P, z, depth, target)
        z_now = fstar.gt_field.numpy()
        rows = []
        for dl in leads:
            j = i_star + int(round(dl / 0.1))
            if j >= len(L):
                break
            gt = L[j].gt_field.numpy()
            pred = render_wave_field(kx, ky, A, phi, depth, gx, gy, float(L[j].t))
            rows.append(dict(lead=dl,
                             fc=(float(np.sqrt(np.mean((pred-gt)**2))/Hs), float(pearson_correlation(pred, gt)), float(skill_score(pred, gt))),
                             pers=(float(np.sqrt(np.mean((z_now-gt)**2))/Hs), float(pearson_correlation(z_now, gt)))))
        out[tag] = rows
        print(f"{tag}:")
        for r in rows:
            print(f"   +{r['lead']:.0f}s  fc nRMSE {r['fc'][0]:.3f} P{r['fc'][1]:.2f} sk{r['fc'][2]:.2f} | pers {r['pers'][0]:.3f} P{r['pers'][1]:.2f}", flush=True)
    json.dump(out, open('results/direction3/forecast_complex.json', 'w'), indent=2)
    print("saved forecast_complex.json")


if __name__ == '__main__':
    main()
