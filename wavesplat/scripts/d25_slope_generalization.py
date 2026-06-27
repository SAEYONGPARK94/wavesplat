"""
Phase 4 — generalization to the slope-raydrop LiDAR dropout model.

The main result used uniform-raydrop episodes. Here we run the same deployable RELAX
estimator on the slope-raydrop versions (a different, geometry-dependent dropout
pattern) to check robustness of the estimator to the LiDAR sampling model. Compare to
the uniform RELAX result (main_results.json) and the slope oracle.

Run: python3 scripts/d25_slope_generalization.py
"""
import os, sys, json
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
os.chdir(os.path.join(os.path.dirname(__file__), '..'))

from wavesplat.data_loader import EpisodeDataLoader
from wavesplat.model.tnse import fit_amplitudes_phases, render_wave_field
from d15_sequential_fusion import gather_frames
from d17_recursive_estimator import refine_params
from d18_relax_estimator import relax_step, MCAP
from d23_main_results import panel, K_TP, N_FRAMES


def slope_file(tag):
    bf, comp, s = tag.split('_')
    return f"episode_{bf}_{comp.replace('c','comp')}_seed{s[1:]}_slope.h5"


def main():
    gp = json.load(open('results/spacetime_gp_30ep.json'))['per_episode']
    uni = json.load(open('results/direction3/main_results.json'))
    tags = list(gp.keys())
    out = {}
    for ti, tag in enumerate(tags):
        try:
            L = EpisodeDataLoader('data/' + slope_file(tag))
        except Exception as e:
            print(f"[{ti+1:2d}/30] {tag:12s} SKIP (corrupt slope file)", flush=True); continue
        depth = L.depth; Hs = L.Hs_true
        wc = np.array(L.wave_components); tkx, tky = wc[:, 2], wc[:, 3]
        gx = L.gt_grid_x.numpy(); gy = L.gt_grid_y.numpy()
        Tobs = K_TP * L.Tp_true; n_back = int(round(Tobs / 0.1))
        nfr = len(L); i0 = n_back + 3; i1 = nfr - 2
        if i1 - i0 < 4:
            i0 = max(n_back + 1, nfr - 60)
        frames = np.linspace(i0, i1, N_FRAMES).astype(int)
        dep, ora = [], []
        for i_star in frames:
            fstar = L[int(i_star)]
            target = {'gx': gx, 'gy': gy, 't': float(fstar.t), 'gt': fstar.gt_field.numpy(), 'Hs': Hs}
            P, z = gather_frames(L, int(i_star), n_back)
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
        out[tag] = dict(deployable={k: agg(dep, k) for k in ['nrmse', 'P', 'VR', 'skill']},
                        oracle={k: agg(ora, k) for k in ['nrmse', 'P']},
                        uniform_nrmse=uni[tag]['deployable']['nrmse'][0])
        d = out[tag]['deployable']
        print(f"[{ti+1:2d}/30] {tag:12s} slope RELAX {d['nrmse'][0]:.3f} P{d['P'][0]:.2f} "
              f"| uniform {uni[tag]['deployable']['nrmse'][0]:.3f} | slope oracle {out[tag]['oracle']['nrmse'][0]:.3f}", flush=True)
    json.dump(out, open('results/direction3/slope_generalization.json', 'w'), indent=2)

    seas = {}
    for tag, r in out.items():
        s = f"{tag.split('_')[0]}_{tag.split('_')[1]}"; seas.setdefault(s, []).append((tag, r))
    print("\n=== per sea-state: slope vs uniform (deployable RELAX) ===")
    print(f"{'sea':9s} {'slope nRMSE':>11} {'slope P':>8} {'uniform nRMSE':>13}")
    sl_all, un_all = [], []
    for s in sorted(seas):
        rs = seas[s]
        sl = np.mean([r['deployable']['nrmse'][0] for _, r in rs])
        Pm = np.mean([r['deployable']['P'][0] for _, r in rs])
        un = np.mean([r['uniform_nrmse'] for _, r in rs])
        print(f"{s:9s} {sl:11.3f} {Pm:8.2f} {un:13.3f}")
        sl_all.append(sl); un_all.append(un)
    print(f"\nOVERALL slope {np.mean(sl_all):.3f}  vs  uniform {np.mean(un_all):.3f}  "
          f"(ST-DF-GP uniform 0.205)")
    print("saved results/direction3/slope_generalization.json")


if __name__ == '__main__':
    main()
