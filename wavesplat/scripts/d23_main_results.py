"""
Phase 1 — consolidated main results (paper centerpiece).

Deployable RELAX sequential estimator across ALL 30 benchmark episodes, multi-frame,
with a period-scaled fusion window (T_obs = k·Tp, per the synthetic-aperture finding),
full metric panel (nRMSE/Pearson/VR/skill), vs ST-DF-GP (fused, from JSON) and the
oracle. Fair (same observations), robust (multi-frame mean±std), honest (full panel).

Run: python3 scripts/d23_main_results.py
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

K_TP = 2.2      # fusion window = K_TP * Tp (period-scaled, synthetic-aperture consistent)
N_FRAMES = 5


def panel(f, gt, Hs):
    return dict(nrmse=float(np.sqrt(np.mean((f - gt)**2)) / max(Hs, 1e-6)),
                P=float(pearson_correlation(f, gt)), VR=float(variance_ratio(f, gt)),
                skill=float(skill_score(f, gt)))


def main():
    gp = json.load(open('results/spacetime_gp_30ep.json'))['per_episode']
    tags = list(gp.keys())
    out = {}
    for ti, tag in enumerate(tags):
        L = EpisodeDataLoader('data/' + tag_to_file(tag)); depth = L.depth; Hs = L.Hs_true
        wc = np.array(L.wave_components); tkx, tky = wc[:, 2], wc[:, 3]
        gx = L.gt_grid_x.numpy(); gy = L.gt_grid_y.numpy()
        Tobs = K_TP * L.Tp_true; n_back = int(round(Tobs / 0.1))
        nfr = len(L)
        i0 = n_back + 3; i1 = nfr - 2
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
        out[tag] = dict(Tp=L.Tp_true, Tobs=Tobs,
                        deployable={k: agg(dep, k) for k in ['nrmse', 'P', 'VR', 'skill']},
                        oracle={k: agg(ora, k) for k in ['nrmse', 'P', 'skill']},
                        stdfgp_fused=gp[tag]['fused']['nRMSE'],
                        stdfgp_skill=gp[tag]['fused'].get('Skill', None))
        d = out[tag]['deployable']
        print(f"[{ti+1:2d}/30] {tag:12s} Tobs={Tobs:.0f}s  RELAX {d['nrmse'][0]:.3f}±{d['nrmse'][1]:.3f} "
              f"P{d['P'][0]:.2f} | GP {gp[tag]['fused']['nRMSE']:.3f} | oracle {out[tag]['oracle']['nrmse'][0]:.3f}",
              flush=True)
    json.dump(out, open('results/direction3/main_results.json', 'w'), indent=2)

    # aggregate per sea-state
    seas = {}
    for tag, r in out.items():
        s = f"{tag.split('_')[0]}_{tag.split('_')[1]}"
        seas.setdefault(s, []).append(r)
    print("\n=== per sea-state (mean over seeds) ===")
    print(f"{'sea':9s} {'RELAX nRMSE':>12} {'RELAX P':>8} {'RELAX skill':>11} {'ST-DF-GP':>9} {'oracle':>7}")
    agg_all = {'rel': [], 'gp': [], 'ora': [], 'P': [], 'sk': []}
    for s in sorted(seas):
        rs = seas[s]
        rel = np.mean([r['deployable']['nrmse'][0] for r in rs])
        Pm = np.mean([r['deployable']['P'][0] for r in rs])
        skm = np.mean([r['deployable']['skill'][0] for r in rs])
        gpm = np.mean([r['stdfgp_fused'] for r in rs])
        oram = np.mean([r['oracle']['nrmse'][0] for r in rs])
        win = '<RELAX' if rel < gpm else ''
        print(f"{s:9s} {rel:12.3f} {Pm:8.2f} {skm:11.2f} {gpm:9.3f} {oram:7.3f} {win}")
        agg_all['rel'].append(rel); agg_all['gp'].append(gpm); agg_all['ora'].append(oram)
        agg_all['P'].append(Pm); agg_all['sk'].append(skm)
    print(f"\nOVERALL  RELAX {np.mean(agg_all['rel']):.3f} (P {np.mean(agg_all['P']):.2f}, "
          f"skill {np.mean(agg_all['sk']):.2f}) | ST-DF-GP {np.mean(agg_all['gp']):.3f} | "
          f"oracle {np.mean(agg_all['ora']):.3f}")
    print("saved results/direction3/main_results.json")


if __name__ == '__main__':
    main()
