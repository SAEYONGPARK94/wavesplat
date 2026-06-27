"""
Main results with 3-pass iterative RELAX (improved component-id) — all 30 episodes.
Same protocol as d23 but with the multi-pass estimator (d26). Saves a parallel JSON.
"""
import os, sys, json
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
os.chdir(os.path.join(os.path.dirname(__file__), '..'))

from wavesplat.data_loader import EpisodeDataLoader
from wavesplat.model.tnse import fit_amplitudes_phases, render_wave_field
from d6_integrated_eval import tag_to_file
from d15_sequential_fusion import gather_frames
from d18_relax_estimator import MCAP
from d23_main_results import panel, K_TP, N_FRAMES
from d26_relax_iterative import iterative_relax

N_OUTER = 3


def main():
    gp = json.load(open('results/spacetime_gp_30ep.json'))['per_episode']
    uni = json.load(open('results/direction3/main_results.json'))
    out = {}
    for ti, tag in enumerate(gp):
        L = EpisodeDataLoader('data/' + tag_to_file(tag)); depth = L.depth; Hs = L.Hs_true
        wc = np.array(L.wave_components); tkx, tky = wc[:, 2], wc[:, 3]
        gx = L.gt_grid_x.numpy(); gy = L.gt_grid_y.numpy()
        n_back = int(round(K_TP * L.Tp_true / 0.1)); nfr = len(L)
        i0 = n_back + 3; i1 = nfr - 2
        if i1 - i0 < 4:
            i0 = max(n_back + 1, nfr - 60)
        frames = np.linspace(i0, i1, N_FRAMES).astype(int)
        dep = []
        for i_star in frames:
            fstar = L[int(i_star)]
            target = {'gx': gx, 'gy': gy, 't': float(fstar.t), 'gt': fstar.gt_field.numpy(), 'Hs': Hs}
            P, z = gather_frames(L, int(i_star), n_back)
            fld, _ = iterative_relax(P, z, depth, target, N_OUTER)
            dep.append(panel(fld, target['gt'], Hs))
        agg = lambda k: (float(np.mean([d[k] for d in dep])), float(np.std([d[k] for d in dep])))
        out[tag] = dict(deployable={k: agg(k) for k in ['nrmse', 'P', 'VR', 'skill']},
                        single_pass=uni[tag]['deployable']['nrmse'][0],
                        stdfgp=gp[tag]['fused']['nRMSE'])
        d = out[tag]['deployable']
        print(f"[{ti+1:2d}/30] {tag:12s} 3p {d['nrmse'][0]:.3f} P{d['P'][0]:.2f} | 1p {uni[tag]['deployable']['nrmse'][0]:.3f} | GP {gp[tag]['fused']['nRMSE']:.3f}", flush=True)
    json.dump(out, open('results/direction3/main_results_iter.json', 'w'), indent=2)
    seas = {}
    for tag, r in out.items():
        s = f"{tag.split('_')[0]}_{tag.split('_')[1]}"; seas.setdefault(s, []).append(r)
    print("\n=== per sea-state: 3-pass vs 1-pass vs ST-DF-GP ===")
    a3, a1, ag = [], [], []
    for s in sorted(seas):
        rs = seas[s]
        m3 = np.mean([r['deployable']['nrmse'][0] for r in rs]); Pm = np.mean([r['deployable']['P'][0] for r in rs])
        m1 = np.mean([r['single_pass'] for r in rs]); g = np.mean([r['stdfgp'] for r in rs])
        print(f"  {s:9s} 3p {m3:.3f} (P{Pm:.2f}) | 1p {m1:.3f} | GP {g:.3f}")
        a3.append(m3); a1.append(m1); ag.append(g)
    print(f"\nOVERALL 3-pass {np.mean(a3):.3f} | 1-pass {np.mean(a1):.3f} | ST-DF-GP {np.mean(ag):.3f}")
    print("saved results/direction3/main_results_iter.json")


if __name__ == '__main__':
    main()
