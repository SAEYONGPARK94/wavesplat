"""
Does active sensing win for the FORECAST-relevant objective (free-aim)?

#1 (d42) showed: for FIELD reconstruction at the measurement time, free-aim random uniform
coverage beats D-optimal/learned concentration (coverage dominates). But FORECASTING
propagates the estimated COMPONENTS forward — it depends on spectrum accuracy, which is
exactly what D-optimal optimises, not spatial coverage. So active sensing may win for
forecast even when it loses for nowcast field reconstruction.

Here: measure over the window (free-aim pool, budget K) with random / learned / greedy
selection; estimate components; FORECAST the field at i_star + horizon; nRMSE vs true future.

Run: python3 scripts/d43_forecast_sensing.py [--horizon 20]
"""
import os, sys, json, argparse
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
os.chdir(os.path.join(os.path.dirname(__file__), '..'))

from wavesplat.data_loader import EpisodeDataLoader
from d6_integrated_eval import tag_to_file
from d23_main_results import K_TP, panel
from d40_learned_acquisition import AcqNet, coarse_theta, greedy_dopt, CKPT, DEVICE
import d42_beam_steering as BS
import d1_crb as C1

EPISODES = ['BF4_15c_s0', 'BF4_30c_s0', 'BF5_15c_s0', 'BF5_30c_s1', 'BF4_15c_s1', 'BF5_15c_s0']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--frames', type=int, default=2)
    ap.add_argument('--npool', type=int, default=1000)
    ap.add_argument('--K', type=int, default=150)
    ap.add_argument('--seedK', type=int, default=80)
    ap.add_argument('--horizon', type=int, default=20)   # frames ahead (0.1s each) = 2 s
    ap.add_argument('--episodes', nargs='+', default=EPISODES)
    a = ap.parse_args()
    model = AcqNet().to(DEVICE); model.load_state_dict(torch.load(CKPT, map_location=DEVICE)); model.eval()
    rng = np.random.default_rng(0)
    out = {}
    for tag in a.episodes:
        L = EpisodeDataLoader('data/' + tag_to_file(tag)); depth = L.depth; Hs = L.Hs_true
        gx = L.gt_grid_x.numpy(); gy = L.gt_grid_y.numpy()
        comp_full = np.array(L.wave_components)
        Tobs = K_TP * L.Tp_true; n_back = int(round(Tobs / 0.1)); nfr = len(L)
        i0 = n_back + 3; i1 = nfr - 2 - a.horizon
        if i1 - i0 < 4:
            i0 = max(n_back + 1, nfr - 60 - a.horizon)
        sel = np.linspace(i0, max(i0 + 1, i1), a.frames).astype(int)
        rec = {k: {'now': [], 'fc': []} for k in ['rand', 'learned', 'greedy']}
        for i_star in sel:
            ff = L[int(i_star + a.horizon)]
            tgt_now = {'gx': gx, 'gy': gy, 't': float(L[int(i_star)].t), 'gt': L[int(i_star)].gt_field.numpy(), 'Hs': Hs}
            tgt_fc = {'gx': gx, 'gy': gy, 't': float(ff.t), 'gt': ff.gt_field.numpy(), 'Hs': Hs}
            pool = BS.aim_pool(L, i_star, n_back, a.npool, rng)
            zpool = BS.measure(comp_full, pool, rng=rng)
            # selections
            sels = {}
            sels['rand'] = rng.choice(a.npool, a.K, replace=False)
            seed = rng.choice(a.npool, a.seedK, replace=False)
            cth = coarse_theta(pool, zpool, depth, seed)
            sc = BS.policy_score(model, cth, pool); sels['learned'] = np.argsort(-sc)[:a.K]
            gt_t = torch.tensor(cth[:, [2, 0, 1, 3]], dtype=C1.DT, device=C1.DEVICE)
            sels['greedy'] = greedy_dopt(gt_t, pool, a.K, depth)
            for k, idx in sels.items():
                # nowcast field AND forecast field from the SAME fitted components
                rec[k]['now'].append(BS.relax_nr(pool[idx], zpool[idx], depth, tgt_now))
                rec[k]['fc'].append(BS.relax_nr(pool[idx], zpool[idx], depth, tgt_fc))
        out[tag] = {k: {'now': float(np.mean(v['now'])), 'fc': float(np.mean(v['fc']))} for k, v in rec.items()}
        o = out[tag]
        print(f"[{tag:12s}] NOWcast  r{o['rand']['now']:.3f} l{o['learned']['now']:.3f} g{o['greedy']['now']:.3f}  "
              f"| FOREcast(+{a.horizon*0.1:.0f}s)  r{o['rand']['fc']:.3f} l{o['learned']['fc']:.3f} g{o['greedy']['fc']:.3f}", flush=True)
    json.dump(out, open('results/direction3/forecast_sensing.json', 'w'), indent=2)
    m = lambda k, w: float(np.mean([out[t][k][w] for t in out]))
    print(f"\n=== OVERALL (free-aim, K={a.K}) ===")
    print(f"  NOWcast field:  rand {m('rand','now'):.3f} | learned {m('learned','now'):.3f} | greedy {m('greedy','now'):.3f}")
    print(f"  FOREcast +{a.horizon*0.1:.0f}s:  rand {m('rand','fc'):.3f} | learned {m('learned','fc'):.3f} | greedy {m('greedy','fc'):.3f}")
    g = lambda k: 100 * (m('rand', k) - min(m('learned', k), m('greedy', k))) / m('rand', k)
    print(f"  active-sensing gain vs random:  nowcast {g('now'):+.0f}%  |  forecast {g('fc'):+.0f}%")
    print("saved results/direction3/forecast_sensing.json")


if __name__ == '__main__':
    main()
