"""A1 step2 — FAIR comparison: the paper's own WaveSplat on the SAME finemesh episodes,
same /Hs normalizer. Pulls both:
  - init_nrmse  = WS nowcast reconstruction at the init frame  (RMSE/Hs)  ← fair vs cond nowcast
  - avg_nrmse   = WS sequential-tracker mean over frames       (RMSE/Hs)  ← the paper's product

Compares to the learned proposal's cond+polish nowcast (a1_cond_finemesh.json). Median over
3 estimator seeds per episode (WS has seed variance; median-of-k reporting convention).
"""
import json
import numpy as np

from wavesplat.run import run_estimation

FINE = 'data_finemesh_2026-07-23/'
SEEDS = (42, 1, 2)
cond = json.load(open('tmp/G1_waveprior_power/a1_cond_finemesh.json'))
EPISODES = list(cond.keys())


def main():
    out = {}
    print(f"{'episode':40s} {'nc':>3s} {'cond_now':>9s} {'WS_init_now':>11s} {'WS_tracked':>10s}")
    for ep in EPISODES:
        inits, tracks = [], []
        for s in SEEDS:
            r = run_estimation(FINE + ep, config={'adaptive_n': True, 'seed': s,
                                                  'device': 'cpu'})
            inits.append(r['init_nrmse'])
            tracks.append(r['avg_nRMSE'])
        ws_init = float(np.median(inits))
        ws_track = float(np.median(tracks))
        c = cond[ep]['cond_polish_nrmse_Hs']
        out[ep] = {'ncomp': cond[ep]['ncomp'], 'cond_nowcast': c,
                   'ws_init_nowcast': ws_init, 'ws_tracked': ws_track}
        print(f"{ep:40s} {cond[ep]['ncomp']:3d} {c:9.3f} {ws_init:11.3f} {ws_track:10.3f}",
              flush=True)

    with open('tmp/G1_waveprior_power/a1_ws_finemesh.json', 'w') as f:
        json.dump(out, f, indent=1)

    cn = np.array([v['cond_nowcast'] for v in out.values()])
    wi = np.array([v['ws_init_nowcast'] for v in out.values()])
    wt = np.array([v['ws_tracked'] for v in out.values()])
    print("\n=== A1 FAIR VERDICT (nRMSE/Hs, lower better) ===")
    print(f"  cond+polish nowcast   mean {cn.mean():.3f}")
    print(f"  WS init nowcast (fair) mean {wi.mean():.3f}   "
          f"(cond beats WS-nowcast by {(wi-cn).mean():+.3f}, cond<WS on {int((cn<wi).sum())}/{len(cn)})")
    print(f"  WS tracked (product)   mean {wt.mean():.3f}   "
          f"(cond beats WS-tracked by {(wt-cn).mean():+.3f})")
    print("\n  by ncomp (sparse→dense):")
    for ep, v in sorted(out.items(), key=lambda kv: kv[1]['ncomp']):
        print(f"    {v['ncomp']:2d}c  cond {v['cond_nowcast']:.3f}  "
              f"WS_init {v['ws_init_nowcast']:.3f}  WS_track {v['ws_tracked']:.3f}")


if __name__ == '__main__':
    main()
