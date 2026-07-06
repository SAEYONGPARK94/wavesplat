"""
d3_e2e_ctrl_tune.py — controller-side fix (2026-07-06 night, queue #3):
the FF low-pass (lp_alpha 0.05 -> ~0.8 Hz cutoff) sits right on the
encounter band (0.5 Hz at 22 kt) => phase lag ~0.17 s eats the preview.
Stage 1: sweep lp_alpha x preview_dt with GT-FF (cheap 15-comp sims) to
find the controller's own optimum margin. Stage 2: rerun Stream-FF at
the best config -> does the recovered fraction rise?

Usage: python3 scripts/d3_e2e_ctrl_tune.py [--seeds 2]
"""
import os
import sys
import json
import argparse

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
os.chdir(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, 'scripts')

from wavesim.wave_generator import jonswap_components
from hydrofoil.config import candela_c8
from hydrofoil.controller import FeedforwardConfig, PDControllerConfig
from hydrofoil.sim import run_simulation, _compute_feedforward_info
from d3_e2e_stream_ff import estimate_components, SEAS

G = 9.81
OUT = 'results/direction3/e2e_ctrl_tune.json'
ALPHAS = (0.05, 0.2, 1.0)
PREVIEWS = (0.2, 0.5, 1.0)


def a_rms(r):
    return float(np.std(r.heave_accel[r.time > 5.0]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', type=int, default=2)
    a = ap.parse_args()
    cfg = candela_c8()
    U = cfg['sim'].speed
    depth = cfg['sim'].depth
    out = {'stage1': [], 'stage2': []}
    # ---------- stage 1: GT-FF controller sweep
    best = {}
    for sea, sc in SEAS.items():
        accs = {}
        for seed in range(a.seeds):
            comps = jonswap_components(Hs=sc['Hs'], Tp=sc['Tp'],
                                       n_components=15, depth=depth,
                                       direction_deg=180.0,
                                       direction_spread_deg=20.0,
                                       seed=7000 + seed)
            pd_ = a_rms(run_simulation(cfg, comps, vessel_heading=0.0,
                                       controller=PDControllerConfig(),
                                       feedforward=None))
            for al in ALPHAS:
                for pv in PREVIEWS:
                    ff = FeedforwardConfig(preview_dt=pv, K_ff=0.8,
                                           lp_alpha=al, noise_std=0.0,
                                           slope_noise_std=0.0)
                    r = a_rms(run_simulation(
                        cfg, comps, vessel_heading=0.0,
                        controller=PDControllerConfig(), feedforward=ff))
                    accs.setdefault((al, pv), []).append(1 - r / pd_)
        for (al, pv), v in accs.items():
            out['stage1'].append(dict(sea=sea, lp_alpha=al, preview=pv,
                                      red_gt=float(np.mean(v))))
        b = max(accs.items(), key=lambda kv: np.mean(kv[1]))
        best[sea] = b[0]
        print(f"[{sea}] GT-FF sweep best: lp_alpha={b[0][0]:g} "
              f"preview={b[0][1]:g} -> -{np.mean(b[1]):.0%} "
              f"(기존 0.05/0.5: -{np.mean(accs[(0.05, 0.5)]):.0%})",
              flush=True)

    # ---------- stage 2: Stream-FF at the tuned controller
    for sea, sc in SEAS.items():
        al, pv = best[sea]
        ff = FeedforwardConfig(preview_dt=pv, K_ff=0.8, lp_alpha=al,
                               noise_std=0.0, slope_noise_std=0.0)
        for seed in range(a.seeds):
            comps = jonswap_components(Hs=sc['Hs'], Tp=sc['Tp'],
                                       n_components=15, depth=depth,
                                       direction_deg=180.0,
                                       direction_spread_deg=20.0,
                                       seed=7000 + seed)
            est = estimate_components(comps, U, sc['Hs'], sc['Tp'],
                                      depth, seed)

            def prov(state, fr, re_, sim, _c, t, hd, fc):
                return _compute_feedforward_info(state, fr, re_, sim,
                                                 est, t, hd, fc)

            pd_ = a_rms(run_simulation(cfg, comps, vessel_heading=0.0,
                                       controller=PDControllerConfig(),
                                       feedforward=None))
            gt_ = a_rms(run_simulation(cfg, comps, vessel_heading=0.0,
                                       controller=PDControllerConfig(),
                                       feedforward=ff))
            st_ = a_rms(run_simulation(cfg, comps, vessel_heading=0.0,
                                       controller=PDControllerConfig(),
                                       feedforward=ff,
                                       ff_info_provider=prov))
            red_gt = 1 - gt_ / pd_
            red_st = 1 - st_ / pd_
            frac = red_st / red_gt if red_gt > 1e-6 else np.nan
            out['stage2'].append(dict(sea=sea, seed=seed, lp_alpha=al,
                                      preview=pv, red_gt=red_gt,
                                      red_st=red_st,
                                      frac=float(frac)))
            print(f"[{sea} seed{seed} tuned] GT -{red_gt:.0%} | "
                  f"Stream -{red_st:.0%} ({frac:+.0%} of perfect)",
                  flush=True)
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"saved {OUT}")


if __name__ == '__main__':
    main()
