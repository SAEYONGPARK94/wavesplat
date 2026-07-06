"""
d3_e2e_velff.py — lever #2-of-queue (2026-07-06 night): the estimator's
slope is its weakest output (corr 0.68, amp 0.62) while vertical
velocity is strong (0.85). Does DOWN-WEIGHTING the slope term in the FF
law (alpha_wave = -lam_s * slope - w/U) recover more of the
perfect-knowledge benefit with the SAME estimator?

Sweep lam_s in {0, 0.5, 1} for the Stream-FF provider (GT-FF stays
lam_s=1), paper scenarios (BF3/BF4 head seas), 2 seeds.

Usage: python3 scripts/d3_e2e_velff.py [--seeds 2]
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
OUT = 'results/direction3/e2e_velff.json'
LAMS = (0.0, 0.5, 1.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', type=int, default=2)
    a = ap.parse_args()
    cfg = candela_c8()
    U = cfg['sim'].speed
    depth = cfg['sim'].depth
    out = {}
    for sea, sc in SEAS.items():
        rows = []
        for seed in range(a.seeds):
            comps = jonswap_components(Hs=sc['Hs'], Tp=sc['Tp'],
                                       n_components=15, depth=depth,
                                       direction_deg=180.0,
                                       direction_spread_deg=20.0,
                                       seed=7000 + seed)
            est = estimate_components(comps, U, sc['Hs'], sc['Tp'],
                                      depth, seed)
            ff = FeedforwardConfig(preview_dt=0.5, K_ff=0.8,
                                   noise_std=0.0, slope_noise_std=0.0)
            res = {}
            r = run_simulation(cfg, comps, vessel_heading=0.0,
                               controller=PDControllerConfig(),
                               feedforward=None)
            res['PD'] = float(np.std(r.heave_accel[r.time > 5.0]))
            r = run_simulation(cfg, comps, vessel_heading=0.0,
                               controller=PDControllerConfig(),
                               feedforward=ff)
            res['GT_FF'] = float(np.std(r.heave_accel[r.time > 5.0]))
            for lam_s in LAMS:
                def prov(state, fr, re_, sim, _c, t, hd, fc,
                         _l=lam_s):
                    info = _compute_feedforward_info(
                        state, fr, re_, sim, est, t, hd, fc)
                    info.slope_front *= _l
                    info.slope_rear *= _l
                    return info
                r = run_simulation(cfg, comps, vessel_heading=0.0,
                                   controller=PDControllerConfig(),
                                   feedforward=ff, ff_info_provider=prov)
                res[f'St_l{lam_s:g}'] = float(
                    np.std(r.heave_accel[r.time > 5.0]))
            red_gt = 1 - res['GT_FF'] / res['PD']
            line = f"[{sea} seed{seed}] PD {res['PD']:.3f} GT -{red_gt:.0%}"
            row = dict(seed=seed, **res, red_gt=red_gt)
            for lam_s in LAMS:
                red = 1 - res[f'St_l{lam_s:g}'] / res['PD']
                frac = red / red_gt if red_gt > 1e-6 else np.nan
                row[f'frac_l{lam_s:g}'] = frac
                line += f" | l={lam_s:g}: -{red:.0%} ({frac:+.0%})"
            rows.append(row)
            print(line, flush=True)
        out[sea] = rows
    for lam_s in LAMS:
        fr = [r[f'frac_l{lam_s:g}'] for sea in out for r in out[sea]]
        print(f"lam_s={lam_s:g}: mean fraction of perfect "
              f"{np.nanmean(fr):+.0%}")
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"saved {OUT}")


if __name__ == '__main__':
    main()
