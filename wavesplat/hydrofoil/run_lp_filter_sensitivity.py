"""
Phase D: LP filter alpha sensitivity for the feedforward controller.

Sweeps the low-pass filter smoothing coefficient alpha across 4 scenarios
(BF3 head/beam, BF4 head/beam) to justify the alpha=0.05 choice.

Output: results/hydrofoil_lp_sensitivity.json and console table.
"""

import json
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + '/..')

from wavesim.wave_generator import jonswap_components
from hydrofoil.config import candela_c8
from hydrofoil.controller import PDControllerConfig, FeedforwardConfig
from hydrofoil.sim import run_simulation


DURATION = 30.0
TRANSIENT = 5.0
SEED = 42
N_COMPONENTS = 15
DIR_SPREAD = 20.0
ALPHAS = [0.01, 0.03, 0.05, 0.10, 0.15, 0.20]
NRMSE = 0.20  # realistic WaveSplat accuracy

SCENARIOS = [
    ('BF3 Head', 0.88, 5.0, 180.0),
    ('BF3 Beam', 0.88, 5.0, 90.0),
    ('BF4 Head', 1.50, 6.5, 180.0),
    ('BF4 Beam', 1.50, 6.5, 90.0),
]


def _run(sc_name, Hs, Tp, direction, alpha):
    cfg = candela_c8()
    cfg['sim'].duration = DURATION
    comps = jonswap_components(Hs=Hs, Tp=Tp, n_components=N_COMPONENTS,
                               depth=cfg['sim'].depth,
                               direction_deg=direction,
                               direction_spread_deg=DIR_SPREAD,
                               seed=SEED)
    np.random.seed(SEED)
    ctrl = PDControllerConfig()

    # PD-only baseline
    r_base = run_simulation(cfg, comps, vessel_heading=0.0,
                            controller=ctrl, feedforward=None)
    mask = r_base.time > TRANSIENT
    z_base = float(np.std(r_base.z[mask]) * 100)
    a_base = float(np.std(r_base.heave_accel[mask]))

    # PD+FF with given alpha
    ff = FeedforwardConfig(
        preview_dt=0.5, K_ff=0.8, lp_alpha=alpha,
        noise_std=Hs * NRMSE, slope_noise_std=NRMSE * 0.1)
    np.random.seed(SEED)
    r = run_simulation(cfg, comps, vessel_heading=0.0,
                       controller=ctrl, feedforward=ff)
    mask = r.time > TRANSIENT
    z_ff = float(np.std(r.z[mask]) * 100)
    a_ff = float(np.std(r.heave_accel[mask]))

    return {
        'z_base_cm': z_base, 'a_base': a_base,
        'z_ff_cm': z_ff, 'a_ff': a_ff,
        'dz_pct': (1 - z_ff / z_base) * 100 if z_base > 0 else 0,
        'da_pct': (1 - a_ff / a_base) * 100 if a_base > 0 else 0,
    }


def main():
    results = {}
    print('=' * 90)
    print(f"LP filter alpha sensitivity (NRMSE = {NRMSE*100:.0f}%)")
    print('=' * 90)

    for sc_name, Hs, Tp, direction in SCENARIOS:
        results[sc_name] = {'Hs': Hs, 'Tp': Tp, 'direction': direction,
                            'by_alpha': {}}
        print(f"\n{sc_name} (Hs={Hs}, Tp={Tp}, dir={direction}°)")
        print(f"  {'alpha':>8} {'z_RMS cm':>10} {'a_RMS':>10} "
              f"{'Δheave':>10} {'Δaccel':>10}")
        for alpha in ALPHAS:
            r = _run(sc_name, Hs, Tp, direction, alpha)
            results[sc_name]['by_alpha'][str(alpha)] = r
            print(f"  {alpha:>8.2f} {r['z_ff_cm']:>10.2f} "
                  f"{r['a_ff']:>10.3f} "
                  f"{r['dz_pct']:>+9.1f}% {r['da_pct']:>+9.1f}%", flush=True)

        # Best
        best = max(results[sc_name]['by_alpha'].items(),
                   key=lambda kv: kv[1]['da_pct'])
        print(f"  >>> best alpha = {best[0]} "
              f"(accel {best[1]['da_pct']:+.1f}%)")

    # Save
    out_path = os.path.join(os.path.dirname(__file__), '..', 'results',
                            'hydrofoil_lp_sensitivity.json')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == '__main__':
    main()
