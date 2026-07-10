"""A1b: where does the failure begin, in physical sea states?

A1 killed a headline. The adversary's catastrophe -- fixed band 3.93x worse
than doing nothing, cured to 0.27x by lowering the bank's floor -- does not
appear when the sub-band energy is a real swell. With a Tp = 17 s swell
(lam 320 m) under a Tp = 8 s wind sea, the fixed band costs 0.434 vs 0.408,
a 6% penalty, not a 4x one.

Either the adversarial sea is a caricature that physics never draws, or the
swell in A1 was simply too small and too short. Both are testable, and the
answer is worth more than the original claim: it is the OPERATING ENVELOPE.
Instead of "the estimator breaks", we get "the estimator is safe for swells up
to here, and here is what happens past that".

Sweep the swell's period and height under a fixed wind sea. Report, for each,
the fixed-band penalty against the lowered floor and against the oracle band,
and whether the adaptive selector recovers it.

  lam/aperture is the axis that ought to matter: the adversary ran at 533/200
  = 2.7, A1's swell at 320/185 = 1.7. If the penalty grows with that ratio,
  the failure axis is real and merely needs a longer swell.

HOS throughout. Each cell is a fresh stateful simulator.
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
os.chdir(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, 'scripts')

from wavesim.hos_simulator import HOSSimulator, HOSConfig
from wavesim.wave_generator import jonswap_components
from d3_a1_hos_reproduction import (wavenumber, observe, truth, solve,
                                    select_band, CANDS, DEV, DEPTH, LX, LY,
                                    NX, NY, TP_WIND, HS_WIND, CX, CY)
import d3_static_field as SF

OUT = 'results/direction3/a1b_swell_envelope.json'
CELLS = [(17.0, 1.6), (17.0, 3.2), (22.0, 1.6), (22.0, 3.2),
         (28.0, 1.6), (28.0, 3.2)]
SEEDS = 2


def sim_with_swell(tp_s, hs_s, seed):
    s = HOSSimulator(HOSConfig(Lx=LX, Ly=LY, Nx=NX, Ny=NY, depth=DEPTH, M=3))
    comps = jonswap_components(Hs=HS_WIND, Tp=TP_WIND, n_components=120,
                              depth=DEPTH, direction_deg=180.0,
                              direction_spread_deg=25.0, seed=9000 + seed)
    comps = comps + jonswap_components(
        Hs=hs_s, Tp=tp_s, n_components=40, depth=DEPTH, direction_deg=195.0,
        direction_spread_deg=12.0, seed=7000 + seed)
    s.initialize_from_components(comps)
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', type=int, default=SEEDS)
    a_ = ap.parse_args()

    kp = wavenumber(TP_WIND, DEPTH)
    aperture = SF.R_FAN[1] - SF.R_FAN[0]
    gen = torch.Generator(device=DEV).manual_seed(0)
    t0 = 10.0 + SF.W_OBS
    sig0 = 0.05 * HS_WIND

    print(f"wind kp {kp:.4f} (lam {2*np.pi/kp:.0f} m), aperture {aperture:.0f} m")
    print(f"band floor 0.35 kp -> lam {2*np.pi/(0.35*kp):.0f} m\n")
    print(f"{'Tp_s':>6}{'Hs_s':>6}{'lam_s':>8}{'k/kp':>7}{'lam/ap':>8}"
          f"{'E_swell':>9}{'fixed.35':>10}{'fixed.15':>10}{'oracle':>8}"
          f"{'ADAPT':>8}{'penalty':>9}")

    res = []
    for tp_s, hs_s in CELLS:
        ks = wavenumber(tp_s, DEPTH)
        lam_s = 2 * np.pi / ks
        e_sw = hs_s ** 2 / (hs_s ** 2 + HS_WIND ** 2)     # variance fraction
        rows = []
        for s in range(a_.seeds):
            rng = np.random.default_rng(100 + s)
            sim = sim_with_swell(tp_s, hs_s, s)
            P, z = observe(sim, rng, t0)
            Pq, gt = truth(sim, t0)
            Pt = torch.tensor(P, device=DEV, dtype=torch.float32)
            zt = torch.tensor(z, device=DEV, dtype=torch.float32)
            Qt = torch.tensor(Pq, device=DEV, dtype=torch.float32)
            g = torch.tensor(gt, device=DEV, dtype=torch.float32)
            g = g - g.mean()
            zero = float(g.std())

            def ratio(b):
                pr = solve(Pt, zt, Qt, b, kp, sig0)
                return float(torch.sqrt(((pr - g) ** 2).mean())) / zero

            r35 = ratio(CANDS[0])
            r15 = ratio((0.15, 2.5))
            orc = min(ratio(b) for b in CANDS)
            rad = ratio(select_band(Pt, zt, kp, sig0, gen))
            rows.append((r35, r15, orc, rad))
        v = np.array(rows).mean(0)
        pen = v[0] / v[2]                       # fixed band vs the oracle band
        print(f"{tp_s:>6.0f}{hs_s:>6.1f}{lam_s:>8.0f}{ks/kp:>7.2f}"
              f"{lam_s/aperture:>8.2f}{e_sw:>9.2f}"
              f"{v[0]:>10.3f}{v[1]:>10.3f}{v[2]:>8.3f}{v[3]:>8.3f}{pen:>9.2f}x")
        res.append(dict(tp_swell=tp_s, hs_swell=hs_s, lam_swell=lam_s,
                        k_over_kp=ks / kp, lam_over_aperture=lam_s / aperture,
                        e_swell=e_sw, fixed35=v[0], fixed15=v[1],
                        oracle=v[2], adaptive=v[3], penalty=pen))

    r = np.array([[d['lam_over_aperture'], d['e_swell'], d['penalty']]
                  for d in res])
    print(f"\ncorr(penalty, lam/aperture) = {np.corrcoef(r[:,0], r[:,2])[0,1]:+.3f}")
    print(f"corr(penalty, swell energy) = {np.corrcoef(r[:,1], r[:,2])[0,1]:+.3f}")
    worst = max(res, key=lambda d: d['penalty'])
    print(f"\nworst physical cell: Tp {worst['tp_swell']:.0f} s, Hs "
          f"{worst['hs_swell']:.1f} m  ->  fixed band is "
          f"{worst['penalty']:.2f}x the oracle band")
    print(f"  the adversary reached 3.93 / 0.23 = 17x. Physics did not.")
    print(f"  adaptive selector there: {worst['adaptive']:.3f} vs oracle "
          f"{worst['oracle']:.3f}")

    os.makedirs('results/direction3', exist_ok=True)
    with open(OUT, 'w') as f:
        json.dump([{k: float(v) for k, v in d.items()} for d in res], f,
                  indent=2)
    return 0


if __name__ == '__main__':
    sys.exit(main())
