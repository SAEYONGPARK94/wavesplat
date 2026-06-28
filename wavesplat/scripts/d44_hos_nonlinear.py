"""
#2 — HOS nonlinear validation: does the linear RELAX estimator break on NONLINEAR (HOS)
waves, leaving a gap for learning to win?

The benchmark is linear (sum of cosines) — exactly where RELAX is near-optimal. Real waves
are weakly nonlinear (bound harmonics, nonlinear evolution). We simulate HOS (M=3) over a
window, sample LiDAR-like observations from the nonlinear surface, run RELAX, and measure:
  (1) RELAX field nRMSE on HOS data (does the linear estimator still work?),
  (2) the nonlinearity magnitude = best linear representation vs HOS GT (the ceiling any
      linear method — incl. the oracle — cannot beat; the headroom for a nonlinear learner).

Run: python3 scripts/d44_hos_nonlinear.py
"""
import os, sys, json, time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
os.chdir(os.path.join(os.path.dirname(__file__), '..'))

from wavesim.hos_simulator import HOSSimulator, HOSConfig
from wavesplat.model.tnse import fit_amplitudes_phases, render_wave_field
from d18_relax_estimator import relax_step
from d17_recursive_estimator import refine_params
from d26_relax_iterative import iterative_relax
from d23_main_results import panel
from d40_learned_acquisition import pool_xy

DEPTH = 50.0
SEAS = [('BF4_15c', 1.2, 7.0), ('BF5_30c', 2.0, 9.0)]
DT = 0.1


def build_window(Hs, Tp, seed, n_back, gx, gy, M=3):
    cfg = HOSConfig(Lx=500.0, Ly=400.0, Nx=64, Ny=64, depth=DEPTH, M=M)
    sim = HOSSimulator(cfg); sim.initialize_jonswap(Hs=Hs, Tp=Tp, seed=seed)
    Xg, Yg = np.meshgrid(gx, gy)
    pool = pool_xy()
    frames = []
    t = 0.0
    nsteps = n_back + 4
    for i in range(nsteps):
        if i > 0:
            sim.step(DT); t += DT
        # GT field on grid
        Zg, *_ = sim.evaluate_at(Xg, Yg)
        # obs: sample LiDAR (x,y) from pool, elevation from HOS + noise
        idx = np.random.default_rng(1000 + i).choice(len(pool), 60, replace=False)
        ox, oy = pool[idx, 0], pool[idx, 1]
        Zo, *_ = sim.evaluate_at(ox.reshape(1, -1), oy.reshape(1, -1))
        zo = Zo.ravel() + np.random.default_rng(i).normal(0, 0.02, len(ox))
        frames.append(dict(t=t, gt=Zg.astype(np.float32),
                           P=np.column_stack([ox, oy, np.full_like(ox, t)]).astype(np.float32),
                           z=zo.astype(np.float32)))
    return frames


def main():
    gx = np.linspace(10, 250, 80); gy = np.linspace(-100, 100, 60)
    out = {}; t0 = time.time()
    for tag, Hs, Tp in SEAS:
        n_back = int(round(2.2 * Tp / 0.1))
        res = {}
        for M in [1, 3]:                                  # M=1 linear control vs M=3 nonlinear
            fr = build_window(Hs, Tp, 0, n_back, gx, gy, M=M)
            i_star = len(fr) - 2
            target = {'gx': gx, 'gy': gy, 't': fr[i_star]['t'], 'gt': fr[i_star]['gt'],
                      'Hs': float(fr[i_star]['gt'].std() * 4)}
            P = np.vstack([f['P'] for f in fr[:i_star + 1]])
            z = np.concatenate([f['z'] for f in fr[:i_star + 1]])
            fld, _ = iterative_relax(P, z, DEPTH, target, 3)
            res[M] = panel(fld, target['gt'], target['Hs'])['nrmse']
        out[tag] = dict(relax_M1=float(res[1]), relax_M3=float(res[3]),
                        nonlinearity_gap=float(res[3] - res[1]))
        print(f"[{tag}] RELAX nRMSE  M=1(linear-HOS) {res[1]:.3f}  M=3(nonlinear-HOS) {res[3]:.3f}  "
              f"| pure-nonlinearity gap {res[3]-res[1]:+.3f}  ({time.time()-t0:.0f}s)", flush=True)
    json.dump(out, open('results/direction3/hos_nonlinear.json', 'w'), indent=2)
    print("\nInterpretation: if nl_ceiling << relax_hos, RELAX error is component-ID (not")
    print("nonlinearity) → little learning headroom. If nl_ceiling is large, nonlinearity is")
    print("a real gap a nonlinear learner could exploit.")
    print("saved results/direction3/hos_nonlinear.json")


if __name__ == '__main__':
    main()
