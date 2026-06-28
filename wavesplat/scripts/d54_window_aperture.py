"""
The last physical lever — does a LONGER window (synthetic aperture, D_eff=L+c_g·T_obs)
lower the continuous-HOS reconstruction error below the ~0.233 single-window info-limit?

Generate one LONG HOS window (5·Tp), then reconstruct (F1 dense+smooth, dispersion-
constrained continuous) using progressively longer sub-windows of obs ending at the target.
If error drops with T_obs, time fusion / synthetic aperture is a real lever — and a
physics-informed temporal network is the honest learning-central contribution (it would
exploit the growing aperture optimally + provide UQ).

Run: python3 scripts/d54_window_aperture.py
"""
import os, sys, json, time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
os.chdir(os.path.join(os.path.dirname(__file__), '..'))

from wavesim.hos_simulator import HOSSimulator, HOSConfig
from d23_main_results import panel
from d26_relax_iterative import iterative_relax
from d51_F1_continuous_density import f1_reconstruct
from d40_learned_acquisition import pool_xy

DEPTH = 50.0; DT = 0.1
SEAS = [('BF4', 1.2, 7.0), ('BF5', 2.0, 9.0)]
TFRACS = [1.0, 2.2, 3.5, 5.0]                 # window length in units of Tp


def main():
    gx = np.linspace(10, 250, 80); gy = np.linspace(-100, 100, 60)
    pool = pool_xy(); out = {}; t0 = time.time()
    for tag, Hs, Tp in SEAS:
        n_steps = int(round(5.0 * Tp / 0.1)) + 4
        cfg = HOSConfig(Lx=500.0, Ly=400.0, Nx=64, Ny=64, depth=DEPTH, M=3)
        sim = HOSSimulator(cfg); sim.initialize_jonswap(Hs=Hs, Tp=Tp, seed=0)
        Xg, Yg = np.meshgrid(gx, gy)
        ts, Ps, zs = [], [], []
        gt_star = None
        for i in range(n_steps):
            if i > 0:
                sim.step(DT)
            t = i * DT
            oi = np.random.default_rng(1000 + i).choice(len(pool), 60, replace=False)
            ox, oy = pool[oi, 0], pool[oi, 1]
            Zo, *_ = sim.evaluate_at(ox.reshape(1, -1), oy.reshape(1, -1))
            zo = Zo.ravel() + np.random.default_rng(i).normal(0, 0.02, len(ox))
            Ps.append(np.column_stack([ox, oy, np.full_like(ox, t)])); zs.append(zo); ts.append(t)
            if i == n_steps - 2:
                Zg, *_ = sim.evaluate_at(Xg, Yg); gt_star = Zg.astype(np.float32); t_star = t
        P = np.vstack(Ps).astype(np.float32); z = np.concatenate(zs).astype(np.float32)
        Hs_eff = float(gt_star.std() * 4)
        target = {'gx': gx, 'gy': gy, 't': t_star, 'gt': gt_star, 'Hs': Hs_eff}
        row = {}
        for tf in TFRACS:
            Tobs = tf * Tp
            mask = P[:, 2] >= (t_star - Tobs - 1e-6)
            Pm, zm = P[mask], z[mask]
            nf = f1_reconstruct(Pm, zm, target, ng=40, lam=5.0)
            nr = panel(iterative_relax(Pm, zm, DEPTH, target, 3)[0], gt_star, Hs_eff)['nrmse']
            row[f'{tf:.1f}Tp'] = dict(F1=float(nf), RELAX=float(nr), n_obs=int(len(zm)))
            print(f"[{tag}] T_obs={tf:.1f}Tp ({Tobs:.0f}s, {len(zm)} obs)  F1 {nf:.3f} | RELAX {nr:.3f}  "
                  f"({time.time()-t0:.0f}s)", flush=True)
        out[tag] = row
    json.dump(out, open('results/direction3/window_aperture.json', 'w'), indent=2)
    print("\n=== F1 nRMSE vs window length (mean over conditions) ===")
    for tf in TFRACS:
        k = f'{tf:.1f}Tp'; mf = np.mean([out[t][k]['F1'] for t in out]); mr = np.mean([out[t][k]['RELAX'] for t in out])
        print(f"  {k:7s}  F1 {mf:.3f} | RELAX {mr:.3f}")
    f_short = np.mean([out[t]['2.2Tp']['F1'] for t in out]); f_long = np.mean([out[t]['5.0Tp']['F1'] for t in out])
    print(f"\n  F1 improves {f_short-f_long:+.3f} from 2.2Tp to 5.0Tp  →")
    print(f"  {'SYNTHETIC APERTURE IS A LEVER (longer window beats 0.233 → physics-temporal net worth it)' if f_long < 0.225 else 'window saturated (~0.233 floor; time not a lever here)'}")
    print("saved results/direction3/window_aperture.json")


if __name__ == '__main__':
    main()
