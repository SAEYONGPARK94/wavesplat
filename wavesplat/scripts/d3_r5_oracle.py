"""R5 — oracle decomposition: WHERE does the HOS error live? (Fable's discriminating test)

R4 only ruled out render-side bound harmonics (which are ~0.3% → near-tautological). R5 actually
separates the 0.25 HOS error into three additive floors, CPU-only, no training/GPU:

  O1_32 : top-32 free modes (HOS init spectrum, kx>0 half-plane) with TRUE A,φ → linear render
          = the floor set by the 32-atom REPRESENTATION itself (perfect learning can't beat it).
  O1_64 : top-64 → truncation sensitivity.
  O2    : true (kx,ky) of top-32, but A,φ RE-FIT from the 400 sparse obs → the floor when k is
          known perfectly but amplitudes/phases come from sparse data (OBS-INFORMATION limit).

Decomposition of a learned method's error L:
  L = O1(representation)  +  (O2−O1)(obs information)  +  (L−O2)(k-identification — the ONLY
      part learning can actually improve).
Gate: if (L−O2) < 0.05 there is little for HOS-augmented training to win → pivot to representation
(bigger M / continuous head) or aperture fusion instead.

Same seas/seeds as d3_hos_eval so O1/O2 are floors for the SAME windows. Re-inits a fresh sim with
the same (Hs,Tp,seed) to read the t=0 free modes (deterministic == build_window's init).

Run: python3 scripts/d3_r5_oracle.py
"""
import os, sys, json, time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
os.chdir(os.path.join(os.path.dirname(__file__), '..'))

from d23_main_results import panel
from wavesplat.model.tnse import fit_amplitudes_phases, render_wave_field
from wavesim.hos_simulator import HOSSimulator, HOSConfig
import d44_hos_nonlinear as H

SEAS = [('BF3', 0.6, 5.5), ('BF4', 1.2, 7.0), ('BF5', 2.0, 9.0)]
SEEDS = [0, 1, 2, 3, 4]
GX = np.linspace(10, 250, 80); GY = np.linspace(-100, 100, 60)
DEPTH = H.DEPTH


def init_modes(Hs, Tp, seed):
    """Deterministic replica of build_window's init → top free modes (kx>0 half-plane)."""
    sim = HOSSimulator(HOSConfig(Lx=500.0, Ly=400.0, Nx=64, Ny=64, depth=DEPTH, M=3))
    sim.initialize_jonswap(Hs=Hs, Tp=Tp, seed=seed)      # same defaults as build_window
    eh = sim.eta_hat; mag = np.where(sim.KX > 1e-9, np.abs(eh), 0.0)
    order = np.argsort(mag.ravel())[::-1]
    kx = sim.KX.ravel()[order]; ky = sim.KY.ravel()[order]
    A = 2.0 * np.abs(eh).ravel()[order]; phi = np.angle(eh).ravel()[order]
    return kx.astype(np.float32), ky.astype(np.float32), A.astype(np.float32), phi.astype(np.float32)


def main():
    print("R5 oracle decomposition (O1_32 repr / O1_64 / O2 obs-info floors)", flush=True)
    print(f"{'window':10s} {'O1_32':>7} {'O1_64':>7} {'O2':>7}", flush=True)
    acc = {'O1_32': [], 'O1_64': [], 'O2': []}; wt = []; t0 = time.time()
    for tag, Hs, Tp in SEAS:
        n_back = int(round(2.2 * Tp / 0.1))
        for seed in SEEDS:
            fr = H.build_window(Hs, Tp, seed, n_back, GX, GY, M=3)
            i_star = len(fr) - 2
            gt = fr[i_star]['gt']; t = fr[i_star]['t']; Hs_t = float(gt.std() * 4)
            P = np.vstack([f['P'] for f in fr[:i_star + 1]]); z = np.concatenate([f['z'] for f in fr[:i_star + 1]])
            idx = np.random.default_rng(seed).choice(len(z), min(400, len(z)), replace=False)
            Ps, zs = P[idx].astype(np.float32), z[idx].astype(np.float32)
            kx, ky, A, phi = init_modes(Hs, Tp, seed)
            # O1: true A,phi, top-N
            for N, key in ((32, 'O1_32'), (64, 'O1_64')):
                f = render_wave_field(kx[:N], ky[:N], A[:N], phi[:N], DEPTH, GX, GY, t)
                acc[key].append(panel(f, gt, Hs_t)['nrmse'])
            # O2: true (kx,ky) top-32, re-fit A,phi from obs
            obs = np.column_stack([Ps[:, 0], Ps[:, 1], Ps[:, 2], zs]).astype(np.float32)
            A2, phi2 = fit_amplitudes_phases(kx[:32], ky[:32], obs, DEPTH, lambda_reg=0.5)
            f2 = render_wave_field(kx[:32], ky[:32], A2, phi2, DEPTH, GX, GY, t)
            acc['O2'].append(panel(f2, gt, Hs_t)['nrmse'])
            wt.append(tag)
            print(f"{tag+' s'+str(seed):10s} {acc['O1_32'][-1]:7.3f} {acc['O1_64'][-1]:7.3f} "
                  f"{acc['O2'][-1]:7.3f}  ({time.time()-t0:.0f}s)", flush=True)
    wt = np.array(wt); seas = [s[0] for s in SEAS]
    print("\n=== R5 floors (mean; per-sea) ===", flush=True)
    summary = {}
    for k in ('O1_32', 'O1_64', 'O2'):
        arr = np.array(acc[k]); ps = {s: float(arr[wt == s].mean()) for s in seas}
        summary[k] = dict(mean=float(arr.mean()), per_sea=ps)
        print(f"  {k:6s} {arr.mean():.3f}  |  " + "  ".join(f"{s} {ps[s]:.3f}" for s in seas), flush=True)
    print("\ninterpret: learned_HOS ≈ 0.25. gap to O2 = k-ID headroom (learnable);"
          " O2−O1_32 = obs-info; O1_32 = representation floor.", flush=True)
    json.dump(dict(summary=summary, win_tags=list(wt)),
              open('results/direction3/r5_oracle.json', 'w'), indent=2)
    print("saved results/direction3/r5_oracle.json", flush=True)


if __name__ == '__main__':
    main()
