"""
Does STRONG nonlinearity (steep waves) create a genuine learning opportunity?

Nonlinear HOS develops BOUND harmonics — energy phase-locked at sum/difference wavenumbers
that lie OFF the linear dispersion shell. RELAX/F1/DPS are ALL dispersion-constrained, so
NONE can represent bound harmonics. At the benchmark steepness (ε~0.25-1%) this is tiny, but
at high steepness it grows — there an UNCONSTRAINED learned field model could win.

Here we sweep steepness (Hs/Tp) and measure, per level:
  - dispersion-constrained CEILING = best linear-dispersion fit to the DENSE HOS GT field
    (what NO dispersion-constrained method can beat = the bound-harmonic fraction),
  - RELAX from sparse obs.
If the ceiling grows with steepness, the strongly-nonlinear regime is where an unconstrained
learner has real headroom.

Run: python3 scripts/d53_steepness_nonlinearity.py
"""
import os, sys, json, time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
os.chdir(os.path.join(os.path.dirname(__file__), '..'))

from wavesim.hos_simulator import HOSSimulator, HOSConfig
from d23_main_results import panel
from d26_relax_iterative import iterative_relax
from d51_F1_continuous_density import disp_grid
from d40_learned_acquisition import pool_xy

G = 9.81; DEPTH = 50.0; DT = 0.1
# (label, Hs, Tp, approx steepness ε≈Hs·kp/2)
LEVELS = [('mild', 1.0, 10.0), ('moderate', 2.0, 8.0), ('steep', 3.0, 7.0), ('very_steep', 3.5, 6.0)]


def dispersion_ceiling(gt, gx, gy, t, lam=5.0, ng=40):
    """Best linear-dispersion fit to the DENSE GT field → residual = bound-harmonic fraction."""
    kx, ky, om = disp_grid(ng)
    Xg, Yg = np.meshgrid(gx, gy); xr = Xg.ravel(); yr = Yg.ravel()
    # subsample dense GT points for the fit
    idx = np.random.default_rng(0).choice(len(xr), 1500, replace=False)
    psi = kx[None]*xr[idx][:,None] + ky[None]*yr[idx][:,None] - om[None]*t
    B = np.concatenate([np.cos(psi), -np.sin(psi)], axis=1)
    A = B.T@B + lam*np.eye(2*len(kx)); coef = np.linalg.solve(A, B.T@gt.ravel()[idx])
    psg = kx[None]*xr[:,None] + ky[None]*yr[:,None] - om[None]*t
    Bg = np.concatenate([np.cos(psg), -np.sin(psg)], axis=1)
    fit = (Bg@coef).reshape(gt.shape)
    Hs = float(gt.std()*4)
    return panel(fit.astype(np.float32), gt, Hs)['nrmse']


def main():
    gx = np.linspace(10, 250, 80); gy = np.linspace(-100, 100, 60)
    pool = pool_xy(); out = {}; t0 = time.time()
    for lab, Hs, Tp in LEVELS:
        cfg = HOSConfig(Lx=500.0, Ly=400.0, Nx=64, Ny=64, depth=DEPTH, M=3)
        try:
            sim = HOSSimulator(cfg); sim.initialize_jonswap(Hs=Hs, Tp=Tp, seed=0)
            n_back = int(round(2.2*Tp/0.1)); Xg, Yg = np.meshgrid(gx, gy)
            frames = []
            for i in range(n_back+4):
                if i > 0:
                    sim.step(DT)
                Zg, *_ = sim.evaluate_at(Xg, Yg)
                oi = np.random.default_rng(1000+i).choice(len(pool), 60, replace=False)
                ox, oy = pool[oi,0], pool[oi,1]
                Zo, *_ = sim.evaluate_at(ox.reshape(1,-1), oy.reshape(1,-1))
                frames.append((i*DT, Zg.astype(np.float32), ox, oy, Zo.ravel()+np.random.default_rng(i).normal(0,0.02,len(ox))))
            i_star = len(frames)-2
            t, gt, *_ = frames[i_star]
            ceil = dispersion_ceiling(gt, gx, gy, t)
            P = np.vstack([np.column_stack([f[2], f[3], np.full_like(f[2], f[0])]) for f in frames[:i_star+1]])
            z = np.concatenate([f[4] for f in frames[:i_star+1]]).astype(np.float32)
            target = {'gx': gx, 'gy': gy, 't': t, 'gt': gt, 'Hs': float(gt.std()*4)}
            relax = panel(iterative_relax(P.astype(np.float32), z, DEPTH, target, 3)[0], gt, target['Hs'])['nrmse']
            out[lab] = dict(Hs=Hs, Tp=Tp, Hs_eff=target['Hs'], disp_ceiling=float(ceil), relax=float(relax),
                            steep=float(Hs*(2*np.pi/(1.56*Tp**2))/2))
            print(f"[{lab:11s}] ε≈{out[lab]['steep']:.2f}  disp-ceiling(bound-harmonic) {ceil:.3f} | "
                  f"RELAX {relax:.3f}  ({time.time()-t0:.0f}s)", flush=True)
        except Exception as e:
            print(f"[{lab}] FAILED: {str(e)[:80]}", flush=True)
    json.dump(out, open('results/direction3/steepness_nonlinearity.json', 'w'), indent=2)
    print("\n→ disp-ceiling이 steepness와 함께 커지면, 강비선형 regime = 분산제약 방법이 막히는 곳 =")
    print("  분산제약 *없는* 학습 필드모델이 bound harmonics를 잡아 이길 수 있는 진짜 학습기회.")
    print("saved results/direction3/steepness_nonlinearity.json")


if __name__ == '__main__':
    main()
