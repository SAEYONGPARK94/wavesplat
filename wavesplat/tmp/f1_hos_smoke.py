"""Smoke test: is HOSSimulator stable at the Door-1 reference sea (Hs 1.2, Tp 8,
depth 50, spread 30, M=3)? Check Hs/skew/kurt evolution, NaN, blowup.
Also time evaluate_at over a fan-sized point set."""
import os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
from wavesim.hos_simulator import HOSSimulator, HOSConfig

G = 9.81
Hs, Tp, DEPTH, SPREAD = 1.2, 8.0, 50.0, 30.0
WDIR = 180.0  # waves travel -x toward a sensor that looks +x
Lx = Ly = 512.0
Nx = Ny = 128

for M in (1, 3):
    t0 = time.time()
    s = HOSSimulator(HOSConfig(Lx=Lx, Ly=Ly, Nx=Nx, Ny=Ny, depth=DEPTH, M=M))
    s.initialize_jonswap(Hs=Hs, Tp=Tp, direction_deg=WDIR,
                         direction_spread_deg=SPREAD, seed=0)
    print(f"\n=== M={M}  init Hs={s.get_Hs():.4f}  build {time.time()-t0:.1f}s ===")
    dt = 0.05
    for tt in np.arange(0, 24.001, 4.0):
        while s.t < tt - 1e-9:
            s.step(min(dt, tt - s.t))
        eta = s.get_eta()
        ex, ey = s.get_surface_gradients()
        sl = ex.ravel()  # x-slope as a proxy
        sk = float(((sl - sl.mean())**3).mean() / sl.std()**3)
        ku = float(((sl - sl.mean())**4).mean() / sl.std()**4 - 3)
        print(f"  t={s.t:5.1f}  Hs={4*eta.std():.4f}  eta_max={eta.max():+.3f} "
              f"eta_min={eta.min():+.3f}  slope_sd={sl.std():.4f} "
              f"skew={sk:+.3f} exkurt={ku:+.3f}  nan={np.isnan(eta).any()}")
    # time evaluate_at on ~11520 fan points
    cx, cy = 156.0, 256.0
    rr = np.random.default_rng(1).uniform(15, 200, 11520)
    aa = np.random.default_rng(2).uniform(-np.pi/3, np.pi/3, 11520)
    xs, ys = cx + rr*np.cos(aa), cy + rr*np.sin(aa)
    te = time.time()
    Z, NX, NY, NZ = s.evaluate_at(xs, ys)
    print(f"  evaluate_at 11520 pts: {time.time()-te:.2f}s  "
          f"n_modes={int((np.abs(s.eta_hat)>1e-15).sum())}")
