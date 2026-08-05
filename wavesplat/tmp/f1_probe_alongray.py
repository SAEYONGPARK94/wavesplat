"""Probe the ALONG-RAY slope on the fan (the quantity the shadow bit uses),
M1 vs M3, pooled over instants + seeds. Whole-grid x-slope cancels skew by
symmetry; the along-ray projection toward the wave direction does not."""
import os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
from scipy.stats import norm
from wavesim.hos_simulator import HOSSimulator, HOSConfig

G = 9.81
Hs, Tp, DEPTH, SPREAD, WDIR = 1.2, 8.0, 50.0, 30.0, 180.0
Lx = Ly = 512.0; Nx = Ny = 128
CX, CY, H = 156.0, 256.0, 5.0
R_MIN, R_MAX = 15.0, 200.0

sys.path.insert(0, 'scripts')
os.chdir(os.path.join(os.path.dirname(__file__), '..'))
from d3_door1b_field_bias import ray_grid  # 240x48 rays, sensor (0,0,5)

o, d, foot = ray_grid()                 # foot = flat-sea footprint xy (sensor origin)
fr = np.hypot(foot[:, 0], foot[:, 1])
inb = (fr > R_MIN) & (fr < R_MAX)
foot = foot[inb]; fr = fr[inb]
ux, uy = foot[:, 0] / fr, foot[:, 1] / fr    # outward horizontal ray unit vec


def along_ray_slopes(sim):
    xs, ys = CX + foot[:, 0], CY + foot[:, 1]
    Z, NX, NY, NZ = sim.evaluate_at(xs, ys)
    ex = -NX / NZ; ey = -NY / NZ         # eta_x, eta_y
    s = ux * ex + uy * ey                # d eta / d(outward range)
    return s, fr


def moments(s):
    m, sd = s.mean(), s.std()
    sk = ((s - m) ** 3).mean() / sd ** 3
    ku = ((s - m) ** 4).mean() / sd ** 3 / sd - 3
    return m, sd, sk, ku


for M in (1, 3):
    S = []
    R = []
    t0 = time.time()
    for seed in range(4):
        sim = HOSSimulator(HOSConfig(Lx=Lx, Ly=Ly, Nx=Nx, Ny=Ny, depth=DEPTH, M=M))
        sim.initialize_jonswap(Hs=Hs, Tp=Tp, direction_deg=WDIR,
                               direction_spread_deg=SPREAD, seed=seed)
        # spin up 16 s, then sample 3 instants 2 s apart
        for tgt in (16.0, 18.0, 20.0):
            while sim.t < tgt - 1e-9:
                sim.step(min(0.05, tgt - sim.t))
            s, r = along_ray_slopes(sim)
            S.append(s); R.append(r)
    S = np.concatenate(S); R = np.concatenate(R)
    m, sd, sk, ku = moments(S)
    print(f"\nM={M}  N={len(S)}  time {time.time()-t0:.1f}s")
    print(f"  along-ray slope: mean {m:+.4f} sd {sd:.4f} skew {sk:+.3f} exkurt {ku:+.3f}")
    # shadow bit at r=25/50/100/150/200 via empirical CDF vs Gaussian
    print(f"  {'r':>4} {'g':>7} {'g/sd':>6} {'-lnP_emp':>9} {'-lnPhi':>8}")
    for rv in (25, 50, 100, 150, 200):
        g = np.arctan(H / rv)
        p_emp = np.mean(S < g)
        bit_emp = -np.log(max(p_emp, 1e-12))
        bit_gauss = -norm.logcdf(g / sd)
        print(f"  {rv:4d} {g:7.4f} {g/sd:6.2f} {bit_emp:9.4f} {bit_gauss:8.4f}")
