"""Positive control for F1: prove the instrument SEES the nonlinearity.
The shadow bit reads the along-ray SLOPE, and a 2nd-order Stokes wave has zero
slope skewness -- bound waves skew the ELEVATION (sharp crests, flat troughs),
not the slope.  So the right nonlinearity witness is ELEVATION skewness, which
must grow with steepness and with HOS order.  Measured on the fan footprints,
pooled over 3 seeds x 3 instants, for hos_m1 / hos_m3 / hos_m3(Hs=3)."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
os.chdir(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, 'scripts')
import numpy as np
from wavesim.hos_simulator import HOSSimulator, HOSConfig
from d3_door1b_field_bias import ray_grid, R_MIN, R_MAX

CX, CY = 156.0, 256.0
o, d, foot = ray_grid(); fr = np.hypot(foot[:, 0], foot[:, 1])
inb = (fr > R_MIN) & (fr < R_MAX); foot = foot[inb]
xs, ys = CX + foot[:, 0], CY + foot[:, 1]


def modes(sim, rel=1e-5):
    a = sim.eta_hat; m = np.abs(a) > rel * np.abs(a).max()
    return sim.KX[m], sim.KY[m], a[m]


def eta_at(sim):
    kx, ky, a = modes(sim); z = np.zeros(len(xs))
    for i in range(0, len(kx), 512):
        e = np.exp(1j*(np.outer(kx[i:i+512], xs)+np.outer(ky[i:i+512], ys)))
        z += np.real(a[i:i+512, None]*e).sum(0)
    return z


def run(M, Hs, seed):
    s = HOSSimulator(HOSConfig(Lx=512, Ly=512, Nx=128, Ny=128, depth=50, M=M))
    s.initialize_jonswap(Hs=Hs, Tp=8.0, direction_deg=180,
                         direction_spread_deg=30, seed=seed)
    while s.t < 12-1e-9:
        s.step(min(0.1, 12-s.t))
    return s


def skewness(z):
    z = z - z.mean(); return float((z**3).mean()/z.std()**3)


for name, M, Hs in (('hos_m1', 1, 1.2), ('hos_m3', 3, 1.2), ('hos_m3_steep', 3, 3.0)):
    Z = []
    for seed in range(3):
        s = run(M, Hs, seed)
        for tgt in (12.0, 14.0, 16.0):
            while s.t < tgt-1e-9:
                s.step(min(0.1, tgt-s.t))
            Z.append(eta_at(s))
    Z = np.concatenate(Z)
    print(f"{name:<14} Hs={Hs} M={M}: elevation skew = {skewness(Z):+.4f}  "
          f"(sd {Z.std():.3f})")
