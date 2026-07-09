"""E1c — factor-2 audit: does A=2|eta_hat| in d3_r5_oracle.init_modes double-count?

Convention chain to verify:
  HOS physical field:  evaluate_at / get_eta_at:  eta(x) = sum_k Re(eta_hat[k] e^{ik.x})
                       => per-mode cosine amplitude = |eta_hat[k]|  (one-sided, Re[] convention,
                          NO Hermitian conjugate pair added)
  render_wave_field:   eta(x) = sum_n A_n cos(kx x + ky y - w t + phi_n)
  => correct extraction: A = |eta_hat|, phi = angle(eta_hat).  A = 2|eta_hat| doubles amplitude.

Tests:
  T1 single mode: set eta_hat to one nonzero coefficient, compare evaluate_at vs render(A=|a|)
     and render(A=2|a|) at t=0.
  T2 full init spectrum (BF4 seed 0, t=0): evaluate_at vs render of ALL kx>0 modes with A=|a|
     vs A=2|a| (residual for A=|a| should be only the kx<=0 leak, ~1e-2 rel or less).
  T3 Hs check: 4*std(render top-32, A=|a|) / 4*std(GT)  -> must be < 1 for a partial mode set.

Run: python3 scripts/e1_factor2_audit.py
"""
import os, sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
os.chdir(os.path.join(os.path.dirname(__file__), '..'))

from wavesim.hos_simulator import HOSSimulator, HOSConfig
from wavesplat.model.tnse import render_wave_field

DEPTH = 50.0
GX = np.linspace(10, 250, 80); GY = np.linspace(-100, 100, 60)
Xg, Yg = np.meshgrid(GX, GY)


def rel(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)) / max(np.sqrt(np.mean(b ** 2)), 1e-12))


# ---- T1: single known mode ----
sim = HOSSimulator(HOSConfig(Lx=500.0, Ly=400.0, Nx=64, Ny=64, depth=DEPTH, M=3))
sim.eta_hat[:] = 0
ix, iy = 5, 3                        # kx>0, ky>0 grid mode
a = 0.37 * np.exp(1j * 1.1)
sim.eta_hat[ix, iy] = a
kx = float(sim.KX[ix, iy]); ky = float(sim.KY[ix, iy])
Z, *_ = sim.evaluate_at(Xg, Yg)

f1 = render_wave_field(np.array([kx]), np.array([ky]), np.array([abs(a)]),
                       np.array([np.angle(a)]), DEPTH, GX, GY, 0.0)
f2 = render_wave_field(np.array([kx]), np.array([ky]), np.array([2 * abs(a)]),
                       np.array([np.angle(a)]), DEPTH, GX, GY, 0.0)
print("T1 single mode (t=0):")
print(f"  rel err  A=|eta_hat|   : {rel(f1, Z):.2e}   (expect ~0)")
print(f"  rel err  A=2|eta_hat|  : {rel(f2, Z):.2e}   (expect ~1.0 => 2x overshoot)")
print(f"  amp check: max|Z|={np.abs(Z).max():.4f}  |a|={abs(a):.4f}  2|a|={2*abs(a):.4f}")

# ---- T2: full JONSWAP init spectrum, t=0 ----
sim2 = HOSSimulator(HOSConfig(Lx=500.0, Ly=400.0, Nx=64, Ny=64, depth=DEPTH, M=3))
sim2.initialize_jonswap(Hs=1.2, Tp=7.0, seed=0)
Zf, *_ = sim2.evaluate_at(Xg, Yg)
eh = sim2.eta_hat
mask = sim2.KX > 1e-9
kxs = sim2.KX[mask]; kys = sim2.KY[mask]
As = np.abs(eh[mask]); phis = np.angle(eh[mask])
nz = As > 1e-15
kxs, kys, As, phis = kxs[nz], kys[nz], As[nz], phis[nz]
g1 = render_wave_field(kxs, kys, As, phis, DEPTH, GX, GY, 0.0)
g2 = render_wave_field(kxs, kys, 2 * As, phis, DEPTH, GX, GY, 0.0)
# energy left outside kx>0 half-plane (should be tiny: spreading kills kx<=0)
E_all = np.sum(np.abs(eh) ** 2); E_pos = np.sum(np.abs(eh[mask]) ** 2)
print("\nT2 full init spectrum BF4 s0 (t=0, all kx>0 modes):")
print(f"  energy outside kx>0 half-plane: {(E_all-E_pos)/E_all:.2e} of total")
print(f"  rel err  A=|eta_hat|   : {rel(g1, Zf):.2e}")
print(f"  rel err  A=2|eta_hat|  : {rel(g2, Zf):.2e}")
print(f"  Hs: GT {4*Zf.std():.3f}  A=|a| {4*g1.std():.3f}  A=2|a| {4*g2.std():.3f}")

# ---- T3: top-32 partial set with corrected amplitude ----
mag = np.where(sim2.KX > 1e-9, np.abs(eh), 0.0)
order = np.argsort(mag.ravel())[::-1][:32]
kx32 = sim2.KX.ravel()[order]; ky32 = sim2.KY.ravel()[order]
A32 = np.abs(eh).ravel()[order]; p32 = np.angle(eh).ravel()[order]
h32 = render_wave_field(kx32, ky32, A32, p32, DEPTH, GX, GY, 0.0)
h32x2 = render_wave_field(kx32, ky32, 2 * A32, p32, DEPTH, GX, GY, 0.0)
print("\nT3 top-32 modes at t=0:")
print(f"  Hs_ratio  A=|a|  : {h32.std()/Zf.std():.3f}   (must be <1 for partial set)")
print(f"  Hs_ratio  A=2|a| : {h32x2.std()/Zf.std():.3f}  (old oracle convention)")
print(f"  nRMSE t=0  A=|a| : {np.sqrt(np.mean((h32-Zf)**2))/(4*Zf.std()):.3f}")
print(f"  nRMSE t=0  A=2|a|: {np.sqrt(np.mean((h32x2-Zf)**2))/(4*Zf.std()):.3f}")
