"""
F1 — continuous spectral DENSITY representation (vs discrete atoms).

Instead of 32 discrete atoms (RELAX/DPS), represent the spectrum as a DENSE fixed grid of
dispersion-constrained wavenumbers with a regularized complex amplitude per node — a
near-continuous density. Field = Σ_grid (a_re·cos(ψ) − a_im·sin(ψ)), ψ=kx·x+ky·y−ω(k)·t.
Fit a (re,im) by ridge/smoothness-regularized least squares (linear, fast). Tests whether a
continuous-density representation beats sparse discrete on continuous (HOS) spectra.

Run: python3 scripts/d51_F1_continuous_density.py [--ng 22] [--lam 0.5]
"""
import os, sys, json, argparse, time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
os.chdir(os.path.join(os.path.dirname(__file__), '..'))

from d23_main_results import panel
from d26_relax_iterative import iterative_relax

G = 9.81; DEPTH = 50.0
CACHE = 'results/direction3/_hos_windows.npz'


def disp_grid(ng, kmax=0.5):
    ax = np.linspace(-kmax, kmax, ng)
    KX, KY = np.meshgrid(ax, ax); kx = KX.ravel(); ky = KY.ravel()
    k = np.sqrt(kx**2 + ky**2)
    keep = (k > 0.015) & (k < kmax)                       # physical band
    kx, ky, k = kx[keep], ky[keep], k[keep]
    om = np.sqrt(G * k * np.tanh(k * DEPTH))
    return kx, ky, om


def f1_reconstruct(P, z, target, ng, lam, m_cap=400):
    if len(z) > m_cap:
        i = np.random.default_rng(0).choice(len(z), m_cap, replace=False); P, z = P[i], z[i]
    kx, ky, om = disp_grid(ng)
    N = len(kx)
    psi = kx[None] * P[:, 0:1] + ky[None] * P[:, 1:2] - om[None] * P[:, 2:3]   # (M,N)
    Bd = np.concatenate([np.cos(psi), -np.sin(psi)], axis=1)                   # (M,2N): [re; im]
    # ridge-regularized least squares
    A = Bd.T @ Bd + lam * np.eye(2 * N)
    coef = np.linalg.solve(A, Bd.T @ z)                                       # (2N,)
    # render at target grid/time
    gx, gy = target['gx'], target['gy']; Xg, Yg = np.meshgrid(gx, gy)
    xr = Xg.ravel(); yr = Yg.ravel(); tt = target['t']
    psg = kx[None] * xr[:, None] + ky[None] * yr[:, None] - om[None] * tt
    Bg = np.concatenate([np.cos(psg), -np.sin(psg)], axis=1)
    fld = (Bg @ coef).reshape(len(gy), len(gx))
    return panel(fld.astype(np.float32), target['gt'], target['Hs'])['nrmse']


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--ng', type=int, default=22)
    ap.add_argument('--lam', type=float, default=0.5); a = ap.parse_args()
    wins = list(np.load(CACHE, allow_pickle=True)['windows'])
    gx = np.linspace(10, 250, 80); gy = np.linspace(-100, 100, 60)
    f1, rel = [], []; t0 = time.time()
    for w in wins:
        target = {'gx': gx, 'gy': gy, 't': float(w['t']), 'gt': w['gt'], 'Hs': float(w['Hs'])}
        f1.append(f1_reconstruct(w['P'], w['z'], target, a.ng, a.lam))
        rel.append(panel(iterative_relax(w['P'], w['z'], DEPTH, target, 3)[0], w['gt'], w['Hs'])['nrmse'])
        print(f"[{w['tag']}] F1(density,{a.ng}²) {f1[-1]:.3f} | RELAX {rel[-1]:.3f}  ({time.time()-t0:.0f}s)", flush=True)
    mf, mr = float(np.mean(f1)), float(np.mean(rel))
    print(f"\nOVERALL (HOS continuous)  F1 {mf:.3f} | RELAX {mr:.3f} | DPS-32 0.234")
    print(f"  → F1 vs DPS-32(0.234): {mf-0.234:+.3f}  ({'F1 BEATS DPS' if mf<0.234 else 'F1 worse than DPS'})")
    json.dump(dict(f1=mf, relax=mr, dps32=0.234, ng=a.ng, lam=a.lam),
              open('results/direction3/F1_density.json', 'w'), indent=2)
    print("saved results/direction3/F1_density.json")


if __name__ == '__main__':
    main()
