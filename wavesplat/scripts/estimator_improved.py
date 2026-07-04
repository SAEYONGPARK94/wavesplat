"""
estimator_improved.py
Hardened RELAX with (1) adaptive klim and (2) cross-validated model order.
Backward-safe: on well-conditioned synthetic seas it reproduces the default RELAX;
on short-wave / smooth-sparse regimes it avoids the klim cap and over-fitting.

  iterative_relax_cv(P, z, depth, target, klim=None, n_outer=2)  -> (field, N)
"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from d18_relax_estimator import beamform_top, ls_fit, omega, MCAP
from d17_recursive_estimator import refine_params

G = 9.81


def _design(P, kx, ky, depth):
    kx = np.asarray(kx); ky = np.asarray(ky)
    k = np.sqrt(kx**2 + ky**2 + 1e-12); om = omega(k, depth)
    psi = kx[None]*P[:, 0:1] + ky[None]*P[:, 1:2] - om[None]*P[:, 2:3]
    return np.concatenate([np.cos(psi), np.sin(psi)], axis=1)


def estimate_klim(P, z, depth, coarse_n=60, floor=0.6, ceil=2.0):
    """Adaptive beamform range: ~3.5x the dominant wavenumber, clipped to [floor, ceil]."""
    bx, by = beamform_top(P, z, depth, klim=ceil, n=coarse_n)
    kp = float(np.hypot(bx, by))
    return float(np.clip(3.5*kp, floor, ceil))


def relax_cv(P, z, depth, klim, max_comp=32, val_frac=0.25, patience=2, seed=0,
             min_comp=1):
    """Grow components; stop when held-out (val) RMSE stops improving -> data-driven model order.
    min_comp: forbid the early-stop before this many components have been tried.
    NOTE (2026-07-04, AA14 moving-track test): forcing growth (patience=5,
    min_comp=8) made the moving-condition field WORSE (swept corr +0.13 -> -0.55
    with N 10-14 -> 21-31): the low N on moving tracks is not a CV bug but an
    identifiability breakdown — greedy components beyond the first few are not
    supported by moving-track samples on broadband seas. Defaults unchanged;
    kwargs kept for experiments."""
    rng = np.random.default_rng(seed); n = len(z)
    if n > MCAP:
        ii = rng.choice(n, MCAP, replace=False); P, z = P[ii], z[ii]; n = MCAP
    vi = rng.choice(n, max(8, int(val_frac*n)), replace=False)
    vmask = np.zeros(n, bool); vmask[vi] = True
    Ptr, ztr, Pv, zv = P[~vmask], z[~vmask], P[vmask], z[vmask]
    kx, ky = [], []
    best_val = np.inf; best = (np.zeros(0), np.zeros(0)); bad = 0
    for _ in range(max_comp):
        if len(kx) == 0:
            bx, by = beamform_top(Ptr, ztr, depth, klim=klim)
        else:
            _, resid, _ = ls_fit(np.array(kx), np.array(ky), Ptr, ztr, depth)
            bx, by = beamform_top(Ptr, resid, depth, klim=klim)
        kx.append(bx); ky.append(by)
        coef, _, _ = ls_fit(np.array(kx), np.array(ky), Ptr, ztr, depth)
        vr = float(np.sqrt(np.mean((_design(Pv, kx, ky, depth) @ coef - zv)**2)))
        if vr < best_val - 1e-4:
            best_val = vr; best = (np.array(kx), np.array(ky)); bad = 0
        else:
            bad += 1
            if bad >= patience and len(kx) >= min_comp:
                break
    return best


def iterative_relax_cv(P, z, depth, target, klim=None, n_outer=2, seed=0,
                       patience=2, min_comp=1):
    if klim is None:
        klim = estimate_klim(P, z, depth)
    kx, ky = relax_cv(P, z, depth, klim, seed=seed, patience=patience,
                      min_comp=min_comp)
    fld = None
    for _ in range(n_outer):
        if len(kx) == 0:
            break
        kx, ky, fld = refine_params(kx, ky, P, z, depth, target, iters=400)
    if fld is None:   # degenerate: fall back to a single beamformed component
        bx, by = beamform_top(P, z, depth, klim=klim)
        kx, ky, fld = refine_params(np.array([bx]), np.array([by]), P, z, depth, target, iters=400)
    return fld, len(kx), klim
