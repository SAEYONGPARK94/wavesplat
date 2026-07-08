"""
d3_sa_nomp.py — NOMP-style CONTINUOUS-k super-resolution solver for the
moving-platform problem, evaluated on the exact continuous-spectrum
achievability harness (same seas / operators / obs draws / scoring pixels as
results/direction3/sa_achievability_gap_cont.json).

Why: b17507a established that every lattice method is capped by k-quantization
(oracle_bins 0.04-0.15) — over T = 10 Tp the time lever arm delta-omega =
c_g * delta-k of even half a bin decoheres the fit. RELAX survives only via
its terminal joint Adam refine, but its GREEDY DETECTION runs on quantized
residuals: after m coarse components the residual is polluted by off-grid
mismatch, so detection m+1 fails on moving tracks (N ~ 4 collapse). The NOMP
recipe (Mamandipoor et al. 2016, freq. estimation) fixes exactly this:

  detect (fine beamform on residual) -> Newton-refine THAT component's
  (kx,ky) continuously -> CYCLIC re-refine all components one at a time
  against their leave-one-out residuals -> refit -> repeat; CV stop.

All refinement is dispersion-constrained: omega = sqrt(g k tanh(k depth)),
so each component has 2 nonlinear dof (kx, ky) + 2 linear (cos/sin).

Success criterion (2026-07-04 queue item 2): mean corr on common-unobserved
pixels > RELAX (~0.20 flat) moving toward oracle_ls2 (0.28 / 0.41 / 0.46
fixed / 0.43cp / 1.0cp).

Usage: python3 scripts/d3_sa_nomp.py [--n_seas 10] [--smoke]
"""
import os
import sys
import json
import time
import argparse

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
os.chdir(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, 'scripts')

from wavesim.wave_generator import jonswap_components
from wavesplat.model.tnse import fit_amplitudes_phases
from estimator_improved import iterative_relax_cv, estimate_klim
from d18_relax_estimator import beamform_top, ls_fit, omega
from d17_recursive_estimator import refine_params
from d3_sa_fim_identifiability import omega_of, render_modes
from d3_sa_achievability_gap import (dispersion_lam, sample_obs,
                                     masks_common_unobs, true_field,
                                     render_from_grid, hashabs, corr,
                                     N_OBS, NOISE_FRAC, TP, HS, DEPTH, G)

OUT = 'results/direction3/sa_nomp.json'
LAM_RIDGE = 1e-2                 # same linear ridge as RELAX's ls_fit


# --------------------------------------------------------------- primitives
def predict(kx, ky, coef, P, depth):
    """Field of the component set at points P (n,3) given stacked cos/sin
    coefficients coef (2N,)."""
    kx = np.asarray(kx); ky = np.asarray(ky)
    om = omega(np.sqrt(kx ** 2 + ky ** 2 + 1e-12), depth)
    psi = kx[None] * P[:, 0:1] + ky[None] * P[:, 1:2] - om[None] * P[:, 2:3]
    N = len(kx)
    return np.cos(psi) @ coef[:N] + np.sin(psi) @ coef[N:]


def profile_power(kx, ky, P, r, depth):
    """Explained energy of a SINGLE dispersion-constrained component at
    (kx, ky) after profiling out its two linear coefficients on residual r.
    Smooth in (kx, ky) -> the Newton objective."""
    om = float(omega(np.hypot(kx, ky) + 1e-12, depth))
    psi = kx * P[:, 0] + ky * P[:, 1] - om * P[:, 2]
    C, S = np.cos(psi), np.sin(psi)
    a11 = C @ C + LAM_RIDGE
    a22 = S @ S + LAM_RIDGE
    a12 = C @ S
    b1, b2 = C @ r, S @ r
    det = a11 * a22 - a12 * a12
    return float((a22 * b1 * b1 - 2 * a12 * b1 * b2 + a11 * b2 * b2)
                 / max(det, 1e-12))


def newton_refine(kx, ky, P, r, depth, h, dk_clamp, iters=8):
    """FD-Newton ascent of profile_power in continuous (kx, ky).
    h: FD step (well inside the coherence basin); dk_clamp: max step per
    iteration (keeps the move inside the detection basin)."""
    cur = profile_power(kx, ky, P, r, depth)
    for _ in range(iters):
        fpx = profile_power(kx + h, ky, P, r, depth)
        fmx = profile_power(kx - h, ky, P, r, depth)
        fpy = profile_power(kx, ky + h, P, r, depth)
        fmy = profile_power(kx, ky - h, P, r, depth)
        gx, gy = (fpx - fmx) / (2 * h), (fpy - fmy) / (2 * h)
        hxx = (fpx - 2 * cur + fmx) / h ** 2
        hyy = (fpy - 2 * cur + fmy) / h ** 2
        fpp = profile_power(kx + h, ky + h, P, r, depth)
        fpm = profile_power(kx + h, ky - h, P, r, depth)
        fmp = profile_power(kx - h, ky + h, P, r, depth)
        fmm = profile_power(kx - h, ky - h, P, r, depth)
        hxy = (fpp - fpm - fmp + fmm) / (4 * h ** 2)
        detH = hxx * hyy - hxy * hxy
        if hxx < 0 and detH > 0:                       # proper maximum step
            dx = -(hyy * gx - hxy * gy) / detH
            dy = -(-hxy * gx + hxx * gy) / detH
        else:                                          # gradient ascent
            gn = np.hypot(gx, gy) + 1e-15
            dx, dy = gx / gn * dk_clamp, gy / gn * dk_clamp
        st = np.hypot(dx, dy)
        if st > dk_clamp:
            dx, dy = dx / st * dk_clamp, dy / st * dk_clamp
        # backtracking
        ok = False
        for _bt in range(6):
            nk, nl = kx + dx, ky + dy
            if np.hypot(nk, nl) > 1e-4:
                val = profile_power(nk, nl, P, r, depth)
                if val > cur + 1e-12:
                    kx, ky, cur, ok = nk, nl, val, True
                    break
            dx, dy = dx / 2, dy / 2
        if not ok:
            break
    return kx, ky, cur


def fit_coef(kx, ky, P, z, depth, lam=1.0):
    """Ridge fit of the linear coefficients. Default lam=1.0 — NOT the
    RELAX 1e-2: the cont-harness oracles showed near-collinear mode pairs
    (JONSWAP peak clustering) blow up under weak ridge (oracle_ls 0.145 at
    lam=0.1 vs oracle_ls2 0.278 at lam=1.0, fixed), and oracle_ref proved
    the under-regularized joint refine collapses even TRUE k to the RELAX
    plateau. Regularization, not detection, guards the ceiling."""
    coef, resid, _ = ls_fit(np.asarray(kx), np.asarray(ky), P, z, depth,
                            lam=lam)
    return coef, resid


def fit_coef_cv(kx, ky, P, z, depth, Pv, zv,
                lams=(0.1, 0.3, 1.0, 3.0, 10.0, 30.0)):
    """Pick the coefficient ridge on held-out RMSE, then fit on all given
    obs. Returns (coef, lam)."""
    best = None
    for lam in lams:
        coef, _ = fit_coef(kx, ky, P, z, depth, lam=lam)
        vr = float(np.sqrt(np.mean((predict(kx, ky, coef, Pv, depth)
                                    - zv) ** 2)))
        if best is None or vr < best[0]:
            best = (vr, lam)
    lam = best[1]
    Pall = np.vstack([P, Pv])
    zall = np.concatenate([z, zv])
    coef, _ = fit_coef(kx, ky, Pall, zall, depth, lam=lam)
    return coef, lam


def cyclic_pass(kx, ky, coef, P, z, depth, h, dk_clamp):
    """One NOMP cyclic pass: re-refine each component against its
    leave-one-out residual, refit all coefficients after each move."""
    kx, ky = list(kx), list(ky)
    N = len(kx)
    for j in range(N):
        pred = predict(kx, ky, coef, P, depth)
        contrib = predict([kx[j]], [ky[j]],
                          np.array([coef[j], coef[N + j]]), P, depth)
        rj = z - pred + contrib
        kx[j], ky[j], _ = newton_refine(kx[j], ky[j], P, rj, depth,
                                        h, dk_clamp, iters=5)
        coef, _ = fit_coef(kx, ky, P, z, depth)
    return kx, ky, coef


def nomp_estimate(P, z, depth, klim=None, max_comp=48, val_frac=0.25,
                  patience=3, seed=0, n_cyclic=1, bf_n=160):
    """Greedy continuous-k NOMP with CV model order (mirrors relax_cv's
    protocol so the comparison isolates the solver, not the stopping rule).
    Returns (kx, ky, coef, N)."""
    if klim is None:
        klim = estimate_klim(P, z, depth)
    rng = np.random.default_rng(seed)
    n = len(z)
    vi = rng.choice(n, max(8, int(val_frac * n)), replace=False)
    vmask = np.zeros(n, bool)
    vmask[vi] = True
    Ptr, ztr, Pv, zv = P[~vmask], z[~vmask], P[vmask], z[vmask]
    # FD step h well inside the temporal coherence basin
    # (basin half-width ~ pi / (c_g T); T*c_g ~ few hundred m -> h ~ 5e-4)
    h = 5e-4
    dk_clamp = 2 * klim / bf_n          # one detection-grid cell
    kx, ky = [], []
    coef = np.zeros(0)
    best = (np.inf, [], [])
    bad = 0
    for _ in range(max_comp):
        r = ztr if len(kx) == 0 else ztr - predict(kx, ky, coef, Ptr, depth)
        bx, by = beamform_top(Ptr.astype(np.float32), r.astype(np.float32),
                              depth, klim=klim, n=bf_n)
        bx, by, _ = newton_refine(bx, by, Ptr, r, depth, h, dk_clamp)
        kx.append(bx)
        ky.append(by)
        coef, _ = fit_coef(kx, ky, Ptr, ztr, depth)
        for _c in range(n_cyclic):
            kx, ky, coef = cyclic_pass(kx, ky, coef, Ptr, ztr, depth,
                                       h, dk_clamp)
        vr = float(np.sqrt(np.mean(
            (predict(kx, ky, coef, Pv, depth) - zv) ** 2)))
        if vr < best[0] - 1e-4:
            best = (vr, list(kx), list(ky))
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                break
    kx, ky = list(best[1]), list(best[2])
    if len(kx) == 0:
        bx, by = beamform_top(P.astype(np.float32), z.astype(np.float32),
                              depth, klim=klim, n=bf_n)
        kx, ky = [bx], [by]
    # final: cyclic Newton on ALL obs, then val-selected ridge refit
    # (no free joint refine — oracle_ref showed it collapses even true k)
    coef, _ = fit_coef(kx, ky, P, z, depth)
    for _c in range(2):
        kx, ky, coef = cyclic_pass(kx, ky, coef, P, z, depth, h, dk_clamp)
    coef, _ = fit_coef_cv(kx, ky, Ptr, ztr, depth, Pv, zv)
    return np.array(kx), np.array(ky), coef, len(kx)


def refine_joint(kx0, ky0, coef0, P, z, depth, iters=120, lr=0.01, lam=1.0):
    """RIDGE-MATCHED joint continuous-k refine: Adam on all (kx, ky, c, s),
    dispersion-constrained, loss = mean sq + (lam/n) * sum(c^2 + s^2) — the
    exact ridge objective that protects oracle_ls2's ceiling (refine_params'
    1e-3 mean-penalty is ~67x weaker at N=10, n=300, and oracle_ref showed
    that free refine collapses even TRUE k to the RELAX plateau).
    Warm-starts c, s from the current ridge coefficients."""
    import torch
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    N, n = len(kx0), len(z)
    kx = torch.tensor(np.asarray(kx0), dtype=torch.float32, device=dev,
                      requires_grad=True)
    ky = torch.tensor(np.asarray(ky0), dtype=torch.float32, device=dev,
                      requires_grad=True)
    c = torch.tensor(np.asarray(coef0[:N]), dtype=torch.float32, device=dev,
                     requires_grad=True)
    s = torch.tensor(np.asarray(coef0[N:]), dtype=torch.float32, device=dev,
                     requires_grad=True)
    Pt = torch.tensor(P, dtype=torch.float32, device=dev)
    zt = torch.tensor(z, dtype=torch.float32, device=dev)
    x, y, t = Pt[:, 0], Pt[:, 1], Pt[:, 2]
    opt = torch.optim.Adam([{'params': [kx, ky], 'lr': lr},
                            {'params': [c, s], 'lr': 0.05}])
    for _ in range(iters):
        k = torch.sqrt(kx ** 2 + ky ** 2 + 1e-9)
        om = torch.sqrt(G * k * torch.tanh(k * depth) + 1e-9)
        psi = kx[None] * x[:, None] + ky[None] * y[:, None] \
            - om[None] * t[:, None]
        pred = (c[None] * torch.cos(psi) + s[None] * torch.sin(psi)).sum(1)
        loss = ((pred - zt) ** 2).mean() \
            + (lam / n) * ((c ** 2).sum() + (s ** 2).sum())
        opt.zero_grad()
        loss.backward()
        opt.step()
    return kx.detach().cpu().numpy().astype(float), \
        ky.detach().cpu().numpy().astype(float)


def nomp_joint_estimate(P, z, depth, target, klim=None, max_comp=48,
                        val_frac=0.25, patience=3, seed=0, bf_n=160,
                        it_grow=120, it_final=300):
    """Joint-refine NOMP: same greedy beamform detection as RELAX, but the
    FULL component set is continuously re-refined (ridge-matched joint Adam
    on all kx,ky + coefs, dispersion-constrained) after EVERY detection, so
    each new detection sees a residual free of off-grid mismatch. This
    isolates the hypothesized RELAX failure mode (greedy detection on
    quantized/unrefined residuals) while replacing its under-regularized
    refine with the ridge-matched one."""
    if klim is None:
        klim = estimate_klim(P, z, depth)
    rng = np.random.default_rng(seed)
    n = len(z)
    vi = rng.choice(n, max(8, int(val_frac * n)), replace=False)
    vmask = np.zeros(n, bool)
    vmask[vi] = True
    Ptr, ztr = P[~vmask].astype(np.float32), z[~vmask].astype(np.float32)
    Pv, zv = P[vmask], z[vmask]
    kx = np.zeros(0)
    ky = np.zeros(0)
    coef = np.zeros(0)
    best = (np.inf, kx, ky)
    bad = 0
    for _ in range(max_comp):
        r = ztr if len(kx) == 0 else \
            (ztr - predict(kx, ky, coef, Ptr, depth)).astype(np.float32)
        bx, by = beamform_top(Ptr, r, depth, klim=klim, n=bf_n)
        kx = np.append(kx, bx)
        ky = np.append(ky, by)
        coef, _ = fit_coef(kx, ky, Ptr, ztr, depth)
        kx, ky = refine_joint(kx, ky, coef, Ptr, ztr, depth, iters=it_grow)
        coef, _ = fit_coef(kx, ky, Ptr, ztr, depth)
        vr = float(np.sqrt(np.mean(
            (predict(kx, ky, coef, Pv, depth) - zv) ** 2)))
        if vr < best[0] - 1e-4:
            best = (vr, kx.copy(), ky.copy())
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                break
    kx, ky = best[1], best[2]
    if len(kx) == 0:
        bx, by = beamform_top(P.astype(np.float32), z.astype(np.float32),
                              depth, klim=klim, n=bf_n)
        kx, ky = np.array([bx]), np.array([by])
    # terminal: ridge-matched refine on ALL obs (regularized, warm-start),
    # then val-selected ridge refit — never the free refine (oracle_ref:
    # free refine collapses even true k)
    coef, _ = fit_coef(kx, ky, P, z, depth)
    kx, ky = refine_joint(kx, ky, coef, P, z, depth, iters=it_final)
    coef, _ = fit_coef_cv(kx, ky, Ptr, ztr, depth, Pv, zv)
    return kx, ky, coef, len(kx)


def beamform_power(P, r, depth, klim, n):
    """Dispersion-constrained beamform power map on the (klim, n) grid."""
    import torch
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    x = torch.tensor(P[:, 0], dtype=torch.float32, device=dev)
    y = torch.tensor(P[:, 1], dtype=torch.float32, device=dev)
    t = torch.tensor(P[:, 2], dtype=torch.float32, device=dev)
    rt = torch.tensor(r, dtype=torch.float32, device=dev)
    ax = torch.linspace(-klim, klim, n, device=dev)
    KX, KY = torch.meshgrid(ax, ax, indexing='xy')
    kx, ky = KX.reshape(-1), KY.reshape(-1)
    k = torch.sqrt(kx ** 2 + ky ** 2 + 1e-9)
    om = torch.sqrt(G * k * torch.tanh(k * depth) + 1e-9)
    pw = torch.empty(kx.shape[0], device=dev)
    B = 2048
    for i in range(0, kx.shape[0], B):
        ps = kx[i:i + B, None] * x[None] + ky[i:i + B, None] * y[None] \
            - om[i:i + B, None] * t[None]
        pw[i:i + B] = (rt[None] * torch.cos(ps)).sum(1) ** 2 \
            + (rt[None] * torch.sin(ps)).sum(1) ** 2
    return (kx.cpu().numpy(), ky.cpu().numpy(), pw.cpu().numpy())


def split_masks(P, mode='legs'):
    """Two-group split for cross-coherent detection. 'legs': group by the
    platform's direction of motion along its principal axis (outbound vs
    return legs of the ping-pong) — a chirp alias changes with the Doppler
    sign, a true mode does not. Falls back to a time-median split when
    there is no detectable back-and-forth (fixed platform)."""
    if mode == 'legs':
        xy = P[:, :2] - P[:, :2].mean(0)
        # principal axis of the trajectory
        _, _, Vt = np.linalg.svd(xy, full_matrices=False)
        s = xy @ Vt[0]
        order = np.argsort(P[:, 2])
        ds = np.gradient(s[order])
        # smooth the velocity sign to be robust to footprint jitter
        w = max(5, len(ds) // 40)
        ker = np.ones(w) / w
        dss = np.convolve(ds, ker, mode='same')
        sgn = dss > 0
        if 0.15 < sgn.mean() < 0.85:      # genuine back-and-forth
            m1 = np.zeros(len(P), bool)
            m1[order[sgn]] = True
            return m1, ~m1
    tmed = np.median(P[:, 2])
    m1 = P[:, 2] <= tmed
    return m1, ~m1


def detect_cross(P, r, depth, klim, n, masks=None):
    """Ghost-suppressed detection: the k-precision probe showed greedy
    beamform peaks under moving sampling sit ~0.05-0.09 rad/m from ANY true
    mode (aliases of the chirped trajectory), while the order-prize only
    needs ~0.01. A trajectory alias is coherent on one subset of the track
    but not the other; a true mode is coherent on both. Score = elementwise
    MIN of the two subset beamform power maps (each normalized), argmax
    over the grid."""
    if masks is None:
        masks = split_masks(P, mode='time')
    m1, m2 = masks
    kx1, ky1, p1 = beamform_power(P[m1], r[m1], depth, klim, n)
    _, _, p2 = beamform_power(P[m2], r[m2], depth, klim, n)
    sc = np.minimum(p1 / max(p1.max(), 1e-12), p2 / max(p2.max(), 1e-12))
    j = int(np.argmax(sc))
    return float(kx1[j]), float(ky1[j])


def nomp_grow_estimate(P, z, depth, klim=None, n_comp=48, val_frac=0.25,
                       seed=0, bf_n=160, n_cyclic=1, cross=None):
    """Forced-growth NOMP: NO early stop — the order probe (sa_orderprobe)
    showed true-k ridge at the CV-stopped order N~10 is already ~RELAX
    (0.15-0.32), while N=48-64 + heavy ridge reaches 0.47-0.60 on moving
    tracks: the prize is MODEL ORDER under regularization, which greedy CV
    stopping can never reach. Grow to n_comp with per-detection Newton
    refine, then hand ALL complexity control to the val-selected ridge."""
    if klim is None:
        klim = estimate_klim(P, z, depth)
    rng = np.random.default_rng(seed)
    n = len(z)
    vi = rng.choice(n, max(8, int(val_frac * n)), replace=False)
    vmask = np.zeros(n, bool)
    vmask[vi] = True
    Ptr, ztr = P[~vmask], z[~vmask]
    Pv, zv = P[vmask], z[vmask]
    h = 5e-4
    dk_clamp = 2 * klim / bf_n
    masks = split_masks(Ptr, mode=cross) if cross else None
    kx, ky = [], []
    coef = np.zeros(0)
    for _ in range(n_comp):
        r = ztr if len(kx) == 0 else ztr - predict(kx, ky, coef, Ptr, depth)
        if cross:
            bx, by = detect_cross(Ptr, r, depth, klim, bf_n, masks=masks)
        else:
            bx, by = beamform_top(Ptr.astype(np.float32),
                                  r.astype(np.float32),
                                  depth, klim=klim, n=bf_n)
        bx, by, _ = newton_refine(bx, by, Ptr, r, depth, h, dk_clamp)
        kx.append(bx)
        ky.append(by)
        coef, _ = fit_coef(kx, ky, Ptr, ztr, depth)
    for _c in range(n_cyclic):
        kx, ky, coef = cyclic_pass(kx, ky, coef, Ptr, ztr, depth, h,
                                   dk_clamp)
    coef, lam_sel = fit_coef_cv(kx, ky, Ptr, ztr, depth, Pv, zv,
                                lams=(0.3, 1.0, 3.0, 10.0, 30.0, 100.0))
    return np.array(kx), np.array(ky), coef, lam_sel


def bank_irls_estimate(P, z, depth, n_sel=48, dk_c=0.015, val_frac=0.25,
                       seed=0, irls_iters=8, n_cyclic=1):
    """Simultaneous sparse fit over a FINE continuous candidate bank —
    removes greedy's sequential error propagation (ghost detections feed on
    residuals polluted by earlier commitments). Bank spacing dk_c=0.015
    puts the nearest candidate within ~0.0075 of any true mode (inside both
    the Newton basin and the ~0.01 tolerance the order-prize needs), unlike
    the lattice zoo's dk_bin=0.041 which quantization-kills (oracle_bins).
    Group-FOCUSS IRLS (w_b = 1/(amp_b + eps)) via Woodbury in obs space,
    lambda0 picked on the val split; top-n_sel groups then Newton cyclic +
    val-selected ridge, complexity handled by regularization not stopping."""
    rng = np.random.default_rng(seed)
    n = len(z)
    vi = rng.choice(n, max(8, int(val_frac * n)), replace=False)
    vmask = np.zeros(n, bool)
    vmask[vi] = True
    Ptr, ztr = P[~vmask], z[~vmask]
    Pv, zv = P[vmask], z[vmask]
    # sea band from the strongest beamform peak
    bx, by = beamform_top(P.astype(np.float32), z.astype(np.float32),
                          depth, klim=2.0, n=120)
    kp = max(np.hypot(bx, by), 1e-3)
    ax = np.arange(-2.5 * kp, 2.5 * kp + dk_c / 2, dk_c)
    KX, KY = np.meshgrid(ax, ax)
    kk = np.hypot(KX, KY)
    sel = (kk > 0.35 * kp) & (kk < 2.5 * kp)
    kxb, kyb = KX[sel].astype(float), KY[sel].astype(float)
    B = len(kxb)
    om = omega(np.hypot(kxb, kyb) + 1e-12, depth)
    psi = kxb[None] * Ptr[:, 0:1] + kyb[None] * Ptr[:, 1:2] \
        - om[None] * Ptr[:, 2:3]
    Phi = np.concatenate([np.cos(psi), np.sin(psi)], axis=1)   # (m, 2B)
    m = len(ztr)
    best = None
    for lam0 in (1.0, 10.0):
        w = np.ones(B)
        eps_amp = None
        for _ in range(irls_iters):
            ginv = np.concatenate([lam0 * w, lam0 * w])
            PhiG = Phi / ginv[None]
            K = PhiG @ Phi.T + np.eye(m)
            alpha = np.linalg.solve(K, ztr)
            c = PhiG.T @ alpha
            amp = np.sqrt(c[:B] ** 2 + c[B:] ** 2)
            # graduated non-convexity: start near-ridge (large eps keeps the
            # weights soft), anneal toward sparsity — plain FOCUSS with a
            # tiny fixed eps entrenches the smeared first-ridge peaks
            eps_amp = (0.2 * amp.max() if eps_amp is None
                       else max(eps_amp * 0.5, 1e-3))
            w = 1.0 / (amp + eps_amp)
        # val score of the top-n_sel restriction (what we will keep)
        top = np.argsort(-amp)[:n_sel]
        coef_t, _ = fit_coef(kxb[top], kyb[top], Ptr, ztr, depth)
        vr = float(np.sqrt(np.mean(
            (predict(kxb[top], kyb[top], coef_t, Pv, depth) - zv) ** 2)))
        if best is None or vr < best[0]:
            best = (vr, lam0, top.copy())
    _, lam0, top = best
    kx, ky = list(kxb[top]), list(kyb[top])
    coef, _ = fit_coef(kx, ky, Ptr, ztr, depth)
    h = 5e-4
    for _c in range(n_cyclic):
        kx, ky, coef = cyclic_pass(kx, ky, coef, Ptr, ztr, depth, h,
                                   dk_c / 2)
    coef, lam_sel = fit_coef_cv(kx, ky, Ptr, ztr, depth, Pv, zv,
                                lams=(0.3, 1.0, 3.0, 10.0, 30.0, 100.0))
    return np.array(kx), np.array(ky), coef, lam_sel


def mmse_bank_estimate(P, z, depth, Pq, sigma0, kp_hint=None, dk_c=None,
                       val_frac=0.25, seed=0, max_em=6, return_var=False,
                       Uc=None, diag_out=None, dual_band=False,
                       gamma_tilt=0.0, band=(0.35, 2.5), bound_wave=False,
                       bw_cross=False):
    """Spectrum-weighted linear MMSE over the fine continuous bank — the
    2026-07-05 winner (10-sea harness: 0.272/0.239/0.241 vs RELAX
    0.199/0.222/0.183). NO identification: prior variance per candidate
    from the (PSF-smeared) beamform power, shape hyperparams (power
    exponent, noise multiplier) val-selected, then val-gated SBL-EM
    refinement of the prior. Pure linear-in-z prediction at Pq.

    sigma0: nominal obs noise std (e.g. 0.05 * Hs_w); the val-selected
    multiplier grid {1,3,10} absorbs misspecification.
    kp_hint: peak wavenumber if known (real records: characterize());
    otherwise estimated by two-stage beamform zoom.
    dk_c: bank spacing; default kp/10 (scale-invariant ~1900 candidates,
    inside the ~pi/(c_g T) coherence basin for T ~ 10 Tp)."""
    rng = np.random.default_rng(seed)
    n = len(z)
    vi = rng.choice(n, max(8, int(val_frac * n)), replace=False)
    vmask = np.zeros(n, bool)
    vmask[vi] = True
    Ptr, ztr = P[~vmask], z[~vmask]
    Pv, zv = P[vmask], z[vmask]
    if kp_hint is None:
        bx, by = beamform_top(P.astype(np.float32), z.astype(np.float32),
                              depth, klim=2.0, n=120)
        kp = max(np.hypot(bx, by), 1e-3)
        bx, by = beamform_top(P.astype(np.float32), z.astype(np.float32),
                              depth, klim=3.5 * kp, n=120)
        kp = max(np.hypot(bx, by), 1e-3)
    else:
        kp = float(kp_hint)
    if dk_c is None:
        dk_c = kp / 10.0
    b_lo, b_hi = band                 # bank annulus in units of kp (widen =
    #                                   richer broadband single-model class)
    ax = np.arange(-b_hi * kp, b_hi * kp + dk_c / 2, dk_c)
    KX, KY = np.meshgrid(ax, ax)
    kk = np.hypot(KX, KY)
    sel = (kk > b_lo * kp) & (kk < b_hi * kp)
    kxb, kyb = KX[sel].astype(float), KY[sel].astype(float)
    if dual_band:
        # wide radial scan for a secondary spectral band (crossing seas:
        # swell + windsea at different |k| — a single kp-annulus clips it)
        kr = np.linspace(0.12 * kp, 3.2 * kp, 40)
        thg = np.linspace(-np.pi, np.pi, 36, endpoint=False)
        KR, TH = np.meshgrid(kr, thg)
        kxs = (KR * np.cos(TH)).ravel()
        kys = (KR * np.sin(TH)).ravel()
        oms = omega(np.hypot(kxs, kys) + 1e-12, depth)
        psi_s = (kxs[:, None] * P[None, :, 0] + kys[:, None] * P[None, :, 1]
                 - oms[:, None] * P[None, :, 2])
        pw_s = ((z[None] * np.cos(psi_s)).sum(1) ** 2
                + (z[None] * np.sin(psi_s)).sum(1) ** 2).reshape(KR.shape)
        prof = pw_s.max(axis=0)                       # radial profile
        inside = (kr > 0.6 * kp) & (kr < 1.5 * kp)
        p_pri = prof[inside].max()
        prof_out = np.where(inside, -np.inf, prof)
        k2 = float(kr[np.argmax(prof_out)])
        if prof_out.max() > 0.25 * p_pri and \
                not (0.35 * kp < k2 < 2.5 * kp):
            dk2 = k2 / 10.0
            ax2 = np.arange(-1.6 * k2, 1.6 * k2 + dk2 / 2, dk2)
            KX2, KY2 = np.meshgrid(ax2, ax2)
            kk2 = np.hypot(KX2, KY2)
            sel2 = ((kk2 > 0.6 * k2) & (kk2 < 1.6 * k2)
                    & ~((kk2 > 0.35 * kp) & (kk2 < 2.5 * kp)))
            kxb = np.concatenate([kxb, KX2[sel2].astype(float)])
            kyb = np.concatenate([kyb, KY2[sel2].astype(float)])
    om = omega(np.hypot(kxb, kyb) + 1e-12, depth)
    n_free = len(kxb)
    if bound_wave:
        # MODEL-CLASS EXTENSION (2026-07-08, A1/#2): 2nd-order super-harmonic
        # bound-wave candidates at (2k, 2*omega) — OFF the free-wave dispersion
        # shell (a free wave at 2k would sit at omega(2k) != 2*omega(k)). This
        # lets the linear MMSE represent the bound-wave energy that a pure
        # free-wave bank cannot (the K3/P5 model-class ceiling on bimodal seas).
        # The beamform-power prior auto-detects them (correlates z with the
        # candidate's own 2*omega phase); they are additionally prior-downscaled
        # (bound waves are O(steepness) smaller) to suppress spurious pickup on
        # clean unimodal records where no real bound-wave energy exists.
        kxb = np.concatenate([kxb, 2.0 * kxb])
        kyb = np.concatenate([kyb, 2.0 * kyb])
        om = np.concatenate([om, 2.0 * om])
        if bw_cross:
            # B1 (2026-07-08): CROSS-interaction bound waves at (k_i+k_j,
            # omega_i+omega_j) for the top-K free-wave power peaks — the
            # dominant bound-wave family for CROSS-Tp bimodal seas (K3), which
            # the self-super-harmonic (2k,2omega) alone does not reach. Detect
            # peaks by a quick beamform power over the free-wave bank.
            kxf, kyf, omf = kxb[:n_free], kyb[:n_free], om[:n_free]
            pwq = np.empty(n_free)
            for i in range(0, n_free, 512):
                psiq = (kxf[i:i + 512, None] * P[None, :, 0]
                        + kyf[i:i + 512, None] * P[None, :, 1]
                        - omf[i:i + 512, None] * P[None, :, 2])
                pwq[i:i + 512] = (z[None] * np.cos(psiq)).sum(1) ** 2 \
                    + (z[None] * np.sin(psiq)).sum(1) ** 2
            top = np.argsort(pwq)[-8:]        # top-8 spectral peaks
            cx, cy, co = [], [], []
            for a in range(len(top)):
                for b in range(a + 1, len(top)):
                    i, j = top[a], top[b]
                    cx += [kxf[i] + kxf[j], kxf[i] - kxf[j]]
                    cy += [kyf[i] + kyf[j], kyf[i] - kyf[j]]
                    co += [omf[i] + omf[j], omf[i] - omf[j]]
            kxb = np.concatenate([kxb, np.array(cx)])
            kyb = np.concatenate([kyb, np.array(cy)])
            om = np.concatenate([om, np.array(co)])
    B = len(kxb)
    if Uc is not None:                    # current Doppler (earth frame)
        om = om + kxb * Uc[0] + kyb * Uc[1]

    def design(Q):
        psi = kxb[None] * Q[:, 0:1] + kyb[None] * Q[:, 1:2] \
            - om[None] * Q[:, 2:3]
        return np.concatenate([np.cos(psi), np.sin(psi)], axis=1)

    Phi_tr = design(Ptr)
    Phi_v = design(Pv)
    # prior shape: beamform power at the bank points (all obs)
    pw = np.empty(B)
    CH = 512
    for i in range(0, B, CH):
        psi = (kxb[i:i + CH, None] * P[None, :, 0]
               + kyb[i:i + CH, None] * P[None, :, 1]
               - om[i:i + CH, None] * P[None, :, 2])
        pw[i:i + CH] = (z[None] * np.cos(psi)).sum(1) ** 2 \
            + (z[None] * np.sin(psi)).sum(1) ** 2
    pw = pw / max(pw.max(), 1e-12)
    if bound_wave and n_free < B:      # physical downscale of bound-wave prior
        pw[n_free:] *= 0.15            # bound waves ~ O(steepness) of free waves
    if gamma_tilt:                # high-k prior lift (slope-aware FF use)
        pw = pw * (np.hypot(kxb, kyb) / kp) ** gamma_tilt
        pw = pw / max(pw.max(), 1e-12)
    m = len(ztr)
    best = None
    for pexp in (0.5, 1.0, 2.0):
        g = pw ** pexp
        g = g / g.sum()
        gam = np.concatenate([g, g]) * np.var(z)
        for smult in (1.0, 3.0, 10.0):
            s2 = (smult * sigma0) ** 2
            K = (Phi_tr * gam[None]) @ Phi_tr.T + s2 * np.eye(m)
            fv = (Phi_v * gam[None]) @ (Phi_tr.T @ np.linalg.solve(K, ztr))
            vr = float(np.sqrt(np.mean((fv - zv) ** 2)))
            if best is None or vr < best[0]:
                best = (vr, pexp, smult, gam)
    vr_best, pexp, smult, gam = best
    s2 = (smult * sigma0) ** 2
    # val-gated SBL-EM refinement of the prior shape (empirical Bayes)
    best_em = (vr_best, gam.copy())
    for _ in range(max_em):
        K = (Phi_tr * gam[None]) @ Phi_tr.T + s2 * np.eye(m)
        Ki_Phi = np.linalg.solve(K, Phi_tr)
        mu = gam * (Phi_tr.T @ np.linalg.solve(K, ztr))
        sig = gam - gam ** 2 * np.einsum('ij,ij->j', Phi_tr, Ki_Phi)
        gb = 0.5 * (mu[:B] ** 2 + mu[B:] ** 2
                    + np.maximum(sig[:B], 0) + np.maximum(sig[B:], 0))
        gam = np.concatenate([gb, gb])
        fv = (Phi_v * gam[None]) @ (Phi_tr.T @ np.linalg.solve(
            (Phi_tr * gam[None]) @ Phi_tr.T + s2 * np.eye(m), ztr))
        vr = float(np.sqrt(np.mean((fv - zv) ** 2)))
        if vr < best_em[0] - 1e-5:
            best_em = (vr, gam.copy())
        else:
            break
    gam = best_em[1]
    if diag_out is not None:      # held-out predictive misfit (gate signal)
        diag_out['vr'] = float(best_em[0])
        diag_out['vr_rel'] = float(best_em[0] / (np.std(zv) + 1e-12))
    Phi_all = design(P)
    K = (Phi_all * gam[None]) @ Phi_all.T + s2 * np.eye(n)
    alpha = np.linalg.solve(K, z)
    w_pred = Phi_all.T @ alpha
    # chunked query render (real grids can be ~1e4 px)
    out = np.empty(len(Pq))
    var = np.empty(len(Pq)) if return_var else None
    Ki = np.linalg.inv(K) if return_var else None
    for i in range(0, len(Pq), 4096):
        Phi_q = design(Pq[i:i + 4096])
        A = Phi_q * gam[None]
        out[i:i + 4096] = A @ w_pred
        if return_var:
            Bq = A @ Phi_all.T                          # (c, n)
            var[i:i + 4096] = np.einsum('ij,ij->i', A, Phi_q) \
                - np.einsum('ij,ij->i', Bq @ Ki, Bq)
    if return_var:
        return out, np.maximum(var, 0.0)
    return out


# ---------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n_seas', type=int, default=10)
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--max_comp', type=int, default=48)
    ap.add_argument('--n_cyclic', type=int, default=1)
    a = ap.parse_args()
    n_seas = 2 if a.smoke else a.n_seas

    lam = dispersion_lam(TP, DEPTH)
    cp = lam / TP
    T = 10 * TP
    fp = 0.18 * lam
    r_fp = fp / 2
    side = 2.0 * lam
    gx = np.linspace(0, side, 96)
    gy = np.linspace(0, side, 96)
    Xg, Yg = np.meshgrid(gx, gy)
    mu = masks_common_unobs(gx, gy, r_fp, 0.6 * fp)
    Xs, Ys = Xg[mu], Yg[mu]
    if len(Xs) > 1500:
        take = np.random.default_rng(0).choice(len(Xs), 1500, replace=False)
        Xs, Ys = Xs[take], Ys[take]
    t_star = T
    print(f"lam={lam:.1f} m cp={cp:.2f} m/s patch {side:.0f} m, T={T:.0f} s, "
          f"score px {len(Xs)}")

    conds = [('fixed', None), ('track043', 0.43), ('track100', 1.0)]
    res = {c: {} for c, _ in conds}
    for si in range(n_seas):
        nc = 200
        c = jonswap_components(Hs=HS, Tp=TP, n_components=nc, depth=DEPTH,
                               seed=4000 + si)
        theta_full = (np.array([x.kx for x in c]), np.array([x.ky for x in c]),
                      np.array([x.A for x in c]), np.array([x.phi for x in c]))
        order = np.argsort(theta_full[2])[::-1]
        theta = tuple(v[order[:32]] for v in theta_full)
        om = omega_of(theta_full[0], theta_full[1], DEPTH)
        gt = true_field(theta_full, DEPTH, Xg, Yg, t_star)
        gt_u = render_modes(*theta_full, DEPTH, Xs, Ys, t_star)
        sigma = NOISE_FRAC * HS
        Hs_w = 4 * gt.std()
        target = {'gx': gx, 'gy': gy, 't': t_star, 'gt': gt, 'Hs': Hs_w}
        for cname, ur in conds:
            rng = np.random.default_rng(9000 + 17 * si + hashabs(cname))
            U = (ur or 0) * cp
            P = sample_obs('fixed' if cname == 'fixed' else 'moving', U, lam,
                           cp, T, gx, gy, r_fp, rng)
            psi = (theta_full[0][None] * P[:, 0:1]
                   + theta_full[1][None] * P[:, 1:2]
                   - om[None] * P[:, 2:3] + theta_full[3][None])
            z = (theta_full[2][None] * np.cos(psi)).sum(1) \
                + rng.normal(0, sigma, N_OBS)
            P32, z32 = P.astype(np.float32), z.astype(np.float32)
            row = res[cname].setdefault(si, {})
            # references on identical obs
            obs = np.column_stack([P32, z32]).astype(np.float32)
            A_l2, phi_l2 = fit_amplitudes_phases(
                theta[0].astype(np.float32), theta[1].astype(np.float32),
                obs, DEPTH, lambda_reg=1.0)
            row['oracle_ls2'] = corr(
                render_modes(theta[0], theta[1], np.asarray(A_l2, float),
                             np.asarray(phi_l2, float), DEPTH, Xs, Ys, t_star),
                gt_u)
            fld, Nr, _ = iterative_relax_cv(P32.copy(), z32.copy(), DEPTH,
                                            target, n_outer=2, seed=si)
            row['relax'] = corr(render_from_grid(fld, gx, gy, Xs, Ys), gt_u)
            row['relax_N'] = Nr
            Pq = np.column_stack([Xs, Ys, np.full(len(Xs), t_star)])
            # NOMP (cyclic per-component FD Newton)
            t0 = time.time()
            nkx, nky, ncoef, Nn = nomp_estimate(P, z, DEPTH,
                                                max_comp=a.max_comp,
                                                seed=si, n_cyclic=a.n_cyclic)
            row['nomp'] = corr(predict(nkx, nky, ncoef, Pq, DEPTH), gt_u)
            row['nomp_N'] = Nn
            row['nomp_sec'] = time.time() - t0
            # joint-NOMP (interleaved joint Adam refine per detection)
            t0 = time.time()
            jkx, jky, jcoef, Nj = nomp_joint_estimate(P, z, DEPTH, target,
                                                      max_comp=a.max_comp,
                                                      seed=si)
            row['nomp_joint'] = corr(predict(jkx, jky, jcoef, Pq, DEPTH),
                                     gt_u)
            row['nomp_joint_N'] = Nj
            row['nomp_joint_sec'] = time.time() - t0
            # forced-growth NOMP (no early stop, lambda-CV complexity)
            for tag, ng, cross in (('grow48', 48, None), ('grow64', 64,
                                                          None),
                                   ('grow48x', 48, 'time'),
                                   ('grow48L', 48, 'legs')):
                t0 = time.time()
                gkx, gky, gcoef, glam = nomp_grow_estimate(P, z, DEPTH,
                                                           n_comp=ng,
                                                           seed=si,
                                                           cross=cross)
                row[tag] = corr(predict(gkx, gky, gcoef, Pq, DEPTH), gt_u)
                row[f'{tag}_lam'] = glam
                row[f'{tag}_sec'] = time.time() - t0
                if tag == 'grow48':
                    g48kx, g48ky = gkx, gky
            # fine-bank simultaneous IRLS (no greedy)
            t0 = time.time()
            bkx, bky, bcoef, blam = bank_irls_estimate(P, z, DEPTH, seed=si)
            row['bank48'] = corr(predict(bkx, bky, bcoef, Pq, DEPTH), gt_u)
            row['bank48_lam'] = blam
            row['bank48_sec'] = time.time() - t0
            # diagnostic: median distance of selected k to nearest true mode
            for tag, tkx, tky in (('bank48', bkx, bky),
                                  ('grow48', g48kx, g48ky)):
                dd = np.sqrt(
                    (np.asarray(tkx)[:, None] - theta_full[0][None]) ** 2
                    + (np.asarray(tky)[:, None] - theta_full[1][None]) ** 2
                ).min(1)
                row[f'{tag}_kerr'] = float(np.median(dd))
        print(f"sea {si}: " + " | ".join(
            f"{cn} oLS2 {res[cn][si]['oracle_ls2']:.2f} "
            f"rlx {res[cn][si]['relax']:.2f}({res[cn][si]['relax_N']}) "
            f"nomp {res[cn][si]['nomp']:.2f}({res[cn][si]['nomp_N']}) "
            f"joint {res[cn][si]['nomp_joint']:.2f}"
            f"({res[cn][si]['nomp_joint_N']}) "
            f"g48 {res[cn][si]['grow48']:.2f} "
            f"g64 {res[cn][si]['grow64']:.2f} "
            f"g48x {res[cn][si]['grow48x']:.2f} "
            f"g48L {res[cn][si]['grow48L']:.2f} "
            f"bank {res[cn][si]['bank48']:.2f}"
            f"({res[cn][si]['bank48_sec']:.0f}s)" for cn, _ in conds),
            flush=True)

    print("\n===== NOMP vs RELAX vs oracle_ls2 (mean corr, common unobs) ====")
    summ = {}
    for cname, _ in conds:
        summ[cname] = {}
        for e in ('oracle_ls2', 'relax', 'nomp', 'nomp_joint', 'grow48',
                  'grow64', 'grow48x', 'grow48L', 'bank48'):
            summ[cname][e] = float(np.nanmean(
                [res[cname][si][e] for si in res[cname]]))
        for e in ('relax_N', 'nomp_N', 'nomp_joint_N'):
            summ[cname][e + '_mean'] = float(np.mean(
                [res[cname][si][e] for si in res[cname]]))
        print(f"{cname:10s} oracle_ls2 {summ[cname]['oracle_ls2']:.3f}  "
              f"relax {summ[cname]['relax']:.3f} "
              f"(N {summ[cname]['relax_N_mean']:.1f})  "
              f"nomp {summ[cname]['nomp']:.3f}  "
              f"joint {summ[cname]['nomp_joint']:.3f}  "
              f"grow48 {summ[cname]['grow48']:.3f}  "
              f"grow64 {summ[cname]['grow64']:.3f}  "
              f"grow48x {summ[cname]['grow48x']:.3f}  "
              f"grow48L {summ[cname]['grow48L']:.3f}  "
              f"bank48 {summ[cname]['bank48']:.3f}")
    json.dump(dict(summary=summ, rows=res, n_seas=n_seas,
                   max_comp=a.max_comp, n_cyclic=a.n_cyclic),
              open(OUT, 'w'), indent=1, default=float)
    print(f"saved {OUT}")


if __name__ == '__main__':
    main()
