#!/usr/bin/env python3
"""W4 `L-VP` arm -- VARIABLE PROJECTION (VARPRO) on the shipped Phase-1 output.

PREREG.md section 2:  "Phase-1 출력에 VARPRO 400스텝 -- (a,b) 를 닫힌형으로 소거하고
k 에서만 하강."   This module is that arm, and nothing else.

  * `wavesplat/**` is NOT modified.  The arm is a method monkeypatch on
    `SequentialUpdater._initialize_learned` installed from the batch directory,
    inside the worker process, and removed in a `finally`.
  * ZERO estimator episodes are executed by this file.  `--selftest` and `--bench`
    are closed-form / synthetic only.

--------------------------------------------------------------------------------
MATH
--------------------------------------------------------------------------------
Shipped forward model (`wavesplat/model/spectral_gaussian.py:118-127`):

    z(x,y,t) = SUM_j  w_j * cos( kx_j x + ky_j y - omega_j t + phi_j )
    omega_j  = sqrt( g |k_j| tanh(|k_j| d) )        (`model/dispersion.py:33`)

With theta_j = kx_j x + ky_j y - omega_j t,

    w cos(theta + phi) = (w cos phi) cos theta + (- w sin phi) sin theta
                       =      a      cos theta +        b      sin theta

so the model is EXACTLY LINEAR in (a,b)   [the three paths that would break that
linearity -- stokes2 / neural corrector / free-omega -- are all off in the shipped
config; `recon/W1_path.md` section 2.2].  The polar map is

    a = w cos phi ,  b = - w sin phi        (forward)
    w = hypot(a,b) ,  phi = atan2(-b, a)    (inverse)

!! `sequential_update.py:1335` uses `atan2(b, a)` and is WRONG for this forward
model.  `model/tnse.py:216` uses `atan2(-b, a)` and is right.  PREREG G-7 gates
this with an identity test; see `selftest()` T2/T3 below.

VARPRO reduced objective (K = (kx,ky) in R^{2N}; the ONLY free variables):

    Phi(K) = [ cos Theta | sin Theta ]                       (M x 2N)
    c(K)   = argmin_c || S ( Phi c - z ) ||^2                (closed form)
           = ( Phi_w^T Phi_w + eps I )^{-1} Phi_w^T z_w ,    Phi_w = S Phi, z_w = S z
    f(K)   = || Phi_w c(K) - z_w ||^2 / M
           = SUM_i conf_i ( zhat_i - z_i )^2 / n_obs         (S = diag(sqrt(conf)))

The last line is IDENTICALLY the shipped data term `_compute_loss` builds at
`sequential_update.py:329-342` (weighted square sum divided by the UNweighted
n_obs).  `selftest()` T6 checks that against the shipped `_compute_loss` itself.

GRADIENT.  torch autograd through `torch.linalg.solve(Phi_w^T Phi_w + eps I, ...)`,
in float64.  This is the FULL Golub-Pereyra (1973) gradient, not the Kaufman
approximation: autograd differentiates the whole composition K -> Phi -> c -> f,
so the d c / d K path is carried, which is exactly the term Kaufman drops.  No
finite differences and no hand-written Jacobian are used in the arm; `selftest()`
T4 verifies the autograd gradient against central finite differences.
(`torch.linalg.lstsq` is avoided -- its autograd support is limited; normal
equations + ridge, per `recon/W2_upper_bound.md` section 4.2.)

WEIGHTING.  Default `weighting='confidence'` = rows scaled by sqrt(conf_i).
Rationale: the arm is judged against `L-SHIP`, whose Phase-1 AND tracking data term
is SUM conf_i r_i^2 (`sequential_update.py:331-334`, `use_confidence=True` by
default at `run.py:62`).  Re-fitting under a DIFFERENT data term would make `L-VP`
a two-factor arm (parameterisation + weighting), and PREREG section 2 declares it
單一 "단일(하강 매개화)".  Recorded per row as `vp_weighting` because the
pre-exposure probe (`recon/probe_varpro.py:_fc`) and W3's `ORA-K-CVX`
(`run_w3.py:479` `weighting:'none'`) were both UNWEIGHTED -- the axis must be
visible, not assumed.  `weighting='none'` reproduces the probe exactly.

STEPS.  `steps` is an argument, default 400 (PREREG froze 400; W2 section 3.5
measured 400->4000 flat).  There is deliberately NO convergence early-exit, so the
executed count is a constant and `vp_steps` records what actually ran, per PREREG
G-3.  Optimiser = Adam(lr=2e-3), inherited unchanged from the pre-exposure probe.

--------------------------------------------------------------------------------
INSERTION POINT
--------------------------------------------------------------------------------
`SequentialUpdater._initialize_learned` (`sequential_update.py:617-673`) is WRAPPED
(original called first, VARPRO applied to its output).  Chosen because:

  * PREREG section 2 says "Phase-1 *출력*에", and the pre-exposure signal
    (PROBE-B, -47.3%) was measured starting from the atom set the estimator ENDS
    WITH -- so descending from the post-Phase-1 k is the faithful online port.
    ==> `L-VP` keeps the shipped 500 Adam iters (`adam_steps_phase1 > 0`, which is
    what PREREG G-2 constrains only for `L-ITER0`/`L-SHIP`), and SECONDARY `S-2`
    (`L-VP < L-ITER0`) is what separates VARPRO from descent-removal.
    If the PREREG owner instead wants VARPRO to REPLACE the descent, set
    `learned_init_iters: 0` in the arm config -- no code change here.
  * `_initialize_inner:517` calls `self._initialize_learned(...)` by instance
    lookup, so a class-level patch reaches it, including every
    `MedianEnsembleUpdater` member (`sequential_update.py:1493`, `:1518`).
  * At return time `_adaptive_prune_grow` (`:660`) has already replaced any
    `nn.Parameter` (`spectral_gaussian.py:191-194`, `:210-213`), so writing with
    in-place `copy_` is safe, and the tracking optimiser built at `:662` keeps
    valid parameter references -- no rebuild needed (its Adam state is still empty).
  * Rows that route to `v2_konly` get NO intervention and record `vp_steps=0`;
    with PREREG G-1 pinning `init_method='learned'` that set is empty by
    construction, and the counter proves it rather than assuming it.

CLAMP.  After install we call `model.clamp_steepness(self.max_steepness)` (the
shipped invariant, applied after every shipped Adam step at `:650`).  Not clamping
would hand Phase 2 a state the shipped code never produces, and `:979` would clamp
it on the first tracked frame anyway.  `vp_clamp_hits` records how many components
the projection actually touched.  Switchable via `clamp=False`.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, '..', '..'))   # /home/vexit/wavesplat/wavesplat
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from wavesplat.model.dispersion import dispersion_omega, G   # noqa: E402

VP_STEPS_DEFAULT = 400        # PREREG section 2, frozen
VP_LR_DEFAULT = 2e-3          # inherited from recon/probe_varpro.py
VP_RIDGE_DEFAULT = 1e-8       # normal-equation ridge (probe used the same)
_DT = torch.float64


# ----------------------------------------------------------------------------
# 1. polar <-> cartesian amplitude convention  (the G-7 surface)
# ----------------------------------------------------------------------------

def wphi_to_ab(w, phi):
    """(w, phi) -> (a, b) for z = SUM w cos(theta + phi) = SUM a cos + b sin."""
    return w * torch.cos(phi), -w * torch.sin(phi)


def ab_to_wphi(a, b):
    """(a, b) -> (w, phi).  phi = atan2(-b, a)  -- `model/tnse.py:216`.

    NOT `atan2(b, a)`: that is `sequential_update.py:1335`, which is sign-flipped
    with respect to the forward model at `spectral_gaussian.py:118-127`.
    """
    return torch.sqrt(a ** 2 + b ** 2), torch.atan2(-b, a)


# ----------------------------------------------------------------------------
# 2. design matrix / reduced objective
# ----------------------------------------------------------------------------

def _omega_of(kx, ky, depth):
    """omega exactly as the shipped model computes it.

    `spectral_gaussian.py:100` forms |k| with a +1e-12 guard inside the sqrt and
    then calls `dispersion_omega` (`dispersion.py:33`), which carries its own
    +1e-12 inside its sqrt.  Both guards are reproduced by calling the package
    function rather than re-deriving the relation.
    """
    k_mag = torch.sqrt(kx ** 2 + ky ** 2 + 1e-12)
    return dispersion_omega(k_mag, depth)


def design_matrix(kx, ky, x, y, t, depth):
    """Phi = [cos Theta | sin Theta], (M, 2N).  Returns (Phi, omega)."""
    om = _omega_of(kx, ky, depth)
    theta = (kx[None, :] * x[:, None]
             + ky[None, :] * y[:, None]
             - om[None, :] * t[:, None])
    return torch.cat([torch.cos(theta), torch.sin(theta)], dim=1), om


def reduced_objective(kx, ky, x, y, t, z, sqrt_c, depth, ridge=VP_RIDGE_DEFAULT):
    """f(K) = ||S(Phi c - z)||^2 / M  with c eliminated in closed form.

    Returns (f, a, b, omega).  Differentiable in (kx, ky) through the solve.
    """
    Phi, om = design_matrix(kx, ky, x, y, t, depth)
    n = kx.shape[0]
    Phi_w = Phi * sqrt_c[:, None]
    z_w = z * sqrt_c
    A = Phi_w.T @ Phi_w + ridge * torch.eye(2 * n, dtype=Phi.dtype, device=Phi.device)
    c = torch.linalg.solve(A, Phi_w.T @ z_w)
    r = Phi_w @ c - z_w
    f = (r ** 2).sum() / z.shape[0]      # divide by UNweighted n_obs -- `:337-341`
    return f, c[:n], c[n:], om


# ----------------------------------------------------------------------------
# 3. the solver
# ----------------------------------------------------------------------------

def varpro_refine(kx0, ky0, x, y, t, z, conf, depth,
                  steps=VP_STEPS_DEFAULT, lr=VP_LR_DEFAULT,
                  ridge=VP_RIDGE_DEFAULT, weighting='confidence'):
    """VARPRO: eliminate (a,b) in closed form, descend on k only.

    All tensors float64 on CPU.  Returns (kx, ky, a, b, info).
    `info['steps_executed']` is the count that actually ran (PREREG G-3).
    """
    if weighting == 'confidence':
        sqrt_c = torch.sqrt(torch.clamp(conf, min=0.0))
    elif weighting == 'none':
        sqrt_c = torch.ones_like(z)
    else:
        raise ValueError("weighting must be 'confidence' or 'none'")

    kx = kx0.clone().detach().requires_grad_(True)
    ky = ky0.clone().detach().requires_grad_(True)
    opt = torch.optim.Adam([kx, ky], lr=lr)

    with torch.no_grad():
        f0, a0, b0, om0 = reduced_objective(kx, ky, x, y, t, z, sqrt_c, depth, ridge)

    t_start = time.time()
    executed = 0
    for _ in range(int(steps)):
        opt.zero_grad(set_to_none=True)
        f, _a, _b, _om = reduced_objective(kx, ky, x, y, t, z, sqrt_c, depth, ridge)
        f.backward()
        opt.step()
        executed += 1
    wall = time.time() - t_start

    with torch.no_grad():
        f1, a1, b1, om1 = reduced_objective(kx, ky, x, y, t, z, sqrt_c, depth, ridge)
        # unweighted window residual RMS as well -- W2 reported the unweighted one
        Phi0, _ = design_matrix(kx0, ky0, x, y, t, depth)
        Phi1, _ = design_matrix(kx.detach(), ky.detach(), x, y, t, depth)
        r0 = Phi0 @ torch.cat([a0, b0]) - z
        r1 = Phi1 @ torch.cat([a1, b1]) - z
        dk = torch.sqrt((kx.detach() - kx0) ** 2 + (ky.detach() - ky0) ** 2)

    info = {
        'steps_executed': executed,
        'lr': float(lr),
        'ridge': float(ridge),
        'weighting': weighting,
        'obj_before': float(f0),          # deployment-aligned (weighted / n_obs)
        'obj_after': float(f1),
        'win_rms_before': float(torch.sqrt((r0 ** 2).mean())),   # unweighted
        'win_rms_after': float(torch.sqrt((r1 ** 2).mean())),
        'k_shift_median': float(dk.median()) if dk.numel() else 0.0,
        'k_shift_max': float(dk.max()) if dk.numel() else 0.0,
        'n_atoms': int(kx0.shape[0]),
        'n_obs': int(z.shape[0]),
        'wall_s': wall,
    }
    return kx.detach(), ky.detach(), a1, b1, info


# ----------------------------------------------------------------------------
# 4. the arm: monkeypatch on the shipped Phase-1
# ----------------------------------------------------------------------------

_ORIG_INIT_LEARNED = None
_CFG: Dict = {}
_CALLS: List[Dict] = []


def _obs_from_frames(frames) -> Optional[Tuple[torch.Tensor, ...]]:
    """Flatten the Phase-1 window exactly as `_compute_loss` sees it (`:327-336`)."""
    xs, ys, zs, ts, cs = [], [], [], [], []
    for f in frames:
        m = int(len(f.x_obs))
        if m == 0:
            continue
        xs.append(f.x_obs.detach().to('cpu', _DT))
        ys.append(f.y_obs.detach().to('cpu', _DT))
        zs.append(f.z_obs.detach().to('cpu', _DT))
        cs.append(f.confidence.detach().to('cpu', _DT))
        ts.append(torch.full((m,), float(f.t), dtype=_DT))
    if not xs:
        return None
    return (torch.cat(xs), torch.cat(ys), torch.cat(ts),
            torch.cat(zs), torch.cat(cs))


def _wrap_pi(d):
    return (d + math.pi) % (2 * math.pi) - math.pi


def apply_varpro(updater, frames, steps=VP_STEPS_DEFAULT, lr=VP_LR_DEFAULT,
                 ridge=VP_RIDGE_DEFAULT, weighting='confidence', clamp=True,
                 verbose=False) -> Dict:
    """Run VARPRO on `updater.model` in place.  Returns the counter record."""
    rec = {'vp_entry': 1, 'vp_steps': 0, 'vp_skipped': None, 'pid': os.getpid()}

    model = getattr(updater, 'model', None)
    if getattr(updater, '_bank', None) is not None:
        rec['vp_skipped'] = 'bank_route'
        return rec
    if model is None or int(getattr(model, 'n_gaussians', 0)) == 0:
        rec['vp_skipped'] = 'no_model'
        return rec
    obs = _obs_from_frames(frames)
    if obs is None:
        rec['vp_skipped'] = 'empty_window'
        return rec
    X, Y, T, Z, C = obs

    # honour the shipped weighting switch; deployment default is True (`run.py:62`)
    eff_weighting = weighting
    if weighting == 'confidence' and not bool(getattr(updater, 'use_confidence', True)):
        eff_weighting = 'none'
        rec['vp_weighting_fallback'] = 'use_confidence=False'

    with torch.no_grad():
        kx0 = model.kx.detach().to('cpu', _DT).clone()
        ky0 = model.ky.detach().to('cpu', _DT).clone()
        w0 = model.w.detach().to('cpu', _DT).clone()
        phi0 = model.phi.detach().to('cpu', _DT).clone()

    kx1, ky1, a1, b1, info = varpro_refine(
        kx0, ky0, X, Y, T, Z, C, float(updater.depth),
        steps=steps, lr=lr, ridge=ridge, weighting=eff_weighting)
    w1, phi1 = ab_to_wphi(a1, b1)

    with torch.no_grad():
        model.kx.copy_(kx1.to(model.kx.dtype).to(model.kx.device))
        model.ky.copy_(ky1.to(model.ky.dtype).to(model.ky.device))
        model.w.copy_(w1.to(model.w.dtype).to(model.w.device))
        model.phi.copy_(phi1.to(model.phi.dtype).to(model.phi.device))
        if getattr(model, 'use_stokes2', False):
            model.update_stokes2_transfer()

        hits = 0
        if clamp:
            k_mag = torch.sqrt(model.kx ** 2 + model.ky ** 2 + 1e-12)
            hits = int((k_mag * torch.abs(model.w)
                        > float(updater.max_steepness)).sum())
            model.clamp_steepness(float(updater.max_steepness))

    rec.update({
        'vp_steps': info['steps_executed'],
        'vp_lr': info['lr'],
        'vp_ridge': info['ridge'],
        'vp_weighting': info['weighting'],
        'vp_clamped': bool(clamp),
        'vp_clamp_hits': hits,
        'vp_n_atoms': info['n_atoms'],
        'vp_n_obs': info['n_obs'],
        'vp_obj_before': info['obj_before'],
        'vp_obj_after': info['obj_after'],
        'vp_win_rms_before': info['win_rms_before'],
        'vp_win_rms_after': info['win_rms_after'],
        'vp_k_shift': info['k_shift_median'],
        'vp_k_shift_max': info['k_shift_max'],
        'vp_dw_max': float(torch.max(torch.abs(w1 - w0))),
        'vp_dphi_max': float(torch.max(torch.abs(_wrap_pi(phi1 - phi0)))),
        'vp_wall_s': info['wall_s'],
    })
    if verbose:
        print("  [L-VP] N=%d M=%d steps=%d obj %.6e -> %.6e | win %.5f -> %.5f "
              "| dk_med %.5f | %.2fs"
              % (rec['vp_n_atoms'], rec['vp_n_obs'], rec['vp_steps'],
                 rec['vp_obj_before'], rec['vp_obj_after'],
                 rec['vp_win_rms_before'], rec['vp_win_rms_after'],
                 rec['vp_k_shift'], rec['vp_wall_s']), flush=True)
    return rec


def install(steps=VP_STEPS_DEFAULT, lr=VP_LR_DEFAULT, ridge=VP_RIDGE_DEFAULT,
            weighting='confidence', clamp=True, verbose=False):
    """Patch `SequentialUpdater._initialize_learned`.  Call INSIDE the worker.

    `recon/W1_path.md` section 5: patch in the worker process, undo in a `finally`
    -- patching the parent lets the arm leak across `fork`.
    """
    global _ORIG_INIT_LEARNED
    if _ORIG_INIT_LEARNED is not None:
        raise RuntimeError('varpro already installed')
    import wavesplat.optimizer.sequential_update as SU

    _ORIG_INIT_LEARNED = SU.SequentialUpdater._initialize_learned
    _CFG.clear()
    _CFG.update(dict(steps=int(steps), lr=float(lr), ridge=float(ridge),
                     weighting=weighting, clamp=bool(clamp), verbose=bool(verbose)))

    def _patched(self, frames, x_range=(10.0, 250.0), y_range=(-100.0, 100.0),
                 verbose=False):
        # signature mirrors `sequential_update.py:617-620` exactly, so the single
        # call site (`:517`, positional) and any keyword call both reach it
        out = _ORIG_INIT_LEARNED(self, frames, x_range, y_range, verbose)
        rec = apply_varpro(self, frames, steps=_CFG['steps'], lr=_CFG['lr'],
                           ridge=_CFG['ridge'], weighting=_CFG['weighting'],
                           clamp=_CFG['clamp'],
                           verbose=_CFG['verbose'] or bool(verbose))
        _CALLS.append(rec)
        if isinstance(out, dict):
            out['varpro'] = rec
        return out

    _patched.__name__ = '_initialize_learned__L_VP'
    SU.SequentialUpdater._initialize_learned = _patched
    return describe()


def uninstall():
    global _ORIG_INIT_LEARNED
    if _ORIG_INIT_LEARNED is None:
        return
    import wavesplat.optimizer.sequential_update as SU
    SU.SequentialUpdater._initialize_learned = _ORIG_INIT_LEARNED
    _ORIG_INIT_LEARNED = None


def reset_stats():
    _CALLS.clear()


def describe() -> Dict:
    """Effective arm config, read back from module state (not an intent dict)."""
    import wavesplat.optimizer.sequential_update as SU
    return {
        'installed': _ORIG_INIT_LEARNED is not None,
        'patched_name': getattr(SU.SequentialUpdater._initialize_learned,
                                '__name__', None),
        'code_md5': _self_md5(),
        **{('vp_cfg_' + k): v for k, v in _CFG.items()},
    }


def stats() -> Dict:
    """Per-run counters for the result row (PREREG section 5 required fields)."""
    entries = len(_CALLS)
    steps = [c.get('vp_steps', 0) for c in _CALLS]
    return {
        'vp_entries': entries,
        'vp_steps': int(sum(steps)),                       # G-3: 400 x entries
        'vp_steps_per_entry': steps,
        'vp_skipped': [c.get('vp_skipped') for c in _CALLS],
        'vp_k_shift': float(np.median([c['vp_k_shift'] for c in _CALLS]))
                      if entries and 'vp_k_shift' in _CALLS[0] else None,
        'vp_dw_max': max([c.get('vp_dw_max', 0.0) for c in _CALLS], default=0.0),
        'vp_dphi_max': max([c.get('vp_dphi_max', 0.0) for c in _CALLS], default=0.0),
        'vp_clamp_hits': int(sum(c.get('vp_clamp_hits', 0) for c in _CALLS)),
        'vp_wall_s': float(sum(c.get('vp_wall_s', 0.0) for c in _CALLS)),
        'vp_calls': list(_CALLS),
    }


def _self_md5() -> str:
    import hashlib
    with open(os.path.abspath(__file__), 'rb') as fh:
        return hashlib.md5(fh.read()).hexdigest()


# ----------------------------------------------------------------------------
# 5. SELF-TEST  (PREREG G-7).  Closed form + synthetic only -- no episode is run.
# ----------------------------------------------------------------------------

def _synthetic_frames(n_comp=7, n_frames=4, m_per=160, seed=0):
    """Random (w, phi, kx, ky) + random observation points/times, float64."""
    rng = np.random.default_rng(seed)
    depth = 50.0
    kmag = rng.uniform(0.04, 0.30, n_comp)
    kang = rng.uniform(-math.pi, math.pi, n_comp)
    kx = torch.tensor(kmag * np.cos(kang), dtype=_DT)
    ky = torch.tensor(kmag * np.sin(kang), dtype=_DT)
    w = torch.tensor(rng.uniform(0.05, 0.6, n_comp), dtype=_DT)
    phi = torch.tensor(rng.uniform(-math.pi, math.pi, n_comp), dtype=_DT)
    fr = []
    for i in range(n_frames):
        x = torch.tensor(rng.uniform(10.0, 250.0, m_per), dtype=_DT)
        y = torch.tensor(rng.uniform(-100.0, 100.0, m_per), dtype=_DT)
        c = torch.tensor(rng.uniform(0.2, 1.0, m_per), dtype=_DT)
        fr.append((float(i) * 0.5, x, y, c))
    return depth, kx, ky, w, phi, fr


def _build_pkg_model(n, depth, kx, ky, w, phi):
    """A REAL `SpectralGaussianModel` in float64 -- the shipped forward code."""
    from wavesplat.model.spectral_gaussian import SpectralGaussianModel
    m = SpectralGaussianModel(n, depth=depth).double()
    with torch.no_grad():
        m.kx.copy_(kx); m.ky.copy_(ky); m.w.copy_(w); m.phi.copy_(phi)
    return m


def _pkg_forward(model, fr):
    with torch.no_grad():
        return torch.cat([model(x, y, t) for (t, x, y, _c) in fr])


def selftest(verbose=True) -> bool:
    """Six checks.  T2 is the PREREG G-7 gate (reconstruction residual <= 1e-12)."""
    TOL = 1e-12
    ok = True
    log = (lambda *a: print(*a, flush=True)) if verbose else (lambda *a: None)

    depth, kx, ky, w, phi, fr = _synthetic_frames()
    n = kx.shape[0]
    X = torch.cat([x for (_t, x, _y, _c) in fr])
    Y = torch.cat([y for (_t, _x, y, _c) in fr])
    C = torch.cat([c for (_t, _x, _y, c) in fr])
    T = torch.cat([torch.full((len(x),), t, dtype=_DT) for (t, x, _y, _c) in fr])

    model = _build_pkg_model(n, depth, kx, ky, w, phi)
    # ground truth from the SHIPPED forward code (detached: model params are leaves)
    z_ref = _pkg_forward(model, fr).detach()

    # a perturbed, noisy problem for the checks that need a NON-zero residual
    # (at the exact truth the reduced objective sits at an exact zero minimum, so
    #  its gradient is analytically 0 and a gradient check there is degenerate)
    _rng0 = np.random.default_rng(11)
    _dk = torch.tensor(_rng0.normal(0.0, 0.004, (2, n)), dtype=_DT)
    kx_p, ky_p = kx + _dk[0], ky + _dk[1]
    z_noisy = z_ref + torch.tensor(_rng0.normal(0.0, 0.002, len(z_ref)), dtype=_DT)

    log('=' * 78)
    log('varpro.py selftest -- torch %s, dtype %s, N=%d, M=%d'
        % (torch.__version__, _DT, n, len(z_ref)))
    log('=' * 78)

    # -- T1: design matrix + (w,phi)->(a,b) reproduces the shipped forward model
    a, b = wphi_to_ab(w, phi)
    Phi, om = design_matrix(kx, ky, X, Y, T, depth)
    e1 = float(torch.max(torch.abs(Phi @ torch.cat([a, b]) - z_ref)))
    p1 = e1 <= TOL
    ok &= p1
    log('T1 design matrix == spectral_gaussian.forward      max|d| = %.3e   %s'
        % (e1, 'PASS' if p1 else 'FAIL'))

    # -- T2: G-7 GATE.  known (w,phi) -> solve (a,b) -> back to (w,phi) -> re-render
    coef = torch.linalg.lstsq(Phi, z_ref.unsqueeze(1)).solution.squeeze(1)
    w_hat, phi_hat = ab_to_wphi(coef[:n], coef[n:])
    z_rec = _pkg_forward(_build_pkg_model(n, depth, kx, ky, w_hat, phi_hat), fr)
    e2 = float(torch.max(torch.abs(z_rec - z_ref)))
    p2 = e2 <= TOL
    ok &= p2
    log('T2 [G-7] (A,phi) -> (a,b) -> (A,phi) round trip     max|d| = %.3e   %s'
        % (e2, 'PASS' if p2 else 'FAIL'))
    e2b = float(torch.max(torch.abs(w_hat - w)))
    e2c = float(torch.max(torch.abs(_wrap_pi(phi_hat - phi))))
    p2b = (e2b <= 1e-10) and (e2c <= 1e-10)
    ok &= p2b
    log('   parameters themselves recovered                 dw=%.3e dphi=%.3e %s'
        % (e2b, e2c, 'PASS' if p2b else 'FAIL'))

    # -- T3: negative control.  The `sequential_update.py:1335` convention must FAIL.
    phi_bug = torch.atan2(coef[n:], coef[:n])                 # atan2(b, a)  <-- BUG
    z_bug = _pkg_forward(_build_pkg_model(n, depth, kx, ky, w_hat, phi_bug), fr)
    e3 = float(torch.max(torch.abs(z_bug - z_ref)))
    p3 = e3 > 1e-3
    ok &= p3
    log('T3 negative control atan2(b,a) (SU.py:1335) breaks  max|d| = %.3e   %s'
        % (e3, 'PASS (test is not vacuous)' if p3 else 'FAIL'))

    # -- T4: autograd gradient == central finite differences (Golub-Pereyra, full)
    kxg = kx_p.clone().requires_grad_(True)
    kyg = ky_p.clone().requires_grad_(True)
    sc = torch.sqrt(C)
    f, _a, _b, _o = reduced_objective(kxg, kyg, X, Y, T, z_noisy, sc, depth)
    f.backward()
    g_auto = torch.cat([kxg.grad, kyg.grad]).clone()
    h = 1e-7
    g_fd = torch.zeros(2 * n, dtype=_DT)
    with torch.no_grad():
        for j in range(2 * n):
            base = kx_p.clone() if j < n else ky_p.clone()
            idx = j if j < n else j - n
            up, dn = base.clone(), base.clone()
            up[idx] += h; dn[idx] -= h
            if j < n:
                fp, *_ = reduced_objective(up, ky_p, X, Y, T, z_noisy, sc, depth)
                fm, *_ = reduced_objective(dn, ky_p, X, Y, T, z_noisy, sc, depth)
            else:
                fp, *_ = reduced_objective(kx_p, up, X, Y, T, z_noisy, sc, depth)
                fm, *_ = reduced_objective(kx_p, dn, X, Y, T, z_noisy, sc, depth)
            g_fd[j] = (fp - fm) / (2 * h)
    denom = float(torch.max(torch.abs(g_fd))) + 1e-300
    e4 = float(torch.max(torch.abs(g_auto - g_fd))) / denom
    p4 = e4 <= 1e-6
    ok &= p4
    log('T4 autograd grad == central FD (rel to max|g_fd|)   rel    = %.3e   %s'
        % (e4, 'PASS' if p4 else 'FAIL'))

    # -- T5: the reduced objective actually descends and pulls k toward truth
    kxv, kyv, av, bv, info = varpro_refine(kx_p, ky_p, X, Y, T, z_noisy, C, depth,
                                           steps=200)
    d_before = float(torch.sqrt((kx_p - kx) ** 2 + (ky_p - ky) ** 2).mean())
    d_after = float(torch.sqrt((kxv - kx) ** 2 + (kyv - ky) ** 2).mean())
    p5 = (info['obj_after'] < info['obj_before']) and (d_after < d_before)
    ok &= p5
    log('T5 200-step descent: obj %.4e -> %.4e | |dk| %.5f -> %.5f   %s'
        % (info['obj_before'], info['obj_after'], d_before, d_after,
           'PASS' if p5 else 'FAIL'))

    # -- T6: our objective == the SHIPPED `_compute_loss` data term (reg=False)
    e6 = _t6_objective_matches_shipped_loss(depth, kx, ky, w, phi, fr, z_noisy)
    p6 = e6 <= 1e-12
    ok &= p6
    log('T6 objective == SU._compute_loss(reg=False)         rel    = %.3e   %s'
        % (e6, 'PASS' if p6 else 'FAIL'))

    # -- T7: install / uninstall leaves the class attribute bit-identical
    import wavesplat.optimizer.sequential_update as SU
    before = SU.SequentialUpdater._initialize_learned
    install(steps=3, verbose=False)
    mid = SU.SequentialUpdater._initialize_learned
    uninstall()
    after = SU.SequentialUpdater._initialize_learned
    p7 = (mid is not before) and (after is before)
    ok &= p7
    log('T7 install/uninstall restores the shipped method                   %s'
        % ('PASS' if p7 else 'FAIL'))

    # -- T8: the intervention actually moves a real model + fills the counters
    rec, loss_pair = _t8_intervention_on_real_objects(depth, kx_p, ky_p, w, phi,
                                                      fr, z_noisy)
    p8 = (rec['vp_steps'] == 50 and rec['vp_dw_max'] > 0 and rec['vp_dphi_max'] > 0
          and rec['vp_k_shift'] > 0 and rec['vp_skipped'] is None
          and loss_pair[1] < loss_pair[0])
    ok &= p8
    log('T8 apply_varpro on a real SequentialUpdater: steps=%d dk=%.5f '
        'loss %.4e -> %.4e   %s'
        % (rec['vp_steps'], rec['vp_k_shift'], loss_pair[0], loss_pair[1],
           'PASS' if p8 else 'FAIL'))

    log('-' * 78)
    log('SELFTEST %s' % ('PASS' if ok else 'FAIL'))
    return ok


def _t6_objective_matches_shipped_loss(depth, kx, ky, w, phi, fr, z_all) -> float:
    """Compare `reduced_objective`'s value at a FIXED (a,b) against the shipped
    weighted data term computed by `SequentialUpdater._compute_loss(reg=False)`
    (`sequential_update.py:313-345`) on the same model and window."""
    from wavesplat.data_loader import FrameData
    import wavesplat.optimizer.sequential_update as SU

    n = kx.shape[0]
    model = _build_pkg_model(n, depth, kx, ky, w, phi)

    frames, off = [], 0
    for (t, x, y, c) in fr:
        m = len(x)
        frames.append(FrameData(t=t, x_obs=x, y_obs=y, z_obs=z_all[off:off + m],
                                confidence=c,
                                incidence_angles=torch.zeros(m, dtype=_DT)))
        off += m

    su = SU.SequentialUpdater({'depth': depth, 'use_confidence': True})
    su.model = model
    with torch.no_grad():
        shipped = float(su._compute_loss(frames, reg=False))

    X = torch.cat([x for (_t, x, _y, _c) in fr])
    Y = torch.cat([y for (_t, _x, y, _c) in fr])
    C = torch.cat([c for (_t, _x, _y, c) in fr])
    T = torch.cat([torch.full((len(x),), t, dtype=_DT) for (t, x, _y, _c) in fr])
    Phi, _om = design_matrix(kx, ky, X, Y, T, depth)
    a, b = wphi_to_ab(w, phi)
    r = (Phi @ torch.cat([a, b]) - z_all)
    ours = float(((C * r ** 2).sum() / len(z_all)))
    return abs(ours - shipped) / (abs(shipped) + 1e-300)


def _t8_intervention_on_real_objects(depth, kx, ky, w, phi, fr, z_all):
    """Drive `apply_varpro` against a genuine `SequentialUpdater` + genuine
    `SpectralGaussianModel` on synthetic frames.  No episode, no learned proposal,
    no `initialize()` -- this exercises exactly the code the patch runs after
    `_initialize_learned` returns, and reports the shipped loss before/after."""
    from wavesplat.data_loader import FrameData
    import wavesplat.optimizer.sequential_update as SU

    n = kx.shape[0]
    model = _build_pkg_model(n, depth, kx, ky, w, phi)
    frames, off = [], 0
    for (t, x, y, c) in fr:
        m = len(x)
        frames.append(FrameData(t=t, x_obs=x, y_obs=y, z_obs=z_all[off:off + m],
                                confidence=c,
                                incidence_angles=torch.zeros(m, dtype=_DT)))
        off += m

    su = SU.SequentialUpdater({'depth': depth, 'use_confidence': True})
    su.model = model
    with torch.no_grad():
        before = float(su._compute_loss(frames, reg=False))
    rec = apply_varpro(su, frames, steps=50, verbose=False)
    with torch.no_grad():
        after = float(su._compute_loss(frames, reg=False))
    return rec, (before, after)


# ----------------------------------------------------------------------------
# 6. synthetic cost benchmark (no episode)
# ----------------------------------------------------------------------------

def bench(steps=VP_STEPS_DEFAULT, sizes=((8080, 20), (8080, 30), (8080, 45)),
          verbose=True):
    """Wall-clock cost of `steps` VARPRO iterations at deployment-scale (M, N)."""
    out = []
    for (M, N) in sizes:
        rng = np.random.default_rng(3)
        depth = 50.0
        kmag = rng.uniform(0.04, 0.30, N); kang = rng.uniform(-math.pi, math.pi, N)
        kx = torch.tensor(kmag * np.cos(kang), dtype=_DT)
        ky = torch.tensor(kmag * np.sin(kang), dtype=_DT)
        X = torch.tensor(rng.uniform(10, 250, M), dtype=_DT)
        Y = torch.tensor(rng.uniform(-100, 100, M), dtype=_DT)
        T = torch.tensor(rng.uniform(0, 8, M), dtype=_DT)
        C = torch.tensor(rng.uniform(0.2, 1.0, M), dtype=_DT)
        Z = torch.tensor(rng.normal(0, 0.5, M), dtype=_DT)
        t0 = time.time()
        _kx, _ky, _a, _b, info = varpro_refine(kx, ky, X, Y, T, Z, C, depth,
                                               steps=steps)
        row = {'M': M, 'N': N, 'steps': info['steps_executed'],
               'wall_s': round(time.time() - t0, 3),
               'threads': torch.get_num_threads()}
        out.append(row)
        if verbose:
            print('  M=%5d N=%3d steps=%d  ->  %6.2f s  (%d threads)'
                  % (M, N, row['steps'], row['wall_s'], row['threads']), flush=True)
    return out


def main():
    ap = argparse.ArgumentParser(description='W4 L-VP (VARPRO) arm')
    ap.add_argument('--selftest', action='store_true')
    ap.add_argument('--bench', action='store_true')
    ap.add_argument('--steps', type=int, default=VP_STEPS_DEFAULT)
    ap.add_argument('--json', type=str, default=None)
    args = ap.parse_args()

    result = {'code_md5': _self_md5(), 'torch': torch.__version__}
    rc = 0
    if args.selftest:
        passed = selftest(verbose=True)
        result['selftest_pass'] = passed
        rc |= 0 if passed else 1
    if args.bench:
        print('\nsynthetic cost benchmark (%d steps, no episode):' % args.steps)
        result['bench'] = bench(steps=args.steps)
    if not args.selftest and not args.bench:
        ap.print_help()
    if args.json:
        with open(args.json, 'w') as fh:
            json.dump(result, fh, indent=2)
    sys.exit(rc)


if __name__ == '__main__':
    main()
