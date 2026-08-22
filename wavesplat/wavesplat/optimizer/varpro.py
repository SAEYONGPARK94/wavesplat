"""VARPRO (variable projection) refinement of the learned Phase-1 initialisation.

★배포 이력 — 2026-08-22. 그전까지 이 코드는 `tmp/W4_SOLVER_2026-08-19/varpro.py` 의
**몽키패치로만** 존재했고 패키지에는 없었다. 즉 **처방은 측정됐는데 배포되지 않은 상태**였다.

측정 근거 (전부 `tmp/` 의 사전등록 배치):
  * `W4` 1,344런 — `L-VP < L-SHIP` 52/64 · -17.53% · p=2.28e-07, 9셸 전부, 비용 +3.8%.
  * `W5`   640런 — 결합 적합(`learned_init_iters`)은 잉여: P-1 58/64 -25.28%,
                    P-2 36/64 널, 벽시계 14.5 s vs 29.7 s.
  * `B`  1,152런 — 배포 라우터를 그대로 두고 ITT: 혼잡 55/60 · -25.28% · p=5.19e-12,
                    두 팔 라우팅 일치 64/64. 희소는 널(바닥 효과).
  ==> 배포 구성 = `learned_init_iters = 0` + VARPRO 400 스텝 (`B` 의 `A-VP` 팔과 동일).

⛔novelty 아님 — Golub & Pereyra, *Inverse Problems* 19 (2003) R1 초록이
"eliminates the linear variables ... results in a better-conditioned problem" 라 적었고
원 방법은 1973년이다. 여기서 사는 것은 적용과 측정이지 기제가 아니다.

아래 수학·규약 주석은 원문(`tmp/W4_SOLVER_2026-08-19/varpro.py`) 그대로다 — 감사 추적을
끊지 않기 위해 축약하지 않았다.
"""
from __future__ import annotations

import math
import os
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from wavesplat.model.dispersion import dispersion_omega, G

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

