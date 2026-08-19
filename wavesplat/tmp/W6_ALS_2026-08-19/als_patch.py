#!/usr/bin/env python3
"""L-ALS: VARPRO with the derivative through the elimination removed.

`varpro.reduced_objective` solves for the linear coefficients in closed form and lets the
gradient flow BACK THROUGH that solve -- its own docstring says "Differentiable in (kx, ky)
through the solve".  That is what makes it variable projection rather than alternating
least squares.

This module wraps that one function so `c` is detached.  The closed-form solve is
unchanged, the step count is unchanged, the weighting is unchanged, the starting point is
unchanged -- only the path of the gradient differs.  It is the single line that separates
the two methods, which is why PREREG section 2 can call the arm single-factor.

    import als_patch; als_patch.install()   # after varpro.install()
"""
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
W4 = os.path.join(ROOT, 'tmp', 'W4_SOLVER_2026-08-19')
for p in (ROOT, W4, HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

import varpro                                                 # noqa: E402

_ORIG = None
_STATS = {'calls': 0}


def _als_objective(kx, ky, x, y, t, z, sqrt_c, depth, ridge=varpro.VP_RIDGE_DEFAULT):
    """Identical to varpro.reduced_objective except `c` carries no gradient."""
    Phi, om = varpro.design_matrix(kx, ky, x, y, t, depth)
    n = kx.shape[0]
    Phi_w = Phi * sqrt_c[:, None]
    z_w = z * sqrt_c
    A = Phi_w.T @ Phi_w + ridge * torch.eye(2 * n, dtype=Phi.dtype, device=Phi.device)
    c = torch.linalg.solve(A, Phi_w.T @ z_w).detach()   # <-- the whole treatment
    r = Phi_w @ c - z_w
    f = (r ** 2).sum() / z.shape[0]
    _STATS['calls'] += 1
    return f, c[:n], c[n:], om


def install():
    global _ORIG
    if _ORIG is not None:
        return
    _ORIG = varpro.reduced_objective
    varpro.reduced_objective = _als_objective


def uninstall():
    global _ORIG
    if _ORIG is not None:
        varpro.reduced_objective = _ORIG
        _ORIG = None


def stats():
    return dict(_STATS)


def selftest(verbose=True):
    """G-3: prove the detach actually changed the gradient.

    VARPRO's gradient matched central differences to rel 1.5e-11 (W4 C2 self-test).  If ALS
    is really a different objective, ITS autograd gradient must NOT match the finite
    difference of the true reduced objective -- a test that passes for both would be vacuous.
    """
    torch.manual_seed(0)
    M, n = 400, 3
    x = torch.rand(M, dtype=torch.float64) * 100
    y = torch.rand(M, dtype=torch.float64) * 100 - 50
    t = torch.rand(M, dtype=torch.float64) * 20
    sc = torch.ones(M, dtype=torch.float64)
    kx0 = torch.tensor([0.05, 0.09, 0.13], dtype=torch.float64)
    ky0 = torch.tensor([0.01, -0.03, 0.02], dtype=torch.float64)
    Phi, _ = varpro.design_matrix(kx0, ky0, x, y, t, 50.0)
    z = Phi @ torch.randn(2 * n, dtype=torch.float64) + 0.02 * torch.randn(M, dtype=torch.float64)
    kx = (kx0 + 0.004).requires_grad_(True)
    ky = (ky0 - 0.003).requires_grad_(True)

    def grad(fn):
        f, *_ = fn(kx, ky, x, y, t, z, sc, 50.0)
        g, = torch.autograd.grad(f, kx, retain_graph=False)
        return g.clone()

    def fd(i, h=1e-6):
        out = []
        for s in (+1, -1):
            k = kx.detach().clone()
            k[i] += s * h
            f, *_ = varpro.reduced_objective(k, ky.detach(), x, y, t, z, sc, 50.0)
            out.append(float(f))
        return (out[0] - out[1]) / (2 * h)

    g_vp = grad(varpro.reduced_objective)
    g_als = grad(_als_objective)
    g_fd = torch.tensor([fd(i) for i in range(n)], dtype=torch.float64)
    rel = lambda a: float((a - g_fd).norm() / g_fd.norm())         # noqa: E731
    r_vp, r_als = rel(g_vp), rel(g_als)
    ok = (r_vp < 1e-6) and (r_als > 1e-3)
    if verbose:
        print('G-3 detach self-test (float64, M=%d, n=%d)' % (M, n))
        print('  VARPRO autograd vs central difference : rel %.3e  (must be < 1e-6)' % r_vp)
        print('  ALS    autograd vs central difference : rel %.3e  (must be > 1e-3)' % r_als)
        print('  -> %s   %s' % ('PASS' if ok else 'FAIL',
                                '' if ok else 'the detach did not change the objective'))
    return ok


if __name__ == '__main__':
    sys.exit(0 if selftest() else 1)
