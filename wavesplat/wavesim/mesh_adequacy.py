"""Raycast-mesh adequacy check.

The LiDAR simulator renders the sea surface as a triangle mesh laid on the
GT grid extent at `mesh.nx x mesh.ny` nodes, then raycasts against it.  If the
node spacing is coarse relative to the wavelength, the mesh cannot represent
the wave at all and the returns carry facet error instead of wave signal.

This is not hypothetical: the shipped default (mesh 40x30 over 240x200 m ->
dx 6.15 m, dy 6.90 m) renders the BF2 1-component sea (lambda 8.2 m) at
1.2 points per wavelength -- below Nyquist.  Every conclusion drawn from those
episodes measured the simulator, not the sea (docs/E2_MESH_AUDIT_prereg_
2026-07-13.md).

ONE criterion, used by both the guard and the audit (reconciled 2026-07-13 after
review found them disagreeing -- the audit judged on the peak wavelength, the
guard on the shortest significant one, so the same episode could be called both
adequate and inadequate).

The physically meaningful quantity is not a wavelength but an energy: **how much
of the sea's variance lives below the mesh's resolving power**.  A mesh that
renders the peak beautifully while dropping a fifth of the variance still feeds
the estimator clutter -- exactly what the BF4 benchmark episodes did (operator
error 0.051 m = 2.4x the nominal noise, despite 9.5 points per peak wavelength).

    var_under = fraction of sum(A^2) held by components with lambda < 6*dx

    var_under <= 5%   and peak above Nyquist  -> ADEQUATE
    var_under <= 25%  and peak above Nyquist  -> DEGRADED (usable, disclose it)
    otherwise, or peak below Nyquist          -> INADEQUATE

Points-per-wavelength (at the peak and at the shortest significant component)
are kept as diagnostics, since they are what you tune the mesh with.
"""
from typing import List, Optional

import numpy as np

PTS_PER_LAMBDA_OK = 6.0       # a wavelength is resolved at >= 6 mesh points
PTS_PER_LAMBDA_NYQUIST = 2.0  # below this a wave does not exist on the mesh
VAR_TAIL = 0.01               # "significant" component: >= 1% of total variance
VAR_UNDER_OK = 0.05           # adequate: <= 5% of variance unresolvable
VAR_UNDER_DEGRADED = 0.25     # degraded but usable: <= 25%

ADEQUATE, DEGRADED, INADEQUATE = 'ADEQUATE', 'DEGRADED', 'INADEQUATE'


def mesh_spacing(cfg) -> tuple:
    """Node spacing [m] of the raycast mesh, from the GT extent and mesh size.

    Mirrors pipeline.run_episode: make_grid(x_range, y_range, mesh.nx, mesh.ny)
    is an endpoint-inclusive linspace, and build_mesh triangulates those nodes
    directly, so node spacing == mesh spacing.
    """
    x0, x1 = cfg.gt_grid.x_range
    y0, y1 = cfg.gt_grid.y_range
    dx = (x1 - x0) / max(cfg.mesh.nx - 1, 1)
    dy = (y1 - y0) / max(cfg.mesh.ny - 1, 1)
    return dx, dy


def check(components: List, cfg, hos_order: Optional[int] = None) -> dict:
    """Report how well the raycast mesh resolves these wave components.

    `hos_order` (M) shrinks the effective wavelength by 1/M: HOS bound
    harmonics live at lambda/2, lambda/3 ... and must be renderable too.
    """
    dx, dy = mesh_spacing(cfg)
    d = max(dx, dy)

    A = np.array([c.A for c in components], float)
    k = np.array([np.hypot(c.kx, c.ky) for c in components], float)
    lam = 2 * np.pi / np.maximum(k, 1e-12)

    var = A ** 2
    keep = var >= VAR_TAIL * var.sum() if var.sum() > 0 else np.ones_like(var, bool)

    # An HOS sea of order M carries bound harmonics at lambda/2 ... lambda/M, so
    # a component counts as rendered only if its SHORTEST harmonic is rendered.
    # Shrinking the wavelengths (rather than only lam_min) keeps the verdict and
    # the variance band on the same footing -- before 2026-07-13 the harmonics
    # shortened lam_min but were invisible to var_under, so a nonlinear episode
    # could be called adequate on content the mesh cannot represent.
    m_ord = float(hos_order) if (hos_order and hos_order > 1) else 1.0
    lam_eff = lam / m_ord

    lam_min = float(lam_eff[keep].min())
    lam_peak = float(lam_eff[int(np.argmax(A))])

    under = float(var[lam_eff < PTS_PER_LAMBDA_OK * d].sum() / var.sum()) \
        if var.sum() else 0.0
    pts_peak = lam_peak / d
    pts_min = lam_min / d
    peak_ok = pts_peak >= PTS_PER_LAMBDA_NYQUIST

    if not peak_ok or under > VAR_UNDER_DEGRADED:
        verdict = INADEQUATE
    elif under > VAR_UNDER_OK:
        verdict = DEGRADED
    else:
        verdict = ADEQUATE

    return dict(mesh_dx=float(dx), mesh_dy=float(dy),
                lambda_peak=lam_peak, lambda_min=lam_min,
                pts_per_lambda_peak=float(pts_peak),
                pts_per_lambda=float(pts_min),      # at the shortest sig. comp
                var_under_resolved=under,
                mesh_verdict=verdict,
                mesh_adequate=bool(verdict == ADEQUATE),
                mesh_above_nyquist=bool(peak_ok))


def enforce(info: dict, strict: bool = False, verbose: bool = True) -> None:
    """Warn on an under-resolved mesh; raise on a sub-Nyquist one when strict."""
    if info['mesh_adequate']:
        return

    msg = (f"raycast mesh under-resolves the sea: "
           f"dx={info['mesh_dx']:.2f} m, dy={info['mesh_dy']:.2f} m vs "
           f"lambda_min={info['lambda_min']:.1f} m "
           f"({info['pts_per_lambda']:.1f} points/wavelength, "
           f"need {PTS_PER_LAMBDA_OK:.0f}); "
           f"{100 * info['var_under_resolved']:.0f}% of wave variance sits "
           f"below the mesh's resolving power")

    if not info['mesh_above_nyquist']:
        msg = ("BELOW NYQUIST -- LiDAR returns will carry facet error, not "
               "wave signal. " + msg)
        if strict:
            raise ValueError(f"[mesh] {msg}")

    if verbose:
        print(f"[mesh][WARNING] {msg}", flush=True)
