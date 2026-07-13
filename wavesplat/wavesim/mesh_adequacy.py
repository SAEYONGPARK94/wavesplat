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

Rule used here:
    >= 6 points per wavelength  -> adequate
    >= 2 points per wavelength  -> representable but degraded (Nyquist floor)
    <  2 points per wavelength  -> the wave does not exist on the mesh
"""
from typing import List, Optional

import numpy as np

PTS_PER_LAMBDA_OK = 6.0       # adequacy target
PTS_PER_LAMBDA_NYQUIST = 2.0  # hard floor
VAR_TAIL = 0.01               # ignore components below 1% of total variance


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

    lam_min = float(lam[keep].min())
    lam_peak = float(lam[int(np.argmax(A))])
    if hos_order and hos_order > 1:
        lam_min /= float(hos_order)      # shortest bound harmonic

    pts = lam_min / d
    under = float(var[lam < PTS_PER_LAMBDA_OK * d].sum() / var.sum()) if var.sum() else 0.0

    return dict(mesh_dx=float(dx), mesh_dy=float(dy),
                lambda_peak=lam_peak, lambda_min=lam_min,
                pts_per_lambda=float(pts),
                var_under_resolved=under,
                mesh_adequate=bool(pts >= PTS_PER_LAMBDA_OK),
                mesh_above_nyquist=bool(pts >= PTS_PER_LAMBDA_NYQUIST))


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
