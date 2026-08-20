#!/usr/bin/env python3
"""Moving-sensor episode generation.  wavesim/** is NOT modified.

Why this is small.  wavesim.pipeline.run_episode pre-generates the LiDAR rays once,
OUTSIDE the frame loop, with ship_x=ship_y=0 hardcoded (pipeline.py:141-142) -- the
comment says "since ship is stationary".  generate_rays already takes ship_x/ship_y/
ship_heading_deg (lidar_simulator.py:275-278); the pose has simply never been non-zero
on any call site in the repository.

For a pure translation at fixed heading, the ray DIRECTIONS are pose-invariant: heading
enters only the direction rotation (lidar_simulator.py:295), and origins are all equal to
the sensor position (:285).  So the entire intervention is: shift `origins` by (U t, 0, 0)
before each frame's raycast.  Everything downstream -- normals, incidence (which reads
ray_dirs, unchanged), raydrop, noise, storage -- then runs unmodified and correctly.

We do that by wrapping the raycast call, which happens exactly once per frame inside the
loop, and writing into `origins` IN PLACE so the caller's later use of the same array
stays consistent.

Two things this cannot do, both of which constrain the batch:
  * data_writer.py:24-25 stores ONE GT grid per episode, so the grid must be WORLD-FIXED
    and wide enough to contain the whole track plus the LiDAR footprint at both ends.
  * There is no per-frame pose field in the HDF5, so the track is reconstructed from
    (U, timestamps) by the analyser, and is asserted here by the counters.
"""
import os
import sys

import numpy as np

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import wavesim.pipeline as P                          # noqa: E402
from wavesim.config import WaveSimConfig              # noqa: E402

_STATS = {}


def reset_stats():
    _STATS.clear()
    _STATS.update(calls=0, ship_x=[], origin_x_first=[], origin_x_last=[])


def stats():
    return dict(_STATS)


class MovingSensor:
    """Context manager that makes run_episode fly the sensor at U m/s along +x."""

    def __init__(self, U, dt, duration):
        self.U = float(U)
        self.timestamps = np.arange(0.0, duration + 1e-9, dt)
        self._saved = {}

    def _wrap(self, real):
        def patched(ray_origins, ray_dirs, *a, **kw):
            i = _STATS['calls']
            # run_episode `continue`s a frame with no hits BEFORE the next raycast, so
            # call index and frame index stay in step; clamp defensively.
            t = self.timestamps[min(i, len(self.timestamps) - 1)]
            dx = self.U * float(t)
            if 'base_x' not in _STATS:
                _STATS['base_x'] = ray_origins[:, 0].copy()
            ray_origins[:, 0] = _STATS['base_x'] + dx      # in place, on purpose
            _STATS['calls'] = i + 1
            _STATS['ship_x'].append(dx)
            _STATS['origin_x_first'].append(float(ray_origins[0, 0]))
            _STATS['origin_x_last'].append(float(ray_origins[-1, 0]))
            return real(ray_origins, ray_dirs, *a, **kw)
        return patched

    def __enter__(self):
        reset_stats()
        for name in ('raycast_surface', 'raycast_surface_gpu'):
            if hasattr(P, name):
                self._saved[name] = getattr(P, name)
                setattr(P, name, self._wrap(self._saved[name]))
        return self

    def __exit__(self, *exc):
        for name, fn in self._saved.items():
            setattr(P, name, fn)
        return False


def world_grid_config(U, duration, base_cfg=None, margin=20.0):
    """Widen the GT/mesh grid so one world-fixed grid covers the whole track.

    The sensor starts at x=0 and ends at x=U*duration; the LiDAR footprint reaches out to
    range_max ahead of wherever it is.  Keeping the same grid SPACING (not the same node
    count) is what preserves the mesh-adequacy verdict, which mesh_adequacy.py:67-68
    judges on spacing.
    """
    cfg = base_cfg or WaveSimConfig()
    x0, x1 = cfg.gt_grid.x_range
    dx_track = U * duration
    new_x = (x0, x1 + dx_track + margin)

    def _rescale(rng_old, rng_new, n_old):
        span_old = rng_old[1] - rng_old[0]
        span_new = rng_new[1] - rng_new[0]
        return int(round((n_old - 1) * span_new / span_old)) + 1

    cfg.gt_grid.nx = _rescale(cfg.gt_grid.x_range, new_x, cfg.gt_grid.nx)
    cfg.gt_grid.x_range = new_x
    if hasattr(cfg, 'mesh'):
        cfg.mesh.nx = _rescale((x0, x1), new_x, cfg.mesh.nx)
    return cfg, new_x


def check(U, duration, dt, tol=1e-6):
    """Post-generation gate: did the sensor actually move, and by exactly U*t?"""
    s = stats()
    if not s.get('calls'):
        return False, 'raycast never called -- the patch did not take'
    obs = np.asarray(s['ship_x'])
    expect = U * np.arange(len(obs)) * dt
    if np.max(np.abs(obs - expect)) > tol:
        return False, 'track deviates from U*t by %.3g m' % np.max(np.abs(obs - expect))
    if U > 0 and obs[-1] <= obs[0]:
        return False, 'sensor did not advance'
    span = float(np.ptp(np.asarray(s['origin_x_first'])))
    if U > 0 and abs(span - U * (len(obs) - 1) * dt) > 1e-3:
        return False, 'ray origins did not follow the track (span %.3f m)' % span
    return True, ('moved %.1f m over %d frames (U=%.2f m/s)'
                  % (obs[-1] - obs[0], len(obs), U))
