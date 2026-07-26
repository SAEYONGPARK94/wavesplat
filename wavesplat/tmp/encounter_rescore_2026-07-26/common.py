"""Shared geometry / GT / origin definitions for the encounter re-score (W9-c).

Everything here is fixed by PREREG.md sections 2-4 and must not be tuned.
"""
import os
import numpy as np
import h5py

DATA = 'tmp/discreteness_2026-07-26/data'
CELLS = ['c', 'cbig', 'a', 'abig']           # decision cells first
SEEDS_OF_CELL = {'c': (10, 11, 12, 13, 14), 'a': (10, 11, 12, 13, 14),
                 'cbig': (20, 21, 22, 23, 24), 'abig': (20, 21, 22, 23, 24)}
ORIGINS = {'a': (157, 188, 219, 250), 'c': (157, 188, 219, 250),
           'abig': (157, 200, 243, 287), 'cbig': (157, 200, 243, 287)}
DISTS = (0.0, 10.0, 22.0, 50.0, 100.0, 200.0)
LEADS = (0.0, 1.0, 2.0, 5.0)
NB_COND = 154            # K_TP(2.2) * Tp(7.0) / 0.1
NF_WS = 141              # shipped init-window frame count (tp_factor 2.0 @ dt 0.1)
HS = 1.2


def episodes(cell):
    return [f'episode_BF4_15comp_seed{s}_{r}.h5'
            for s in SEEDS_OF_CELL[cell] for r in ('uniform', 'slope')]


def all_jobs(cells=CELLS):
    for c in cells:
        for ep in episodes(c):
            for i_ref in ORIGINS[c]:
                yield c, ep, i_ref


def path_of(cell, ep):
    return f'{DATA}/cell_{cell}/{ep}'


class EpisodeGeom:
    """Analytic GT + footprint geometry for one episode. No estimator involved."""

    def __init__(self, cell, ep):
        self.path = path_of(cell, ep)
        with h5py.File(self.path, 'r') as f:
            wc = f['config/wave_components'][:]
            self.A, self.om, self.kx, self.ky, self.phi = [wc[:, i] for i in range(5)]
            self.ts = f['frames/timestamps'][:]
            self.pts = f['frames/lidar_points'][:]
            self.msk = f['frames/lidar_valid_mask'][:]
            self.gx = f['frames/gt_grid_x'][:]
            self.gy = f['frames/gt_grid_y'][:]
        self.dt = float(self.ts[1] - self.ts[0])

    # ---- ground truth (verified == stored gt_wave_field to 2.2e-15) ----
    def eta(self, x, y, t):
        x = np.atleast_1d(np.asarray(x, float))
        y = np.atleast_1d(np.asarray(y, float))
        ph = (self.kx[None, :] * x[:, None] + self.ky[None, :] * y[:, None]
              - self.om[None, :] * float(t) + self.phi[None, :])
        return (self.A[None, :] * np.cos(ph)).sum(1)

    def eta_grid(self, gx, gy, t):
        X, Y = np.meshgrid(gx, gy)
        return self.eta(X.ravel(), Y.ravel(), t).reshape(len(gy), len(gx))

    # ---- footprint geometry, from the COMMON 155-frame union window ----
    def edge(self, i_ref):
        lo = i_ref - NB_COND
        q = self.pts[lo:i_ref + 1][self.msk[lo:i_ref + 1]]
        x_edge = float(np.percentile(q[:, 0], 95))
        y_c = float(np.median(q[:, 1]))
        frac_beyond = {d: float((q[:, 0] > x_edge + d).mean()) for d in DISTS}
        return x_edge, y_c, frac_beyond, len(q)

    def targets(self, i_ref):
        x_edge, y_c, frac, n = self.edge(i_ref)
        xs = np.array([x_edge + d for d in DISTS])
        t_ref = float(self.ts[i_ref])
        gt = np.array([[float(self.eta(xs[j], y_c, t_ref + L)[0])
                        for j in range(len(DISTS))] for L in LEADS])   # (nlead, ndist)
        return dict(x_edge=x_edge, y_c=y_c, xs=xs, t_ref=t_ref, gt=gt,
                    frac_beyond=frac, n_win_obs=n)
