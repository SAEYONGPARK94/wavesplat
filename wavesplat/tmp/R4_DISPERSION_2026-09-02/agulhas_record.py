"""AgulhasRecord — MRecord-compatible reader for the AAS_4434 WASS netCDF-3 files.

Duck-types the 8 attributes tmp/B2_REAL_2026-08-23/scene_station.py touches:
    dt · T_all · gx · gy · dx · dy · t · window()
(+ .valid, used by MRecord internals and by callers that mask.)
"""
import numpy as np
from scipy.io import netcdf_file


class AgulhasRecord:
    def __init__(self, path, stride=1, valid_frac_min=0.90, n_probe=120):
        self.f = netcdf_file(path, 'r', mmap=True)
        V = self.f.variables
        self.zd = V['Z']                                   # (T, nY, nX) int16, mm, fill=0
        self.T_all, self.n0, self.n1 = self.zd.shape
        self.stride = s = int(stride)
        # NOTE: axis roles are TRANSPOSED vs the Bergamasco files.
        # 'X_grid' varies along axis1; 'Y_grid' varies along axis0.  Both are mm.
        self.gx = np.asarray(V['X_grid'][0, ::s], float) / 1000.0     # (n1,)
        self.gy = np.asarray(V['Y_grid'][::s, 0], float) / 1000.0     # (n0,)
        self.t = np.asarray(V['time'][:], float).ravel()              # (T,) — 2-D in file
        self.dt = float(np.median(np.diff(self.t)))
        self.dx = float(np.median(np.diff(self.gx)))
        self.dy = float(np.median(np.diff(self.gy)))
        if s > 1:
            self.n0, self.n1 = len(self.gy), len(self.gx)
        # mask_Z ("no NaN anywhere in the record") is 0.00-0.35 here and ALL-ZERO on
        # the 1150 record -> unusable.  Use an occupancy mask from probe frames.
        idx = np.linspace(0, self.T_all - 1, n_probe).astype(int)
        fr = np.stack([np.asarray(self.zd[int(i)])[::s, ::s] for i in idx])
        self.valid = (fr != 0).mean(axis=0) >= valid_frac_min

    def window(self, i0, n_frames):
        s = self.stride
        Z = np.asarray(self.zd[i0:i0 + n_frames, ::s, ::s]).astype(np.float32)
        Z = np.where(Z == 0, np.nan, Z) / 1000.0           # fill=0, mm -> m
        Z[:, ~self.valid] = np.nan
        Z -= np.nanmean(Z, axis=0, keepdims=True)          # same stereo-bias removal
        return Z
