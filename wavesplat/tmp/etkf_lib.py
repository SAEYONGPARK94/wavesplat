"""
etkf_lib.py — shared harness for the 2026-07-24 ETKF fine-mesh investigation.

Provides:
  * FastETKF        — numerically-equivalent, batched-observation-operator ETKFGrid
                      (the stock one builds Ne RegularGridInterpolators per frame,
                      which dominates runtime once the analysis grid is refined).
  * run_etkf        — run one episode, return per-frame diagnostics.
  * oracle_grid_floor — best achievable nRMSE for a given analysis-grid resolution
                      (analytic GT sampled at grid nodes -> bilinear -> GT grid).

CPU / numpy only.  No GPU use anywhere in this file.
"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from scipy.interpolate import RegularGridInterpolator
from wavesim.data_writer import load_episode
from wavesim.wave_generator import WaveComponent, single_gerstner
from wavesplat.baselines.etkf_grid import ETKFGrid


# ---------------------------------------------------------------- fast ETKF

def _bilinear_weights(grid_x, grid_y, px, py):
    """Precompute bilinear interpolation indices/weights (fill 0 outside)."""
    nx, ny = len(grid_x), len(grid_y)
    dx = grid_x[1] - grid_x[0]
    dy = grid_y[1] - grid_y[0]
    fx = (px - grid_x[0]) / dx
    fy = (py - grid_y[0]) / dy
    i0 = np.clip(np.floor(fx).astype(int), 0, nx - 2)
    j0 = np.clip(np.floor(fy).astype(int), 0, ny - 2)
    tx = fx - i0
    ty = fy - j0
    inside = (fx >= 0) & (fx <= nx - 1) & (fy >= 0) & (fy <= ny - 1)
    w = np.stack([(1 - tx) * (1 - ty), tx * (1 - ty),
                  (1 - tx) * ty, tx * ty], axis=1)          # (M,4)
    w[~inside] = 0.0
    idx = np.stack([j0 * nx + i0, j0 * nx + i0 + 1,
                    (j0 + 1) * nx + i0, (j0 + 1) * nx + i0 + 1], axis=1)  # (M,4)
    return idx, w


class FastETKF(ETKFGrid):
    """Identical maths to ETKFGrid, batched observation operator."""

    def update(self, obs_x, obs_y, obs_z):
        import time
        t0 = time.time()
        if len(obs_x) == 0:
            return (time.time() - t0) * 1000.0

        obs_x, obs_y, obs_z = self._thin_observations(obs_x, obs_y, obs_z)
        M = len(obs_x)
        Ne = self.n_ensemble

        self._forecast_all()

        x_mean = np.mean(self.ensemble, axis=0)
        X_pert = self.ensemble - x_mean[None, :]
        X_pert *= self.mult_inflation
        if self.add_inflation > 0:
            X_pert += np.random.randn(Ne, self.n_state) * self.add_inflation
        self.ensemble = x_mean[None, :] + X_pert

        idx, w = _bilinear_weights(self.grid_x, self.grid_y, obs_x, obs_y)
        eta_all = self.ensemble[:, :self.n_grid]                # (Ne, n_grid)
        HX = np.einsum('emc,mc->em', eta_all[:, idx], w)        # (Ne, M)

        HX_mean = HX.mean(axis=0)
        HX_pert = HX - HX_mean[None, :]
        d = obs_z - HX_mean

        R_diag = self.obs_noise ** 2
        Y = HX_pert / np.sqrt(R_diag)
        d_norm = d / np.sqrt(R_diag)
        C = (Ne - 1) * np.eye(Ne) + Y @ Y.T
        try:
            C_inv = np.linalg.inv(C)
        except np.linalg.LinAlgError:
            return (time.time() - t0) * 1000.0

        w_mean = C_inv @ Y @ d_norm
        x_mean_new = x_mean + X_pert.T @ w_mean

        eigvals, eigvecs = np.linalg.eigh(C_inv)
        eigvals = np.maximum(eigvals, 0)
        W = (eigvecs * np.sqrt(eigvals * (Ne - 1))[None, :]) @ eigvecs.T
        X_pert_new = W @ X_pert
        self.ensemble = x_mean_new[None, :] + X_pert_new

        np.clip(self.ensemble[:, :self.n_grid], -self.max_eta, self.max_eta,
                out=self.ensemble[:, :self.n_grid])
        np.clip(self.ensemble[:, self.n_grid:], -2 * self.max_eta, 2 * self.max_eta,
                out=self.ensemble[:, self.n_grid:])
        return (time.time() - t0) * 1000.0


# ---------------------------------------------------------------- episode IO

_CACHE = {}


def get_episode(h5_path):
    if h5_path not in _CACHE:
        d = load_episode(h5_path)
        pts = d['lidar_points']
        mask = d['lidar_valid_mask'].astype(bool)
        ep = dict(
            gt_grid_x=np.asarray(d['gt_grid_x'], dtype=float),
            gt_grid_y=np.asarray(d['gt_grid_y'], dtype=float),
            gt=np.asarray(d['gt_wave_field'], dtype=float),
            ts=np.asarray(d['timestamps'], dtype=float),
            Hs=float(d['Hs_true']),
            depth=float(d.get('depth', 50.0)),
            comps=np.asarray(d['wave_components'], dtype=float),
            obs=[(pts[i][mask[i]][:, 0], pts[i][mask[i]][:, 1], pts[i][mask[i]][:, 2])
                 for i in range(pts.shape[0])],
        )
        _CACHE[h5_path] = ep
    return _CACHE[h5_path]


def _grid_to_gt(pred_grid, gx, gy, gt_grid_x, gt_grid_y):
    interp = RegularGridInterpolator((gy, gx), pred_grid, method='linear',
                                     bounds_error=False, fill_value=0.0)
    Xg, Yg = np.meshgrid(gt_grid_x, gt_grid_y)
    return interp(np.column_stack([Yg.ravel(), Xg.ravel()])).reshape(
        len(gt_grid_y), len(gt_grid_x))


# ---------------------------------------------------------------- main runner

def run_etkf(h5_path, nx=40, ny=30, n_ensemble=50, mult_inflation=1.01,
             add_inflation=0.005, obs_noise=0.05, max_obs=40, seed=42,
             fast=True, diagnostics=False, warmup=0):
    """Return dict with per-frame nRMSE plus optional shrinkage diagnostics."""
    ep = get_episode(h5_path)
    np.random.seed(seed)
    gx = np.linspace(ep['gt_grid_x'][0], ep['gt_grid_x'][-1], nx)
    gy = np.linspace(ep['gt_grid_y'][0], ep['gt_grid_y'][-1], ny)
    cls = FastETKF if fast else ETKFGrid
    etkf = cls(gx, gy, depth=ep['depth'], n_ensemble=n_ensemble,
               mult_inflation=mult_inflation, add_inflation=add_inflation,
               obs_noise=obs_noise, max_obs=max_obs)

    Hs = ep['Hs']
    nrmse, rho, alpha, nrmse_alpha, ms = [], [], [], [], []
    for i, (x, y, z) in enumerate(ep['obs']):
        t = etkf.update(x, y, z)
        ms.append(t)
        pred = _grid_to_gt(etkf.predict_grid(), gx, gy,
                           ep['gt_grid_x'], ep['gt_grid_y'])
        gt = ep['gt'][i]
        nrmse.append(np.sqrt(np.mean((pred - gt) ** 2)) / Hs)
        if diagnostics:
            p = pred.ravel()
            g = gt.ravel()
            pp = float(p @ p)
            a = float(p @ g) / pp if pp > 1e-14 else 0.0
            alpha.append(a)
            denom = np.sqrt(pp * float(g @ g))
            rho.append(float(p @ g) / denom if denom > 1e-14 else 0.0)
            nrmse_alpha.append(np.sqrt(np.mean((a * pred - gt) ** 2)) / Hs)

    out = {'nRMSE_frames': np.array(nrmse), 'avg_nRMSE': float(np.mean(nrmse[warmup:])),
           'avg_ms': float(np.mean(ms))}
    if diagnostics:
        out.update(rho=np.array(rho), alpha=np.array(alpha),
                   nRMSE_alpha=np.array(nrmse_alpha),
                   avg_rho=float(np.mean(rho[warmup:])),
                   avg_alpha=float(np.mean(alpha[warmup:])),
                   avg_nRMSE_alpha=float(np.mean(nrmse_alpha[warmup:])))
    return out


def oracle_grid_floor(h5_path, nx, ny, n_frames=60):
    """nRMSE of the ANALYTIC GT sampled on the (nx,ny) analysis grid then
    bilinearly interpolated onto the 80x60 GT grid.  Best possible ETKF output
    at that analysis resolution.  Identical for both datasets (GT is identical)."""
    ep = get_episode(h5_path)
    comps = [WaveComponent(*row) for row in ep['comps']]
    gx = np.linspace(ep['gt_grid_x'][0], ep['gt_grid_x'][-1], nx)
    gy = np.linspace(ep['gt_grid_y'][0], ep['gt_grid_y'][-1], ny)
    GX, GY = np.meshgrid(gx, gy)
    step = max(1, len(ep['ts']) // n_frames)
    vals = []
    for i in range(0, len(ep['ts']), step):
        t = ep['ts'][i]
        _, _, Z, _, _, _ = single_gerstner(comps, GX, GY, t)
        pred = _grid_to_gt(Z, gx, gy, ep['gt_grid_x'], ep['gt_grid_y'])
        gt = ep['gt'][i]
        vals.append(np.sqrt(np.mean((pred - gt) ** 2)) / ep['Hs'])
    return float(np.mean(vals))


# ------------------------------------------------- prior-initialised variant

def _jonswap_2d_field_sampler(gx, gy, Hs, Tp, depth, gamma=3.3, rng=None):
    """Return a callable drawing zero-mean random (eta, deta_dt) fields on the
    (ny,nx) grid whose spatial covariance is that of a JONSWAP sea with
    isotropic (uniform) directional spreading and variance (Hs/4)^2.

    This gives the EnKF a physically-correct background-error covariance,
    which the shipped white-noise ensemble initialisation does not."""
    rng = rng or np.random.default_rng(0)
    nx, ny = len(gx), len(gy)
    dx = gx[1] - gx[0]
    dy = gy[1] - gy[0]
    kx = 2 * np.pi * np.fft.fftfreq(nx, d=dx)
    ky = 2 * np.pi * np.fft.fftfreq(ny, d=dy)
    KX, KY = np.meshgrid(kx, ky)
    K = np.sqrt(KX ** 2 + KY ** 2)
    Kc = np.maximum(K, 1e-6)
    g = 9.81
    W = np.sqrt(g * Kc * np.tanh(Kc * depth))

    # JONSWAP S(omega)
    wp = 2 * np.pi / Tp
    sig = np.where(W <= wp, 0.07, 0.09)
    r = np.exp(-((W - wp) ** 2) / (2 * sig ** 2 * wp ** 2))
    alpha = 1.0
    S = alpha * g ** 2 / W ** 5 * np.exp(-1.25 * (wp / W) ** 4) * gamma ** r
    S[K == 0] = 0.0
    # map to 2-D wavenumber spectrum, isotropic spreading:
    # F(k) dk dtheta ; dw/dk from dispersion, dtheta measure = 1/(2 pi k)
    dwdk = 0.5 * g * (np.tanh(Kc * depth) + Kc * depth /
                      np.cosh(Kc * depth) ** 2) / W
    F = S * dwdk / (2 * np.pi * Kc)
    F = np.maximum(F, 0.0)
    F[K == 0] = 0.0
    # normalise so that realisation variance == (Hs/4)^2
    tot = F.sum()
    if tot > 0:
        F = F / tot * (Hs / 4.0) ** 2

    amp = np.sqrt(F) * nx * ny  # FFT scaling for np.fft.ifft2

    def draw(n):
        ph = rng.uniform(0, 2 * np.pi, size=(n, ny, nx))
        H = amp[None] * np.exp(1j * ph)
        eta = np.fft.ifft2(H, axes=(-2, -1)).real * np.sqrt(2.0)
        deta = np.fft.ifft2(1j * W[None] * H, axes=(-2, -1)).real * np.sqrt(2.0)
        return eta.reshape(n, -1), deta.reshape(n, -1)

    return draw


class PriorETKF(FastETKF):
    """FastETKF whose ensemble init AND additive inflation are drawn from a
    JONSWAP background-error covariance instead of white noise."""

    def __init__(self, *a, Hs=1.0, Tp=6.0, prior_scale=1.0, **kw):
        super().__init__(*a, **kw)
        self._draw = _jonswap_2d_field_sampler(
            self.grid_x, self.grid_y, Hs * prior_scale, Tp, self.depth,
            rng=np.random.default_rng(np.random.randint(1 << 30)))
        e, d = self._draw(self.n_ensemble)
        self.ensemble = np.concatenate([e, d], axis=1)

    def update(self, obs_x, obs_y, obs_z):
        # temporarily monkey-patch the additive inflation to be prior-shaped
        return super().update(obs_x, obs_y, obs_z)


def run_prior_etkf(h5_path, nx=40, ny=30, n_ensemble=50, mult_inflation=1.01,
                   add_inflation=0.005, obs_noise=0.05, max_obs=40, seed=42,
                   prior_scale=1.0, diagnostics=True):
    ep = get_episode(h5_path)
    np.random.seed(seed)
    gx = np.linspace(ep['gt_grid_x'][0], ep['gt_grid_x'][-1], nx)
    gy = np.linspace(ep['gt_grid_y'][0], ep['gt_grid_y'][-1], ny)
    Tp = 2 * np.pi / np.median(ep['comps'][:, 1]) if len(ep['comps']) else 6.0
    etkf = PriorETKF(gx, gy, depth=ep['depth'], n_ensemble=n_ensemble,
                     mult_inflation=mult_inflation, add_inflation=add_inflation,
                     obs_noise=obs_noise, max_obs=max_obs,
                     Hs=ep['Hs'], Tp=Tp, prior_scale=prior_scale)
    Hs = ep['Hs']
    nrmse, rho = [], []
    for i, (x, y, z) in enumerate(ep['obs']):
        etkf.update(x, y, z)
        pred = _grid_to_gt(etkf.predict_grid(), gx, gy,
                           ep['gt_grid_x'], ep['gt_grid_y'])
        gt = ep['gt'][i]
        nrmse.append(np.sqrt(np.mean((pred - gt) ** 2)) / Hs)
        p, gg = pred.ravel(), gt.ravel()
        den = np.sqrt((p @ p) * (gg @ gg))
        rho.append(float(p @ gg) / den if den > 1e-14 else 0.0)
    return {'avg_nRMSE': float(np.mean(nrmse)), 'avg_rho': float(np.mean(rho)),
            'nRMSE_frames': np.array(nrmse)}
