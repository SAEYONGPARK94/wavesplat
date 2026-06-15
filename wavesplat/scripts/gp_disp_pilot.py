"""
gp_disp_pilot.py
Direction A Pilot: GP with Dispersion-Informed Kernel (K_disp)

Two variants:
  1. Oracle K_disp: uses exact wave components (A_n, kx_n, ky_n) from episode
  2. JONSWAP K_disp: uses only Hs, Tp (practical, no GT components needed)

K_disp formula (spatial covariance of linear wave field at fixed t):
  K(xi, xj) = sum_n (A_n^2 / 2) * cos(kx_n*(xi_x - xj_x) + ky_n*(xi_y - xj_y))
           = spectral mixture kernel with waves on dispersion manifold

GP posterior mean (closed-form):
  alpha = solve(K_obs_obs + sigma^2*I, z_obs)
  mu_post(x*) = K_pred_obs @ alpha

This is O(N_obs^3) for solve + O(N_grid * N_obs) for predict.
With N_obs ~ 50 (1% of 80x60=4800): extremely fast.

Theoretical expectation:
  - VR ~ 1.0 (correct prior variance from K_disp)
  - Pearson > 0.5 on complex sea (GP is convex = no local optima)
  - nRMSE competitive with WavePINO (0.222)
"""

import sys
import os
import json
import numpy as np
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from wavesplat.data_loader import EpisodeDataLoader

G = 9.81
DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')

TEST_EPISODES = [
    'episode_BF3_5comp_seed0_slope.h5',
    'episode_BF4_5comp_seed0_uniform.h5',
    'episode_BF4_15comp_seed0_uniform.h5',
    'episode_BF4_30comp_seed0_uniform.h5',
    'episode_BF5_15comp_seed0_uniform.h5',
    'episode_BF5_30comp_seed0_uniform.h5',
]

# Full evaluation: seed 0-4, uniform for all categories
TEST_EPISODES_FULL = []
for _seed in range(5):
    TEST_EPISODES_FULL.append(f'episode_BF3_5comp_seed{_seed}_uniform.h5')
    TEST_EPISODES_FULL.append(f'episode_BF4_5comp_seed{_seed}_uniform.h5')
    TEST_EPISODES_FULL.append(f'episode_BF4_15comp_seed{_seed}_uniform.h5')
    TEST_EPISODES_FULL.append(f'episode_BF4_30comp_seed{_seed}_uniform.h5')
    TEST_EPISODES_FULL.append(f'episode_BF5_15comp_seed{_seed}_uniform.h5')
    TEST_EPISODES_FULL.append(f'episode_BF5_30comp_seed{_seed}_uniform.h5')

N_FRAMES = 30   # frames per episode (same as pino_vs_wavesplat eval)
SIGMA_OBS = 0.02  # observation noise [m]


# ── Kernel builders ──────────────────────────────────────────────────────────

def build_K_disp_oracle(xi, xj, wave_components):
    """
    Build K_disp from exact wave components.

    K(xi, xj) = sum_n (A_n^2 / 2) * cos(kx_n*(xi_x - xj_x) + ky_n*(xi_y - xj_y))

    Args:
        xi: (N, 2) positions
        xj: (M, 2) positions
        wave_components: (n_comp, 6) array [A, omega, kx, ky, phi, Q]
    Returns:
        K: (N, M) covariance matrix
    """
    dx = xi[:, 0:1] - xj[None, :, 0]  # (N, M)
    dy = xi[:, 1:2] - xj[None, :, 1]  # (N, M)

    A = wave_components[:, 0]   # (n_comp,)
    kx = wave_components[:, 2]  # (n_comp,)
    ky = wave_components[:, 3]  # (n_comp,)

    K = np.zeros((len(xi), len(xj)))
    for n in range(len(A)):
        phase = kx[n] * dx + ky[n] * dy  # (N, M)
        K += 0.5 * A[n]**2 * np.cos(phase)

    return K


def build_K_disp_jonswap_scale(Hs, Tp, depth=50.0, n_k=64):
    """
    Compute JONSWAP K_disp S(k) array and normalization scale.
    Returns (k_arr, S_k_normalized) so K_oo and K_po use the same scale.
    """
    from scipy.special import j0

    omega_p = 2 * np.pi / Tp
    k_p = omega_p**2 / G

    k_arr = np.linspace(0.3 * k_p, 5.0 * k_p, n_k)
    gamma_jonswap = 3.3

    def jonswap_k(k):
        omega = np.sqrt(G * k * np.tanh(k * depth))
        sigma = np.where(omega <= omega_p, 0.07, 0.09)
        r_exp = np.exp(-0.5 * ((omega - omega_p) / (sigma * omega_p))**2)
        # JONSWAP S(omega) — use Hs²-normalized form
        S_omega = (5.0/16.0) * Hs**2 * omega_p**4 / omega**5 * \
                  np.exp(-1.25 * (omega_p / omega)**4) * gamma_jonswap**r_exp
        th = np.tanh(k * depth)
        d_omega_dk = 0.5 * G * (th + k * depth * (1 - th**2)) / np.sqrt(G * k * th)
        return S_omega * d_omega_dk

    S_k = jonswap_k(k_arr)
    dk = k_arr[1] - k_arr[0]

    # Raw variance = K(r=0) = integral S(k) * J0(0) * k dk = integral S(k) * k dk
    raw_var = float(np.sum(S_k * k_arr * dk))
    target_var = (Hs / 4.0)**2  # Hs = 4*std -> std^2 = Hs^2/16
    scale = target_var / max(raw_var, 1e-12)

    return k_arr, S_k * scale, dk


def build_K_disp_jonswap(xi, xj, Hs, Tp, depth=50.0, n_k=64,
                          _cache={}):
    """
    Build K_disp from JONSWAP spectrum (practical, no wave components needed).

    K(r) = integral_0^inf S_1D(k) J_0(k*r) k dk   [isotropic spreading]

    Normalization: K(r=0) = Hs^2/16 (correct wave variance).
    Same normalization applied consistently to K_oo and K_po.

    Args:
        xi: (N, 2) positions
        xj: (M, 2) positions
        Hs, Tp: wave parameters
        depth: water depth [m]
    Returns:
        K: (N, M) covariance matrix
    """
    from scipy.special import j0

    cache_key = (round(Hs, 4), round(Tp, 4), round(depth, 1), n_k)
    if cache_key not in _cache:
        _cache[cache_key] = build_K_disp_jonswap_scale(Hs, Tp, depth, n_k)
    k_arr, S_k_norm, dk = _cache[cache_key]

    dx = xi[:, 0:1] - xj[None, :, 0]  # (N, M)
    dy = xi[:, 1:2] - xj[None, :, 1]  # (N, M)
    r = np.sqrt(dx**2 + dy**2)         # (N, M)

    K = np.zeros((len(xi), len(xj)))
    for i, k in enumerate(k_arr):
        K += S_k_norm[i] * j0(k * r) * k * dk

    return K


# ── GP posterior mean ─────────────────────────────────────────────────────────

def gp_predict(K_obs_obs, K_pred_obs, z_obs, sigma_obs):
    """
    GP posterior mean prediction.

    Args:
        K_obs_obs: (N_obs, N_obs) observation covariance
        K_pred_obs: (N_pred, N_obs) cross-covariance
        z_obs: (N_obs,) observations
        sigma_obs: observation noise std
    Returns:
        mu_post: (N_pred,) posterior mean
    """
    N = len(z_obs)
    A = K_obs_obs + sigma_obs**2 * np.eye(N)

    # Solve A @ alpha = z_obs
    try:
        alpha = np.linalg.solve(A, z_obs)
    except np.linalg.LinAlgError:
        alpha = np.linalg.lstsq(A, z_obs, rcond=None)[0]

    mu_post = K_pred_obs @ alpha
    return mu_post


# ── Metrics (inline to avoid import issues) ──────────────────────────────────

def compute_metrics_simple(pred, gt, Hs):
    nRMSE = float(np.sqrt(np.mean((pred - gt)**2)) / max(Hs, 1e-6))
    std_pred = float(np.std(pred))
    std_gt = float(np.std(gt))
    VR = (std_pred / max(std_gt, 1e-6))**2
    rho = float(np.corrcoef(pred.ravel(), gt.ravel())[0, 1]) if std_pred > 1e-8 and std_gt > 1e-8 else 0.0
    ss_res = np.sum((pred - gt)**2)
    ss_tot = np.sum((gt - np.mean(gt))**2)
    skill = float(1.0 - ss_res / max(ss_tot, 1e-12))
    Hs_pred = 4.0 * std_pred
    Hs_err = abs(Hs_pred - Hs) / max(Hs, 1e-6)
    return dict(nRMSE=nRMSE, VR=VR, Pearson=rho, Skill=skill, Hs_err=Hs_err)


# ── Episode evaluation ────────────────────────────────────────────────────────

def estimate_components_beamform(loader, n_gaussians=15):
    """
    Estimate wave components from init window via v2 beamforming.
    Returns pseudo wave_components array: (n_gaussians, 6) = [A, 0, kx, ky, phi, 0]
    """
    from wavesplat.optimizer.initializer import initialize_from_frames_v2
    init_frames = loader.get_init_window(tp_factor=2.0)
    if len(init_frames) < 3:
        init_frames = [loader[i] for i in range(min(10, loader.n_frames))]

    init_kx, init_ky, init_w, init_phi = initialize_from_frames_v2(
        init_frames, n_gaussians=n_gaussians, depth=loader.depth
    )
    A = init_w.numpy()
    kx = init_kx.numpy()
    ky = init_ky.numpy()
    phi = init_phi.numpy()
    comps = np.zeros((n_gaussians, 6))
    comps[:, 0] = A
    comps[:, 2] = kx
    comps[:, 3] = ky
    comps[:, 4] = phi
    return comps


def _neg_log_ml(K_unit_oo, z_obs, log_s, sigma_obs):
    """Negative log marginal likelihood for scalar scale s."""
    s = np.exp(log_s)
    K = s * K_unit_oo + sigma_obs**2 * np.eye(len(z_obs))
    try:
        L = np.linalg.cholesky(K + 1e-8 * np.eye(len(K)))
        alpha = np.linalg.solve(L.T, np.linalg.solve(L, z_obs))
        log_det = 2.0 * np.sum(np.log(np.diag(L)))
        return 0.5 * float(z_obs @ alpha) + 0.5 * log_det
    except np.linalg.LinAlgError:
        return 1e10


def _neg_log_ml_per_comp(K_bases, z_obs, log_s_vec, sigma_obs):
    """Negative log marginal likelihood for per-component scales s_n."""
    s = np.exp(log_s_vec)
    K = sum(s[n] * K_bases[n] for n in range(len(s)))
    K += sigma_obs**2 * np.eye(len(z_obs))
    try:
        L = np.linalg.cholesky(K + 1e-8 * np.eye(len(K)))
        alpha = np.linalg.solve(L.T, np.linalg.solve(L, z_obs))
        log_det = 2.0 * np.sum(np.log(np.diag(L)))
        return 0.5 * float(z_obs @ alpha) + 0.5 * log_det
    except np.linalg.LinAlgError:
        return 1e10


def estimate_components_ml(loader, mode='ml_scale', n_gaussians=15):
    """
    Estimate K_disp components via Marginal Likelihood optimization.

    mode='ml_scale': fix kx/ky/direction from beamforming,
                     optimize a single global scale s (K = s * K_unit).
    mode='ml_amp':   fix kx/ky from beamforming,
                     optimize per-component amplitudes A_n independently.

    Uses concatenated init window observations for estimation.
    """
    from scipy.optimize import minimize_scalar, minimize

    # Step 1: Get kx, ky from beamforming
    beam_comps = estimate_components_beamform(loader, n_gaussians=n_gaussians)
    kx = beam_comps[:, 2]
    ky = beam_comps[:, 3]

    # Step 2: Collect init window observations
    init_frames = loader.get_init_window(tp_factor=2.0)
    if len(init_frames) < 3:
        init_frames = [loader[i] for i in range(min(10, loader.n_frames))]

    all_x, all_y, all_z = [], [], []
    for f in init_frames[:10]:  # cap at 10 frames
        all_x.append(f.x_obs.cpu().numpy())
        all_y.append(f.y_obs.cpu().numpy())
        all_z.append(f.z_obs.cpu().numpy())
    X_obs = np.column_stack([np.concatenate(all_x), np.concatenate(all_y)])
    z_obs = np.concatenate(all_z)

    # Subsample if too many points (> 300 is slow for cholesky)
    if len(z_obs) > 300:
        idx = np.random.default_rng(42).choice(len(z_obs), 300, replace=False)
        X_obs = X_obs[idx]
        z_obs = z_obs[idx]

    dx = X_obs[:, 0:1] - X_obs[None, :, 0]  # (N, N)
    dy = X_obs[:, 1:2] - X_obs[None, :, 1]

    if mode == 'ml_scale':
        # Build unit-amplitude kernel (A_n=1 for all n)
        K_unit = np.zeros((len(X_obs), len(X_obs)))
        for n in range(n_gaussians):
            K_unit += 0.5 * np.cos(kx[n] * dx + ky[n] * dy)

        result = minimize_scalar(
            lambda log_s: _neg_log_ml(K_unit, z_obs, log_s, SIGMA_OBS),
            bounds=(-6, 6), method='bounded'
        )
        s_opt = np.exp(result.x)  # K = s_opt * K_unit
        # A_n^2/2 = s_opt / n_gaussians for uniform split
        A_opt = np.full(n_gaussians, np.sqrt(2.0 * s_opt / n_gaussians))

    elif mode == 'ml_amp':
        # Build per-component basis kernels: K_base_n = cos(k_n . Delta)
        K_bases = []
        for n in range(n_gaussians):
            K_bases.append(0.5 * np.cos(kx[n] * dx + ky[n] * dy))

        # Initialize from beamform amplitudes
        A_init = np.maximum(beam_comps[:, 0], 0.01)
        log_s_init = np.log(np.maximum(A_init**2 / 2.0, 1e-6))

        result = minimize(
            lambda log_s: _neg_log_ml_per_comp(K_bases, z_obs, log_s, SIGMA_OBS),
            log_s_init,
            method='L-BFGS-B',
            options={'maxiter': 200, 'ftol': 1e-9}
        )
        s_opt = np.exp(result.x)
        A_opt = np.sqrt(2.0 * np.maximum(s_opt, 1e-8))

    comps = np.zeros((n_gaussians, 6))
    comps[:, 0] = A_opt
    comps[:, 2] = kx
    comps[:, 3] = ky
    return comps


def estimate_components_ml_full(loader, n_gaussians=10):
    """
    Full ML optimization: optimize (k_n, theta_n, A_n) jointly.
    Phase is marginalized out (unlike WaveSplat which estimates phase).

    Parameters: [log_k_1..N, theta_1..N, log_A_1..N]  (3*N total)

    Key advantage over WaveSplat: no phase estimation needed.
    The ML objective smoothly penalizes both wrong directions AND wrong amplitudes.
    """
    from scipy.optimize import minimize

    # Collect init window observations
    init_frames = loader.get_init_window(tp_factor=2.0)
    if len(init_frames) < 3:
        init_frames = [loader[i] for i in range(min(10, loader.n_frames))]

    all_x, all_y, all_z = [], [], []
    for f in init_frames[:10]:
        all_x.append(f.x_obs.cpu().numpy())
        all_y.append(f.y_obs.cpu().numpy())
        all_z.append(f.z_obs.cpu().numpy())
    X_obs = np.column_stack([np.concatenate(all_x), np.concatenate(all_y)])
    z_obs = np.concatenate(all_z)

    if len(z_obs) > 300:
        idx = np.random.default_rng(42).choice(len(z_obs), 300, replace=False)
        X_obs = X_obs[idx]
        z_obs = z_obs[idx]

    dx = X_obs[:, 0:1] - X_obs[None, :, 0]
    dy = X_obs[:, 1:2] - X_obs[None, :, 1]
    N = len(z_obs)
    I = np.eye(N)

    # Initialize from beamforming
    beam_comps = estimate_components_beamform(loader, n_gaussians=n_gaussians)
    kx0 = beam_comps[:, 2]
    ky0 = beam_comps[:, 3]
    A0  = np.maximum(beam_comps[:, 0], 0.01)
    k0     = np.sqrt(kx0**2 + ky0**2 + 1e-8)
    theta0 = np.arctan2(ky0, kx0)

    p0 = np.concatenate([np.log(np.maximum(k0, 1e-4)),
                         theta0,
                         np.log(np.maximum(A0, 0.01))])

    def obj(p):
        log_k  = p[:n_gaussians]
        theta  = p[n_gaussians:2*n_gaussians]
        log_A  = p[2*n_gaussians:]
        k_n    = np.exp(log_k)
        A_n    = np.exp(log_A)
        kx_n   = k_n * np.cos(theta)
        ky_n   = k_n * np.sin(theta)

        K = np.zeros((N, N))
        for n in range(n_gaussians):
            K += 0.5 * A_n[n]**2 * np.cos(kx_n[n] * dx + ky_n[n] * dy)
        K += SIGMA_OBS**2 * I
        try:
            K += 1e-8 * I
            L = np.linalg.cholesky(K)
            alpha = np.linalg.solve(L.T, np.linalg.solve(L, z_obs))
            log_det = 2.0 * np.sum(np.log(np.diag(L)))
            return 0.5 * float(z_obs @ alpha) + 0.5 * log_det
        except np.linalg.LinAlgError:
            return 1e10

    result = minimize(obj, p0, method='L-BFGS-B',
                      options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-6})

    log_k_opt  = result.x[:n_gaussians]
    theta_opt  = result.x[n_gaussians:2*n_gaussians]
    log_A_opt  = result.x[2*n_gaussians:]
    k_opt  = np.exp(log_k_opt)
    A_opt  = np.exp(log_A_opt)
    kx_opt = k_opt * np.cos(theta_opt)
    ky_opt = k_opt * np.sin(theta_opt)

    comps = np.zeros((n_gaussians, 6))
    comps[:, 0] = A_opt
    comps[:, 2] = kx_opt
    comps[:, 3] = ky_opt
    return comps


def _collect_init_obs(init_frames, max_pts=300, rng_seed=42):
    """Collect and subsample init window observations."""
    all_x, all_y, all_z = [], [], []
    for f in init_frames[:10]:
        all_x.append(f.x_obs.cpu().numpy())
        all_y.append(f.y_obs.cpu().numpy())
        all_z.append(f.z_obs.cpu().numpy())
    X_obs = np.column_stack([np.concatenate(all_x), np.concatenate(all_y)])
    z_obs = np.concatenate(all_z)
    if len(z_obs) > max_pts:
        idx = np.random.default_rng(rng_seed).choice(len(z_obs), max_pts, replace=False)
        X_obs = X_obs[idx]; z_obs = z_obs[idx]
    return X_obs, z_obs


def _apply_ml_scale(kx, ky, X_obs, z_obs, sigma_obs=None):
    """Apply single ML global scale to WaveSplat kx/ky directions."""
    from scipy.optimize import minimize_scalar
    _s = sigma_obs if sigma_obs is not None else SIGMA_OBS
    n_comp = len(kx)
    dx = X_obs[:, 0:1] - X_obs[None, :, 0]
    dy = X_obs[:, 1:2] - X_obs[None, :, 1]
    K_unit = np.zeros((len(X_obs), len(X_obs)))
    for ni in range(n_comp):
        K_unit += 0.5 * np.cos(kx[ni] * dx + ky[ni] * dy)
    result = minimize_scalar(
        lambda log_s: _neg_log_ml(K_unit, z_obs, log_s, _s),
        bounds=(-6, 6), method='bounded'
    )
    s_opt = np.exp(result.x)
    nml = result.fun
    A_opt = np.full(n_comp, np.sqrt(2.0 * s_opt / max(n_comp, 1)))
    return A_opt, nml


def _apply_reg_ml_amp(kx, ky, X_obs, z_obs, lambda_reg=2.0, sigma_obs=None):
    """
    Regularized per-component amplitude via MAP with L2 prior.

    Objective: NML(K | obs) + lambda * sum_n (log_s_n - log_s_prior)^2
    Prior center: global ML scale (= 1D bounded optimization result / n_comp)
    This shrinks per-component scales toward the global ML estimate,
    preventing overfitting while allowing spectral shape variation.

    Returns: (A_opt, nml)
    """
    from scipy.optimize import minimize, minimize_scalar
    _s = sigma_obs if sigma_obs is not None else SIGMA_OBS
    n_comp = len(kx)
    dx = X_obs[:, 0:1] - X_obs[None, :, 0]
    dy = X_obs[:, 1:2] - X_obs[None, :, 1]

    # Step 1: global scale as prior center
    K_unit = np.zeros((len(X_obs), len(X_obs)))
    for ni in range(n_comp):
        K_unit += 0.5 * np.cos(kx[ni] * dx + ky[ni] * dy)
    res_scalar = minimize_scalar(
        lambda log_s: _neg_log_ml(K_unit, z_obs, log_s, _s),
        bounds=(-6, 6), method='bounded'
    )
    s_global = np.exp(res_scalar.x)
    log_s_prior = np.full(n_comp, np.log(max(s_global / n_comp, 1e-8)))

    # Step 2: per-component basis kernels
    K_bases = [0.5 * np.cos(kx[ni] * dx + ky[ni] * dy) for ni in range(n_comp)]

    def obj(log_s_vec):
        nml = _neg_log_ml_per_comp(K_bases, z_obs, log_s_vec, _s)
        reg = lambda_reg * float(np.sum((log_s_vec - log_s_prior)**2))
        return nml + reg

    result = minimize(
        obj,
        log_s_prior.copy(),
        method='L-BFGS-B',
        options={'maxiter': 400, 'ftol': 1e-10}
    )
    s_opt = np.exp(result.x)
    A_opt = np.sqrt(2.0 * np.maximum(s_opt, 1e-8))
    nml = float(_neg_log_ml_per_comp(K_bases, z_obs, result.x, _s))  # NML only (no reg)
    return A_opt, nml


def estimate_components_wavesplat_opt(loader, apply_scale=False, apply_amp=False,
                                      apply_reg_amp=False, n_starts=1, lambda_reg=2.0,
                                      init_tp_factor=2.0, sigma_obs=None):
    """
    Estimate K_disp components using WaveSplat's full Adam optimization.

    Runs Phase 1 (coarse-to-fine init) on the init window and extracts
    converged (kx_n, ky_n, A_n).  Optionally applies marginal-likelihood
    amplitude correction.

    apply_scale=False, apply_amp=False  → ws_opt   (WaveSplat kx/ky/A directly)
    apply_scale=True                    → ws_opt_scale (WaveSplat kx/ky + single ML scale)
    apply_amp=True                      → ws_opt_amp (WaveSplat kx/ky + per-component ML A_n)
    apply_reg_amp=True                  → ws_reg_amp (WaveSplat kx/ky + regularized per-comp amp)
    n_starts > 1                        → multi-start: run N seeds, pick best ML
    """
    from wavesplat.optimizer.sequential_update import SequentialUpdater
    from scipy.optimize import minimize_scalar, minimize

    _sigma = sigma_obs if sigma_obs is not None else SIGMA_OBS
    init_frames = loader.get_init_window(tp_factor=init_tp_factor)
    if len(init_frames) < 3:
        init_frames = [loader[i] for i in range(min(10, loader.n_frames))]

    gt_x = loader.gt_grid_x.cpu().numpy()
    gt_y = loader.gt_grid_y.cpu().numpy()
    x_range = (float(gt_x[0]), float(gt_x[-1]))
    y_range = (float(gt_y[0]), float(gt_y[-1]))

    # For multi-start or apply_scale: collect obs once
    X_obs, z_obs = _collect_init_obs(init_frames)

    seeds = list(range(n_starts)) if n_starts > 1 else [42]
    best_kx, best_ky, best_A = None, None, None
    best_nml = np.inf

    for seed in seeds:
        cfg = {
            'n_gaussians': [3, 8, 15],
            'depth': loader.depth,
            'lr': 0.01,
            'lambda_l1': 0.001,
            'lambda_smooth': 0.0001,
            'tracking_steps': 10,
            'tracking_mode': 'w_phi',
            'tracking_lr': 0.005,
            'init_method': 'v2_konly',
            'use_dispersion': True,
            'use_confidence': True,
            'adaptive_n': False,
            'device': 'cpu',
            'seed': seed,
        }
        updater = SequentialUpdater(cfg)
        updater.initialize(init_frames, x_range=x_range, y_range=y_range)

        kx_s = updater.model.kx.detach().cpu().numpy()
        ky_s = updater.model.ky.detach().cpu().numpy()
        A_s  = np.abs(updater.model.w.detach().cpu().numpy())

        if apply_reg_amp:
            A_s, nml = _apply_reg_ml_amp(kx_s, ky_s, X_obs, z_obs,
                                          lambda_reg=lambda_reg, sigma_obs=_sigma)
        elif apply_scale or n_starts > 1:
            A_s, nml = _apply_ml_scale(kx_s, ky_s, X_obs, z_obs, sigma_obs=_sigma)
        else:
            # Evaluate NML with raw WaveSplat A for multi-start selection
            nml = 0.0  # not compared

        if nml < best_nml:
            best_nml = nml
            best_kx, best_ky, best_A = kx_s, ky_s, A_s

    kx, ky, A = best_kx, best_ky, best_A
    n_comp = len(kx)

    if apply_scale and n_starts == 1:
        # Already applied in the loop above
        pass

    if apply_amp:
        # Per-component amplitude optimization via ML (WaveSplat kx/ky fixed)
        dx = X_obs[:, 0:1] - X_obs[None, :, 0]
        dy = X_obs[:, 1:2] - X_obs[None, :, 1]
        K_bases = [0.5 * np.cos(kx[ni] * dx + ky[ni] * dy) for ni in range(n_comp)]

        # Init: use WaveSplat's A values (log space)
        log_s_init = np.log(np.maximum(A**2 / 2.0, 1e-6))
        result = minimize(
            lambda log_s: _neg_log_ml_per_comp(K_bases, z_obs, log_s, SIGMA_OBS),
            log_s_init,
            method='L-BFGS-B',
            options={'maxiter': 300, 'ftol': 1e-10}
        )
        s_opt = np.exp(result.x)
        A = np.sqrt(2.0 * np.maximum(s_opt, 1e-8))

    comps = np.zeros((n_comp, 6))
    comps[:, 0] = A
    comps[:, 2] = kx
    comps[:, 3] = ky
    return comps


def eval_episode_wavesplat(h5_path, n_frames=N_FRAMES):
    """Run WaveSplat on first n_frames tracking frames — same protocol as GP eval."""
    from wavesplat.run import run_estimation
    import torch

    loader = EpisodeDataLoader(h5_path)
    Hs = loader.Hs_true
    gt_x = loader.gt_grid_x.cpu().numpy()
    gt_y = loader.gt_grid_y.cpu().numpy()
    Xg, Yg = np.meshgrid(gt_x, gt_y)

    cfg = {'seed': 42, 'adaptive_n': False}
    result = run_estimation(h5_path, config=cfg, verbose=False)

    frame_results = result.get('frame_results', [])  # tracking frames
    if not frame_results:
        return None

    frame_metrics = []
    for fr in frame_results[:n_frames]:
        m = dict(
            nRMSE=fr.get('nRMSE', float('nan')),
            VR=fr.get('Variance_Ratio', float('nan')),
            Pearson=fr.get('Pearson_rho', float('nan')),
            Skill=fr.get('Skill_Score', float('nan')),
            Hs_err=fr.get('Hs_error', float('nan')),
            elapsed_ms=fr.get('elapsed_ms', 0.0),
        )
        frame_metrics.append(m)

    if not frame_metrics:
        return None
    avg = {k: float(np.nanmean([f[k] for f in frame_metrics])) for k in frame_metrics[0]}
    return avg


def eval_episode(h5_path, mode='oracle', n_frames=N_FRAMES, lambda_reg=2.0,
                 sigma_obs=None, init_tp_factor=2.0):
    loader = EpisodeDataLoader(h5_path)
    gt_x = loader.gt_grid_x.cpu().numpy()
    gt_y = loader.gt_grid_y.cpu().numpy()
    Hs = loader.Hs_true
    depth = loader.depth

    # Build prediction grid coordinates
    Xg, Yg = np.meshgrid(gt_x, gt_y)
    X_pred = np.column_stack([Xg.ravel(), Yg.ravel()])  # (N_grid, 2)

    wave_components = loader.wave_components  # oracle: (n_comp, 6)

    # Estimate components once per episode
    beam_components = None
    if mode == 'beamform':
        beam_components = estimate_components_beamform(loader, n_gaussians=15)
    elif mode in ('ml_scale', 'ml_amp'):
        beam_components = estimate_components_ml(loader, mode=mode, n_gaussians=15)
    elif mode == 'ml_full':
        beam_components = estimate_components_ml_full(loader, n_gaussians=10)
    elif mode == 'ws_opt':
        beam_components = estimate_components_wavesplat_opt(loader, apply_scale=False)
    elif mode == 'ws_opt_scale':
        beam_components = estimate_components_wavesplat_opt(loader, apply_scale=True)
    elif mode == 'ws_opt_amp':
        beam_components = estimate_components_wavesplat_opt(loader, apply_amp=True)
    elif mode == 'ws_multistart':
        beam_components = estimate_components_wavesplat_opt(loader, apply_scale=True,
                                                            n_starts=5)
    elif mode == 'ws_multistart_10':
        beam_components = estimate_components_wavesplat_opt(loader, apply_scale=True,
                                                            n_starts=10)
    elif mode == 'ws_reg_amp':
        # n_starts=1: test amplitude calibration independently of multi-start direction search
        beam_components = estimate_components_wavesplat_opt(loader, apply_reg_amp=True,
                                                            n_starts=1, lambda_reg=lambda_reg,
                                                            init_tp_factor=init_tp_factor,
                                                            sigma_obs=sigma_obs)
    elif mode == 'ws_ms5_reg_amp':
        # n_starts=5 + per-component regularized amplitudes (combined best of both)
        beam_components = estimate_components_wavesplat_opt(loader, apply_reg_amp=True,
                                                            n_starts=5,
                                                            lambda_reg=lambda_reg,
                                                            init_tp_factor=init_tp_factor,
                                                            sigma_obs=sigma_obs)
    elif mode == 'ws_ms10_reg_amp':
        # n_starts=10: extra seeds for catastrophic failure recovery
        beam_components = estimate_components_wavesplat_opt(loader, apply_reg_amp=True,
                                                            n_starts=10,
                                                            lambda_reg=lambda_reg,
                                                            init_tp_factor=init_tp_factor,
                                                            sigma_obs=sigma_obs)

    frame_metrics = []
    frame_count = 0

    # Use tracking frames only (after init window) for fair comparison with WaveSplat
    eval_frames = loader.get_tracking_frames(tp_factor=2.0)

    for frame in eval_frames:
        if frame_count >= n_frames:
            break

        x_obs = frame.x_obs.cpu().numpy()
        y_obs = frame.y_obs.cpu().numpy()
        z_obs = frame.z_obs.cpu().numpy()
        gt = frame.gt_field.cpu().numpy()  # (Ny, Nx)

        if len(x_obs) < 5:
            frame_count += 1
            continue

        X_obs = np.column_stack([x_obs, y_obs])  # (N_obs, 2)

        t0 = time.time()

        if mode == 'oracle' and wave_components is not None:
            K_oo = build_K_disp_oracle(X_obs, X_obs, wave_components)
            K_po = build_K_disp_oracle(X_pred, X_obs, wave_components)
        elif mode in ('beamform', 'ml_scale', 'ml_amp', 'ml_full',
                      'ws_opt', 'ws_opt_scale', 'ws_opt_amp',
                      'ws_multistart', 'ws_multistart_10', 'ws_reg_amp',
                      'ws_ms5_reg_amp', 'ws_ms10_reg_amp') and beam_components is not None:
            K_oo = build_K_disp_oracle(X_obs, X_obs, beam_components)
            K_po = build_K_disp_oracle(X_pred, X_obs, beam_components)
        elif mode == 'jonswap':
            K_oo = build_K_disp_jonswap(X_obs, X_obs, Hs, loader.Tp_true, depth)
            K_po = build_K_disp_jonswap(X_pred, X_obs, Hs, loader.Tp_true, depth)
        else:
            raise ValueError(f"Unknown mode: {mode}")

        _so = sigma_obs if sigma_obs is not None else SIGMA_OBS
        mu = gp_predict(K_oo, K_po, z_obs, _so)
        elapsed_ms = (time.time() - t0) * 1000

        pred = mu.reshape(len(gt_y), len(gt_x))
        m = compute_metrics_simple(pred, gt, Hs)
        m['elapsed_ms'] = elapsed_ms
        frame_metrics.append(m)
        frame_count += 1

    if not frame_metrics:
        return None

    avg = {k: float(np.mean([f[k] for f in frame_metrics]))
           for k in frame_metrics[0]}
    return avg


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("Direction A Pilot: GP with K_disp")
    print("=" * 70)

    out_path = os.path.join(os.path.dirname(__file__), '..', 'results',
                            'gp_disp_full_eval.json')

    # Load existing results if available (to avoid re-running oracle/ws_multistart)
    if os.path.exists(out_path):
        with open(out_path) as f:
            results = json.load(f)
        print(f"Loaded existing results from {out_path}")
        # Add wavesplat key if missing
        if 'wavesplat' not in results:
            results['wavesplat'] = {}
    else:
        results = {'oracle': {}, 'ws_opt': {}, 'ws_opt_scale': {}, 'ws_opt_amp': {},
                   'ws_multistart': {}, 'ws_multistart_10': {}, 'ws_reg_amp': {},
                   'ws_ms5_reg_amp': {},
                   'wavesplat': {},
                   'ml_full': {}, 'ml_scale': {}, 'ml_amp': {}, 'beamform': {}, 'jonswap': {}}

    # Skip modes that already have results
    # Note: ws_multistart_10 is handled separately by eval_nstarts_comparison.py
    modes_to_run = [m for m in ['oracle', 'ws_multistart', 'ws_reg_amp', 'ws_ms5_reg_amp']
                    if 'average' not in results.get(m, {})]
    print(f"Modes to run: {modes_to_run if modes_to_run else '(none — already done)'}")

    for mode in modes_to_run:
        print(f"\n--- Mode: {mode} ---")
        per_ep = {}

        for ep_name in TEST_EPISODES_FULL:

            h5_path = os.path.join(DATA_DIR, ep_name)
            if not os.path.exists(h5_path):
                print(f"  SKIP (not found): {ep_name}")
                continue

            ep_key = ep_name.replace('episode_', '').replace('.h5', '')
            print(f"  {ep_key}...", end='', flush=True)

            try:
                m = eval_episode(h5_path, mode=mode)
                per_ep[ep_key] = m
                print(f" nRMSE={m['nRMSE']:.4f} VR={m['VR']:.3f} "
                      f"Pearson={m['Pearson']:.3f} Skill={m['Skill']:.3f} "
                      f"({m['elapsed_ms']:.0f}ms/frame)")
                # Intermediate save: protect against crash loss
                if mode not in results:
                    results[mode] = {}
                results[mode]['per_episode'] = dict(per_ep)
                with open(out_path, 'w') as _f:
                    json.dump(results, _f, indent=2)
            except Exception as e:
                print(f" ERROR: {e}")
                import traceback; traceback.print_exc()

        if per_ep:
            avg_keys = list(next(iter(per_ep.values())).keys())
            avg = {k: float(np.mean([v[k] for v in per_ep.values()
                                     if v is not None]))
                   for k in avg_keys}
            if mode not in results:
                results[mode] = {}
            results[mode]['per_episode'] = per_ep
            results[mode]['average'] = avg
            print(f"\n  AVERAGE [{mode}]:")
            print(f"    nRMSE  = {avg['nRMSE']:.4f}")
            print(f"    VR     = {avg['VR']:.4f}")
            print(f"    Pearson= {avg['Pearson']:.4f}")
            print(f"    Skill  = {avg['Skill']:.4f}")

    # WaveSplat baseline on same 30 tracking frames
    print(f"\n--- Mode: wavesplat (baseline) ---")
    ws_per_ep = {}
    for ep_name in TEST_EPISODES_FULL:
        h5_path = os.path.join(DATA_DIR, ep_name)
        if not os.path.exists(h5_path):
            continue
        ep_key = ep_name.replace('episode_', '').replace('.h5', '')
        print(f"  {ep_key}...", end='', flush=True)
        try:
            m = eval_episode_wavesplat(h5_path)
            ws_per_ep[ep_key] = m
            if m:
                print(f" nRMSE={m['nRMSE']:.4f} VR={m['VR']:.3f} "
                      f"Pearson={m['Pearson']:.3f} Skill={m['Skill']:.3f}")
            else:
                print(" EMPTY")
        except Exception as e:
            print(f" ERROR: {e}")
            import traceback; traceback.print_exc()

    if ws_per_ep:
        valid = [v for v in ws_per_ep.values() if v is not None]
        avg_keys = list(valid[0].keys()) if valid else []
        ws_avg = {k: float(np.nanmean([v[k] for v in valid])) for k in avg_keys}
        results['wavesplat'] = {'per_episode': ws_per_ep, 'average': ws_avg}
        print(f"\n  AVERAGE [wavesplat]:")
        print(f"    nRMSE  = {ws_avg['nRMSE']:.4f}")
        print(f"    VR     = {ws_avg['VR']:.4f}")
        print(f"    Pearson= {ws_avg['Pearson']:.4f}")
        print(f"    Skill  = {ws_avg['Skill']:.4f}")

    # Save results
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved: {out_path}")

    # Summary comparison
    print("\n" + "=" * 70)
    print(f"SUMMARY — same 30 tracking frames, {len(TEST_EPISODES_FULL)} episodes (seed 0-4)")
    print("=" * 70)
    print(f"{'Method':<28} {'nRMSE':>8} {'VR':>7} {'Pearson':>9} {'Skill':>8}")
    print("-" * 65)
    if 'average' in results.get('wavesplat', {}):
        a = results['wavesplat']['average']
        print(f"{'WaveSplat (this eval)':<28} {a['nRMSE']:>8.4f} {a['VR']:>7.3f}"
              f" {a['Pearson']:>9.3f} {a['Skill']:>8.3f}")
    for mode in ['oracle', 'ws_multistart', 'ws_multistart_10', 'ws_reg_amp', 'ws_ms5_reg_amp']:
        if 'average' in results.get(mode, {}):
            a = results[mode]['average']
            labels = {
                'oracle': 'oracle',
                'ws_multistart': 'ws_ms5',
                'ws_multistart_10': 'ws_ms10',
                'ws_reg_amp': 'ws_reg_amp',
                'ws_ms5_reg_amp': 'ws_ms5_reg_amp',
            }
            label = labels.get(mode, mode)
            print(f"{'GP K_disp ['+label+']':<30} {a['nRMSE']:>8.4f} {a['VR']:>7.3f}"
                  f" {a['Pearson']:>9.3f} {a['Skill']:>8.3f}")


if __name__ == '__main__':
    main()
