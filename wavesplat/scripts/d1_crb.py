"""
d1_crb.py — Direction I: Cramér–Rao Bound for space-time wave field estimation.

Observation model (matched to wavesplat.model.spectral_gaussian):
    z(x,y,t) = Σᵢ wᵢ·cos(ψᵢ),   ψᵢ = kxᵢ·x + kyᵢ·y − ωᵢ·t + φᵢ
    ωᵢ = √(g·kᵢ·tanh(kᵢ·d)),     kᵢ = √(kxᵢ²+kyᵢ²)   (dispersion: hard constraint)

Free parameters per component: θᵢ = (wᵢ, kxᵢ, kyᵢ, φᵢ).  ωᵢ is NOT free.
Noise: zₘ = μₘ(θ) + εₘ,  εₘ ~ N(0, σ²) iid  (σ ≈ LiDAR range noise on elevation).

Fisher Information Matrix (Gaussian mean model, known σ²):
    F = (1/σ²) · Jᵀ J,   J[m, a] = ∂μₘ/∂θ_a      (M × 4N Jacobian)

Key derivative structure (the synthetic-aperture term):
    ∂ψ/∂kx = x − c_g·(kx/k)·t        ∂ψ/∂ky = y − c_g·(ky/k)·t
    c_g = dω/dk = group velocity.
  → the effective sampling position at time t is r − c_g·k̂·t, displaced along
    the propagation direction by c_g·t.  Over a window T_obs the point cloud is
    extended ALONG k by c_g·T_obs ⇒ effective aperture  D_eff = L_foot + c_g·T_obs.

Field-reconstruction CRB (any unbiased estimator):
    ẑ(x,y,t*) − z ≈ g(x,y,t*)ᵀ δθ,  g_a = ∂z/∂θ_a
    E[(ẑ−z)²] averaged over a grid  ≥  tr(F⁻¹ · G),   G = ⟨ g gᵀ ⟩_grid
  → field-RMSE CRB = √(tr(F⁻¹ G)),  field-nRMSE CRB = that / Hs.

Everything runs in float64 on GPU.
"""

import os
import sys
import json
import math
import argparse
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from wavesim.data_writer import load_episode
from wavesplat.model.dispersion import G

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DT = torch.float64


# ----------------------------------------------------------------------------
# Physics primitives (float64, autograd-friendly)
# ----------------------------------------------------------------------------
def omega_of_k(k, depth):
    return torch.sqrt(G * k * torch.tanh(k * depth) + 1e-30)


def cg_of_k(k, depth):
    kd = k * depth
    th = torch.tanh(kd)
    sech2 = 1.0 - th ** 2
    w = torch.sqrt(G * k * th + 1e-30)
    return G * (th + kd * sech2) / (2.0 * w + 1e-30)


# ----------------------------------------------------------------------------
# Forward + analytic Jacobian
# ----------------------------------------------------------------------------
def predict(theta, x, y, t, depth):
    """theta: (N,4) = [w, kx, ky, phi]. x,y,t: (M,). Returns mu: (M,)."""
    w, kx, ky, phi = theta[:, 0], theta[:, 1], theta[:, 2], theta[:, 3]
    k = torch.sqrt(kx ** 2 + ky ** 2 + 1e-30)
    omega = omega_of_k(k, depth)
    psi = (kx[None, :] * x[:, None] + ky[None, :] * y[:, None]
           - omega[None, :] * t[:, None] + phi[None, :])     # (M,N)
    return (w[None, :] * torch.cos(psi)).sum(dim=1)


def jacobian(theta, x, y, t, depth):
    """Analytic Jacobian ∂μ/∂θ. theta:(N,4); returns J:(M, 4N) col order
    [w_0..w_{N-1}, kx_0.., ky_0.., phi_0..]."""
    w, kx, ky, phi = theta[:, 0], theta[:, 1], theta[:, 2], theta[:, 3]
    k = torch.sqrt(kx ** 2 + ky ** 2 + 1e-30)
    omega = omega_of_k(k, depth)
    cg = cg_of_k(k, depth)
    psi = (kx[None, :] * x[:, None] + ky[None, :] * y[:, None]
           - omega[None, :] * t[:, None] + phi[None, :])      # (M,N)
    cos_p, sin_p = torch.cos(psi), torch.sin(psi)
    # ∂ψ/∂kx = x − c_g (kx/k) t ; ∂ψ/∂ky = y − c_g (ky/k) t
    dpsi_dkx = x[:, None] - (cg * kx / k)[None, :] * t[:, None]
    dpsi_dky = y[:, None] - (cg * ky / k)[None, :] * t[:, None]
    J_w = cos_p
    J_kx = -w[None, :] * sin_p * dpsi_dkx
    J_ky = -w[None, :] * sin_p * dpsi_dky
    J_phi = -w[None, :] * sin_p
    return torch.cat([J_w, J_kx, J_ky, J_phi], dim=1)         # (M, 4N)


# ----------------------------------------------------------------------------
# FIM / CRB
# ----------------------------------------------------------------------------
def fisher(theta, x, y, t, depth, sigma):
    J = jacobian(theta, x, y, t, depth)
    return (J.T @ J) / (sigma ** 2)


def field_design(theta, gx, gy, t_star, depth):
    """G = ⟨ g gᵀ ⟩ averaged over reconstruction grid (gx,gy) at time t_star."""
    Xg, Yg = torch.meshgrid(gx, gy, indexing='xy')
    xf, yf = Xg.reshape(-1), Yg.reshape(-1)
    tf = torch.full_like(xf, float(t_star))
    Jg = jacobian(theta, xf, yf, tf, depth)                   # (P, 4N)
    return (Jg.T @ Jg) / Jg.shape[0]


def field_nrmse_crb(F, Gmat, Hs, ridge_rel=1e-10):
    """√(tr(F⁻¹ G)) / Hs. Symmetric solve with tiny relative ridge for stability."""
    n = F.shape[0]
    lam = ridge_rel * torch.trace(F) / n
    Finv = torch.linalg.inv(F + lam * torch.eye(n, dtype=DT, device=F.device))
    var = torch.trace(Finv @ Gmat)
    return float(torch.sqrt(torch.clamp(var, min=0.0))) / Hs, Finv


def per_component_crb(Finv, N):
    """Return per-component std on (w, kx, ky, phi) and derived |k|, direction."""
    d = torch.diagonal(Finv)
    sw, skx, sky, sphi = d[:N], d[N:2*N], d[2*N:3*N], d[3*N:4*N]
    return (torch.sqrt(torch.clamp(sw, min=0)),
            torch.sqrt(torch.clamp(skx, min=0)),
            torch.sqrt(torch.clamp(sky, min=0)),
            torch.sqrt(torch.clamp(sphi, min=0)))


def direction_crb_deg(theta, Finv, N):
    """CRB std on each component's propagation direction α=atan2(ky,kx), in degrees.
    σ_α² = Jᵀ Cov[kx,ky] J  with J = ∂α/∂(kx,ky) = (−ky, kx)/k²."""
    kx, ky = theta[:, 1], theta[:, 2]
    k2 = kx ** 2 + ky ** 2 + 1e-30
    out = []
    for i in range(N):
        gi = torch.zeros(4 * N, dtype=DT, device=Finv.device)
        gi[N + i] = -ky[i] / k2[i]       # ∂α/∂kx
        gi[2 * N + i] = kx[i] / k2[i]    # ∂α/∂ky
        var = gi @ Finv @ gi
        out.append(float(torch.sqrt(torch.clamp(var, min=0))) * 180.0 / math.pi)
    return np.array(out)


# ----------------------------------------------------------------------------
# Episode helpers
# ----------------------------------------------------------------------------
def load_ep(path, depth=50.0):
    d = load_episode(path)
    wc = np.array(d['wave_components'])           # [A, omega, kx, ky, phi, Q]
    theta = torch.tensor(np.stack([wc[:, 0], wc[:, 2], wc[:, 3], wc[:, 4]], 1),
                         dtype=DT, device=DEVICE)
    ts = np.array(d['timestamps'])
    lp = np.array(d['lidar_points']); vm = np.array(d['lidar_valid_mask']).astype(bool)
    return dict(
        theta=theta, ts=ts, lp=lp, vm=vm,
        Hs=float(d['Hs_true']), Tp=float(d['Tp_true']), depth=depth,
        gx=torch.tensor(d['gt_grid_x'], dtype=DT, device=DEVICE),
        gy=torch.tensor(d['gt_grid_y'], dtype=DT, device=DEVICE),
    )


def collect_obs(ep, t_obs):
    """All observation points with timestamp in [0, t_obs]."""
    xs, ys, tsr = [], [], []
    for fi, tt in enumerate(ep['ts']):
        if tt > t_obs + 1e-9:
            break
        m = ep['vm'][fi]
        pts = ep['lp'][fi][m]
        xs.append(pts[:, 0]); ys.append(pts[:, 1])
        tsr.append(np.full(pts.shape[0], tt))
    x = torch.tensor(np.concatenate(xs), dtype=DT, device=DEVICE)
    y = torch.tensor(np.concatenate(ys), dtype=DT, device=DEVICE)
    t = torch.tensor(np.concatenate(tsr), dtype=DT, device=DEVICE)
    return x, y, t


def footprint_extent(ep):
    """Along-/cross-propagation footprint extent (m) using dominant direction."""
    m0 = ep['vm'][0]; p0 = ep['lp'][0][m0]
    # dominant propagation direction = energy-weighted mean of components
    w = ep['theta'][:, 0].cpu().numpy()
    kx = ep['theta'][:, 1].cpu().numpy(); ky = ep['theta'][:, 2].cpu().numpy()
    ang = math.atan2(np.sum(w ** 2 * ky), np.sum(w ** 2 * kx))
    khat = np.array([math.cos(ang), math.sin(ang)])
    kperp = np.array([-math.sin(ang), math.cos(ang)])
    s_par = p0[:, :2] @ khat
    s_perp = p0[:, :2] @ kperp
    return float(s_par.max() - s_par.min()), float(s_perp.max() - s_perp.min()), ang


# ----------------------------------------------------------------------------
# Validation: analytic Jacobian vs autograd
# ----------------------------------------------------------------------------
def gradcheck(depth=50.0):
    torch.manual_seed(0)
    N, M = 4, 60
    theta = torch.randn(N, 4, dtype=DT, device=DEVICE)
    theta[:, 0] = theta[:, 0].abs() * 0.2          # w
    theta[:, 1:3] = theta[:, 1:3] * 0.1            # kx, ky
    x = torch.rand(M, dtype=DT, device=DEVICE) * 80
    y = (torch.rand(M, dtype=DT, device=DEVICE) - 0.5) * 80
    t = torch.rand(M, dtype=DT, device=DEVICE) * 5
    Ja = jacobian(theta, x, y, t, depth)
    th = theta.clone().requires_grad_(True)
    Jauto = torch.autograd.functional.jacobian(
        lambda p: predict(p, x, y, t, depth), th)              # (M,N,4)
    # reorder autograd (M,N,4) -> (M,4N) col [w,kx,ky,phi]
    Jr = torch.cat([Jauto[:, :, j] for j in range(4)], dim=1)
    err = (Ja - Jr).abs().max().item()
    return err


# ----------------------------------------------------------------------------
# Main experiment
# ----------------------------------------------------------------------------
def run_episode(path, t_grid, sigma=0.02, depth=50.0):
    ep = load_ep(path, depth)
    N = ep['theta'].shape[0]
    L_par, L_perp, ang = footprint_extent(ep)
    # dominant component (max energy) for k / cg reference
    w = ep['theta'][:, 0]
    di = int(torch.argmax(w ** 2))
    kx_d, ky_d = float(ep['theta'][di, 1]), float(ep['theta'][di, 2])
    k_d = math.hypot(kx_d, ky_d)
    cg_d = float(cg_of_k(torch.tensor(k_d, dtype=DT), depth))
    Tp = ep['Tp']

    curves = dict(t_obs=[], n_obs=[], field_nrmse_crb=[], dir_crb_deg=[],
                  sigma_k_dom=[], D_eff=[], cond=[])
    for t_obs in t_grid:
        if t_obs < ep['ts'][0] - 1e-9:
            continue
        x, y, t = collect_obs(ep, t_obs)
        if x.numel() < 4 * N:                      # under-determined: CRB = ∞
            continue
        F = fisher(ep['theta'], x, y, t, depth, sigma)
        Gmat = field_design(ep['theta'], ep['gx'], ep['gy'], t_obs, depth)
        nrmse_crb, Finv = field_nrmse_crb(F, Gmat, ep['Hs'])
        dirs = direction_crb_deg(ep['theta'], Finv, N)
        # dominant |k| std via delta method
        kx, ky = ep['theta'][:, 1], ep['theta'][:, 2]
        k = torch.sqrt(kx ** 2 + ky ** 2)
        gk = torch.zeros(4 * N, dtype=DT, device=Finv.device)
        gk[N + di] = kx[di] / k[di]; gk[2 * N + di] = ky[di] / k[di]
        sigma_k = float(torch.sqrt(torch.clamp(gk @ Finv @ gk, min=0)))
        curves['t_obs'].append(float(t_obs))
        curves['n_obs'].append(int(x.numel()))
        curves['field_nrmse_crb'].append(nrmse_crb)
        curves['dir_crb_deg'].append(float(np.median(dirs)))
        curves['sigma_k_dom'].append(sigma_k)
        curves['D_eff'].append(L_par + cg_d * t_obs)
        curves['cond'].append(float(torch.linalg.cond(F)))
    twall = info_wall_time(curves['t_obs'], curves['field_nrmse_crb'], 0.25)
    t10 = info_wall_time(curves['t_obs'], curves['field_nrmse_crb'], 0.10)
    return dict(N=N, Hs=ep['Hs'], Tp=Tp, depth=depth, L_par=L_par, L_perp=L_perp,
                k_dom=k_d, cg_dom=cg_d, t_cross=L_par / cg_d,
                T_wall=twall, T_below10=t10, curves=curves)


def info_wall_time(t_list, nrmse_list, thresh):
    """Smallest T_obs at which the unbiased CRB on field nRMSE drops below thresh.
    thresh=0.25 ≈ the zero-predictor baseline: below the wall, NO unbiased
    estimator can beat predicting zero."""
    t = np.array(t_list); v = np.array(nrmse_list)
    idx = np.where(v < thresh)[0]
    return float(t[idx[0]]) if len(idx) else None


def validate_aperture_law(out, sigma=0.02, depth=50.0, M=300, L=50.0,
                          k=0.09, w=0.30, seed=0):
    """Controlled single-component test: hold #observations M and footprint L
    FIXED, distribute the M points uniformly over [0, T_obs], vary T_obs only.
    This isolates the synthetic-aperture effect from sample-count (√M) growth and
    from multi-component decoupling, so the CRB on |k| should follow 1/D_eff with
    D_eff = L + c_g·T_obs."""
    rng = np.random.default_rng(seed)
    cg = float(cg_of_k(torch.tensor(k, dtype=DT), depth))
    theta = torch.tensor([[w, k, 0.0, 0.0]], dtype=DT, device=DEVICE)   # +x prop.
    # fixed spatial footprint (same for every T_obs)
    xs = torch.tensor(rng.uniform(0, L, M), dtype=DT, device=DEVICE)
    ys = torch.tensor(rng.uniform(-L / 2, L / 2, M), dtype=DT, device=DEVICE)
    frac = torch.tensor(rng.uniform(0, 1, M), dtype=DT, device=DEVICE)  # time fractions
    T_list = np.linspace(0.0, 30.0, 31)
    rows = dict(T_obs=[], D_eff=[], sigma_kx=[], sigma_dir_deg=[])
    for T in T_list:
        t = frac * T
        F = fisher(theta, xs, ys, t, depth, sigma)
        Finv = torch.linalg.inv(F + 1e-12 * torch.trace(F) / 4 * torch.eye(4, dtype=DT, device=F.device))
        sig_kx = float(torch.sqrt(torch.clamp(Finv[1, 1], min=0)))   # along-prop |k|
        sig_dir = float(torch.sqrt(torch.clamp(Finv[2, 2], min=0)) / k) * 180 / math.pi  # ky/k
        rows['T_obs'].append(float(T)); rows['D_eff'].append(L + cg * T)
        rows['sigma_kx'].append(sig_kx); rows['sigma_dir_deg'].append(sig_dir)
    # fit log-log slope of sigma_kx vs D_eff over the aperture-dominated regime
    De = np.array(rows['D_eff']); sk = np.array(rows['sigma_kx'])
    sl = np.polyfit(np.log(De[5:]), np.log(sk[5:]), 1)[0]
    rows['loglog_slope'] = float(sl); rows['cg'] = cg; rows['L'] = L; rows['M'] = M

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 5))
    a1.loglog(De, sk, 'o-', color='tab:blue', label='CRB std on $|k|$ (fixed M, footprint)')
    a1.loglog(De, sk[0] * De[0] / De, 'k--', lw=1, label=r'$\propto 1/D_{eff}$')
    a1.axvline(L, color='gray', ls=':', label=f'footprint L={L:.0f}m')
    a1.set_xlabel(r'$D_{eff}=L+c_g T_{obs}$ [m]'); a1.set_ylabel(r'$\sigma_{|k|}$ [rad/m]')
    a1.set_title(f'Aperture law verified: slope={sl:.2f} (ideal −1)')
    a1.grid(True, which='both', alpha=0.25); a1.legend(fontsize=8)
    a2.plot(rows['T_obs'], rows['sigma_dir_deg'], 'o-', color='tab:purple')
    a2.axvline(L / cg, color='gray', ls='--', label=f'$L/c_g$={L/cg:.1f}s')
    a2.set_xlabel('$T_{obs}$ [s]'); a2.set_ylabel('CRB std on direction [deg]')
    a2.set_title('Direction CRB (single comp, fixed M)')
    a2.grid(True, alpha=0.25); a2.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(f'{out}/fig_aperture_law.png', dpi=140); plt.close(fig)
    return rows


# estimator results (ST-DF-GP, from results/spacetime_gp_30ep.json), per-episode s0
ESTIMATORS = {
    'BF4_5comp_seed0_uniform':  dict(perframe=0.0877, fused=0.0700, oracle=0.0458),
    'BF4_15comp_seed0_uniform': dict(perframe=0.3260, fused=0.2715, oracle=0.0393),
    'BF4_30comp_seed0_uniform': dict(perframe=0.3564, fused=0.3078, oracle=0.0981),
}
W_ESTIMATOR_SEC = 1.0   # ST-DF-GP uses W=10 frames @ dt=0.1s = 1.0 s window


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='results/direction1')
    ap.add_argument('--sigma', type=float, default=0.02)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    ge = gradcheck()
    print(f'[gradcheck] max |analytic - autograd| = {ge:.2e}  '
          f'({"PASS" if ge < 1e-6 else "FAIL"})')

    eps = ['BF4_5comp_seed0_uniform', 'BF4_15comp_seed0_uniform',
           'BF4_30comp_seed0_uniform']
    # T_obs grid: 1 frame (0.1s) .. 30s, log-spaced
    t_grid = np.unique(np.round(np.concatenate([
        np.array([0.1, 0.2, 0.3, 0.5, 0.7, 1.0]),
        np.geomspace(1.0, 30.0, 22)]), 2))

    results = dict(sigma=args.sigma, gradcheck_err=ge, episodes={})
    for tag in eps:
        path = f'data/episode_{tag}.h5'
        r = run_episode(path, t_grid, sigma=args.sigma)
        r['estimator'] = ESTIMATORS[tag]
        r['W_estimator_sec'] = W_ESTIMATOR_SEC
        results['episodes'][tag] = r
        c = r['curves']
        print(f'\n=== {tag} (N={r["N"]}, Hs={r["Hs"]}, Tp={r["Tp"]}s) ===')
        print(f'  footprint  L_par={r["L_par"]:.1f}m  L_perp={r["L_perp"]:.1f}m  '
              f'k_dom={r["k_dom"]:.3f}  c_g={r["cg_dom"]:.2f}m/s  '
              f'T_cross(L_par/c_g)={r["t_cross"]:.1f}s')
        for j in range(len(c['t_obs'])):
            if c['t_obs'][j] in (0.1, 1.0, 3.0) or j == len(c['t_obs']) - 1:
                print(f'   T_obs={c["t_obs"][j]:5.1f}s  M={c["n_obs"][j]:5d}  '
                      f'CRB_nRMSE={c["field_nrmse_crb"][j]:.4f}  '
                      f'dirCRB={c["dir_crb_deg"][j]:6.2f}°  '
                      f'sig_k={c["sigma_k_dom"][j]:.2e}  '
                      f'D_eff={c["D_eff"][j]:6.1f}m')

    print('\n=== Controlled aperture-law validation (single comp, fixed M) ===')
    ap_rows = validate_aperture_law(args.out)
    results['aperture_law'] = ap_rows
    print(f'  log-log slope σ_|k| vs D_eff = {ap_rows["loglog_slope"]:.3f} '
          f'(ideal −1.0; c_g={ap_rows["cg"]:.2f}m/s, L={ap_rows["L"]:.0f}m, M={ap_rows["M"]})')

    print('\n=== Information wall summary (T_obs to reach CRB threshold) ===')
    for tag, r in results['episodes'].items():
        print(f'  {tag:30s} N={r["N"]:2d}  T_wall(CRB<0.25)={r["T_wall"]}s  '
              f'T(CRB<0.10)={r["T_below10"]}s  L/c_g={r["t_cross"]:.1f}s')

    with open(os.path.join(args.out, 'crb_results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    print(f'\nSaved {args.out}/crb_results.json')
    make_figures(results, args.out)


def make_figures(results, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    eps = list(results['episodes'].keys())
    colors = {'BF4_5comp_seed0_uniform': 'tab:green',
              'BF4_15comp_seed0_uniform': 'tab:orange',
              'BF4_30comp_seed0_uniform': 'tab:red'}
    labels = {'BF4_5comp_seed0_uniform': 'BF4 5-comp',
              'BF4_15comp_seed0_uniform': 'BF4 15-comp',
              'BF4_30comp_seed0_uniform': 'BF4 30-comp'}

    # --- Figure 1: information wall (field nRMSE CRB vs T_obs) + estimators ---
    fig, ax = plt.subplots(figsize=(8.2, 6))
    for tag in eps:
        r = results['episodes'][tag]; c = r['curves']; col = colors[tag]
        ax.plot(c['t_obs'], c['field_nrmse_crb'], '-', color=col, lw=2,
                label=f'{labels[tag]}  CRB')
        est = r['estimator']
        ax.scatter([r['W_estimator_sec']], [est['perframe']], marker='v',
                   s=70, color=col, zorder=5)
        ax.scatter([r['W_estimator_sec']], [est['fused']], marker='o',
                   s=70, color=col, edgecolor='k', zorder=5)
        ax.axhline(est['oracle'], color=col, ls=':', lw=1, alpha=0.6)
        ax.axvline(r['t_cross'], color=col, ls='--', lw=1, alpha=0.4)
    # zero-predictor baseline + per-episode information-wall markers
    ax.axhline(0.25, color='k', lw=1.4, alpha=0.8)
    ax.text(0.11, 0.265, 'zero-predictor baseline (nRMSE≈0.25)', fontsize=8)
    for tag in eps:
        r = results['episodes'][tag]
        if r['T_wall'] is not None:
            ax.scatter([r['T_wall']], [0.25], marker='*', s=140,
                       color=colors[tag], edgecolor='k', zorder=6)
    ax.scatter([], [], marker='v', color='gray', label='ST-DF-GP per-frame')
    ax.scatter([], [], marker='o', color='gray', edgecolor='k', label='ST-DF-GP fused')
    ax.scatter([], [], marker='*', color='gray', edgecolor='k',
               label=r'info-wall $T_{wall}$ (CRB=0.25)')
    ax.plot([], [], ':', color='gray', label='ST-DF-GP oracle')
    ax.plot([], [], '--', color='gray', label=r'synth-aperture $L/c_g$')
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_ylim(2e-3, 5.0)
    ax.set_xlabel('observation window $T_{obs}$ [s]')
    ax.set_ylabel('field nRMSE (CRB lower bound / estimator)')
    ax.set_title('Direction I — the information wall:\n'
                 'CRB on field reconstruction vs observation time')
    ax.grid(True, which='both', alpha=0.25); ax.legend(fontsize=8, loc='upper right')
    fig.tight_layout(); fig.savefig(f'{out}/fig_information_wall.png', dpi=140)
    plt.close(fig)

    # --- Figure 2: synthetic aperture — sigma_k vs D_eff, dir CRB vs T_obs ---
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5.2))
    for tag in eps:
        r = results['episodes'][tag]; c = r['curves']; col = colors[tag]
        a1.loglog(c['D_eff'], c['sigma_k_dom'], 'o-', color=col, ms=4,
                  label=labels[tag])
        a2.semilogx(c['t_obs'], c['dir_crb_deg'], 'o-', color=col, ms=4,
                    label=labels[tag])
    # reference 1/D_eff slope on a1
    de = np.array(results['episodes'][eps[0]]['curves']['D_eff'])
    sk = np.array(results['episodes'][eps[0]]['curves']['sigma_k_dom'])
    a1.loglog(de, sk[0] * de[0] / de, 'k--', lw=1, label=r'$\propto 1/D_{eff}$')
    a1.set_xlabel(r'effective aperture $D_{eff}=L_{foot}+c_g T_{obs}$ [m]')
    a1.set_ylabel(r'CRB std on dominant $|k|$  [rad/m]')
    a1.set_title('Synthetic aperture: wavenumber CRB $\\propto 1/D_{eff}$')
    a1.grid(True, which='both', alpha=0.25); a1.legend(fontsize=8)
    for tag in eps:
        r = results['episodes'][tag]
        a2.axvline(r['t_cross'], color=colors[tag], ls='--', lw=1, alpha=0.4)
    a2.set_xlabel('observation window $T_{obs}$ [s]')
    a2.set_ylabel('CRB std on direction (median comp) [deg]')
    a2.set_title('Direction CRB: footprint-limited (cross-range), plateaus')
    a2.grid(True, which='both', alpha=0.25); a2.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(f'{out}/fig_synthetic_aperture.png', dpi=140)
    plt.close(fig)
    print(f'Saved figures to {out}/')


if __name__ == '__main__':
    main()
