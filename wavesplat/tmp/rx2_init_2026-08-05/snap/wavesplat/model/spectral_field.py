"""
spectral_field.py — Direction II: directional amplitude spectrum representation
and the GP-marginal-likelihood forward model (phase marginalised analytically).

Spectrum S(kx,ky): non-negative energy on a 64x64 (kx,ky) grid.
A spectrum defines a space-time GP kernel
    K_S((x,t),(x',t')) = Σ_grid S(k)·cos(k·Δx − ω(k)·Δt),  ω=√(g|k|tanh|k|d)
and the sparse-LiDAR observations have closed-form marginal likelihood
    p(z|S) = N(z; 0, K_S(X,X) + σ²I),
so phases are marginalised (no phase nonlinearity — the T-NSE failure mode).
"""

import numpy as np

G = 9.81

# ── Spectrum grid ─────────────────────────────────────────────────────────────
K_LIM = 1.1
N_GRID = 64
_kax = np.linspace(-K_LIM, K_LIM, N_GRID).astype(np.float32)
KX_GRID, KY_GRID = np.meshgrid(_kax, _kax)          # (G,G) each, [row=ky, col=kx]
KMAG_GRID = np.sqrt(KX_GRID**2 + KY_GRID**2)
DK = float(_kax[1] - _kax[0])


def omega_grid(depth):
    """Dispersion ω on the spectrum grid."""
    k = KMAG_GRID
    return np.sqrt(G * k * np.tanh(k * depth) + 1e-12)


def rasterize_components(components, splat_sigma_cells=1.0):
    """
    Components (N,>=4: A,ω,kx,ky,...) → energy spectrum S(kx,ky) on the grid.
    Each component deposits A² (energy) at its (kx,ky), Gaussian-splatted.
    """
    S = np.zeros((N_GRID, N_GRID), dtype=np.float32)
    A = components[:, 0]; kx = components[:, 2]; ky = components[:, 3]
    s = splat_sigma_cells * DK
    for n in range(len(A)):
        d2 = (KX_GRID - kx[n])**2 + (KY_GRID - ky[n])**2
        g = np.exp(-0.5 * d2 / s**2)
        if g.sum() > 0:
            S += (A[n]**2) * g / g.sum()
    return S


def spectrum_to_components(S, max_comp=30, rel_thresh=0.02):
    """Peak-pick a spectrum back into components (A,ω,kx,ky) — for rendering/eval."""
    Sf = S.flatten()
    if Sf.max() <= 0:
        return np.zeros((0, 4), dtype=np.float32)
    order = np.argsort(-Sf)
    out = []
    thr = rel_thresh * Sf.max()
    for idx in order[:max_comp]:
        if Sf[idx] < thr:
            break
        r, c = idx // N_GRID, idx % N_GRID
        kx, ky = KX_GRID[r, c], KY_GRID[r, c]
        A = np.sqrt(max(Sf[idx], 0.0))
        out.append([A, 0.0, kx, ky])
    return np.array(out, dtype=np.float32) if out else np.zeros((0, 4), np.float32)


def gp_kernel_from_spectrum(S, P1, P2, depth, active_thresh=1e-6):
    """
    K_S(P1,P2) = Σ_grid S(k)·cos(kx·Δx + ky·Δy − ω·Δt).
    Only grid cells with S>thresh contribute (realistic spectra are sparse).
    P1,P2: (n,3) columns [x,y,t].
    """
    om = omega_grid(depth)
    mask = S > active_thresh
    s = S[mask]; kx = KX_GRID[mask]; ky = KY_GRID[mask]; om = om[mask]
    dx = P1[:, 0:1] - P2[None, :, 0]
    dy = P1[:, 1:2] - P2[None, :, 1]
    dt = P1[:, 2:3] - P2[None, :, 2]
    K = np.zeros((P1.shape[0], P2.shape[0]), dtype=np.float64)
    for i in range(len(s)):
        K += s[i] * np.cos(kx[i]*dx + ky[i]*dy - om[i]*dt)
    return K


def gp_marginal_loglik(S, obs_xyz, z, depth, sigma_obs=0.02, jitter=1e-6):
    """
    log N(z; 0, K_S(X,X) + σ²I). Higher = spectrum S explains observations better.
    obs_xyz: (M,3) [x,y,t]; z: (M,).
    """
    K = gp_kernel_from_spectrum(S, obs_xyz, obs_xyz, depth)
    M = len(z)
    A = K + (sigma_obs**2 + jitter) * np.eye(M)
    try:
        L = np.linalg.cholesky(A)
    except np.linalg.LinAlgError:
        A = A + 1e-3 * np.eye(M)
        L = np.linalg.cholesky(A)
    alpha = np.linalg.solve(L, z)
    quad = -0.5 * float(alpha @ alpha)
    logdet = -np.sum(np.log(np.diag(L)))
    const = -0.5 * M * np.log(2 * np.pi)
    return quad + logdet + const
