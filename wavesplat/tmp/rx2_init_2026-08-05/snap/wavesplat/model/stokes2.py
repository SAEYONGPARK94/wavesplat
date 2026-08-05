"""
stokes2.py
Stokes 2nd-order wave interaction transfer functions and surface elevation correction.

Theory: Sharma & Dean (1981), finite-depth directional 2nd-order theory.
Reference: Kim, Platt, Jonkman (2014) — OpenFAST WAVE2 documentation,
           Eqs. 3.24–3.30.

Second-order surface elevation:
    eta2(x,t) = sum_n sum_m w_n w_m [L+_nm cos(theta_n + theta_m)
                                    + L-_nm cos(theta_n - theta_m)]

where theta_n = kx_n*x + ky_n*y - omega_n*t + phi_n.

The L+/L- transfer functions depend only on (kx, ky, depth) and can be
precomputed once per spectral configuration. During WaveSplat tracking
(w_phi mode), kx/ky are fixed, so L+/L- are constant.

Memory-efficient evaluation via trig identity:
    cos(A+B) = cosA cosB - sinA sinB
    cos(A-B) = cosA cosB + sinA sinB
avoids materializing the O(M x N x N) tensor.
"""

import torch
import numpy as np
import math

G = 9.81


def _compute_transfer_functions_np(kx: np.ndarray, ky: np.ndarray,
                                   depth: float) -> tuple:
    """
    Compute Sharma-Dean L+/L- transfer function matrices.

    Args:
        kx: (N,) x-wavenumber [rad/m]
        ky: (N,) y-wavenumber [rad/m]
        depth: water depth [m]

    Returns:
        L_plus:  (N, N) sum-interaction transfer function
        L_minus: (N, N) difference-interaction transfer function
    """
    N = len(kx)
    h = depth

    # Per-component quantities
    k_mag = np.sqrt(kx**2 + ky**2)            # (N,)
    kh = k_mag * h
    tanh_kh = np.tanh(np.clip(kh, 0, 20))     # clip to avoid overflow
    R = k_mag * tanh_kh                        # R_n = |k_n| tanh(|k_n| h)
    sqrt_R = np.sqrt(np.maximum(R, 1e-12))     # (N,)

    # Pairwise quantities — (N, N)
    # cos(theta_dir_n - theta_dir_m) = dot product of unit wave vectors
    k_safe = np.maximum(k_mag, 1e-10)
    ux = kx / k_safe  # unit direction x
    uy = ky / k_safe  # unit direction y
    cos_dtheta = ux[:, None] * ux[None, :] + uy[:, None] * uy[None, :]  # (N,N)
    cos_dtheta = np.clip(cos_dtheta, -1.0, 1.0)

    # k_n * k_m * cos(delta_theta)
    kn_km_cos = k_mag[:, None] * k_mag[None, :] * cos_dtheta  # (N,N)

    # R_n * R_m
    Rn_Rm = R[:, None] * R[None, :]  # (N,N)

    # sqrt(R_n * R_m)
    sqrt_Rn_Rm = sqrt_R[:, None] * sqrt_R[None, :]  # (N,N)

    # k_n^2 - R_n^2
    k2_minus_R2 = k_mag**2 - R**2  # (N,)

    L_plus = np.zeros((N, N))
    L_minus = np.zeros((N, N))

    for sign_idx, sign in enumerate([+1, -1]):
        # sign=+1 → sum interaction (L+)
        # sign=-1 → difference interaction (L-)

        # sqrt(R_n) ± sqrt(R_m)
        sqR_pm = sqrt_R[:, None] + sign * sqrt_R[None, :]  # (N,N)
        sqR_pm_sq = sqR_pm**2

        # k±_nm = |k_vec_n ± k_vec_m|
        kx_pm = kx[:, None] + sign * kx[None, :]
        ky_pm = ky[:, None] + sign * ky[None, :]
        k_pm = np.sqrt(kx_pm**2 + ky_pm**2)  # (N,N)

        # k±_nm * tanh(k±_nm * h)
        k_pm_h = k_pm * h
        k_pm_tanh = k_pm * np.tanh(np.clip(k_pm_h, 0, 20))

        # Denominator for D±: (sqrt(Rn) ± sqrt(Rm))^2 - k±_nm tanh(k±_nm h)
        denom = sqR_pm_sq - k_pm_tanh
        # Regularize to avoid division by zero (near-resonance)
        denom = np.where(np.abs(denom) < 1e-8, np.sign(denom + 1e-20) * 1e-8, denom)

        # Numerator term 1 of D±:
        # (sqrt(Rn) ± sqrt(Rm)) * [sqrt(Rm)*(kn^2 - Rn^2) ± sqrt(Rn)*(km^2 - Rm^2)]
        bracket1 = (sqrt_R[None, :] * k2_minus_R2[:, None]
                    + sign * sqrt_R[:, None] * k2_minus_R2[None, :])
        num1 = sqR_pm * bracket1

        # Numerator term 2 of D±:
        # 2*(sqrt(Rn) ± sqrt(Rm))^2 * [kn*km*cos(dtheta) ∓ Rn*Rm]
        # Note: ∓ is opposite sign from ±
        inner2 = kn_km_cos - sign * Rn_Rm  # ∓ means opposite of sign
        num2 = 2.0 * sqR_pm_sq * inner2

        D_pm = (num1 + num2) / denom  # (N,N)

        # L± = 1/4 * [D± - (kn*km*cos(dtheta) ∓ Rn*Rm) / sqrt(Rn*Rm) + (Rn + Rm)]
        # The ∓ in the middle term is again opposite of the ± sign
        middle_term = (kn_km_cos - sign * Rn_Rm) / np.maximum(sqrt_Rn_Rm, 1e-12)
        R_sum = R[:, None] + R[None, :]

        L = 0.25 * (D_pm - middle_term + R_sum)

        if sign == +1:
            L_plus = L
        else:
            L_minus = L

    # Handle difference self-interaction (n=m): set to 0
    # The n=m term gives constant mean set-down (time-independent offset).
    # It doesn't affect wave shape, so we exclude it.
    np.fill_diagonal(L_minus, 0.0)

    # Handle components with k≈0 (mean level): zero out their interactions
    zero_k = k_mag < 1e-8
    L_plus[zero_k, :] = 0.0
    L_plus[:, zero_k] = 0.0
    L_minus[zero_k, :] = 0.0
    L_minus[:, zero_k] = 0.0

    return L_plus, L_minus


class Stokes2Transfer:
    """
    Precomputed Stokes 2nd-order transfer functions for a given spectral config.

    Usage:
        transfer = Stokes2Transfer(kx, ky, depth)
        z2 = transfer.correction(x, y, t, w, kx_t, ky_t, phi)
    """

    def __init__(self, kx: np.ndarray, ky: np.ndarray, depth: float):
        """
        Args:
            kx: (N,) x-wavenumber [rad/m]
            ky: (N,) y-wavenumber [rad/m]
            depth: water depth [m]
        """
        self.depth = depth
        self._kx_np = kx.copy()
        self._ky_np = ky.copy()
        L_plus_np, L_minus_np = _compute_transfer_functions_np(kx, ky, depth)
        self._L_plus_np = L_plus_np
        self._L_minus_np = L_minus_np
        # Lazy torch conversion
        self._L_plus = None
        self._L_minus = None
        self._device = None

    def _ensure_torch(self, device):
        if self._L_plus is None or self._device != device:
            self._L_plus = torch.tensor(self._L_plus_np, dtype=torch.float32,
                                        device=device)
            self._L_minus = torch.tensor(self._L_minus_np, dtype=torch.float32,
                                         device=device)
            self._device = device

    @property
    def L_plus(self) -> np.ndarray:
        return self._L_plus_np

    @property
    def L_minus(self) -> np.ndarray:
        return self._L_minus_np

    def correction(self, x: torch.Tensor, y: torch.Tensor, t: float,
                   w: torch.Tensor, kx: torch.Tensor, ky: torch.Tensor,
                   phi: torch.Tensor, omega: torch.Tensor) -> torch.Tensor:
        """
        Compute 2nd-order surface elevation correction at query points.

        Uses trig identities to avoid O(M*N*N) memory:
            cos(A+B) = cosA cosB - sinA sinB
            cos(A-B) = cosA cosB + sinA sinB

        Args:
            x: (M,) x-coordinates [m]
            y: (M,) y-coordinates [m]
            t: time [s]
            w: (N,) amplitudes [m] — differentiable
            kx: (N,) x-wavenumber [rad/m]
            ky: (N,) y-wavenumber [rad/m]
            phi: (N,) phase [rad] — differentiable
            omega: (N,) angular frequency [rad/s]

        Returns:
            z2: (M,) second-order elevation correction [m]
        """
        device = w.device
        self._ensure_torch(device)

        # Phase: theta_n = kx_n*x + ky_n*y - omega_n*t + phi_n  → (M, N)
        phase = (kx[None, :] * x[:, None]
                 + ky[None, :] * y[:, None]
                 - omega[None, :] * t
                 + phi[None, :])

        C = torch.cos(phase)  # (M, N)
        S = torch.sin(phase)  # (M, N)

        # Effective weight matrices: W±[n,m] = w[n] * w[m] * L±[n,m]
        ww = w[:, None] * w[None, :]  # (N, N)
        W_plus = ww * self._L_plus    # (N, N)
        W_minus = ww * self._L_minus  # (N, N)

        # Sum interaction: Σ W+[n,m] cos(θn+θm)
        #   = Σ W+[n,m] (Cn*Cm - Sn*Sm)
        #   = (C @ W+) ⊙ C .sum(1) - (S @ W+) ⊙ S .sum(1)
        CW_plus = C @ W_plus   # (M, N)
        SW_plus = S @ W_plus   # (M, N)
        z_sum = (CW_plus * C).sum(dim=1) - (SW_plus * S).sum(dim=1)

        # Diff interaction: Σ W-[n,m] cos(θn-θm)
        #   = Σ W-[n,m] (Cn*Cm + Sn*Sm)
        #   = (C @ W-) ⊙ C .sum(1) + (S @ W-) ⊙ S .sum(1)
        CW_minus = C @ W_minus  # (M, N)
        SW_minus = S @ W_minus  # (M, N)
        z_diff = (CW_minus * C).sum(dim=1) + (SW_minus * S).sum(dim=1)

        return z_sum + z_diff

    def correction_np(self, x: np.ndarray, y: np.ndarray, t: float,
                      w: np.ndarray, kx: np.ndarray, ky: np.ndarray,
                      phi: np.ndarray, omega: np.ndarray) -> np.ndarray:
        """NumPy version for non-differentiable evaluation (e.g., gap experiments)."""
        N = len(w)
        phase = (kx[None, :] * x[:, None]
                 + ky[None, :] * y[:, None]
                 - omega[None, :] * t
                 + phi[None, :])  # (M, N)

        C = np.cos(phase)
        S = np.sin(phase)

        ww = w[:, None] * w[None, :]
        W_plus = ww * self._L_plus_np
        W_minus = ww * self._L_minus_np

        CW_plus = C @ W_plus
        SW_plus = S @ W_plus
        z_sum = np.sum(CW_plus * C, axis=1) - np.sum(SW_plus * S, axis=1)

        CW_minus = C @ W_minus
        SW_minus = S @ W_minus
        z_diff = np.sum(CW_minus * C, axis=1) + np.sum(SW_minus * S, axis=1)

        return z_sum + z_diff
