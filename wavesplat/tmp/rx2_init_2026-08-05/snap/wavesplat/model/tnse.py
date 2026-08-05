"""
tnse.py — Temporal Neural Spectral Estimator (T-NSE)

Two-stage design:
  Stage 1 (neural):    W frames of sparse LiDAR → {kx_n, ky_n}  (hard: aperture-limited)
  Stage 2 (algebraic): {kx_n, ky_n} + observations → {A_n, φ_n}  (fast linear LS)

Key insight: given k-vectors, the amplitude/phase fit is a linear system (no local optima).
The neural part only needs to learn direction/frequency — temporal consistency
constrains the aperture ambiguity because waves propagate at known phase velocities.

Architecture: SetTransformer-style (permutation-invariant over points within each frame,
attention across frames for temporal context).
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from wavesplat.model.dispersion import dispersion_omega, G


# ── Hyperparameters (default) ─────────────────────────────────────────────────
D_MODEL   = 128    # transformer hidden dim
N_HEADS   = 4      # attention heads
N_LAYERS  = 4      # transformer layers
D_FF      = 256    # feed-forward dim
MAX_PTS   = 128    # max LiDAR points per frame (pad/truncate)
K_MAX     = 0.60   # maximum wavenumber [rad/m] (wavelength ≥ 10m)
K_MIN     = 0.02   # minimum wavenumber [rad/m] (wavelength ≤ 314m)


class PointEncoder(nn.Module):
    """Maps (x_norm, y_norm, t_norm, z_norm, conf) → d_model embedding."""
    def __init__(self, d_model=D_MODEL):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(5, 64), nn.GELU(),
            nn.Linear(64, d_model), nn.GELU(),
            nn.Linear(d_model, d_model),
        )

    def forward(self, x):  # x: (..., 5)
        return self.net(x)


class TransformerBlock(nn.Module):
    def __init__(self, d_model=D_MODEL, n_heads=N_HEADS, d_ff=D_FF, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ff   = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.GELU(), nn.Linear(d_ff, d_model)
        )
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, key_padding_mask=None):
        h, _ = self.attn(x, x, x, key_padding_mask=key_padding_mask)
        x = self.ln1(x + self.drop(h))
        x = self.ln2(x + self.drop(self.ff(x)))
        return x


class TNSE(nn.Module):
    """
    Temporal Neural Spectral Estimator.

    Input:  W frames, each with up to MAX_PTS observations (x, y, t, z, conf)
    Output: N wave components as (kx, ky) — wavenumber vectors
            Amplitudes/phases are solved algebraically (see fit_amplitudes_phases)

    Forward returns raw (kx, ky) predictions in [K_MIN, K_MAX] range.
    """

    def __init__(self, n_components=15, depth=50.0,
                 d_model=D_MODEL, n_heads=N_HEADS, n_layers=N_LAYERS, d_ff=D_FF,
                 dropout=0.1):
        super().__init__()
        self.n_components = n_components
        self.depth = depth
        self.d_model = d_model

        # Encoder
        self.point_enc = PointEncoder(d_model)

        # Learnable frame position embedding (up to 32 frames)
        self.frame_pos_emb = nn.Embedding(32, d_model)

        # Transformer over all tokens (W × MAX_PTS)
        self.transformer = nn.Sequential(
            *[TransformerBlock(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)]
        )

        # Set decoder (DETR-style): N learnable component queries cross-attend to
        # the encoded point tokens. Global average pooling collapses to an
        # input-invariant constant predictor (diagnosed: kx_std≈0 across inputs);
        # per-component queries let each output extract an input-specific direction.
        self.query_emb  = nn.Embedding(n_components, d_model)
        self.cross_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout,
                                                batch_first=True)
        self.q_ln1 = nn.LayerNorm(d_model)
        self.q_ln2 = nn.LayerNorm(d_model)
        self.q_ff  = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.GELU(), nn.Linear(d_ff, d_model)
        )
        # Per-query head → (kx_raw, ky_raw, conf_logit). The confidence logit is
        # the DETR "objectness": Hungarian matching supervises matched queries to
        # confidence 1 and unmatched ("no-object") queries to 0, so the model can
        # leave spare queries idle instead of forcing every query onto a component.
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model // 2), nn.GELU(),
            nn.Linear(d_model // 2, 3),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, frames_tokens, frames_mask, frame_ids):
        """
        Args:
            frames_tokens: (B, W, MAX_PTS, 5) — normalised (x,y,t,z,conf)
            frames_mask:   (B, W, MAX_PTS) bool — True = padding (ignored)
            frame_ids:     (B, W) int — frame indices for positional embedding

        Returns:
            kx: (B, N) wavenumber x-components [rad/m]
            ky: (B, N) wavenumber y-components [rad/m]
        """
        B, W, P, _ = frames_tokens.shape

        # Encode each point
        tokens = self.point_enc(frames_tokens)   # (B, W, P, D)

        # Add frame positional embedding
        frame_emb = self.frame_pos_emb(frame_ids)  # (B, W, D)
        tokens = tokens + frame_emb.unsqueeze(2)   # broadcast over points

        # Flatten W×P → sequence
        tokens = tokens.view(B, W * P, self.d_model)
        mask   = frames_mask.view(B, W * P)         # True = ignore

        # Transformer (apply each layer; nn.Sequential doesn't support mask kwarg)
        for layer in self.transformer:
            tokens = layer(tokens, key_padding_mask=mask)

        # Set decoder: component queries cross-attend to the encoded points.
        q = self.query_emb.weight.unsqueeze(0).expand(B, -1, -1)  # (B, N, D)
        attn, _ = self.cross_attn(q, tokens, tokens, key_padding_mask=mask)  # (B, N, D)
        q = self.q_ln1(q + attn)
        q = self.q_ln2(q + self.q_ff(q))

        # Per-query → (kx_raw, ky_raw, conf_logit)
        out = self.head(q)  # (B, N, 3)

        # Map to physically valid wavenumber range via sigmoid
        k_mag_raw = torch.sigmoid(out[..., 0]) * (K_MAX - K_MIN) + K_MIN  # (B, N)
        theta_raw = torch.tanh(out[..., 1]) * math.pi                      # (B, N) ∈ (-π, π)

        kx = k_mag_raw * torch.cos(theta_raw)
        ky = k_mag_raw * torch.sin(theta_raw)
        conf = out[..., 2]   # raw objectness logit (caller applies sigmoid)

        return kx, ky, conf


# ── Algebraic amplitude/phase fitting ─────────────────────────────────────────

def fit_amplitudes_phases(kx, ky, obs_xyz, depth, lambda_reg=0.5):
    """
    Given wavenumber vectors {kx_n, ky_n} and observations (x_i, y_i, t_i, z_i),
    fit amplitudes A_n and phases φ_n via regularised linear least squares.

    Model: z(x,y,t) = Σ_n [a_n cos(ψ_n) + b_n sin(ψ_n)]
           where ψ_n = kx_n*x + ky_n*y - ω_n*t, ω_n = dispersion(|k_n|, d)
           A_n = sqrt(a_n² + b_n²),  φ_n = atan2(-b_n, a_n)

    Args:
        kx, ky:   (N,)  numpy arrays, wavenumber vectors [rad/m]
        obs_xyz:  (M, 4) numpy array, columns: [x, y, t, z]
        depth:    float, water depth [m]
        lambda_reg: L2 regularisation strength

    Returns:
        A:   (N,)  amplitudes [m]
        phi: (N,)  phases [rad]
    """
    import numpy as np

    N = len(kx)
    M = len(obs_xyz)
    x, y, t, z = obs_xyz[:, 0], obs_xyz[:, 1], obs_xyz[:, 2], obs_xyz[:, 3]

    k_mag = np.sqrt(kx**2 + ky**2 + 1e-12)
    g = G
    omega = np.sqrt(g * k_mag * np.tanh(k_mag * depth))

    # Build design matrix [cos(ψ_1),...,cos(ψ_N), sin(ψ_1),...,sin(ψ_N)]  (M × 2N)
    psi = kx[None, :] * x[:, None] + ky[None, :] * y[:, None] - omega[None, :] * t[:, None]
    Phi = np.concatenate([np.cos(psi), np.sin(psi)], axis=1)  # (M, 2N)

    # Regularised LS: (Phi^T Phi + λI) coef = Phi^T z
    A_mat = Phi.T @ Phi + lambda_reg * np.eye(2 * N)
    b_vec = Phi.T @ z
    coef = np.linalg.solve(A_mat, b_vec)  # (2N,)

    a = coef[:N]; b = coef[N:]
    A   = np.sqrt(a**2 + b**2)
    phi = np.arctan2(-b, a)

    return A, phi


def render_wave_field(kx, ky, A, phi, depth, grid_x, grid_y, t):
    """
    Render wave elevation on a grid.

    Args:
        kx, ky, A, phi: (N,) numpy arrays
        grid_x: (Nx,)  x-coordinates
        grid_y: (Ny,)  y-coordinates
        t:      float, time [s]

    Returns:
        eta: (Ny, Nx) wave elevation [m]
    """
    import numpy as np
    k_mag = np.sqrt(kx**2 + ky**2 + 1e-12)
    omega = np.sqrt(G * k_mag * np.tanh(k_mag * depth))

    Xg, Yg = np.meshgrid(grid_x, grid_y)
    eta = np.zeros_like(Xg)
    for n in range(len(kx)):
        psi = kx[n] * Xg + ky[n] * Yg - omega[n] * t + phi[n]
        eta += A[n] * np.cos(psi)
    return eta


# ── Wave field rendering (differentiable, for training loss) ──────────────────

def render_wave_field_torch(kx, ky, obs_x, obs_y, obs_t, depth, A=None, phi=None):
    """
    Differentiable rendering at observation points.
    If A and phi are None, only returns the design matrix (for LS fitting).

    Args:
        kx, ky:       (B, N)  wavenumber vectors
        obs_x/y/t:   (M,) or (B, M) observation coords
        depth:        float

    Returns:
        eta_pred: (B, M) predicted elevations (if A, phi given)
        or Phi:   (B, M, 2N) design matrix (if A, phi is None)
    """
    k_mag = torch.sqrt(kx**2 + ky**2 + 1e-12)        # (B, N)
    omega = dispersion_omega(k_mag, depth)              # (B, N)

    # Broadcast
    if obs_x.dim() == 1:
        obs_x = obs_x.unsqueeze(0); obs_y = obs_y.unsqueeze(0); obs_t = obs_t.unsqueeze(0)

    psi = (kx.unsqueeze(2) * obs_x.unsqueeze(1) +
           ky.unsqueeze(2) * obs_y.unsqueeze(1) -
           omega.unsqueeze(2) * obs_t.unsqueeze(1))    # (B, N, M)

    if A is not None and phi is not None:
        eta = (A.unsqueeze(2) * torch.cos(psi + phi.unsqueeze(2))).sum(1)  # (B, M)
        return eta

    # Return design matrix for differentiable LS
    Phi = torch.cat([torch.cos(psi), torch.sin(psi)], dim=1)  # (B, 2N, M)
    return Phi.permute(0, 2, 1)  # (B, M, 2N)


def differentiable_ls_fit(Phi, z, lambda_reg=0.5):
    """
    Solve (Phi^T Phi + λI) coef = Phi^T z via Cholesky (differentiable).

    Args:
        Phi: (B, M, 2N)
        z:   (B, M)
        lambda_reg: float

    Returns:
        coef: (B, 2N)
    """
    B, M, K = Phi.shape
    A_mat = Phi.transpose(1, 2) @ Phi + lambda_reg * torch.eye(K, device=Phi.device).unsqueeze(0)
    b_vec = Phi.transpose(1, 2) @ z.unsqueeze(2)  # (B, 2N, 1)
    coef = torch.linalg.solve(A_mat, b_vec).squeeze(2)  # (B, 2N)
    return coef


def tnse_forward_with_ls(kx, ky, obs_x, obs_y, obs_t, obs_z, depth, lambda_reg=0.5):
    """
    Full forward pass: kx/ky → LS-fit A/phi → render at obs points.
    Differentiable end-to-end.

    Returns:
        z_pred:  (B, M) predictions at observation points
        coef:    (B, 2N) [a_1..a_N, b_1..b_N]
    """
    Phi = render_wave_field_torch(kx, ky, obs_x, obs_y, obs_t, depth)  # (B, M, 2N)
    coef = differentiable_ls_fit(Phi, obs_z, lambda_reg)                # (B, 2N)
    z_pred = (Phi @ coef.unsqueeze(2)).squeeze(2)                       # (B, M)
    return z_pred, coef
