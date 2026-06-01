"""
DCSpecNet model: Set Transformer encoder + Dispersion-Constrained spectral decoder.

Architecture:
  Input:  {(x_i, y_i, z_i)} sparse LiDAR point cloud (variable size)
  Output: complex spectral coefficients on a fixed (kx, ky) grid

The dispersion relation w = sqrt(g * |k| * tanh(|k| * d)) is enforced as a
hard architectural constraint (Beucler et al. 2021), never predicted freely.

Key insight: predicting (A, phi) on a FIXED wavenumber grid makes
reconstruction LINEAR in (A*cos(phi), A*sin(phi)), avoiding the
non-convex loss landscape of free (kx, ky) prediction.

References:
  - Set Transformer: Lee et al., ICML 2019
  - Hard physics constraints: Beucler et al., PRL 2021
  - Dispersion: full linear dispersion for finite depth
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Tuple, List

G = 9.81


@dataclass
class DCSpecNetConfig:
    """Configuration for DCSpecNet."""
    # Input
    input_dim: int = 3              # (x, y, z) per point
    # Encoder
    embed_dim: int = 128            # embedding dimension
    n_heads: int = 4                # attention heads
    n_isab: int = 2                 # number of ISAB layers
    n_inducing: int = 32            # inducing points for ISAB
    # Spectral grid
    k_grid_n: int = 11              # grid points per axis → total k_grid_n^2
    k_max: float = 1.0              # max wavenumber [rad/m]
    # Physics
    depth: float = 50.0             # water depth [m]
    g: float = G


# ─── Attention building blocks (Set Transformer) ─────────────────────────

class MultiHeadAttention(nn.Module):
    """Standard multi-head attention."""

    def __init__(self, dim: int, n_heads: int):
        super().__init__()
        assert dim % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.W_q = nn.Linear(dim, dim)
        self.W_k = nn.Linear(dim, dim)
        self.W_v = nn.Linear(dim, dim)
        self.W_o = nn.Linear(dim, dim)

    def forward(self, Q: torch.Tensor, K: torch.Tensor,
                V: torch.Tensor) -> torch.Tensor:
        B, Nq, D = Q.shape
        _, Nk, _ = K.shape
        h = self.n_heads

        q = self.W_q(Q).view(B, Nq, h, self.head_dim).transpose(1, 2)
        k = self.W_k(K).view(B, Nk, h, self.head_dim).transpose(1, 2)
        v = self.W_v(V).view(B, Nk, h, self.head_dim).transpose(1, 2)

        attn = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, v)  # (B, h, Nq, head_dim)
        out = out.transpose(1, 2).contiguous().view(B, Nq, D)
        return self.W_o(out)


class MAB(nn.Module):
    """Multi-head Attention Block: LayerNorm + MHA + FF."""

    def __init__(self, dim: int, n_heads: int):
        super().__init__()
        self.mha = MultiHeadAttention(dim, n_heads)
        self.ln1 = nn.LayerNorm(dim)
        self.ln2 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Linear(dim * 2, dim),
        )

    def forward(self, Q: torch.Tensor, K: torch.Tensor) -> torch.Tensor:
        h = self.ln1(Q + self.mha(Q, K, K))
        return self.ln2(h + self.ff(h))


class SAB(nn.Module):
    """Set Attention Block: self-attention over set elements."""

    def __init__(self, dim: int, n_heads: int):
        super().__init__()
        self.mab = MAB(dim, n_heads)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return self.mab(X, X)


class ISAB(nn.Module):
    """Inducing-point Set Attention Block: O(nm) complexity."""

    def __init__(self, dim: int, n_heads: int, n_inducing: int):
        super().__init__()
        self.inducing = nn.Parameter(torch.randn(1, n_inducing, dim) * 0.02)
        self.mab1 = MAB(dim, n_heads)
        self.mab2 = MAB(dim, n_heads)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        B = X.shape[0]
        I = self.inducing.expand(B, -1, -1)
        H = self.mab1(I, X)      # (B, m, dim) — inducing points attend to input
        return self.mab2(X, H)   # (B, n, dim) — input attends to inducing points


class PMA(nn.Module):
    """Pooling by Multi-head Attention: learnable query vectors."""

    def __init__(self, dim: int, n_heads: int, n_outputs: int):
        super().__init__()
        self.seeds = nn.Parameter(torch.randn(1, n_outputs, dim) * 0.02)
        self.mab = MAB(dim, n_heads)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        B = X.shape[0]
        S = self.seeds.expand(B, -1, -1)
        return self.mab(S, X)    # (B, n_outputs, dim)


# ─── Dispersion constraint layer ─────────────────────────────────────────

def dispersion_omega(kx: torch.Tensor, ky: torch.Tensor,
                     depth: float, g: float = G) -> torch.Tensor:
    """
    Hard dispersion constraint: w = sqrt(g * |k| * tanh(|k| * d)).

    This is computed deterministically from (kx, ky) — never predicted.
    Follows Beucler et al. (2021) approach of architectural constraints.
    """
    k_mag = torch.sqrt(kx**2 + ky**2 + 1e-12)
    kd = k_mag * depth
    return torch.sqrt(g * k_mag * torch.tanh(torch.clamp(kd, max=20.0)))


# ─── DCSpecNet main model ────────────────────────────────────────────────

class DCSpecNet(nn.Module):
    """
    Dispersion-Constrained Neural Spectral Estimator.

    Takes sparse LiDAR point cloud {(x, y, z)} and predicts complex
    spectral coefficients on a fixed (kx, ky) grid.

    The loss is convex in the output coefficients because reconstruction
    is linear: eta = sum (cr * cos + ci * sin).
    """

    def __init__(self, config: DCSpecNetConfig):
        super().__init__()
        self.cfg = config
        D = config.embed_dim
        K = config.k_grid_n ** 2  # total grid points

        # Build fixed wavenumber grid
        kx_1d = torch.linspace(-config.k_max, config.k_max, config.k_grid_n)
        ky_1d = torch.linspace(-config.k_max, config.k_max, config.k_grid_n)
        kx_grid, ky_grid = torch.meshgrid(kx_1d, ky_1d, indexing='ij')
        self.register_buffer('kx_grid', kx_grid.reshape(-1))  # (K,)
        self.register_buffer('ky_grid', ky_grid.reshape(-1))  # (K,)

        # Precompute omega for the grid
        omega_grid = dispersion_omega(self.kx_grid, self.ky_grid, config.depth)
        self.register_buffer('omega_grid', omega_grid)  # (K,)

        # Point embedding: (x, y, z) → D-dim
        self.point_embed = nn.Sequential(
            nn.Linear(config.input_dim, D),
            nn.GELU(),
            nn.Linear(D, D),
        )

        # Set Transformer encoder
        self.encoder = nn.ModuleList()
        for _ in range(config.n_isab):
            self.encoder.append(ISAB(D, config.n_heads, config.n_inducing))

        # Pool to single global vector
        self.pma = PMA(D, config.n_heads, 1)

        # Decode to 2*K coefficients (real, imag per grid point)
        self.decoder = nn.Sequential(
            nn.Linear(D, D),
            nn.GELU(),
            nn.Linear(D, 2 * K),
        )

    def forward(self, points: torch.Tensor) -> dict:
        """
        Forward pass.

        Args:
            points: (B, N, 3) LiDAR point cloud [x, y, z]

        Returns:
            dict with keys:
                c_real, c_imag: (B, K) complex coefficients
                kx, ky: (K,) fixed grid wavenumbers
                omega: (K,) dispersion frequencies
                A: (B, K) amplitudes = sqrt(cr^2 + ci^2)
                phi: (B, K) phases = atan2(ci, cr)
        """
        B, N, _ = points.shape
        cfg = self.cfg
        K = cfg.k_grid_n ** 2

        # Encode points
        h = self.point_embed(points)        # (B, N, D)
        for isab in self.encoder:
            h = isab(h)                     # (B, N, D)

        # Pool to global representation
        g = self.pma(h).squeeze(1)          # (B, D)

        # Decode spectral coefficients
        raw = self.decoder(g)               # (B, 2K)
        c_real = raw[:, :K]                 # (B, K)
        c_imag = raw[:, K:]                 # (B, K)

        # Derived quantities
        A = torch.sqrt(c_real**2 + c_imag**2 + 1e-12)
        phi = torch.atan2(c_imag, c_real)

        return {
            'c_real': c_real, 'c_imag': c_imag,
            'kx': self.kx_grid, 'ky': self.ky_grid,
            'omega': self.omega_grid,
            'A': A, 'phi': phi,
        }

    def reconstruct(self, params: dict, X: torch.Tensor,
                    Y: torch.Tensor, t: float = 0.0) -> torch.Tensor:
        """
        Reconstruct wave field from spectral coefficients.

        eta(x,y,t) = sum_j [cr_j * cos(kx_j*x + ky_j*y - w_j*t)
                           - ci_j * sin(kx_j*x + ky_j*y - w_j*t)]

        This is LINEAR in (cr, ci) — convex loss landscape.
        """
        cr = params['c_real']       # (B, K)
        ci = params['c_imag']
        kx = params['kx']           # (K,)
        ky = params['ky']
        omega = params['omega']

        # phase: (B, K, M)
        phase = (kx.unsqueeze(0).unsqueeze(-1) * X.unsqueeze(1)
                 + ky.unsqueeze(0).unsqueeze(-1) * Y.unsqueeze(1)
                 - omega.unsqueeze(0).unsqueeze(-1) * t)

        # Linear reconstruction: (B, M)
        eta = (cr.unsqueeze(-1) * torch.cos(phase)
               - ci.unsqueeze(-1) * torch.sin(phase)).sum(dim=1)
        return eta

    def top_components(self, params: dict, n_top: int = 5) -> dict:
        """Extract top-N components by amplitude for WaveSplat init."""
        A = params['A']  # (B, K)
        B = A.shape[0]

        _, idx = A.topk(n_top, dim=1)  # (B, n_top)

        kx = params['kx'][None, :].expand(B, -1)
        ky = params['ky'][None, :].expand(B, -1)
        omega = params['omega'][None, :].expand(B, -1)

        return {
            'kx': torch.gather(kx, 1, idx),
            'ky': torch.gather(ky, 1, idx),
            'A': torch.gather(params['A'], 1, idx),
            'phi': torch.gather(params['phi'], 1, idx),
            'omega': torch.gather(omega, 1, idx),
        }


class DCSpecNetLoss(nn.Module):
    """
    Training loss for DCSpecNet.

    Reconstruction MSE + L1 sparsity on coefficients.
    """

    def __init__(self, lambda_sparse: float = 0.001,
                 lambda_smooth: float = 0.0):
        super().__init__()
        self.lambda_sparse = lambda_sparse
        self.lambda_smooth = lambda_smooth

    def forward(self, params: dict, points: torch.Tensor,
                t: float = 0.0, epoch: int = 0,
                n_valid: torch.Tensor = None) -> Tuple[torch.Tensor, dict]:
        B, N, _ = points.shape
        x = points[..., 0]     # (B, N)
        y = points[..., 1]
        z = points[..., 2]     # observed elevation

        cr = params['c_real']   # (B, K)
        ci = params['c_imag']
        kx = params['kx']       # (K,)
        ky = params['ky']
        omega = params['omega']

        # phase: (B, K, N)
        phase = (kx.unsqueeze(0).unsqueeze(-1) * x.unsqueeze(1)
                 + ky.unsqueeze(0).unsqueeze(-1) * y.unsqueeze(1)
                 - omega.unsqueeze(0).unsqueeze(-1) * t)

        eta_pred = (cr.unsqueeze(-1) * torch.cos(phase)
                    - ci.unsqueeze(-1) * torch.sin(phase)).sum(dim=1)

        # 1. Reconstruction MSE (masked if n_valid provided)
        if n_valid is not None:
            mask = torch.arange(N, device=points.device).unsqueeze(0) < n_valid.unsqueeze(1)
            diff_sq = (eta_pred - z) ** 2
            loss_recon = (diff_sq * mask).sum() / mask.sum().clamp(min=1)
        else:
            loss_recon = F.mse_loss(eta_pred, z)

        # 2. L1 sparsity on coefficients (encourage few active modes)
        loss_sparse = params['A'].mean()

        total = loss_recon + self.lambda_sparse * loss_sparse

        n_active = (params['A'] > 0.01).float().sum(dim=-1).mean().item()

        return total, {
            'recon': loss_recon.item(),
            'sparse': loss_sparse.item(),
            'total': total.item(),
            'n_active': n_active,
        }
