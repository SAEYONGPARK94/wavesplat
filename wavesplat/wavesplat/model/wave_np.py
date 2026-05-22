"""
wave_np.py
Neural Process for ocean wave field estimation from sparse LiDAR.

Architecture (adapted from Gridded TNP, Ashman et al. 2024):
  Context {(x,y,z)} → CrossAttention → Grid Pseudo-Tokens → SelfAttention → CrossAttention → Targets

Input:  sparse LiDAR points (x, y, elevation)
Output: Normal distribution over elevation at arbitrary query points

Key design decisions:
  - Per-frame prediction (no temporal dimension — each frame is one task)
  - 2D pseudo-token grid matching spatial domain
  - Fourier positional encoding for sub-wavelength resolution
  - Heteroscedastic Gaussian output (mean + variance)

v2 additions:
  - Dispersion spectral consistency loss (physics constraint)
  - Multi-frame temporal context via frame stacking
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

G = 9.81  # gravitational acceleration [m/s²]


class FourierEncoding(nn.Module):
    """Fourier positional encoding for spatial coordinates.

    Maps (x, y) to [x, y, sin(f1*x), cos(f1*x), sin(f1*y), cos(f1*y), ...]
    with log-spaced frequencies.
    """

    def __init__(self, n_freq=8, min_freq=0.1, max_freq=5.0):
        super().__init__()
        freqs = torch.exp(torch.linspace(
            math.log(min_freq), math.log(max_freq), n_freq
        ))
        self.register_buffer('freqs', freqs)
        self.out_dim = 2 + 4 * n_freq  # x, y + sin/cos for x and y

    def forward(self, xy):
        """xy: [..., 2] -> [..., out_dim]"""
        x, y = xy[..., 0:1], xy[..., 1:2]
        enc = [xy]
        for f in self.freqs:
            enc.extend([torch.sin(f * x), torch.cos(f * x),
                        torch.sin(f * y), torch.cos(f * y)])
        return torch.cat(enc, dim=-1)


class CrossAttention(nn.Module):
    """Multi-head cross-attention: queries attend to keys/values."""

    def __init__(self, dim, n_heads=4, dropout=0.0):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.scale = self.head_dim ** -0.5

        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_k = nn.Linear(dim, dim, bias=False)
        self.to_v = nn.Linear(dim, dim, bias=False)
        self.to_out = nn.Linear(dim, dim)
        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, kv):
        """q: [B, Nq, D], kv: [B, Nkv, D] -> [B, Nq, D]"""
        B, Nq, D = q.shape
        q_normed = self.norm_q(q)
        kv_normed = self.norm_kv(kv)

        Q = self.to_q(q_normed).view(B, Nq, self.n_heads, self.head_dim).transpose(1, 2)
        K = self.to_k(kv_normed).view(B, -1, self.n_heads, self.head_dim).transpose(1, 2)
        V = self.to_v(kv_normed).view(B, -1, self.n_heads, self.head_dim).transpose(1, 2)

        attn = (Q @ K.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = (attn @ V).transpose(1, 2).contiguous().view(B, Nq, D)
        return q + self.to_out(out)


class SelfAttention(nn.Module):
    """Multi-head self-attention with FFN."""

    def __init__(self, dim, n_heads=4, ff_mult=2, dropout=0.0):
        super().__init__()
        self.attn = CrossAttention(dim, n_heads, dropout)
        self.norm = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, dim * ff_mult),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * ff_mult, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        """x: [B, N, D] -> [B, N, D]"""
        x = self.attn(x, x)
        x = x + self.ff(self.norm(x))
        return x


class WaveNP(nn.Module):
    """Neural Process for ocean wave field estimation.

    Architecture:
      1. Fourier encoding of spatial coordinates
      2. MLP encoder for context (obs) and target (query) tokens
      3. Cross-attention: grid pseudo-tokens ← context tokens
      4. Self-attention layers on grid pseudo-tokens
      5. Cross-attention: target tokens ← grid pseudo-tokens
      6. MLP decoder → mean + log_var

    Args:
        embed_dim: hidden dimension
        n_heads: attention heads
        n_layers: self-attention layers on grid
        grid_nx, grid_ny: pseudo-token grid resolution
        x_range, y_range: spatial domain bounds
        n_freq: Fourier encoding frequencies
    """

    def __init__(self, embed_dim=128, n_heads=4, n_layers=4,
                 grid_nx=16, grid_ny=12,
                 x_range=(10.0, 250.0), y_range=(-100.0, 100.0),
                 n_freq=8, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.grid_nx = grid_nx
        self.grid_ny = grid_ny

        # Fourier positional encoding
        self.pos_enc = FourierEncoding(n_freq=n_freq)
        pos_dim = self.pos_enc.out_dim

        # Context encoder: (pos_encoding, elevation, indicator) -> embed_dim
        self.context_enc = nn.Sequential(
            nn.Linear(pos_dim + 2, embed_dim),  # +1 elevation, +1 indicator
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )

        # Target encoder: (pos_encoding, indicator) -> embed_dim
        self.target_enc = nn.Sequential(
            nn.Linear(pos_dim + 1, embed_dim),  # +1 indicator (no elevation)
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )

        # Grid pseudo-tokens (learnable)
        gx = torch.linspace(x_range[0], x_range[1], grid_nx)
        gy = torch.linspace(y_range[0], y_range[1], grid_ny)
        grid_xy = torch.stack(torch.meshgrid(gx, gy, indexing='ij'), dim=-1)
        grid_xy = grid_xy.reshape(-1, 2)  # [M, 2]
        self.register_buffer('grid_xy', grid_xy)
        self.grid_tokens = nn.Parameter(torch.randn(grid_nx * grid_ny, embed_dim) * 0.02)

        # Grid encoder: cross-attention from grid ← context
        self.grid_encoder = CrossAttention(embed_dim, n_heads, dropout)

        # Grid processor: self-attention layers
        self.processor = nn.ModuleList([
            SelfAttention(embed_dim, n_heads, dropout=dropout)
            for _ in range(n_layers)
        ])

        # Target decoder: cross-attention from target ← grid
        self.target_decoder = CrossAttention(embed_dim, n_heads, dropout)

        # Output: embed_dim -> mean + log_var
        self.z_decoder = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Linear(embed_dim // 2, 2),  # mean, log_var
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, xc, yc, xt):
        """
        Args:
            xc: [B, Nc, 2] context spatial coordinates (x, y)
            yc: [B, Nc, 1] context observations (sea surface elevation)
            xt: [B, Nt, 2] target spatial coordinates (x, y)

        Returns:
            Normal distribution with shape [B, Nt, 1]
        """
        B = xc.shape[0]

        # Encode positions
        pc = self.pos_enc(xc)  # [B, Nc, pos_dim]
        pt = self.pos_enc(xt)  # [B, Nt, pos_dim]

        # Context tokens: [pos_enc, elevation, indicator=0]
        ctx_input = torch.cat([pc, yc, torch.zeros_like(yc)], dim=-1)
        ctx_tokens = self.context_enc(ctx_input)  # [B, Nc, D]

        # Target tokens: [pos_enc, indicator=1]
        tgt_indicator = torch.ones(B, xt.shape[1], 1, device=xt.device)
        tgt_input = torch.cat([pt, tgt_indicator], dim=-1)
        tgt_tokens = self.target_enc(tgt_input)  # [B, Nt, D]

        # Grid pseudo-tokens (broadcast to batch)
        grid_tokens = self.grid_tokens.unsqueeze(0).expand(B, -1, -1)  # [B, M, D]

        # Encode grid positions for the grid tokens
        pg = self.pos_enc(self.grid_xy.unsqueeze(0).expand(B, -1, -1))
        grid_pos_tokens = self.context_enc(
            torch.cat([pg, torch.zeros(B, pg.shape[1], 2, device=xc.device)], dim=-1)
        )
        grid_tokens = grid_tokens + grid_pos_tokens

        # Grid encoder: grid ← context via cross-attention
        grid_tokens = self.grid_encoder(grid_tokens, ctx_tokens)

        # Process grid with self-attention
        for layer in self.processor:
            grid_tokens = layer(grid_tokens)

        # Target decoder: target ← grid via cross-attention
        tgt_tokens = self.target_decoder(tgt_tokens, grid_tokens)

        # Decode to mean + log_var
        out = self.z_decoder(tgt_tokens)  # [B, Nt, 2]
        mean = out[..., 0:1]
        log_var = out[..., 1:2].clamp(-10, 5)
        std = (0.5 * log_var).exp()

        return Normal(mean, std + 1e-6)

    def predict(self, xc, yc, xt):
        """Predict mean field (no gradient)."""
        with torch.no_grad():
            dist = self.forward(xc, yc, xt)
            return dist.mean

    @property
    def n_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ─── Dispersion Spectral Consistency Loss ─────────────────────────────────────

def dispersion_spectral_loss(pred_field_2d, dx, dy, depth=50.0,
                             k_min=0.02, k_max=0.5):
    """Penalize spectral energy outside the physically plausible wavenumber band.

    Ocean surface waves satisfy the dispersion relation ω = √(g|k|tanh(|k|d)).
    For a given sea state, energy concentrates in a wavenumber band [k_min, k_max].
    This loss penalizes predicted fields that contain significant spectral energy
    outside this band, enforcing physical consistency without requiring time info.

    Args:
        pred_field_2d: [B, Nx, Ny] predicted wave elevation field
        dx: grid spacing in x [m]
        dy: grid spacing in y [m]
        depth: water depth [m]
        k_min: minimum plausible wavenumber [rad/m] (from longest wave)
        k_max: maximum plausible wavenumber [rad/m] (from shortest wave)

    Returns:
        scalar loss: fraction of energy outside the plausible band
    """
    B, Nx, Ny = pred_field_2d.shape

    # 2D FFT
    F_pred = torch.fft.fft2(pred_field_2d)
    power = (F_pred.real ** 2 + F_pred.imag ** 2)  # [B, Nx, Ny]

    # Wavenumber grids
    kx_freq = torch.fft.fftfreq(Nx, d=dx, device=pred_field_2d.device) * 2 * math.pi
    ky_freq = torch.fft.fftfreq(Ny, d=dy, device=pred_field_2d.device) * 2 * math.pi
    KX, KY = torch.meshgrid(kx_freq, ky_freq, indexing='ij')
    K_mag = torch.sqrt(KX ** 2 + KY ** 2)  # [Nx, Ny]

    # Soft bandpass mask: Gaussian rolloff outside [k_min, k_max]
    # mask = 1 inside band, smooth decay outside
    sigma = 0.03  # rolloff width in rad/m
    mask_low = torch.sigmoid((K_mag - k_min) / sigma)
    mask_high = torch.sigmoid((k_max - K_mag) / sigma)
    mask = mask_low * mask_high  # [Nx, Ny], ~1 in band, ~0 outside

    # DC component (k=0) is always allowed
    mask[0, 0] = 1.0

    # Loss: fraction of energy outside the band
    total_energy = power.sum(dim=(-2, -1)) + 1e-12
    outside_energy = (power * (1.0 - mask.unsqueeze(0))).sum(dim=(-2, -1))

    return (outside_energy / total_energy).mean()


def dispersion_gradient_loss(pred_field_2d, dx, dy, depth=50.0):
    """Smoothness loss based on dispersion-implied spatial gradients.

    Penalizes excessively high spatial frequency content that exceeds
    what the dispersion relation allows for gravity waves.

    For deep water: shortest gravity wave ~ 1.7m (capillary-gravity transition).
    k_capillary ≈ 2π/0.017 ≈ 370 rad/m — far above our grid resolution.
    So we use a Laplacian-based smoothness that penalizes grid-scale noise.

    Args:
        pred_field_2d: [B, Nx, Ny] predicted wave elevation field
        dx, dy: grid spacing [m]

    Returns:
        scalar loss: normalized Laplacian magnitude
    """
    # Finite difference Laplacian
    d2x = (pred_field_2d[:, 2:, 1:-1] - 2 * pred_field_2d[:, 1:-1, 1:-1]
            + pred_field_2d[:, :-2, 1:-1]) / (dx ** 2)
    d2y = (pred_field_2d[:, 1:-1, 2:] - 2 * pred_field_2d[:, 1:-1, 1:-1]
            + pred_field_2d[:, 1:-1, :-2]) / (dy ** 2)
    laplacian = d2x + d2y

    # Normalize by field variance to make it scale-invariant
    field_var = pred_field_2d.var(dim=(-2, -1), keepdim=True).clamp(min=1e-8)
    return (laplacian ** 2).mean() / field_var.mean()


# ─── Multi-frame Context Wrapper ──────────────────────────────────────────────

class WaveNPMultiFrame(nn.Module):
    """WaveNP with multi-frame temporal context.

    Stacks observations from multiple frames as context, adding a time
    encoding to distinguish frames. This gives the model temporal coherence
    information without changing the core architecture.

    Args:
        base_model: WaveNP instance
        n_context_frames: number of historical frames to use as context
    """

    def __init__(self, base_model, n_context_frames=3):
        super().__init__()
        self.model = base_model
        self.n_context_frames = n_context_frames
        # Time encoding: small MLP that modulates context tokens
        self.time_enc = nn.Sequential(
            nn.Linear(1, base_model.embed_dim),
            nn.GELU(),
            nn.Linear(base_model.embed_dim, base_model.embed_dim),
        )

    def forward(self, xc_frames, yc_frames, dt_frames, xt):
        """
        Args:
            xc_frames: list of [B, Nc_i, 2] context coordinates per frame
            yc_frames: list of [B, Nc_i, 1] context observations per frame
            dt_frames: list of [B, 1] relative time offsets (0 = current)
            xt: [B, Nt, 2] target coordinates

        Returns:
            Normal distribution with shape [B, Nt, 1]
        """
        B = xt.shape[0]
        model = self.model

        # Encode all context frames and concatenate
        all_ctx = []
        for xc, yc, dt in zip(xc_frames, yc_frames, dt_frames):
            pc = model.pos_enc(xc)
            ctx_input = torch.cat([pc, yc, torch.zeros_like(yc)], dim=-1)
            ctx_tokens = model.context_enc(ctx_input)  # [B, Nc_i, D]

            # Add time encoding
            t_emb = self.time_enc(dt.unsqueeze(-1))  # [B, 1, D]
            ctx_tokens = ctx_tokens + t_emb

            all_ctx.append(ctx_tokens)

        ctx_tokens = torch.cat(all_ctx, dim=1)  # [B, sum(Nc_i), D]

        # Target tokens
        pt = model.pos_enc(xt)
        tgt_indicator = torch.ones(B, xt.shape[1], 1, device=xt.device)
        tgt_input = torch.cat([pt, tgt_indicator], dim=-1)
        tgt_tokens = model.target_enc(tgt_input)

        # Grid pseudo-tokens
        grid_tokens = model.grid_tokens.unsqueeze(0).expand(B, -1, -1)
        pg = model.pos_enc(model.grid_xy.unsqueeze(0).expand(B, -1, -1))
        grid_pos_tokens = model.context_enc(
            torch.cat([pg, torch.zeros(B, pg.shape[1], 2, device=xt.device)], dim=-1)
        )
        grid_tokens = grid_tokens + grid_pos_tokens

        # Grid ← context
        grid_tokens = model.grid_encoder(grid_tokens, ctx_tokens)
        for layer in model.processor:
            grid_tokens = layer(grid_tokens)

        # Target ← grid
        tgt_tokens = model.target_decoder(tgt_tokens, grid_tokens)

        # Decode
        out = model.z_decoder(tgt_tokens)
        mean = out[..., 0:1]
        log_var = out[..., 1:2].clamp(-10, 5)
        std = (0.5 * log_var).exp()

        return Normal(mean, std + 1e-6)

    def predict(self, xc_frames, yc_frames, dt_frames, xt):
        with torch.no_grad():
            dist = self.forward(xc_frames, yc_frames, dt_frames, xt)
            return dist.mean

    @property
    def n_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
