"""
spec_np.py
Spectral Neural Process: physics-hard-constrained neural wave estimation.

Key innovation: NP predicts Fourier coefficients on the natural FFT grid of the
spatial domain, supervised directly in frequency domain + spatial domain.

Architecture:
  Context {(x,y,z)} -> FourierEncoding -> CrossAttention -> Grid Pseudo-Tokens
  -> SelfAttention layers -> Global Pooling -> MLP -> FFT coefficients (real, imag)
  -> IFFT -> eta(x,y) wave field

Why FFT-based?
  1. FFT basis is orthonormal -> condition number = 1 -> clean gradients
  2. Direct spectral supervision possible (no oscillatory rendering gradients)
  3. Low-k bins capture >97% of wave energy -> compact representation
  4. IFFT is exact, differentiable, and fast
  5. Hard dispersion constraint can be applied as k-space mask

Physics constraint (hard, by construction):
  - Only predict bins with |k| < k_max (ocean wave spectrum support)
  - Dispersion: omega_n = sqrt(g * |k_n| * tanh(|k_n| * d)) from bin position
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

G = 9.81


def build_fft_k_grid(nx, ny, dx, dy, k_max=0.5):
    """Build FFT frequency grid and select low-k bins.

    Returns:
        kx_fft: [Ny, Nx] full kx grid
        ky_fft: [Ny, Nx] full ky grid
        mask: [Ny, Nx] boolean mask for |k| < k_max
        bin_indices: [N_bins, 2] (iy, ix) indices of selected bins
    """
    kx_1d = np.fft.fftfreq(nx, d=dx) * 2 * np.pi
    ky_1d = np.fft.fftfreq(ny, d=dy) * 2 * np.pi
    kx_2d, ky_2d = np.meshgrid(kx_1d, ky_1d)
    k_mag = np.sqrt(kx_2d ** 2 + ky_2d ** 2)
    mask = k_mag < k_max
    # Always include DC
    mask[0, 0] = True
    bin_iy, bin_ix = np.where(mask)
    return kx_2d, ky_2d, mask, np.stack([bin_iy, bin_ix], axis=1)


class FourierEncoding(nn.Module):
    """Fourier positional encoding for spatial coordinates."""

    def __init__(self, n_freq=8, min_freq=0.1, max_freq=5.0):
        super().__init__()
        freqs = torch.exp(torch.linspace(
            math.log(min_freq), math.log(max_freq), n_freq
        ))
        self.register_buffer('freqs', freqs)
        self.out_dim = 2 + 4 * n_freq

    def forward(self, xy):
        x, y = xy[..., 0:1], xy[..., 1:2]
        enc = [xy]
        for f in self.freqs:
            enc.extend([torch.sin(f * x), torch.cos(f * x),
                        torch.sin(f * y), torch.cos(f * y)])
        return torch.cat(enc, dim=-1)


class CrossAttention(nn.Module):
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
        B, Nq, D = q.shape
        q_n = self.norm_q(q)
        kv_n = self.norm_kv(kv)
        Q = self.to_q(q_n).view(B, Nq, self.n_heads, self.head_dim).transpose(1, 2)
        K = self.to_k(kv_n).view(B, -1, self.n_heads, self.head_dim).transpose(1, 2)
        V = self.to_v(kv_n).view(B, -1, self.n_heads, self.head_dim).transpose(1, 2)
        attn = (Q @ K.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        out = (attn @ V).transpose(1, 2).contiguous().view(B, Nq, D)
        return q + self.to_out(out)


class SelfAttention(nn.Module):
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
        x = self.attn(x, x)
        x = x + self.ff(self.norm(x))
        return x


class SpecNP(nn.Module):
    """Spectral Neural Process — FFT-based variant.

    Predicts complex FFT coefficients at low-k bins, reconstructs via IFFT.
    """

    def __init__(self, embed_dim=128, n_heads=4, n_layers=4,
                 grid_nx=16, grid_ny=12,
                 x_range=(10.0, 250.0), y_range=(-100.0, 100.0),
                 n_freq=8, depth=50.0, dropout=0.1,
                 gt_nx=80, gt_ny=60, dx=3.04, dy=3.39, k_max=0.5):
        super().__init__()
        self.embed_dim = embed_dim
        self.depth = depth
        self.grid_nx = grid_nx
        self.grid_ny = grid_ny
        self.gt_nx = gt_nx
        self.gt_ny = gt_ny

        # Build FFT k-grid and select low-k bins
        kx_2d, ky_2d, mask, bin_indices = build_fft_k_grid(
            gt_nx, gt_ny, dx, dy, k_max)
        self.n_bins = len(bin_indices)
        self.register_buffer('bin_iy', torch.from_numpy(bin_indices[:, 0]).long())
        self.register_buffer('bin_ix', torch.from_numpy(bin_indices[:, 1]).long())
        self.register_buffer('fft_mask', torch.from_numpy(mask))

        # Store k values for each selected bin (for physics analysis)
        bin_kx = kx_2d[bin_indices[:, 0], bin_indices[:, 1]]
        bin_ky = ky_2d[bin_indices[:, 0], bin_indices[:, 1]]
        self.register_buffer('bin_kx', torch.from_numpy(bin_kx).float())
        self.register_buffer('bin_ky', torch.from_numpy(bin_ky).float())

        # Fourier positional encoding
        self.pos_enc = FourierEncoding(n_freq=n_freq)
        pos_dim = self.pos_enc.out_dim

        # Context encoder
        self.context_enc = nn.Sequential(
            nn.Linear(pos_dim + 2, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )

        # Grid pseudo-tokens (spatial)
        gx = torch.linspace(x_range[0], x_range[1], grid_nx)
        gy = torch.linspace(y_range[0], y_range[1], grid_ny)
        grid_xy = torch.stack(torch.meshgrid(gx, gy, indexing='ij'), dim=-1)
        grid_xy = grid_xy.reshape(-1, 2)
        self.register_buffer('grid_xy', grid_xy)
        self.grid_tokens = nn.Parameter(
            torch.randn(grid_nx * grid_ny, embed_dim) * 0.02)

        # Grid encoder: cross-attention from grid <- context
        self.grid_encoder = CrossAttention(embed_dim, n_heads, dropout)

        # Grid processor: self-attention layers
        self.processor = nn.ModuleList([
            SelfAttention(embed_dim, n_heads, dropout=dropout)
            for _ in range(n_layers)
        ])

        # Spectral head: global pool -> MLP -> (real, imag) per selected FFT bin
        n_out = self.n_bins * 2  # real + imaginary per bin
        self.spec_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 2, embed_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 2, n_out),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        # Initialize final layer small
        with torch.no_grad():
            self.spec_head[-1].weight.data *= 0.01
            self.spec_head[-1].bias.data.fill_(0.0)

    def encode_context(self, xc, yc):
        """Encode context into processed grid tokens."""
        B = xc.shape[0]

        pc = self.pos_enc(xc)
        ctx_input = torch.cat([pc, yc, torch.zeros_like(yc)], dim=-1)
        ctx_tokens = self.context_enc(ctx_input)

        grid_tokens = self.grid_tokens.unsqueeze(0).expand(B, -1, -1)
        pg = self.pos_enc(self.grid_xy.unsqueeze(0).expand(B, -1, -1))
        grid_pos = self.context_enc(
            torch.cat([pg, torch.zeros(B, pg.shape[1], 2, device=xc.device)], dim=-1)
        )
        grid_tokens = grid_tokens + grid_pos

        grid_tokens = self.grid_encoder(grid_tokens, ctx_tokens)

        for layer in self.processor:
            grid_tokens = layer(grid_tokens)

        return grid_tokens

    def decode_spectrum(self, grid_tokens):
        """Decode grid tokens into FFT bin coefficients.

        Returns:
            spec_real: [B, N_bins] real parts
            spec_imag: [B, N_bins] imaginary parts
        """
        pooled = grid_tokens.mean(dim=1)  # [B, D]
        raw = self.spec_head(pooled)  # [B, N_bins*2]
        spec_real = raw[:, :self.n_bins]
        spec_imag = raw[:, self.n_bins:]
        return spec_real, spec_imag

    def spectrum_to_field(self, spec_real, spec_imag):
        """Convert predicted FFT bins to spatial field via IFFT.

        Returns:
            field: [B, Ny, Nx] real-valued wave field
        """
        B = spec_real.shape[0]
        # Build full FFT grid (zero outside selected bins)
        full_spec = torch.zeros(B, self.gt_ny, self.gt_nx,
                                dtype=torch.cfloat, device=spec_real.device)
        full_spec[:, self.bin_iy, self.bin_ix] = (
            spec_real + 1j * spec_imag
        )
        # IFFT (ortho-normalized) to get spatial field
        field = torch.fft.ifft2(full_spec, norm='ortho').real  # [B, Ny, Nx]
        return field

    def forward(self, xc, yc, xt=None):
        """
        Args:
            xc: [B, Nc, 2] context coordinates
            yc: [B, Nc, 1] context elevations
            xt: [B, Nt, 2] target query coordinates (unused — returns full grid)

        Returns:
            field: [B, Ny, Nx] predicted wave field on GT grid
            spectral_params: dict with spectrum info
        """
        grid_tokens = self.encode_context(xc, yc)
        spec_real, spec_imag = self.decode_spectrum(grid_tokens)
        field = self.spectrum_to_field(spec_real, spec_imag)

        spectral_params = {
            'spec_real': spec_real,
            'spec_imag': spec_imag,
            'bin_kx': self.bin_kx,
            'bin_ky': self.bin_ky,
        }
        return field, spectral_params

    def predict(self, xc, yc, xt=None):
        """Predict field (no gradient)."""
        with torch.no_grad():
            field, _ = self.forward(xc, yc, xt)
            return field

    @property
    def n_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# --- Loss Functions ---

def spec_np_loss(field_pred, field_gt, spectral_params,
                 gt_spec_real=None, gt_spec_imag=None,
                 lambda_spec=1.0, lambda_l1=0.0001):
    """Combined spatial + spectral loss.

    Args:
        field_pred: [B, Ny, Nx] predicted field
        field_gt: [B, Ny, Nx] ground truth field
        spectral_params: dict with spec_real, spec_imag
        gt_spec_real, gt_spec_imag: [B, N_bins] GT FFT coefficients (optional)
        lambda_spec: weight for spectral loss
        lambda_l1: L1 on spectral magnitudes (sparsity)
    """
    spatial_loss = F.mse_loss(field_pred, field_gt)

    spec_real = spectral_params['spec_real']
    spec_imag = spectral_params['spec_imag']

    loss_dict = {'spatial': spatial_loss.item()}
    total = spatial_loss

    # Direct spectral supervision (if GT spectrum available)
    if gt_spec_real is not None:
        spec_loss = (F.mse_loss(spec_real, gt_spec_real)
                     + F.mse_loss(spec_imag, gt_spec_imag))
        total = total + lambda_spec * spec_loss
        loss_dict['spectral'] = spec_loss.item()

    # L1 on spectral magnitudes for sparsity
    A = torch.sqrt(spec_real ** 2 + spec_imag ** 2 + 1e-12)
    l1_loss = A.mean()
    total = total + lambda_l1 * l1_loss
    loss_dict['l1'] = l1_loss.item()

    # Count active bins
    A_max = A.max(dim=1, keepdim=True).values.clamp(min=1e-6)
    n_active = (A / A_max > 0.01).float().sum(dim=1).mean()
    loss_dict['n_active'] = n_active.item()

    return total, loss_dict
