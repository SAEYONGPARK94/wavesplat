"""
wave_pino.py
Physics-Informed Neural Operator (FNO-based) for ocean wave field estimation.

Architecture (WavePINO):
  Sparse LiDAR → SetConv → Lift → N × FNOBlock → Output head → bilinear interp

Key difference from ConvCNP:
  - FNO operates globally in Fourier space (k-space) instead of locally in spatial
    domain. Ocean waves are superpositions of sinusoids → k-space is the natural
    domain.
  - Spectral physics loss: penalizes energy at wavenumbers outside the physical
    ocean wave range defined by dispersion ω² = g|k|tanh(|k|d).

References:
  Li et al. (2021) "Fourier Neural Operator for Parametric PDEs", ICLR 2021.
  Raissi et al. (2019) "Physics-informed neural networks", JCP.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

G = 9.81  # gravitational acceleration [m/s²]


class SpectralConv2d(nn.Module):
    """2D truncated spectral convolution — the core FNO operation.

    Computes a global convolution in Fourier space using a limited set of
    low-frequency modes. For a wave field (sum of sinusoids), this is the
    natural mixing operation: each Fourier mode is a learnable combination
    of input spectral components.

    Args:
        in_ch:   input channel count
        out_ch:  output channel count
        modes_h: number of Fourier modes retained in H (x) dimension
                 (top + bottom, total 2*modes_h rows in the FFT)
        modes_w: number of Fourier modes retained in W//2+1 (y, rfft) dimension
    """

    def __init__(self, in_ch: int, out_ch: int, modes_h: int, modes_w: int):
        super().__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.modes_h = modes_h
        self.modes_w = modes_w

        # Two weight matrices: top (positive kx) and bottom (negative kx) blocks
        # Stored as real [2] to allow view_as_complex; shape [..., 2] = [re, im]
        scale = 1.0 / (in_ch * out_ch)
        self.w_top = nn.Parameter(
            scale * torch.randn(in_ch, out_ch, modes_h, modes_w, 2))
        self.w_bot = nn.Parameter(
            scale * torch.randn(in_ch, out_ch, modes_h, modes_w, 2))

    @staticmethod
    def _cmul(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        """Complex einsum: [B,in,H,W] × [in,out,H,W] → [B,out,H,W]."""
        return torch.einsum('bixy,ioxy->boxy', x, w)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, in_ch, H, W]  →  [B, out_ch, H, W]"""
        B, C, H, W = x.shape

        # Real 2D FFT: [B, C, H, W//2+1] complex
        x_ft = torch.fft.rfft2(x)

        # Initialize output Fourier tensor (zeros = discard high-freq modes)
        out_ft = torch.zeros(B, self.out_ch, H, W // 2 + 1,
                             dtype=torch.cfloat, device=x.device)

        w_top = torch.view_as_complex(self.w_top.contiguous())  # [in,out,mh,mw]
        w_bot = torch.view_as_complex(self.w_bot.contiguous())

        # Low positive kx (top rows) × low ky (left cols)
        out_ft[:, :, :self.modes_h, :self.modes_w] = self._cmul(
            x_ft[:, :, :self.modes_h, :self.modes_w], w_top)

        # Low negative kx (bottom rows) × low ky
        out_ft[:, :, -self.modes_h:, :self.modes_w] = self._cmul(
            x_ft[:, :, -self.modes_h:, :self.modes_w], w_bot)

        # Inverse real FFT back to spatial domain
        return torch.fft.irfft2(out_ft, s=(H, W))


class FNOBlock(nn.Module):
    """One FNO layer: spectral global conv + local bypass + norm + activation.

    out = GELU(GroupNorm(SpectralConv(x) + W_bypass(x)))

    The bypass (1×1 conv) ensures local mixing and prevents gradient
    vanishing when spectral modes don't capture all relevant structures.
    """

    def __init__(self, channels: int, modes_h: int, modes_w: int):
        super().__init__()
        self.spectral = SpectralConv2d(channels, channels, modes_h, modes_w)
        self.bypass   = nn.Conv2d(channels, channels, 1)
        # GroupNorm: stable for small batches; use min(8, channels) groups
        self.norm = nn.GroupNorm(min(8, channels), channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.gelu(self.norm(self.spectral(x) + self.bypass(x)))


class WavePINO(nn.Module):
    """Physics-Informed Neural Operator for ocean wave estimation.

    Architecture:
      1. SetConv: scatter sparse LiDAR observations onto a regular grid
         via learned RBF aggregation (identical to ConvCNP SetConv)
      2. Lifting: 2-channel grid (signal + density) → fno_width channels
      3. FNO blocks: global k-space mixing via SpectralConv2d
      4. Output head: 2-layer CNN to predict scalar elevation field
      5. Bilinear interpolation: grid output → arbitrary query points

    Args:
        grid_nx, grid_ny: internal grid resolution
        x_range, y_range: physical domain bounds [m]
        fno_width:   channel count inside FNO layers
        fno_layers:  number of FNO blocks
        fno_modes_h: Fourier modes retained in x (H) dimension
        fno_modes_w: Fourier modes retained in y rfft (W) dimension
        init_lengthscale: SetConv RBF initial lengthscale [m]
    """

    def __init__(
        self,
        grid_nx: int = 64,
        grid_ny: int = 48,
        x_range=(10.0, 250.0),
        y_range=(-100.0, 100.0),
        fno_width: int = 16,
        fno_layers: int = 4,
        fno_modes_h: int = 16,
        fno_modes_w: int = 12,
        init_lengthscale: float = 10.0,
    ):
        super().__init__()
        self.grid_nx  = grid_nx
        self.grid_ny  = grid_ny
        self.x_range  = x_range
        self.y_range  = y_range
        self.fno_width = fno_width

        # --- SetConv grid coordinates ---
        gx = torch.linspace(x_range[0], x_range[1], grid_nx)
        gy = torch.linspace(y_range[0], y_range[1], grid_ny)
        xx, yy = torch.meshgrid(gx, gy, indexing='ij')
        # [Gx*Gy, 2]: flattened (x, y) for each grid cell
        self.register_buffer('grid_xy',
                             torch.stack([xx.flatten(), yy.flatten()], dim=-1))

        # SetConv: learnable RBF lengthscale
        self.log_lengthscale = nn.Parameter(
            torch.tensor(math.log(init_lengthscale)))

        # --- FNO backbone ---
        # Lifting: 2 (signal + density) → fno_width
        self.lifting = nn.Sequential(
            nn.Conv2d(2, fno_width, 1),
            nn.GELU(),
        )

        # Stacked FNO blocks
        self.fno_blocks = nn.ModuleList([
            FNOBlock(fno_width, fno_modes_h, fno_modes_w)
            for _ in range(fno_layers)
        ])

        # Output head: fno_width → 1 (elevation)
        self.output_head = nn.Sequential(
            nn.Conv2d(fno_width, fno_width * 2, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(fno_width * 2, 1, 1),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='linear')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    # ------------------------------------------------------------------
    # SetConv
    # ------------------------------------------------------------------
    def _setconv(
        self,
        obs_xy: torch.Tensor,   # [B, N, 2]
        obs_z:  torch.Tensor,   # [B, N, 1]
    ) -> torch.Tensor:          # [B, 2, Gx, Gy]
        """Scatter sparse observations to grid via RBF aggregation."""
        ls = self.log_lengthscale.exp()

        # Pairwise squared distances [B, G, N]
        diff    = (self.grid_xy.unsqueeze(0).unsqueeze(2)
                   - obs_xy.unsqueeze(1))          # [B, G, N, 2]
        sq_dist = (diff ** 2).sum(-1)              # [B, G, N]
        weights = torch.exp(-0.5 * sq_dist / (ls ** 2))  # [B, G, N]

        density = weights.sum(-1, keepdim=True)    # [B, G, 1]
        signal  = (weights.unsqueeze(-1) * obs_z.unsqueeze(1)).sum(2)  # [B,G,1]
        signal  = signal / (density + 1e-8)

        features = torch.cat([signal, density], dim=-1)  # [B, G, 2]
        B = obs_xy.shape[0]
        return features.permute(0, 2, 1).reshape(B, 2, self.grid_nx, self.grid_ny)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(
        self,
        xc: torch.Tensor,   # [B, Nc, 2]
        yc: torch.Tensor,   # [B, Nc, 1]
        xt: torch.Tensor,   # [B, Nt, 2]
    ) -> torch.Tensor:      # [B, Nt, 1]
        grid = self._grid_predict(xc, yc)    # [B, 1, Gx, Gy]
        return self._interp_to_targets(grid, xt)

    def predict_grid(
        self,
        xc: torch.Tensor,   # [B, Nc, 2]
        yc: torch.Tensor,   # [B, Nc, 1]
    ) -> torch.Tensor:      # [B, 1, Gx, Gy]
        """Return raw grid prediction (used for physics loss)."""
        return self._grid_predict(xc, yc)

    def _grid_predict(self, xc, yc):
        z = self.lifting(self._setconv(xc, yc))  # [B, W, Gx, Gy]
        for block in self.fno_blocks:
            z = block(z)
        return self.output_head(z)               # [B, 1, Gx, Gy]

    def _interp_to_targets(
        self,
        grid_field: torch.Tensor,   # [B, 1, Gx, Gy]
        xt: torch.Tensor,           # [B, Nt, 2]
    ) -> torch.Tensor:              # [B, Nt, 1]
        """Bilinear interpolation from grid to arbitrary target points.

        grid_field layout: dim-2 = x (H), dim-3 = y (W)
        PyTorch grid_sample: grid[...,0] → W direction (y), grid[...,1] → H (x)
        """
        x_norm = (2.0 * (xt[..., 0] - self.x_range[0])
                  / (self.x_range[1] - self.x_range[0]) - 1.0)
        y_norm = (2.0 * (xt[..., 1] - self.y_range[0])
                  / (self.y_range[1] - self.y_range[0]) - 1.0)

        # grid_sample convention: (W=y, H=x) → stack (y_norm, x_norm)
        sample_grid = torch.stack([y_norm, x_norm], dim=-1).unsqueeze(2)

        sampled = F.grid_sample(
            grid_field, sample_grid,
            mode='bilinear', align_corners=True, padding_mode='border')
        return sampled.squeeze(1).squeeze(-1).unsqueeze(-1)  # [B, Nt, 1]

    def predict(self, xc, yc, xt):
        """Inference-only forward (no gradient)."""
        with torch.no_grad():
            return self.forward(xc, yc, xt)

    @property
    def n_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Physics-informed spectral loss
# ---------------------------------------------------------------------------

def spectral_physics_loss(
    pred_grid: torch.Tensor,
    x_extent: float,
    y_extent: float,
    depth: float = 50.0,
    T_min: float = 4.0,
    T_max: float = 25.0,
) -> torch.Tensor:
    """Penalize spectral energy at physically implausible wavenumbers.

    Ocean wave fields should have energy concentrated in the ring
        k_min ≤ |k| ≤ k_max
    where k_min/k_max correspond to the longest/shortest physically
    occurring wave periods via the dispersion relation
        ω² = g |k| tanh(|k| d)
    (deep-water approx used for bounds: k = ω²/g).

    Energy outside this ring is penalized as a fraction of total power.
    This discourages:
      - High-frequency noise (|k| > k_max): fine-scale artifacts
      - DC offset and very-long-wave drift (|k| < k_min)

    Args:
        pred_grid: [B, 1, Gx, Gy] predicted elevation field (normalized)
        x_extent:  physical x extent of grid [m]
        y_extent:  physical y extent of grid [m]
        depth:     mean water depth [m]
        T_min:     minimum physically expected wave period [s]
        T_max:     maximum physically expected wave period [s]

    Returns:
        scalar in [0, 1]: fraction of power outside physical k range
    """
    B, _, H, W = pred_grid.shape
    field = pred_grid[:, 0]               # [B, H=Gx, W=Gy]

    # 2D real FFT: [B, H, W//2+1] complex
    fft   = torch.fft.rfft2(field)
    power = fft.real ** 2 + fft.imag ** 2  # [B, H, W//2+1]

    # Wavenumber grids [rad/m]
    # x direction (H=Gx): full fft frequencies
    kx = (2.0 * math.pi
          * torch.fft.fftfreq(H, d=x_extent / H).to(pred_grid.device))   # [H]
    # y direction (W=Gy): rfft frequencies (one-sided)
    ky = (2.0 * math.pi
          * torch.fft.rfftfreq(W, d=y_extent / W).to(pred_grid.device))  # [W//2+1]

    KX, KY = torch.meshgrid(kx, ky, indexing='ij')   # [H, W//2+1]
    k_mag  = torch.sqrt(KX ** 2 + KY ** 2 + 1e-12)  # [H, W//2+1]

    # Physical wavenumber bounds (deep-water: k = ω²/g)
    omega_max = 2.0 * math.pi / T_min   # shortest period → highest k
    omega_min = 2.0 * math.pi / T_max   # longest period  → lowest k
    k_max = omega_max ** 2 / G
    k_min = omega_min ** 2 / G

    in_range  = (k_mag >= k_min) & (k_mag <= k_max)          # [H, W//2+1]
    out_power = power * (~in_range).float().unsqueeze(0)      # [B, H, W//2+1]

    total = power.sum(dim=(-2, -1)) + 1e-8                    # [B]
    return (out_power.sum(dim=(-2, -1)) / total).mean()
