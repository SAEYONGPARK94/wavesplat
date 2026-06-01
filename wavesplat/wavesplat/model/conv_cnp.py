"""
conv_cnp.py
Convolutional Conditional Neural Process for ocean wave field estimation.

Architecture (adapted from Gordon et al. 2020, "Convolutional CNPs"):
  Sparse LiDAR {(x,y,z)} → SetConv (scatter to grid) → UNet → grid field → interpolate to targets

Key advantages over attention-based NP (wave_np.py):
  - Translation equivariance: wave patterns don't depend on absolute position
  - Multi-scale processing: UNet captures both long and short waves
  - No global bottleneck: information flows through spatial convolutions
  - Density channel: model knows where observations are sparse/dense

Input:  sparse LiDAR points (x, y, elevation)
Output: predicted elevation at arbitrary query points
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SetConv(nn.Module):
    """Scatter sparse observations onto a regular grid using learned RBF kernels.

    For each grid cell, aggregates nearby observations weighted by distance:
      h_grid[i,j] = sum_n w(||grid_xy[i,j] - obs_xy[n]||) * [z_n, 1]

    The density channel (sum of weights) tells the model where data exists.
    """

    def __init__(self, init_lengthscale=10.0):
        super().__init__()
        # Learnable log-lengthscale for RBF kernel
        self.log_lengthscale = nn.Parameter(
            torch.tensor(math.log(init_lengthscale)))

    def forward(self, obs_xy, obs_z, grid_xy):
        """
        Args:
            obs_xy: [B, N, 2] observation coordinates
            obs_z:  [B, N, 1] observation values
            grid_xy: [Gx*Gy, 2] grid coordinates (shared across batch)

        Returns:
            grid_features: [B, 2, Gx, Gy] - (weighted_z, density)
        """
        B, N, _ = obs_xy.shape
        G = grid_xy.shape[0]

        lengthscale = self.log_lengthscale.exp()

        # Pairwise squared distances: [B, G, N]
        # grid_xy: [G, 2] -> [1, G, 1, 2]
        # obs_xy:  [B, N, 2] -> [B, 1, N, 2]
        diff = grid_xy.unsqueeze(0).unsqueeze(2) - obs_xy.unsqueeze(1)  # [B, G, N, 2]
        sq_dist = (diff ** 2).sum(-1)  # [B, G, N]

        # RBF weights
        weights = torch.exp(-0.5 * sq_dist / (lengthscale ** 2))  # [B, G, N]

        # Weighted aggregation
        density = weights.sum(-1, keepdim=True)  # [B, G, 1]
        weighted_z = (weights.unsqueeze(-1) * obs_z.unsqueeze(1)).sum(2)  # [B, G, 1]

        # Normalize by density (avoid div-by-zero)
        signal = weighted_z / (density + 1e-8)

        # Stack: [signal, density]
        features = torch.cat([signal, density], dim=-1)  # [B, G, 2]

        return features


class ConvBlock(nn.Module):
    """Conv2D + BatchNorm + GELU."""

    def __init__(self, in_ch, out_ch, kernel_size=3, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, padding=padding)
        self.bn = nn.BatchNorm2d(out_ch)

    def forward(self, x):
        return F.gelu(self.bn(self.conv(x)))


class ResBlock(nn.Module):
    """Residual block with two convolutions."""

    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        out = F.gelu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.gelu(out + residual)


class UNet(nn.Module):
    """Simple UNet with skip connections for multi-scale processing.

    Encoder: downsample 3 times (stride 2)
    Decoder: upsample 3 times with skip connections
    """

    def __init__(self, in_ch=2, base_ch=64, out_ch=1):
        super().__init__()

        # Encoder
        self.enc1 = nn.Sequential(ConvBlock(in_ch, base_ch), ResBlock(base_ch))
        self.enc2 = nn.Sequential(ConvBlock(base_ch, base_ch * 2, 3, 1),
                                   ResBlock(base_ch * 2))
        self.enc3 = nn.Sequential(ConvBlock(base_ch * 2, base_ch * 4, 3, 1),
                                   ResBlock(base_ch * 4))

        # Bottleneck
        self.bottleneck = nn.Sequential(
            ConvBlock(base_ch * 4, base_ch * 4, 3, 1),
            ResBlock(base_ch * 4),
        )

        # Decoder (with skip connections)
        self.up3 = nn.ConvTranspose2d(base_ch * 4, base_ch * 4, 2, stride=2)
        self.dec3 = nn.Sequential(ConvBlock(base_ch * 8, base_ch * 2, 3, 1),
                                   ResBlock(base_ch * 2))

        self.up2 = nn.ConvTranspose2d(base_ch * 2, base_ch * 2, 2, stride=2)
        self.dec2 = nn.Sequential(ConvBlock(base_ch * 4, base_ch, 3, 1),
                                   ResBlock(base_ch))

        self.up1 = nn.ConvTranspose2d(base_ch, base_ch, 2, stride=2)
        self.dec1 = nn.Sequential(ConvBlock(base_ch * 2, base_ch, 3, 1),
                                   ResBlock(base_ch))

        # Output
        self.out_conv = nn.Conv2d(base_ch, out_ch, 1)

        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        """x: [B, in_ch, H, W] -> [B, out_ch, H, W]"""
        # Encoder
        e1 = self.enc1(x)       # [B, C, H, W]
        e2 = self.enc2(self.pool(e1))  # [B, 2C, H/2, W/2]
        e3 = self.enc3(self.pool(e2))  # [B, 4C, H/4, W/4]

        # Bottleneck
        b = self.bottleneck(self.pool(e3))  # [B, 4C, H/8, W/8]

        # Decoder with skip connections
        d3 = self.up3(b)  # [B, 4C, H/4, W/4]
        # Handle size mismatch from non-power-of-2 grids
        d3 = _match_size(d3, e3)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))

        d2 = self.up2(d3)
        d2 = _match_size(d2, e2)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))

        d1 = self.up1(d2)
        d1 = _match_size(d1, e1)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))

        return self.out_conv(d1)


def _match_size(x, target):
    """Crop or pad x to match target's spatial dimensions."""
    _, _, H, W = target.shape
    _, _, Hx, Wx = x.shape
    if Hx == H and Wx == W:
        return x
    # Crop if larger
    if Hx >= H and Wx >= W:
        return x[:, :, :H, :W]
    # Pad if smaller
    pad_h = H - Hx
    pad_w = W - Wx
    return F.pad(x, (0, pad_w, 0, pad_h))


class ConvCNP(nn.Module):
    """Convolutional Conditional Neural Process for ocean wave field estimation.

    Architecture:
      1. SetConv: scatter sparse observations to a regular grid
      2. UNet: multi-scale CNN processing (captures both long and short waves)
      3. Bilinear interpolation: grid output -> query points

    The grid resolution is independent of the GT grid resolution, allowing
    the model to operate at its own optimal resolution.

    Args:
        grid_nx, grid_ny: internal processing grid resolution
        x_range, y_range: spatial domain bounds [m]
        base_ch: UNet base channel width
        init_lengthscale: initial RBF lengthscale for SetConv [m]
    """

    def __init__(self, grid_nx=64, grid_ny=48,
                 x_range=(10.0, 250.0), y_range=(-100.0, 100.0),
                 base_ch=64, init_lengthscale=10.0):
        super().__init__()
        self.grid_nx = grid_nx
        self.grid_ny = grid_ny
        self.x_range = x_range
        self.y_range = y_range

        # Build grid coordinates
        gx = torch.linspace(x_range[0], x_range[1], grid_nx)
        gy = torch.linspace(y_range[0], y_range[1], grid_ny)
        xx, yy = torch.meshgrid(gx, gy, indexing='ij')
        grid_xy = torch.stack([xx.flatten(), yy.flatten()], dim=-1)  # [Gx*Gy, 2]
        self.register_buffer('grid_xy', grid_xy)

        # SetConv encoder
        self.setconv = SetConv(init_lengthscale=init_lengthscale)

        # UNet backbone: 2 channels in (signal + density), 1 channel out
        self.unet = UNet(in_ch=2, base_ch=base_ch, out_ch=1)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='linear')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, xc, yc, xt):
        """
        Args:
            xc: [B, Nc, 2] context spatial coordinates
            yc: [B, Nc, 1] context observations (elevation)
            xt: [B, Nt, 2] target spatial coordinates

        Returns:
            pred: [B, Nt, 1] predicted elevation at targets
        """
        B = xc.shape[0]

        # 1. SetConv: scatter observations to grid
        grid_feat = self.setconv(xc, yc, self.grid_xy)  # [B, G, 2]

        # Reshape to image: [B, 2, Gx, Gy]
        grid_img = grid_feat.permute(0, 2, 1).reshape(
            B, 2, self.grid_nx, self.grid_ny)

        # 2. UNet processing
        out_img = self.unet(grid_img)  # [B, 1, Gx, Gy]

        # 3. Interpolate to target points using grid_sample
        pred = self._interpolate_to_targets(out_img, xt)  # [B, Nt, 1]

        return pred

    def _interpolate_to_targets(self, grid_field, xt):
        """Bilinear interpolation from grid to arbitrary target points.

        Args:
            grid_field: [B, 1, Gx, Gy]
            xt: [B, Nt, 2] target coordinates in physical space

        Returns:
            [B, Nt, 1] interpolated values
        """
        B, _, Gx, Gy = grid_field.shape

        # Normalize target coordinates to [-1, 1] for grid_sample
        x_norm = 2.0 * (xt[..., 0] - self.x_range[0]) / (
            self.x_range[1] - self.x_range[0]) - 1.0
        y_norm = 2.0 * (xt[..., 1] - self.y_range[0]) / (
            self.y_range[1] - self.y_range[0]) - 1.0

        # grid_sample expects [B, Nt, 1, 2] with (x, y) in normalized coords
        # Note: grid_sample uses (x=width, y=height) convention
        # Our grid is [Gx, Gy] where Gx is height, Gy is width
        sample_grid = torch.stack([y_norm, x_norm], dim=-1).unsqueeze(2)  # [B, Nt, 1, 2]

        # Interpolate
        sampled = F.grid_sample(
            grid_field, sample_grid,
            mode='bilinear', align_corners=True, padding_mode='border'
        )  # [B, 1, Nt, 1]

        return sampled.squeeze(1).squeeze(-1).unsqueeze(-1)  # [B, Nt, 1]

    def predict(self, xc, yc, xt):
        """Predict field (no gradient)."""
        with torch.no_grad():
            return self.forward(xc, yc, xt)

    @property
    def n_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
