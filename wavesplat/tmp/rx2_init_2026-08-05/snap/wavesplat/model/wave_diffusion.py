"""
wave_diffusion.py
Conditional Denoising Diffusion Probabilistic Model (DDPM) for ocean wave field
estimation from sparse LiDAR observations.

Architecture:
  - Sparse LiDAR {(x,y,z)} → SetConv → condition grid [2, H, W]
  - Noisy wave field [1, H, W] + condition → UNet → denoised field
  - DDPM noise schedule for training/sampling

Key insight: Diffusion models sample from the posterior p(field | obs),
not the conditional mean E[field | obs]. This naturally preserves wave
amplitude instead of "hedging" toward zero (the MSE-optimal solution).

Reference:
  - Ho et al. 2020, "Denoising Diffusion Probabilistic Models"
  - Zhuang & Cheng 2024, "Spatially-Aware Diffusion for Field Reconstruction"
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ========== SetConv (reuse from conv_cnp.py pattern) ==========

class SetConv(nn.Module):
    """Scatter sparse observations onto a regular grid using learned RBF kernel."""

    def __init__(self, init_lengthscale=10.0):
        super().__init__()
        self.log_lengthscale = nn.Parameter(
            torch.tensor(math.log(init_lengthscale)))

    def forward(self, obs_xy, obs_z, grid_xy):
        """
        Args:
            obs_xy: [B, N, 2] observation coordinates
            obs_z:  [B, N, 1] observation values
            grid_xy: [G, 2] grid coordinates

        Returns:
            [B, G, 2] — (signal, density) per grid cell
        """
        lengthscale = self.log_lengthscale.exp()

        diff = grid_xy.unsqueeze(0).unsqueeze(2) - obs_xy.unsqueeze(1)  # [B, G, N, 2]
        sq_dist = (diff ** 2).sum(-1)  # [B, G, N]
        weights = torch.exp(-0.5 * sq_dist / (lengthscale ** 2))  # [B, G, N]

        density = weights.sum(-1, keepdim=True)  # [B, G, 1]
        weighted_z = (weights.unsqueeze(-1) * obs_z.unsqueeze(1)).sum(2)  # [B, G, 1]
        signal = weighted_z / (density + 1e-8)

        return torch.cat([signal, density], dim=-1)  # [B, G, 2]


# ========== Sinusoidal timestep embedding ==========

class SinusoidalPosEmb(nn.Module):
    """Sinusoidal positional embedding for diffusion timestep."""

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        """t: [B] integer timesteps → [B, dim] embedding."""
        device = t.device
        half = self.dim // 2
        emb = math.log(10000) / (half - 1)
        emb = torch.exp(torch.arange(half, device=device) * -emb)
        emb = t[:, None].float() * emb[None, :]
        return torch.cat([emb.sin(), emb.cos()], dim=-1)


# ========== Conditional UNet for Diffusion ==========

class ConvBlock(nn.Module):
    """Conv + GroupNorm + SiLU (no BatchNorm — important for diffusion)."""

    def __init__(self, in_ch, out_ch, time_dim=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.gn1 = nn.GroupNorm(min(8, out_ch), out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.gn2 = nn.GroupNorm(min(8, out_ch), out_ch)

        # Time conditioning via scale+shift
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_dim, out_ch * 2),
        ) if time_dim is not None else None

        # Residual projection if channels change
        self.residual = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, t_emb=None):
        h = F.silu(self.gn1(self.conv1(x)))

        # Add time embedding via scale+shift
        if self.time_mlp is not None and t_emb is not None:
            t_out = self.time_mlp(t_emb)[:, :, None, None]
            scale, shift = t_out.chunk(2, dim=1)
            h = h * (1 + scale) + shift

        h = F.silu(self.gn2(self.conv2(h)))
        return h + self.residual(x)


class SelfAttention2d(nn.Module):
    """Multi-head self-attention for 2D feature maps (bottleneck only)."""

    def __init__(self, channels, n_heads=4):
        super().__init__()
        self.n_heads = n_heads
        self.norm = nn.GroupNorm(min(8, channels), channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.proj = nn.Conv2d(channels, channels, 1)

    def forward(self, x):
        B, C, H, W = x.shape
        h = self.norm(x)
        qkv = self.qkv(h).reshape(B, 3, self.n_heads, C // self.n_heads, H * W)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]  # [B, heads, C//heads, HW]
        scale = (C // self.n_heads) ** -0.5
        attn = torch.einsum('bhci,bhcj->bhij', q, k) * scale
        attn = attn.softmax(dim=-1)
        out = torch.einsum('bhij,bhcj->bhci', attn, v)
        out = out.reshape(B, C, H, W)
        return x + self.proj(out)


class DiffusionUNet(nn.Module):
    """UNet with time conditioning for DDPM.

    Input: [B, in_ch, H, W] where in_ch = 1 (noisy field) + 2 (SetConv condition)
    Output: [B, 1, H, W] predicted noise/v/x0

    Args:
        in_ch: input channels (1 noisy + 2 cond + optional 1 self-cond)
        base_ch: base channel width
        time_dim: timestep embedding dimension
        use_attn: add self-attention at bottleneck
        attn_levels: list of levels to add attention ('bot', 'enc3', 'dec3')
    """

    def __init__(self, in_ch=3, base_ch=64, time_dim=128, use_attn=False,
                 attn_levels=None):
        super().__init__()

        # attn_levels overrides use_attn for fine-grained control
        if attn_levels is None:
            attn_levels = ['bot'] if use_attn else []

        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )

        # Encoder
        self.enc1 = ConvBlock(in_ch, base_ch, time_dim)
        self.enc2 = ConvBlock(base_ch, base_ch * 2, time_dim)
        self.enc3 = ConvBlock(base_ch * 2, base_ch * 4, time_dim)
        self.enc3_attn = SelfAttention2d(base_ch * 4) if 'enc3' in attn_levels else None

        # Bottleneck with optional self-attention
        self.bot = ConvBlock(base_ch * 4, base_ch * 4, time_dim)
        self.bot_attn = SelfAttention2d(base_ch * 4) if 'bot' in attn_levels else None

        # Decoder
        self.up3 = nn.ConvTranspose2d(base_ch * 4, base_ch * 4, 2, stride=2)
        self.dec3 = ConvBlock(base_ch * 8, base_ch * 2, time_dim)
        self.dec3_attn = SelfAttention2d(base_ch * 2) if 'dec3' in attn_levels else None

        self.up2 = nn.ConvTranspose2d(base_ch * 2, base_ch * 2, 2, stride=2)
        self.dec2 = ConvBlock(base_ch * 4, base_ch, time_dim)

        self.up1 = nn.ConvTranspose2d(base_ch, base_ch, 2, stride=2)
        self.dec1 = ConvBlock(base_ch * 2, base_ch, time_dim)

        self.out_conv = nn.Conv2d(base_ch, 1, 1)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x, t, cond):
        """
        Args:
            x: [B, 1, H, W] noisy wave field
            t: [B] integer timesteps
            cond: [B, C, H, W] condition (2 for SetConv, +1 if self-cond)

        Returns:
            [B, 1, H, W] predicted output
        """
        t_emb = self.time_mlp(t)  # [B, time_dim]

        # Concatenate noisy field with condition
        inp = torch.cat([x, cond], dim=1)  # [B, in_ch, H, W]

        # Encoder
        e1 = self.enc1(inp, t_emb)
        e2 = self.enc2(self.pool(e1), t_emb)
        e3 = self.enc3(self.pool(e2), t_emb)
        if self.enc3_attn is not None:
            e3 = self.enc3_attn(e3)

        # Bottleneck
        b = self.bot(self.pool(e3), t_emb)
        if self.bot_attn is not None:
            b = self.bot_attn(b)

        # Decoder with skip connections
        d3 = self.up3(b)
        d3 = _match_size(d3, e3)
        d3 = self.dec3(torch.cat([d3, e3], dim=1), t_emb)
        if self.dec3_attn is not None:
            d3 = self.dec3_attn(d3)

        d2 = self.up2(d3)
        d2 = _match_size(d2, e2)
        d2 = self.dec2(torch.cat([d2, e2], dim=1), t_emb)

        d1 = self.up1(d2)
        d1 = _match_size(d1, e1)
        d1 = self.dec1(torch.cat([d1, e1], dim=1), t_emb)

        return self.out_conv(d1)


def _match_size(x, target):
    """Crop or pad x to match target's spatial dimensions."""
    _, _, H, W = target.shape
    _, _, Hx, Wx = x.shape
    if Hx == H and Wx == W:
        return x
    if Hx >= H and Wx >= W:
        return x[:, :, :H, :W]
    pad_h = max(0, H - Hx)
    pad_w = max(0, W - Wx)
    return F.pad(x, (0, pad_w, 0, pad_h))


# ========== DDPM Noise Schedule ==========

class GaussianDiffusion:
    """DDPM forward/reverse process with linear beta schedule."""

    def __init__(self, n_timesteps=1000, beta_start=1e-4, beta_end=0.02,
                 device='cpu'):
        self.n_timesteps = n_timesteps
        self.device = device

        # Linear schedule
        betas = torch.linspace(beta_start, beta_end, n_timesteps,
                               dtype=torch.float32)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        self.betas = betas.to(device)
        self.alphas = alphas.to(device)
        self.alphas_cumprod = alphas_cumprod.to(device)
        self.sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod).to(device)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(
            1.0 - alphas_cumprod).to(device)

        # For sampling
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
        self.posterior_variance = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        ).to(device)

    def q_sample(self, x0, t, noise=None):
        """Forward diffusion: add noise to x0 at timestep t.

        Args:
            x0: [B, 1, H, W] clean wave field
            t: [B] integer timesteps
            noise: optional pre-generated noise

        Returns:
            x_t: [B, 1, H, W] noisy field
            noise: [B, 1, H, W] the noise that was added
        """
        if noise is None:
            noise = torch.randn_like(x0)

        sqrt_alpha = self.sqrt_alphas_cumprod[t][:, None, None, None]
        sqrt_one_minus = self.sqrt_one_minus_alphas_cumprod[t][:, None, None, None]

        x_t = sqrt_alpha * x0 + sqrt_one_minus * noise
        return x_t, noise

    @torch.no_grad()
    def p_sample(self, model, x_t, t, cond):
        """Single reverse diffusion step."""
        B = x_t.shape[0]

        # Predict noise
        pred_noise = model(x_t, t, cond)

        alpha = self.alphas[t][:, None, None, None]
        alpha_bar = self.alphas_cumprod[t][:, None, None, None]
        sqrt_one_minus = self.sqrt_one_minus_alphas_cumprod[t][:, None, None, None]

        # Predicted x0
        x0_pred = (x_t - sqrt_one_minus * pred_noise) / torch.sqrt(alpha_bar)

        # Posterior mean
        coeff1 = self.betas[t][:, None, None, None] / sqrt_one_minus
        mean = (x_t - coeff1 * pred_noise) / torch.sqrt(alpha)

        if t[0] > 0:
            var = self.posterior_variance[t][:, None, None, None]
            noise = torch.randn_like(x_t)
            return mean + torch.sqrt(var) * noise
        else:
            return mean

    @torch.no_grad()
    def sample(self, model, cond, shape):
        """Full reverse diffusion: generate wave field from noise.

        Args:
            model: DiffusionUNet
            cond: [B, 2, H, W] SetConv condition
            shape: (B, 1, H, W)

        Returns:
            [B, 1, H, W] generated wave field
        """
        device = cond.device
        x = torch.randn(shape, device=device)

        for i in reversed(range(self.n_timesteps)):
            t = torch.full((shape[0],), i, device=device, dtype=torch.long)
            x = self.p_sample(model, x, t, cond)

        return x

    @torch.no_grad()
    def sample_ddim(self, model, cond, shape, n_steps=50, guidance_scale=1.0):
        """DDIM sampling with optional classifier-free guidance.

        Args:
            model: DiffusionUNet
            cond: [B, 2, H, W]
            shape: (B, 1, H, W)
            n_steps: number of DDIM steps (default 50)
            guidance_scale: CFG scale (1.0 = no guidance, >1.0 = stronger conditioning)

        Returns:
            [B, 1, H, W] generated wave field
        """
        device = cond.device

        # Evenly spaced timestep subsequence
        step_size = self.n_timesteps // n_steps
        timesteps = list(range(0, self.n_timesteps, step_size))
        timesteps = list(reversed(timesteps))

        x = torch.randn(shape, device=device)
        uncond = torch.zeros_like(cond) if guidance_scale > 1.0 else None

        for i, t_cur in enumerate(timesteps):
            t = torch.full((shape[0],), t_cur, device=device, dtype=torch.long)

            if guidance_scale > 1.0 and uncond is not None:
                # Classifier-free guidance: combine cond and uncond predictions
                noise_cond = model(x, t, cond)
                noise_uncond = model(x, t, uncond)
                pred_noise = noise_uncond + guidance_scale * (noise_cond - noise_uncond)
            else:
                pred_noise = model(x, t, cond)

            alpha_bar = self.alphas_cumprod[t_cur]
            x0_pred = (x - torch.sqrt(1 - alpha_bar) * pred_noise) / torch.sqrt(alpha_bar)

            if i < len(timesteps) - 1:
                t_next = timesteps[i + 1]
                alpha_bar_next = self.alphas_cumprod[t_next]
                x = (torch.sqrt(alpha_bar_next) * x0_pred +
                     torch.sqrt(1 - alpha_bar_next) * pred_noise)
            else:
                x = x0_pred

        return x

    @torch.no_grad()
    def sample_ddim_guided(self, model, cond, shape, n_steps=50,
                           guidance_scale=1.0,
                           obs_mask=None, obs_values=None,
                           k_min=0.02, k_max=0.5, disp_strength=0.3):
        """DDIM sampling with observation guidance (RePaint) and dispersion projection.

        Args:
            model: DiffusionUNet
            cond: [B, 2, H, W]
            shape: (B, 1, H, W)
            n_steps: number of DDIM steps
            guidance_scale: CFG scale
            obs_mask: [B, 1, H, W] binary mask (1 where observations exist)
            obs_values: [B, 1, H, W] observed values on grid (Hs-normalized)
            k_min: min wavenumber for dispersion bandpass [rad/m]
            k_max: max wavenumber for dispersion bandpass [rad/m]
            disp_strength: strength of dispersion projection (0=off, 1=hard)

        Returns:
            [B, 1, H, W] generated wave field
        """
        device = cond.device
        B = shape[0]
        H, W = shape[2], shape[3]

        step_size = self.n_timesteps // n_steps
        timesteps = list(range(0, self.n_timesteps, step_size))
        timesteps = list(reversed(timesteps))

        x = torch.randn(shape, device=device)
        uncond = torch.zeros_like(cond) if guidance_scale > 1.0 else None

        # Pre-compute dispersion bandpass mask in frequency domain
        disp_mask = None
        if disp_strength > 0:
            disp_mask = _make_dispersion_mask(H, W, k_min, k_max, device)

        for i, t_cur in enumerate(timesteps):
            t = torch.full((B,), t_cur, device=device, dtype=torch.long)

            # --- RePaint: observation replacement ---
            if obs_mask is not None and obs_values is not None and t_cur > 0:
                alpha_bar = self.alphas_cumprod[t_cur]
                noise = torch.randn_like(x)
                obs_noisy = (torch.sqrt(alpha_bar) * obs_values +
                             torch.sqrt(1 - alpha_bar) * noise)
                x = obs_mask * obs_noisy + (1 - obs_mask) * x

            # --- Noise prediction with CFG ---
            if guidance_scale > 1.0 and uncond is not None:
                noise_cond = model(x, t, cond)
                noise_uncond = model(x, t, uncond)
                pred_noise = noise_uncond + guidance_scale * (noise_cond - noise_uncond)
            else:
                pred_noise = model(x, t, cond)

            alpha_bar = self.alphas_cumprod[t_cur]
            x0_pred = (x - torch.sqrt(1 - alpha_bar) * pred_noise) / torch.sqrt(alpha_bar)

            # --- Dispersion projection on x0_pred ---
            if disp_mask is not None and disp_strength > 0:
                x0_pred = _apply_dispersion_filter(x0_pred, disp_mask, disp_strength)

            if i < len(timesteps) - 1:
                t_next = timesteps[i + 1]
                alpha_bar_next = self.alphas_cumprod[t_next]
                x = (torch.sqrt(alpha_bar_next) * x0_pred +
                     torch.sqrt(1 - alpha_bar_next) * pred_noise)
            else:
                x = x0_pred

        return x

    def sample_ddim_dpps(self, model, cond, shape, n_steps=50,
                         guidance_scale=1.0,
                         obs_mask=None, obs_values=None,
                         k_min=0.02, k_max=0.5, disp_strength=0.3,
                         obs_guidance_scale=1.0):
        """DDIM sampling with Dispersion-Projected Posterior Sampling (DPPS).

        Combines DPS observation guidance gradient with dispersion spectral
        projection at each denoising step. No retraining required.

        Algorithm at each step:
          1. Predict x0 from noise prediction (with CFG)
          2. DPS: compute ∇_{x_t} ||z_obs - x0_pred[obs]||² and update x_t
          3. Dispersion projection: FFT bandpass on x0_pred
          4. DDIM re-noise for next step

        Args:
            model: DiffusionUNet
            cond: [B, 2, H, W]
            shape: (B, 1, H, W)
            n_steps: number of DDIM steps
            guidance_scale: CFG scale
            obs_mask: [B, 1, H, W] binary mask (1 where observations exist)
            obs_values: [B, 1, H, W] observed values on grid (Hs-normalized)
            k_min: min wavenumber for dispersion bandpass [rad/m]
            k_max: max wavenumber for dispersion bandpass [rad/m]
            disp_strength: strength of dispersion projection (0=off, 1=hard)
            obs_guidance_scale: DPS guidance scale ζ

        Returns:
            [B, 1, H, W] generated wave field
        """
        device = cond.device
        B = shape[0]
        H, W = shape[2], shape[3]

        step_size = self.n_timesteps // n_steps
        timesteps = list(range(0, self.n_timesteps, step_size))
        timesteps = list(reversed(timesteps))

        x = torch.randn(shape, device=device)
        uncond = torch.zeros_like(cond) if guidance_scale > 1.0 else None

        # Pre-compute dispersion bandpass mask
        disp_mask = None
        if disp_strength > 0:
            disp_mask = _make_dispersion_mask(H, W, k_min, k_max, device)

        has_obs = (obs_mask is not None and obs_values is not None
                   and obs_guidance_scale > 0)

        for i, t_cur in enumerate(timesteps):
            t = torch.full((B,), t_cur, device=device, dtype=torch.long)

            # --- DPS: observation guidance via gradient ---
            if has_obs and t_cur > 0:
                x.requires_grad_(True)

                with torch.enable_grad():
                    # Noise prediction with CFG
                    if guidance_scale > 1.0 and uncond is not None:
                        noise_cond = model(x, t, cond)
                        noise_uncond = model(x, t, uncond)
                        pred_noise_g = (noise_uncond +
                                        guidance_scale * (noise_cond - noise_uncond))
                    else:
                        pred_noise_g = model(x, t, cond)

                    alpha_bar = self.alphas_cumprod[t_cur]
                    x0_hat = ((x - torch.sqrt(1 - alpha_bar) * pred_noise_g)
                              / torch.sqrt(alpha_bar))

                    # Observation likelihood: ||z_obs - x0_hat[obs_locs]||²
                    diff = (obs_values - x0_hat) * obs_mask
                    obs_loss = (diff ** 2).sum()

                    # Gradient of observation loss w.r.t. x_t
                    grad = torch.autograd.grad(obs_loss, x)[0]

                x = x.detach()

                # DPS update: step toward observation consistency
                # Normalize by number of observed pixels to stabilize
                n_obs = obs_mask.sum().clamp(min=1.0)
                x = x - obs_guidance_scale * grad / n_obs

            # --- Standard noise prediction (without grad) ---
            with torch.no_grad():
                if guidance_scale > 1.0 and uncond is not None:
                    noise_cond = model(x, t, cond)
                    noise_uncond = model(x, t, uncond)
                    pred_noise = (noise_uncond +
                                  guidance_scale * (noise_cond - noise_uncond))
                else:
                    pred_noise = model(x, t, cond)

                alpha_bar = self.alphas_cumprod[t_cur]
                x0_pred = ((x - torch.sqrt(1 - alpha_bar) * pred_noise)
                           / torch.sqrt(alpha_bar))

                # --- Dispersion projection on x0_pred ---
                if disp_mask is not None and disp_strength > 0:
                    x0_pred = _apply_dispersion_filter(x0_pred, disp_mask,
                                                       disp_strength)

                if i < len(timesteps) - 1:
                    t_next = timesteps[i + 1]
                    alpha_bar_next = self.alphas_cumprod[t_next]
                    x = (torch.sqrt(alpha_bar_next) * x0_pred +
                         torch.sqrt(1 - alpha_bar_next) * pred_noise)
                else:
                    x = x0_pred

        return x


# ========== V4: Cosine schedule + v-prediction + self-conditioning + min-SNR ==========

def cosine_beta_schedule(n_timesteps, s=0.008):
    """Cosine noise schedule (Nichol & Dhariwal 2021).

    Produces a smoother distribution of noise levels than linear,
    avoiding wasted capacity at extreme noise levels.
    """
    steps = torch.arange(n_timesteps + 1, dtype=torch.float64)
    f = torch.cos((steps / n_timesteps + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = f / f[0]
    betas = 1 - alphas_cumprod[1:] / alphas_cumprod[:-1]
    return torch.clamp(betas, 0.0001, 0.999).float()


class GaussianDiffusionV4:
    """V4 diffusion: cosine schedule + v-prediction + min-SNR weighting.

    v-prediction (Salimans & Ho 2022):
        v = sqrt(alpha_bar) * noise - sqrt(1 - alpha_bar) * x0
        More stable x0 recovery: x0 = sqrt(alpha_bar) * x_t - sqrt(1-alpha_bar) * v

    Min-SNR-gamma weighting for v-prediction (Hang et al. 2023):
        weight = min(SNR, gamma) / (SNR + 1)
        For v-prediction, divides by (SNR+1) not SNR, because v-loss has
        implicit (SNR+1)^{-1} x0-weighting vs SNR^{-1} for ε-prediction.
    """

    def __init__(self, n_timesteps=1000, device='cpu', min_snr_gamma=5.0):
        self.n_timesteps = n_timesteps
        self.device = device
        self.min_snr_gamma = min_snr_gamma

        # Cosine schedule
        betas = cosine_beta_schedule(n_timesteps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        self.betas = betas.to(device)
        self.alphas = alphas.to(device)
        self.alphas_cumprod = alphas_cumprod.to(device)
        self.sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod).to(device)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(
            1.0 - alphas_cumprod).to(device)

        # SNR for min-SNR weighting (v-prediction formula)
        snr = alphas_cumprod / (1.0 - alphas_cumprod + 1e-20)
        self.snr_weights = (torch.clamp(snr, max=min_snr_gamma) / (snr + 1.0)).to(device)

        # For DDPM sampling
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
        self.posterior_variance = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        ).to(device)

    def q_sample(self, x0, t, noise=None):
        """Forward diffusion."""
        if noise is None:
            noise = torch.randn_like(x0)
        sqrt_alpha = self.sqrt_alphas_cumprod[t][:, None, None, None]
        sqrt_one_minus = self.sqrt_one_minus_alphas_cumprod[t][:, None, None, None]
        x_t = sqrt_alpha * x0 + sqrt_one_minus * noise
        return x_t, noise

    def get_v_target(self, x0, noise, t):
        """Compute v-prediction target: v = sqrt(alpha_bar) * noise - sqrt(1-alpha_bar) * x0."""
        sqrt_alpha = self.sqrt_alphas_cumprod[t][:, None, None, None]
        sqrt_one_minus = self.sqrt_one_minus_alphas_cumprod[t][:, None, None, None]
        return sqrt_alpha * noise - sqrt_one_minus * x0

    def predict_x0_from_v(self, x_t, v, t):
        """Recover x0 from v-prediction."""
        sqrt_alpha = self.sqrt_alphas_cumprod[t][:, None, None, None]
        sqrt_one_minus = self.sqrt_one_minus_alphas_cumprod[t][:, None, None, None]
        return sqrt_alpha * x_t - sqrt_one_minus * v

    def predict_noise_from_v(self, x_t, v, t):
        """Recover noise from v-prediction."""
        sqrt_alpha = self.sqrt_alphas_cumprod[t][:, None, None, None]
        sqrt_one_minus = self.sqrt_one_minus_alphas_cumprod[t][:, None, None, None]
        return sqrt_one_minus * x_t + sqrt_alpha * v

    def get_loss_weight(self, t):
        """Min-SNR-gamma loss weighting."""
        return self.snr_weights[t]

    def sample_ddim_dpps_v4(self, model, cond, shape, n_steps=50,
                            guidance_scale=1.0,
                            obs_mask=None, obs_values=None,
                            k_min=0.02, k_max=0.5, disp_strength=0.3,
                            obs_guidance_scale=1.0):
        """DPPS adapted for v-prediction (GaussianDiffusionV4).

        Same algorithm as GaussianDiffusion.sample_ddim_dpps but uses
        v-prediction to recover x0_hat for the DPS gradient.

        Args:
            model: DiffusionUNet (outputs v-prediction)
            cond: [B, 2, H, W]
            shape: (B, 1, H, W)
            n_steps: DDIM steps
            guidance_scale: CFG scale
            obs_mask: [B, 1, H, W] binary mask
            obs_values: [B, 1, H, W] observed values (Hs-normalized)
            k_min/k_max: dispersion wavenumber bounds [rad/m]
            disp_strength: dispersion projection strength
            obs_guidance_scale: DPS step size ζ

        Returns:
            [B, 1, H, W] generated wave field
        """
        device = cond.device
        B = shape[0]
        H, W = shape[2], shape[3]

        step_size = self.n_timesteps // n_steps
        timesteps = list(range(0, self.n_timesteps, step_size))
        timesteps = list(reversed(timesteps))

        x = torch.randn(shape, device=device)
        uncond = torch.zeros_like(cond) if guidance_scale > 1.0 else None

        # Pre-compute dispersion bandpass mask
        disp_mask_tensor = None
        if disp_strength > 0:
            disp_mask_tensor = _make_dispersion_mask(H, W, k_min, k_max, device)

        has_obs = (obs_mask is not None and obs_values is not None
                   and obs_guidance_scale > 0)

        for i, t_cur in enumerate(timesteps):
            t = torch.full((B,), t_cur, device=device, dtype=torch.long)
            sqrt_alpha = self.sqrt_alphas_cumprod[t_cur]
            sqrt_one_minus = self.sqrt_one_minus_alphas_cumprod[t_cur]

            # --- DPS: observation guidance via gradient (v-prediction) ---
            if has_obs and t_cur > 0:
                x.requires_grad_(True)

                with torch.enable_grad():
                    if guidance_scale > 1.0 and uncond is not None:
                        v_cond = model(x, t, cond)
                        v_uncond = model(x, t, uncond)
                        v_g = v_uncond + guidance_scale * (v_cond - v_uncond)
                    else:
                        v_g = model(x, t, cond)

                    # x0 from v-prediction: x0 = sqrt_alpha*x_t - sqrt_one_minus*v
                    x0_hat = sqrt_alpha * x - sqrt_one_minus * v_g

                    diff = (obs_values - x0_hat) * obs_mask
                    obs_loss = (diff ** 2).sum()
                    grad = torch.autograd.grad(obs_loss, x)[0]

                x = x.detach()
                n_obs = obs_mask.sum().clamp(min=1.0)
                x = x - obs_guidance_scale * grad / n_obs

            # --- Standard noise prediction (no grad) ---
            with torch.no_grad():
                if guidance_scale > 1.0 and uncond is not None:
                    v_cond = model(x, t, cond)
                    v_uncond = model(x, t, uncond)
                    v_pred = v_uncond + guidance_scale * (v_cond - v_uncond)
                else:
                    v_pred = model(x, t, cond)

                # Recover x0 and noise from v
                x0_pred = sqrt_alpha * x - sqrt_one_minus * v_pred
                pred_noise = sqrt_one_minus * x + sqrt_alpha * v_pred

                # Dispersion projection on x0_pred
                if disp_mask_tensor is not None and disp_strength > 0:
                    x0_pred = _apply_dispersion_filter(x0_pred, disp_mask_tensor,
                                                       disp_strength)

                if i < len(timesteps) - 1:
                    t_next = timesteps[i + 1]
                    alpha_bar_next = self.alphas_cumprod[t_next]
                    x = (torch.sqrt(alpha_bar_next) * x0_pred +
                         torch.sqrt(1 - alpha_bar_next) * pred_noise)
                else:
                    x = x0_pred

        return x

    @torch.no_grad()
    def sample_ddim_v4(self, model, cond, shape, n_steps=50,
                       guidance_scale=1.0, self_cond=True,
                       disp_mask=None, disp_strength=0.3):
        """DDIM sampling with v-prediction, self-conditioning, and dispersion projection.

        Args:
            model: DiffusionUNet (outputs v-prediction)
            cond: [B, 2, H, W] SetConv condition
            shape: (B, 1, H, W)
            n_steps: DDIM steps
            guidance_scale: CFG scale
            self_cond: enable self-conditioning
            disp_mask: precomputed dispersion bandpass mask [H, W]
            disp_strength: dispersion filter strength
        """
        device = cond.device
        B = shape[0]

        step_size = self.n_timesteps // n_steps
        timesteps = list(range(0, self.n_timesteps, step_size))
        timesteps = list(reversed(timesteps))

        x = torch.randn(shape, device=device)
        uncond = torch.zeros_like(cond) if guidance_scale > 1.0 else None

        # Self-conditioning: start with zeros
        x0_prev = torch.zeros(shape, device=device) if self_cond else None

        for i, t_cur in enumerate(timesteps):
            t = torch.full((B,), t_cur, device=device, dtype=torch.long)

            # Build condition (with optional self-cond channel)
            if self_cond and x0_prev is not None:
                full_cond = torch.cat([cond, x0_prev], dim=1)
                full_uncond = torch.cat([uncond, torch.zeros_like(x0_prev)], dim=1) if uncond is not None else None
            else:
                full_cond = cond
                full_uncond = uncond

            # v-prediction with CFG
            if guidance_scale > 1.0 and full_uncond is not None:
                v_cond = model(x, t, full_cond)
                v_uncond = model(x, t, full_uncond)
                v_pred = v_uncond + guidance_scale * (v_cond - v_uncond)
            else:
                v_pred = model(x, t, full_cond)

            # Recover x0 and noise from v
            x0_pred = self.predict_x0_from_v(x_t=x, v=v_pred, t=t)
            pred_noise = self.predict_noise_from_v(x_t=x, v=v_pred, t=t)

            # Dispersion projection on x0
            if disp_mask is not None and disp_strength > 0:
                x0_pred = _apply_dispersion_filter(x0_pred, disp_mask, disp_strength)

            # Update self-conditioning
            if self_cond:
                x0_prev = x0_pred.detach()

            # DDIM step
            alpha_bar = self.alphas_cumprod[t_cur]
            if i < len(timesteps) - 1:
                t_next = timesteps[i + 1]
                alpha_bar_next = self.alphas_cumprod[t_next]
                x = (torch.sqrt(alpha_bar_next) * x0_pred +
                     torch.sqrt(1 - alpha_bar_next) * pred_noise)
            else:
                x = x0_pred

        return x


def _make_dispersion_mask(H, W, k_min, k_max, device):
    """Create wavenumber bandpass mask for dispersion filtering.

    Args:
        H, W: spatial dimensions (grid_nx, grid_ny)
        k_min, k_max: wavenumber bounds [rad/m]
        device: torch device

    Returns:
        [H, W] soft bandpass mask in frequency domain
    """
    # Wavenumber grid — fft2 on [B, H, W] gives spectrum of shape [H, W]
    freq_h = torch.fft.fftfreq(H, d=1.0, device=device)  # [H] cycles/pixel
    freq_w = torch.fft.fftfreq(W, d=1.0, device=device)  # [W] cycles/pixel
    fh, fw = torch.meshgrid(freq_h, freq_w, indexing='ij')  # [H, W]
    k_mag = 2 * math.pi * torch.sqrt(fh**2 + fw**2)  # rad/pixel

    # Typical grid spacing: ~3m (240m / 80 pixels)
    dx = 3.0  # approximate
    k_physical = k_mag / dx  # rad/m

    # Soft sigmoid bandpass
    sharpness = 30.0
    low_pass = torch.sigmoid(sharpness * (k_physical - k_min))
    high_pass = torch.sigmoid(sharpness * (k_max - k_physical))
    mask = low_pass * high_pass

    # Keep DC component (mean)
    mask[0, 0] = 1.0

    return mask


def _apply_dispersion_filter(x0_pred, disp_mask, strength):
    """Apply wavenumber bandpass filter to predicted x0.

    Soft blending: x0_filtered = strength * filtered + (1-strength) * original
    """
    B = x0_pred.shape[0]
    # x0_pred is [B, 1, H, W], need FFT on last two dims
    x_2d = x0_pred[:, 0]  # [B, H, W]

    # 2D FFT
    X_freq = torch.fft.fft2(x_2d)

    # Apply mask (broadcast over batch)
    X_filtered = X_freq * disp_mask.unsqueeze(0)

    # IFFT back
    x_filtered = torch.fft.ifft2(X_filtered).real

    # Soft blend
    x_out = (1 - strength) * x_2d + strength * x_filtered

    return x_out.unsqueeze(1)


# ========== Full Model ==========

class WaveDiffusion(nn.Module):
    """Conditional DDPM for ocean wave field estimation.

    Combines:
      1. SetConv encoder: sparse LiDAR → grid condition
      2. DiffusionUNet: conditional noise prediction
      3. GaussianDiffusion: noise schedule + sampling
    """

    def __init__(self, grid_nx=80, grid_ny=60,
                 x_range=(10.0, 250.0), y_range=(-100.0, 100.0),
                 base_ch=64, init_lengthscale=10.0,
                 n_timesteps=1000, self_cond=False, use_attn=False,
                 attn_levels=None):
        super().__init__()

        self.grid_nx = grid_nx
        self.grid_ny = grid_ny
        self.x_range = x_range
        self.y_range = y_range
        self.n_timesteps = n_timesteps
        self.self_cond = self_cond

        # Grid coordinates (match GT grid: x is columns, y is rows)
        gx = torch.linspace(x_range[0], x_range[1], grid_nx)
        gy = torch.linspace(y_range[0], y_range[1], grid_ny)
        xx, yy = torch.meshgrid(gx, gy, indexing='ij')
        grid_xy = torch.stack([xx.flatten(), yy.flatten()], dim=-1)
        self.register_buffer('grid_xy', grid_xy)

        # SetConv for sparse obs → condition grid
        self.setconv = SetConv(init_lengthscale=init_lengthscale)

        # Conditional UNet: in_ch = 1 (noisy) + 2 (cond) + optional 1 (self-cond)
        in_ch = 3 + (1 if self_cond else 0)
        self.unet = DiffusionUNet(in_ch=in_ch, base_ch=base_ch, time_dim=128,
                                   use_attn=use_attn, attn_levels=attn_levels)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, nonlinearity='linear')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def get_condition(self, obs_xy, obs_z):
        """Compute SetConv condition grid from sparse observations.

        Args:
            obs_xy: [B, N, 2]
            obs_z: [B, N, 1]

        Returns:
            [B, 2, Gx, Gy] condition grid (signal + density)
        """
        B = obs_xy.shape[0]
        grid_feat = self.setconv(obs_xy, obs_z, self.grid_xy)  # [B, G, 2]
        return grid_feat.permute(0, 2, 1).reshape(B, 2, self.grid_nx, self.grid_ny)

    def forward(self, x_noisy, t, cond):
        """Predict noise. Used during training.

        Args:
            x_noisy: [B, 1, Gx, Gy] noisy wave field
            t: [B] timesteps
            cond: [B, 2, Gx, Gy] condition

        Returns:
            [B, 1, Gx, Gy] predicted noise
        """
        return self.unet(x_noisy, t, cond)

    def make_obs_grid(self, obs_xy, obs_z, radius=2):
        """Scatter sparse observations onto the model grid for RePaint guidance.

        Args:
            obs_xy: [B, N, 2] observation coordinates
            obs_z: [B, N, 1] observation values (Hs-normalized)
            radius: scatter radius in grid cells

        Returns:
            obs_mask: [B, 1, Gx, Gy] binary mask
            obs_values: [B, 1, Gx, Gy] observation values on grid
        """
        B, N, _ = obs_xy.shape
        device = obs_xy.device

        gx = torch.linspace(self.x_range[0], self.x_range[1], self.grid_nx, device=device)
        gy = torch.linspace(self.y_range[0], self.y_range[1], self.grid_ny, device=device)
        dx = (self.x_range[1] - self.x_range[0]) / (self.grid_nx - 1)
        dy = (self.y_range[1] - self.y_range[0]) / (self.grid_ny - 1)

        obs_mask = torch.zeros(B, 1, self.grid_nx, self.grid_ny, device=device)
        obs_values = torch.zeros(B, 1, self.grid_nx, self.grid_ny, device=device)
        weight_sum = torch.zeros(B, 1, self.grid_nx, self.grid_ny, device=device)

        for b in range(B):
            for n in range(N):
                ox, oy = obs_xy[b, n, 0].item(), obs_xy[b, n, 1].item()
                oz = obs_z[b, n, 0].item()
                if ox == 0 and oy == 0 and oz == 0:
                    continue  # skip padding

                ix = int(round((ox - self.x_range[0]) / dx))
                iy = int(round((oy - self.y_range[0]) / dy))

                for di in range(-radius, radius + 1):
                    for dj in range(-radius, radius + 1):
                        gi, gj = ix + di, iy + dj
                        if 0 <= gi < self.grid_nx and 0 <= gj < self.grid_ny:
                            dist = math.sqrt(di**2 + dj**2)
                            if dist <= radius:
                                w = max(0, 1 - dist / max(radius, 1))
                                obs_values[b, 0, gi, gj] += w * oz
                                weight_sum[b, 0, gi, gj] += w
                                obs_mask[b, 0, gi, gj] = 1.0

        # Average where multiple obs overlap
        valid = weight_sum > 0
        obs_values[valid] = obs_values[valid] / weight_sum[valid]

        return obs_mask, obs_values

    @property
    def n_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
