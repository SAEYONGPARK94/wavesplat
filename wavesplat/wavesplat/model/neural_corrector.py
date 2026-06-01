"""
neural_corrector.py
Neural residual correction for 3rd+ order nonlinear wave effects.

Two architectures:

1. NeuralCorrector (per-component): DeepSets → (delta_omega, delta_w)
   - Limited: can only modify existing spectral components
   - Optimal R² ≈ 0.11 on 30-component basis

2. NeuralFieldCorrector (phase-aware field): computes per-component phases
   at query points, learns nonlinear correction via DeepSets aggregation.

   Key insight: the nonlinear residual depends on PHASE INTERFERENCE between
   components. The model must see the actual phases at each query point to
   learn meaningful corrections. Without phases, the best prediction is zero
   (the ensemble mean over random phases).

   Architecture per query point (x, y, t):
     - For each component n: compute phase_n = kx_n*x + ky_n*y - ω_n*t + φ_n
     - Per-component features: [cos(phase_n), sin(phase_n), w_n, steepness_n, ...]
     - MLP: features → hidden
     - Sum over components → aggregate
     - Decoder MLP → δη(x, y, t)

   The linear model η = Σ w_n cos(phase_n) is a special case where the NN
   outputs w_n for cos and 0 for sin. The NN learns nonlinear modifications.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

G = 9.81


class NeuralCorrector(nn.Module):
    """Per-component corrector (delta_omega, delta_w). Limited capacity."""

    def __init__(self, hidden_dim: int = 64, n_layers: int = 3,
                 input_dim: int = 5):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim

        layers_phi = [nn.Linear(input_dim, hidden_dim), nn.GELU()]
        for _ in range(n_layers - 1):
            layers_phi.extend([nn.Linear(hidden_dim, hidden_dim), nn.GELU()])
        self.phi_net = nn.Sequential(*layers_phi)

        layers_rho = [nn.Linear(hidden_dim * 2, hidden_dim), nn.GELU()]
        for _ in range(n_layers - 1):
            layers_rho.extend([nn.Linear(hidden_dim, hidden_dim), nn.GELU()])
        layers_rho.append(nn.Linear(hidden_dim, 2))
        self.rho_net = nn.Sequential(*layers_rho)

        with torch.no_grad():
            self.rho_net[-1].weight.mul_(0.01)
            self.rho_net[-1].bias.zero_()

    def _build_features(self, kx, ky, w, depth):
        k_mag = torch.sqrt(kx**2 + ky**2 + 1e-12)
        k_safe = torch.clamp(k_mag, min=1e-6)
        k_peak = k_mag[torch.argmax(torch.abs(w))]
        k_peak = torch.clamp(k_peak, min=0.01)
        f_k = k_mag / k_peak
        f_cos = kx / k_safe
        f_sin = ky / k_safe
        f_steep = k_mag * torch.abs(w)
        w_scale = 4.0 * torch.std(w) + 1e-6
        f_w = w / w_scale
        return torch.stack([f_k, f_cos, f_sin, f_steep, f_w], dim=-1)

    def forward(self, kx, ky, w, depth):
        features = self._build_features(kx, ky, w, depth)
        h = self.phi_net(features)
        global_pool = h.sum(dim=0, keepdim=True).expand_as(h)
        combined = torch.cat([h, global_pool], dim=-1)
        output = self.rho_net(combined)
        delta_omega = output[:, 0]
        delta_w = output[:, 1]
        k_mag = torch.sqrt(kx**2 + ky**2 + 1e-12)
        omega = torch.sqrt(G * k_mag * torch.tanh(k_mag * depth) + 1e-12)
        delta_omega = delta_omega * omega
        delta_w = delta_w * torch.abs(w)
        return delta_omega, delta_w

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class NeuralFieldCorrector(nn.Module):
    """
    Phase-aware field-level neural corrector.

    For each query point, computes per-component phase features,
    processes through an MLP, and aggregates via sum (DeepSets).
    """

    def __init__(self, hidden_dim: int = 64, n_layers: int = 2,
                 per_comp_features: int = 6):
        """
        Args:
            hidden_dim: width of hidden layers
            n_layers: number of hidden layers
            per_comp_features: number of per-component input features
        """
        super().__init__()
        self.hidden_dim = hidden_dim

        # Per-component encoder: features → hidden vector (to be summed)
        enc_layers = [nn.Linear(per_comp_features, hidden_dim), nn.GELU()]
        for _ in range(n_layers - 1):
            enc_layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.GELU()])
        self.encoder = nn.Sequential(*enc_layers)

        # Decoder: aggregated hidden → scalar delta_eta
        # This allows nonlinear combination of aggregated features
        dec_layers = [nn.Linear(hidden_dim, hidden_dim), nn.GELU()]
        dec_layers.append(nn.Linear(hidden_dim, 1))
        self.decoder = nn.Sequential(*dec_layers)

        with torch.no_grad():
            self.decoder[-1].bias.zero_()

    def forward(self, kx, ky, w, phi, omega, depth, x, y, t):
        """
        Compute nonlinear correction field.

        Args:
            kx, ky, w, phi, omega: (N,) spectral parameters
            depth: water depth [m]
            x, y: (M,) query coordinates [m]
            t: scalar time [s]

        Returns:
            delta_eta: (M,) correction at each query point [m]
        """
        N = len(kx)
        M = len(x)

        # Compute per-component phases at each query point
        # phase[m, n] = kx_n * x_m + ky_n * y_m - omega_n * t + phi_n
        phase = (kx[None, :] * x[:, None]
                 + ky[None, :] * y[:, None]
                 - omega[None, :] * t
                 + phi[None, :])  # (M, N)

        cos_phase = torch.cos(phase)  # (M, N)
        sin_phase = torch.sin(phase)  # (M, N)

        # Per-component static features (broadcast to all query points)
        k_mag = torch.sqrt(kx**2 + ky**2 + 1e-12)
        k_peak = k_mag[torch.argmax(torch.abs(w))]
        k_peak = torch.clamp(k_peak, min=0.01)

        steepness = (k_mag * torch.abs(w))[None, :].expand(M, N)  # (M, N)
        k_rel = (k_mag / k_peak)[None, :].expand(M, N)  # (M, N)
        w_scale = 4.0 * torch.std(w) + 1e-6
        w_norm = (w / w_scale)[None, :].expand(M, N)  # (M, N)
        w_broad = w[None, :].expand(M, N)  # (M, N)

        # Stack features: (M, N, 6)
        features = torch.stack([
            cos_phase, sin_phase, w_broad, steepness, k_rel, w_norm
        ], dim=-1)

        # Encode: (M*N, 6) → (M*N, hidden_dim)
        features_flat = features.reshape(M * N, -1)
        h_flat = self.encoder(features_flat)  # (M*N, hidden_dim)
        h = h_flat.reshape(M, N, -1)  # (M, N, hidden_dim)

        # Sum over components → (M, hidden_dim)
        h_agg = h.sum(dim=1)  # (M, hidden_dim)

        # Decode to scalar correction
        delta_eta = self.decoder(h_agg).squeeze(-1)  # (M,)

        return delta_eta

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ─── Set Transformer building blocks ──────────────────────────────────────

class _MultiHeadAttention(nn.Module):
    """Standard multi-head attention (no batch dim required)."""

    def __init__(self, dim: int, n_heads: int):
        super().__init__()
        assert dim % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.W_q = nn.Linear(dim, dim)
        self.W_k = nn.Linear(dim, dim)
        self.W_v = nn.Linear(dim, dim)
        self.W_o = nn.Linear(dim, dim)

    def forward(self, Q, K, V):
        """Q: (B, Nq, D), K/V: (B, Nk, D) → (B, Nq, D)"""
        B, Nq, D = Q.shape
        Nk = K.shape[1]
        h = self.n_heads

        q = self.W_q(Q).view(B, Nq, h, self.head_dim).transpose(1, 2)
        k = self.W_k(K).view(B, Nk, h, self.head_dim).transpose(1, 2)
        v = self.W_v(V).view(B, Nk, h, self.head_dim).transpose(1, 2)

        attn = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(B, Nq, D)
        return self.W_o(out)


class _SAB(nn.Module):
    """Set Attention Block: self-attention + FFN with residual connections."""

    def __init__(self, dim: int, n_heads: int):
        super().__init__()
        self.mha = _MultiHeadAttention(dim, n_heads)
        self.ln1 = nn.LayerNorm(dim)
        self.ln2 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Linear(dim * 2, dim),
        )

    def forward(self, X):
        """X: (B, N, D) → (B, N, D)"""
        h = self.ln1(X + self.mha(X, X, X))
        return self.ln2(h + self.ff(h))


class NeuralFieldCorrectorV2(nn.Module):
    """
    Phase-aware field corrector with Set Transformer-style self-attention.

    Key difference from V1 (DeepSets): components interact via self-attention
    BEFORE sum pooling. This captures pairwise (and higher-order via stacking)
    interactions between wave components — critical for modeling nonlinear
    wave-wave coupling like cos(θ_n ± θ_m) effects.

    Architecture per query point (x, y, t):
      1. Per-component features: [cos(θ_n), sin(θ_n), w_n, steepness_n, ...]
      2. Linear projection to hidden_dim
      3. Self-attention blocks (SAB) for cross-component interaction
      4. Sum pooling over components
      5. Decoder MLP → δη(x, y, t)
    """

    def __init__(self, hidden_dim: int = 64, n_heads: int = 4,
                 n_attn_layers: int = 2, per_comp_features: int = 6):
        super().__init__()
        self.hidden_dim = hidden_dim

        # Input projection
        self.input_proj = nn.Linear(per_comp_features, hidden_dim)

        # Self-attention blocks for cross-component interaction
        self.attn_layers = nn.ModuleList([
            _SAB(hidden_dim, n_heads) for _ in range(n_attn_layers)
        ])

        # Decoder: aggregated hidden → scalar delta_eta
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

        # Initialize output near zero
        with torch.no_grad():
            self.decoder[-1].bias.zero_()
            self.decoder[-1].weight.mul_(0.01)

    def forward(self, kx, ky, w, phi, omega, depth, x, y, t):
        """
        Compute nonlinear correction field with cross-component attention.

        Args:
            kx, ky, w, phi, omega: (N,) spectral parameters
            depth: water depth [m]
            x, y: (M,) query coordinates [m]
            t: scalar time [s]

        Returns:
            delta_eta: (M,) correction at each query point [m]
        """
        N = len(kx)
        M = len(x)

        # Phase at each query point: (M, N)
        phase = (kx[None, :] * x[:, None]
                 + ky[None, :] * y[:, None]
                 - omega[None, :] * t
                 + phi[None, :])

        cos_phase = torch.cos(phase)
        sin_phase = torch.sin(phase)

        # Per-component static features
        k_mag = torch.sqrt(kx**2 + ky**2 + 1e-12)
        k_peak = k_mag[torch.argmax(torch.abs(w))]
        k_peak = torch.clamp(k_peak, min=0.01)

        steepness = (k_mag * torch.abs(w))[None, :].expand(M, N)
        k_rel = (k_mag / k_peak)[None, :].expand(M, N)
        w_scale = 4.0 * torch.std(w) + 1e-6
        w_norm = (w / w_scale)[None, :].expand(M, N)
        w_broad = w[None, :].expand(M, N)

        # (M, N, 6)
        features = torch.stack([
            cos_phase, sin_phase, w_broad, steepness, k_rel, w_norm
        ], dim=-1)

        # Project to hidden_dim: (M, N, hidden_dim)
        h = self.input_proj(features)

        # Self-attention over components (treat M query points as batch)
        # h: (M, N, D) — M is batch, N is set size
        for sab in self.attn_layers:
            h = sab(h)  # (M, N, D)

        # Sum pooling over components → (M, hidden_dim)
        h_agg = h.sum(dim=1)

        # Decode to scalar correction
        delta_eta = self.decoder(h_agg).squeeze(-1)  # (M,)

        return delta_eta

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
