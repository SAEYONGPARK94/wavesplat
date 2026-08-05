"""
kalmannet.py
KalmanNet (Revach et al. 2022) for WaveSplat tracking.

Drop-in replacement for the Adam-based update step in
sequential_update.update().  Only the *gain matrix* is learned —
dynamics f(·) = dispersion phase advance, observation h(·) = linear
sinusoidal forward model are kept from WaveSplat (analytical).

State formulation (w_phi tracking mode):
    x_t = (w_n, φ_n) for n = 1..N    (variable N, padded to N_max)
    f(x_{t-1}) = analytical phase advance: φ_n ← φ_n + ω_n·Δt
    h(x_t)     = forward_lidar at obs points

KalmanNet update:
    innovation = y_t - h(x_t|t-1)            (~50-dim, padded)
    state_feat = state diffs (F3, F4)        (2N_max)
    Δx_t       = GRU(innovation, state_feat) (2N_max)
    x_t|t      = x_t|t-1 + mask ⊙ Δx_t       (active components only)
"""
import torch
import torch.nn as nn


class KalmanNet(nn.Module):
    """KalmanNet gain module for WaveSplat tracking.

    Inputs per step:
        innov         (B, M_max)         — η_obs − η_pred at obs points
        innov_mask    (B, M_max)         — 1 for valid obs, 0 for padding
        state_prev    (B, 2N_max)        — concat(w_prev, phi_prev)
        state_pred    (B, 2N_max)        — concat(w, phi after phase advance)
        state_mask    (B, 2N_max)        — 1 for active component
        hidden        (1, B, hidden_dim) — GRU hidden state

    Output:
        delta_state   (B, 2N_max) — (Δw, Δφ) update, masked
        hidden_new    (1, B, hidden_dim)
    """

    def __init__(
        self,
        n_max: int = 30,
        obs_max: int = 80,           # max LiDAR points per frame
        embed_dim: int = 64,
        hidden_dim: int = 128,
        n_gru_layers: int = 1,
    ):
        super().__init__()
        self.n_max = n_max
        self.obs_max = obs_max
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim

        # ── Encoders ──
        # Innovation features: (innov, innov_mask) → embed_dim
        self.innov_encoder = nn.Sequential(
            nn.Linear(obs_max * 2, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
        )

        # State features: (state_prev, state_pred, state_mask) → embed_dim
        self.state_encoder = nn.Sequential(
            nn.Linear(2 * n_max * 3, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
        )

        # ── GRU ──
        self.gru = nn.GRU(
            input_size=2 * embed_dim,
            hidden_size=hidden_dim,
            num_layers=n_gru_layers,
            batch_first=False,
        )

        # ── Decoder: hidden → state delta ──
        self.delta_decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2 * n_max),
        )

    def init_hidden(self, batch_size: int, device: str = 'cpu') -> torch.Tensor:
        return torch.zeros(1, batch_size, self.hidden_dim, device=device)

    def forward(
        self,
        innov: torch.Tensor,        # (B, M_max)
        innov_mask: torch.Tensor,   # (B, M_max)
        state_prev: torch.Tensor,   # (B, 2N_max)
        state_pred: torch.Tensor,   # (B, 2N_max)
        state_mask: torch.Tensor,   # (B, 2N_max)
        hidden: torch.Tensor,       # (1, B, H)
    ):
        # Mask innovation & encode
        innov_masked = innov * innov_mask
        innov_in = torch.cat([innov_masked, innov_mask], dim=-1)  # (B, 2M_max)
        e_innov = self.innov_encoder(innov_in)                    # (B, embed)

        # State features
        state_in = torch.cat([state_prev, state_pred, state_mask], dim=-1)  # (B, 6N_max)
        e_state = self.state_encoder(state_in)                              # (B, embed)

        # GRU expects (seq=1, B, feat)
        gru_in = torch.cat([e_innov, e_state], dim=-1).unsqueeze(0)         # (1, B, 2*embed)
        gru_out, hidden_new = self.gru(gru_in, hidden)                      # (1, B, H)
        gru_out = gru_out.squeeze(0)                                        # (B, H)

        # Decode delta state
        delta = self.delta_decoder(gru_out)                                 # (B, 2N_max)

        # Mask out inactive components
        delta = delta * state_mask

        return delta, hidden_new
