"""
kalmannet_sanity.py

Stage 1 Sanity Test — Go/No-go #1 for KalmanNet × WaveSplat tracking.

Goal:
  1. Build sequence from one episode (kalmannet_dataset)
  2. Instantiate KalmanNet GRU module
  3. Forward pass on each frame (innovation → state update)
  4. Compute observation NLL on tracking sequence
  5. Backward + Adam step several times → check NLL decreases
"""
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from kalmannet_dataset import build_sequence_from_episode, forward_lidar  # noqa: E402
from wavesplat.optimizer.kalmannet import KalmanNet  # noqa: E402
from wavesplat.model.dispersion import dispersion_omega  # noqa: E402


def pad_to(t: torch.Tensor, n: int, dim: int = -1) -> torch.Tensor:
    """Zero-pad along last dim to length n."""
    cur = t.shape[dim]
    if cur >= n:
        return t[..., :n]
    pad_shape = list(t.shape)
    pad_shape[dim] = n - cur
    pad = torch.zeros(pad_shape, dtype=t.dtype, device=t.device)
    return torch.cat([t, pad], dim=dim)


def make_mask(n_active: int, n_max: int) -> torch.Tensor:
    m = torch.zeros(n_max)
    m[:n_active] = 1.0
    return m


def run_sequence(model: KalmanNet, sequence: dict, n_max: int, obs_max: int):
    """Roll out KalmanNet on a tracking sequence.

    Returns:
        total_nll: scalar — sum of -log p(y_t | x_t|t) over sequence
        traj: list of state snapshots
    """
    init = sequence['init_state']
    depth = sequence['meta']['depth']

    # ── State (cloned, requires_grad NOT needed — KalmanNet decides updates) ──
    kx_t  = init['kx'].detach()
    ky_t  = init['ky'].detach()
    w_t   = init['w'].detach()
    phi_t = init['phi'].detach()
    n_active = kx_t.shape[0]

    state_mask = make_mask(n_active, n_max).repeat(2)  # 2N_max
    state_mask_b = state_mask.unsqueeze(0)             # (1, 2N_max)

    # Hidden state
    hidden = model.init_hidden(batch_size=1)
    total_nll = 0.0
    n_frames_used = 0

    state_prev = torch.cat([
        pad_to(w_t, n_max), pad_to(phi_t, n_max),
    ]).unsqueeze(0)  # (1, 2N_max)

    for frame in sequence['frames']:
        dt = frame['dt']
        x_obs = frame['x_obs']
        y_obs = frame['y_obs']
        eta_obs = frame['eta_obs']
        t = frame['t']

        # ── Step 1: Predict (analytical phase advance) ──
        k_mag = torch.sqrt(kx_t**2 + ky_t**2 + 1e-12)
        omega = dispersion_omega(k_mag, depth)
        phi_t_pred = phi_t + omega * dt
        # w unchanged in predict

        state_pred = torch.cat([
            pad_to(w_t, n_max), pad_to(phi_t_pred, n_max),
        ]).unsqueeze(0)  # (1, 2N_max)

        # ── Step 2: Innovation at LiDAR points ──
        eta_pred = forward_lidar(kx_t, ky_t, w_t, phi_t_pred, omega,
                                  x_obs, y_obs, t)
        innov = eta_obs - eta_pred                              # (M,)
        m_obs = innov.shape[0]
        innov_padded = pad_to(innov, obs_max).unsqueeze(0)      # (1, M_max)
        innov_mask = make_mask(m_obs, obs_max).unsqueeze(0)     # (1, M_max)

        # ── Step 3: KalmanNet RNN gain ──
        delta, hidden = model(
            innov=innov_padded,
            innov_mask=innov_mask,
            state_prev=state_prev,
            state_pred=state_pred,
            state_mask=state_mask_b,
            hidden=hidden,
        )
        # delta: (1, 2N_max)
        delta = delta.squeeze(0)

        # ── Step 4: Apply update (only active components) ──
        dw = delta[:n_active]
        dphi = delta[n_max:n_max + n_active]
        w_t = w_t + dw
        phi_t = phi_t_pred + dphi

        # ── Step 5: Observation NLL after update ──
        eta_pred_post = forward_lidar(kx_t, ky_t, w_t, phi_t, omega,
                                       x_obs, y_obs, t)
        sigma = 0.05  # fixed obs noise for sanity test
        nll = 0.5 * ((eta_obs - eta_pred_post) / sigma) ** 2
        total_nll = total_nll + nll.sum()
        n_frames_used += 1

        # Update state_prev for next iter
        state_prev = torch.cat([
            pad_to(w_t, n_max), pad_to(phi_t, n_max),
        ]).unsqueeze(0)

    return total_nll / max(n_frames_used, 1), n_frames_used


def main():
    print('=' * 64)
    print('KalmanNet × WaveSplat Stage 1 Sanity Test')
    print('=' * 64)

    DATA = os.path.join(os.path.dirname(__file__), '..', 'data',
                        'episode_BF3_2comp_seed0_uniform.h5')

    # ── 1) Build sequence ──
    print('\n[Sequence build]')
    sequence = build_sequence_from_episode(DATA,
                                            tp_factor=2.0,
                                            n_tracking_frames=10)
    n_active = sequence['init_state']['kx'].shape[0]
    n_frames = len(sequence['frames'])
    print(f'  Episode: {sequence["meta"]["episode"]}')
    print(f'  N active components: {n_active}')
    print(f'  Tracking frames: {n_frames}')

    # ── 2) Instantiate model ──
    N_MAX = 30
    OBS_MAX = 80
    model = KalmanNet(n_max=N_MAX, obs_max=OBS_MAX,
                      embed_dim=64, hidden_dim=128, n_gru_layers=1)
    n_params = sum(p.numel() for p in model.parameters())
    print(f'\n[Model] KalmanNet, {n_params:,} parameters ({n_params/1e3:.1f}K)')

    # ── 3) Forward pass ──
    print('\n[Forward pass on sequence]')
    model.eval()
    t0 = time.time()
    with torch.no_grad():
        nll_init, _ = run_sequence(model, sequence, N_MAX, OBS_MAX)
    dt = time.time() - t0
    print(f'  Avg NLL per frame: {nll_init.item():.4f}')
    print(f'  Sequence wall time: {dt*1000:.1f} ms (CPU)')

    # ── 4) Backward + Adam steps ──
    print('\n[Train step — Go/No-go #1]')
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    losses = []
    for step in range(8):
        optimizer.zero_grad()
        nll, _ = run_sequence(model, sequence, N_MAX, OBS_MAX)
        nll.backward()
        optimizer.step()
        losses.append(nll.item())
        print(f'  step {step}: NLL = {nll.item():.4f}')

    diff = losses[0] - losses[-1]
    if diff > 0:
        print(f'\n✓ Loss decreasing ({losses[0]:.4f} → {losses[-1]:.4f}, '
              f'Δ={diff:.4f}) — GRADIENT FLOWS')
        print('✓ Go/No-go #1 PASSED — KalmanNet can be trained on WaveSplat tracking')
    else:
        print(f'\n✗ Loss not decreasing — needs investigation')


if __name__ == '__main__':
    main()
