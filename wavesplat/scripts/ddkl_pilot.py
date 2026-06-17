"""
ddkl_pilot.py — Deep Dispersion Kernel Learning (DDKL) Pilot v3

Two approaches:
  A) End-to-end: SetConv → CNN → predict K_disp params (v1/v2, failed)
  B) Neural refinement: Beamforming init + SetConv features → refined K_disp params

Current: Approach B.
  Input: beamforming {k_n, A_n} + sparse obs (via SetConv grid)
  Output: refined {k_n, A_n} for GP K_disp
  Loss: GP reconstruction MSE vs GT field (supervised)

Approach B amortizes the per-episode optimization of WaveSplat into a learned
correction, using beamforming as a strong initialization.
"""

import os
import sys
import json
import time
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
os.chdir(os.path.join(os.path.dirname(__file__), '..'))

from wavesplat.data_loader import EpisodeDataLoader

G = 9.81
SIGMA_OBS = 0.02
N_COMP = 15
N_FRAMES_TRAIN = 5    # frames per episode per epoch
N_FRAMES_EVAL = 30

DATA_DIR = 'data'

# Test episodes (same as GP K_disp eval)
TEST_EPISODES = [
    'episode_BF3_5comp_seed0_slope.h5',
    'episode_BF4_5comp_seed0_uniform.h5',
    'episode_BF4_15comp_seed0_uniform.h5',
    'episode_BF4_30comp_seed0_uniform.h5',
    'episode_BF5_15comp_seed0_uniform.h5',
    'episode_BF5_30comp_seed0_uniform.h5',
]

# Train episodes — seeds 1-4 ONLY (test is all seed0).
# Verified: seedN_uniform and seedN_slope share the SAME wave realization
# (corr=1.0), so excluding seed0 entirely guarantees disjoint wave fields
# between train and the seed0 test set (no leakage).
TRAIN_EPISODES = []
for bf in [3, 4, 5]:
    for nc in [1, 2, 5, 15, 30]:
        for seed in range(1, 5):  # seeds 1-4, seed0 reserved for test
            fname = f'episode_BF{bf}_{nc}comp_seed{seed}_uniform.h5'
            if fname in TEST_EPISODES:
                continue
            if os.path.exists(os.path.join(DATA_DIR, fname)):
                TRAIN_EPISODES.append(fname)
# Add slope-sampled variants (seeds 1-4) for sampling-pattern diversity
for bf in [3, 4, 5]:
    for nc in [1, 2, 5, 15, 30]:
        for seed in range(1, 5):
            fname = f'episode_BF{bf}_{nc}comp_seed{seed}_slope.h5'
            if fname in TEST_EPISODES:
                continue
            if os.path.exists(os.path.join(DATA_DIR, fname)):
                TRAIN_EPISODES.append(fname)


# ── Beamforming helper ────────────────────────────────────────────────────────

def get_beamform_init(loader, n_gaussians=N_COMP):
    """Run beamforming to get initial wave component estimates."""
    from wavesplat.optimizer.initializer import initialize_from_frames_v2
    init_frames = loader.get_init_window(tp_factor=2.0)
    if len(init_frames) < 3:
        init_frames = [loader[i] for i in range(min(10, loader.n_frames))]
    init_kx, init_ky, init_w, init_phi = initialize_from_frames_v2(
        init_frames, n_gaussians=n_gaussians, depth=loader.depth
    )
    k_mag = torch.sqrt(init_kx**2 + init_ky**2 + 1e-8)
    theta = torch.atan2(init_ky, init_kx)
    A = torch.abs(init_w)
    return k_mag, theta, A  # all [n_comp]


# ── Model ─────────────────────────────────────────────────────────────────────

class DDKLRefiner(nn.Module):
    """Refine beamforming K_disp estimates using spatial context.

    Input: beamforming {k, θ, A} + SetConv grid from sparse obs
    Output: refined {k, θ, A}

    Architecture:
      1. SetConv(obs) → grid [2, Gx, Gy]
      2. CNN backbone → global features [hidden_dim]
      3. Concatenate with beamform params [3*N_comp]
      4. MLP → delta corrections for (log_k, θ, log_A)
    """

    def __init__(self, n_comp=15, grid_nx=64, grid_ny=48,
                 x_range=(10.0, 250.0), y_range=(-100.0, 100.0),
                 init_lengthscale=10.0, hidden_dim=64,
                 refine_mode='full'):
        # refine_mode: 'full' = correct (k, theta, A); 'amp' = correct A only
        super().__init__()
        self.n_comp = n_comp
        self.grid_nx = grid_nx
        self.grid_ny = grid_ny
        self.refine_mode = refine_mode

        # SetConv grid
        gx = torch.linspace(x_range[0], x_range[1], grid_nx)
        gy = torch.linspace(y_range[0], y_range[1], grid_ny)
        xx, yy = torch.meshgrid(gx, gy, indexing='ij')
        self.register_buffer('grid_xy',
                             torch.stack([xx.flatten(), yy.flatten()], dim=-1))
        self.log_lengthscale = nn.Parameter(torch.tensor(math.log(init_lengthscale)))

        # CNN backbone: 2ch → hidden_dim
        self.backbone = nn.Sequential(
            nn.Conv2d(2, 32, 5, stride=2, padding=2),
            nn.GELU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(64, hidden_dim, 3, stride=2, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
        )

        # Fusion MLP: grid features + beamform params → delta corrections
        input_dim = hidden_dim + 3 * n_comp  # spatial feat + (k, θ, A) init
        self.refiner = nn.Sequential(
            nn.Linear(input_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 3 * n_comp),  # delta (log_k, theta, log_A)
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, nonlinearity='linear')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        # Initialize refiner output near zero (identity mapping at start)
        nn.init.zeros_(self.refiner[-1].weight)
        nn.init.zeros_(self.refiner[-1].bias)

    def _setconv(self, obs_xy, obs_z):
        """Sparse obs → grid."""
        ls = self.log_lengthscale.exp()
        diff = self.grid_xy.unsqueeze(1) - obs_xy.unsqueeze(0)
        sq_dist = (diff ** 2).sum(-1)
        weights = torch.exp(-0.5 * sq_dist / (ls ** 2))
        density = weights.sum(-1, keepdim=True)
        signal = (weights * obs_z.unsqueeze(0)).sum(-1, keepdim=True)
        signal = signal / (density + 1e-8)
        features = torch.cat([signal, density], dim=-1)
        return features.permute(1, 0).reshape(1, 2, self.grid_nx, self.grid_ny)

    def forward(self, obs_xy, obs_z, beam_k, beam_theta, beam_A):
        """
        Args:
            obs_xy: [N_obs, 2]
            obs_z: [N_obs]
            beam_k: [n_comp] beamform wavenumber magnitudes
            beam_theta: [n_comp] beamform directions
            beam_A: [n_comp] beamform amplitudes
        Returns:
            k_mag, theta, A: refined [n_comp] each
        """
        grid = self._setconv(obs_xy, obs_z)
        feat = self.backbone(grid).flatten(1)  # [1, hidden_dim]

        # Normalize beamform params for input
        beam_input = torch.cat([
            torch.log(beam_k + 1e-6),
            beam_theta,
            torch.log(beam_A + 1e-6),
        ]).unsqueeze(0)  # [1, 3*n_comp]

        combined = torch.cat([feat, beam_input], dim=1)  # [1, hidden+3*n]
        delta = self.refiner(combined).squeeze(0)  # [3*n_comp]

        # Apply residual corrections
        d_log_k = delta[:self.n_comp]
        d_theta = delta[self.n_comp:2*self.n_comp]
        d_log_A = delta[2*self.n_comp:]

        # Refined params = init + learned correction
        A = beam_A * torch.exp(torch.clamp(d_log_A, -3, 3))
        A = torch.clamp(A, 1e-4, 5.0)

        if self.refine_mode == 'amp':
            # Amplitude-only calibration: freeze beamform directions/wavenumbers.
            # Tests whether the harm in 'full' mode comes from directional moves.
            return beam_k, beam_theta, A

        k_mag = beam_k * torch.exp(torch.clamp(d_log_k, -2, 2))
        k_mag = torch.clamp(k_mag, 0.005, 0.5)
        theta = beam_theta + torch.tanh(d_theta) * 0.5  # max ±0.5 rad correction

        return k_mag, theta, A


# ── K_disp and NML ────────────────────────────────────────────────────────────

def build_K_disp_torch(X1, X2, k_mag, theta, A):
    """Build K_disp kernel matrix in PyTorch (differentiable).

    Args:
        X1: [N1, 2] positions
        X2: [N2, 2] positions
        k_mag: [n_comp] wavenumber magnitudes
        theta: [n_comp] propagation directions
        A: [n_comp] amplitudes
    Returns:
        K: [N1, N2] covariance matrix
    """
    kx = k_mag * torch.cos(theta)  # [n_comp]
    ky = k_mag * torch.sin(theta)  # [n_comp]

    dx = X1[:, 0:1] - X2[:, 0].unsqueeze(0)  # [N1, N2]
    dy = X1[:, 1:2] - X2[:, 1].unsqueeze(0)  # [N1, N2]

    # K = sum_n (A_n^2 / 2) * cos(kx_n * dx + ky_n * dy)
    K = torch.zeros(X1.shape[0], X2.shape[0], device=X1.device)
    for n in range(len(A)):
        phase = kx[n] * dx + ky[n] * dy
        K = K + 0.5 * A[n]**2 * torch.cos(phase)

    return K


def nml_loss(K_oo, z_obs, sigma_obs=SIGMA_OBS):
    """Negative marginal log-likelihood (differentiable via Cholesky).

    NML = 0.5 * z^T (K + σ²I)^{-1} z + 0.5 * log|K + σ²I| + const

    Args:
        K_oo: [N, N] obs-obs covariance
        z_obs: [N] observations
        sigma_obs: observation noise std
    Returns:
        nml: scalar loss
    """
    N = K_oo.shape[0]
    K_noisy = K_oo + sigma_obs**2 * torch.eye(N, device=K_oo.device)
    # Jitter for numerical stability
    K_noisy = K_noisy + 1e-5 * torch.eye(N, device=K_oo.device)

    try:
        L = torch.linalg.cholesky(K_noisy)
    except RuntimeError:
        # Fallback: add more jitter
        K_noisy = K_noisy + 1e-3 * torch.eye(N, device=K_oo.device)
        L = torch.linalg.cholesky(K_noisy)

    # Solve L @ alpha_half = z, then L^T @ alpha = alpha_half
    alpha = torch.cholesky_solve(z_obs.unsqueeze(1), L).squeeze(1)
    data_fit = 0.5 * torch.dot(z_obs, alpha)
    log_det = torch.sum(torch.log(torch.diag(L)))  # 0.5 * log|K| = sum log diag(L)

    return data_fit + log_det


def gp_predict_torch(K_oo, K_po, z_obs, sigma_obs=SIGMA_OBS):
    """GP posterior mean (differentiable)."""
    N = K_oo.shape[0]
    K_noisy = K_oo + (sigma_obs**2 + 1e-5) * torch.eye(N, device=K_oo.device)
    try:
        L = torch.linalg.cholesky(K_noisy)
    except RuntimeError:
        K_noisy = K_noisy + 1e-3 * torch.eye(N, device=K_oo.device)
        L = torch.linalg.cholesky(K_noisy)
    alpha = torch.cholesky_solve(z_obs.unsqueeze(1), L).squeeze(1)
    return K_po @ alpha


# ── Data loading ──────────────────────────────────────────────────────────────

def load_episode_data(h5_path, tp_factor=2.0):
    """Load episode and return init obs + beamform init + eval frames."""
    loader = EpisodeDataLoader(h5_path)
    init_frames = loader.get_init_window(tp_factor=tp_factor)
    if len(init_frames) < 3:
        init_frames = [loader[i] for i in range(min(10, loader.n_frames))]

    # Concatenate init window observations
    all_x, all_y, all_z = [], [], []
    for f in init_frames[:10]:
        all_x.append(f.x_obs.cpu().numpy())
        all_y.append(f.y_obs.cpu().numpy())
        all_z.append(f.z_obs.cpu().numpy())
    init_xy = np.column_stack([np.concatenate(all_x), np.concatenate(all_y)])
    init_z = np.concatenate(all_z)

    # Subsample if too many
    if len(init_z) > 500:
        idx = np.random.default_rng(42).choice(len(init_z), 500, replace=False)
        init_xy = init_xy[idx]
        init_z = init_z[idx]

    # Beamforming initialization
    beam_k, beam_theta, beam_A = get_beamform_init(loader, n_gaussians=N_COMP)

    eval_frames = loader.get_tracking_frames(tp_factor=tp_factor)

    return {
        'init_xy': init_xy,
        'init_z': init_z,
        'beam_k': beam_k.numpy(),
        'beam_theta': beam_theta.numpy(),
        'beam_A': beam_A.numpy(),
        'eval_frames': eval_frames,
        'gt_grid_x': loader.gt_grid_x.cpu().numpy(),
        'gt_grid_y': loader.gt_grid_y.cpu().numpy(),
        'Hs': loader.Hs_true,
        'depth': loader.depth,
    }


# ── Training ──────────────────────────────────────────────────────────────────

def train_ddkl(model, train_episodes, n_epochs=150, lr=1e-3, device='cpu',
               n_grid_subsample=400, lambda_nml=0.1):
    """Train DDKL refiner with GP reconstruction loss + NML regularizer."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)

    print(f"Loading {len(train_episodes)} training episodes...")
    train_data = []
    for h5_name in train_episodes:
        h5_path = os.path.join(DATA_DIR, h5_name)
        if not os.path.exists(h5_path):
            continue
        try:
            d = load_episode_data(h5_path)
            train_data.append(d)
        except Exception as e:
            print(f"  Skip {h5_name}: {e}")
    print(f"  Loaded {len(train_data)} episodes")

    for d in train_data:
        gt_x, gt_y = d['gt_grid_x'], d['gt_grid_y']
        Xg, Yg = np.meshgrid(gt_x, gt_y)
        d['X_pred_full'] = np.column_stack([Xg.ravel(), Yg.ravel()])

    model.to(device)
    model.train()

    best_loss = float('inf')
    best_state = None
    history = []

    for epoch in range(n_epochs):
        epoch_losses = []
        epoch_recon = []
        np.random.shuffle(train_data)

        for ep_data in train_data:
            init_xy_t = torch.tensor(ep_data['init_xy'], dtype=torch.float32, device=device)
            init_z_t = torch.tensor(ep_data['init_z'], dtype=torch.float32, device=device)
            beam_k_t = torch.tensor(ep_data['beam_k'], dtype=torch.float32, device=device)
            beam_theta_t = torch.tensor(ep_data['beam_theta'], dtype=torch.float32, device=device)
            beam_A_t = torch.tensor(ep_data['beam_A'], dtype=torch.float32, device=device)

            k_mag, theta, A = model(init_xy_t, init_z_t, beam_k_t, beam_theta_t, beam_A_t)

            eval_frames = ep_data['eval_frames']
            n_train_frames = min(N_FRAMES_TRAIN, len(eval_frames))
            frame_indices = np.random.choice(len(eval_frames), n_train_frames, replace=False)

            frame_losses = []
            frame_recon = []

            for fi in frame_indices:
                frame = eval_frames[fi]
                x_obs = frame.x_obs.cpu().numpy()
                y_obs = frame.y_obs.cpu().numpy()
                z_obs_np = frame.z_obs.cpu().numpy()
                gt_field = frame.gt_field.cpu().numpy()
                if len(x_obs) < 5:
                    continue

                X_obs = torch.tensor(
                    np.column_stack([x_obs, y_obs]),
                    dtype=torch.float32, device=device
                )
                z_obs = torch.tensor(z_obs_np, dtype=torch.float32, device=device)

                K_oo = build_K_disp_torch(X_obs, X_obs, k_mag, theta, A)

                # NML
                nml = nml_loss(K_oo, z_obs)

                # Reconstruction loss
                n_total = len(ep_data['X_pred_full'])
                sub_idx = np.random.choice(n_total, min(n_grid_subsample, n_total), replace=False)
                X_sub = torch.tensor(ep_data['X_pred_full'][sub_idx],
                                     dtype=torch.float32, device=device)
                gt_sub = torch.tensor(gt_field.ravel()[sub_idx],
                                      dtype=torch.float32, device=device)

                K_po = build_K_disp_torch(X_sub, X_obs, k_mag, theta, A)
                mu = gp_predict_torch(K_oo, K_po, z_obs)

                gt_var = gt_sub.var() + 1e-8
                recon = F.mse_loss(mu, gt_sub) / gt_var

                if torch.isfinite(nml) and torch.isfinite(recon):
                    loss = recon + lambda_nml * nml / max(len(z_obs), 1)
                    frame_losses.append(loss)
                    frame_recon.append(recon.item())

            if not frame_losses:
                continue

            total_loss = torch.stack(frame_losses).mean()

            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            epoch_losses.append(total_loss.item())
            epoch_recon.extend(frame_recon)

        scheduler.step()

        if epoch_losses:
            mean_loss = np.mean(epoch_losses)
            mean_recon = np.mean(epoch_recon) if epoch_recon else 0
            history.append(mean_loss)

            if mean_loss < best_loss:
                best_loss = mean_loss
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

            if epoch % 10 == 0 or epoch == n_epochs - 1:
                print(f"  Epoch {epoch:3d}/{n_epochs}  loss={mean_loss:.4f}  "
                      f"recon={mean_recon:.4f}  best={best_loss:.4f}  "
                      f"lr={scheduler.get_last_lr()[0]:.6f}")

    if best_state is not None:
        model.load_state_dict(best_state)
    model.to(device)

    return history, best_loss


# ── Evaluation ────────────────────────────────────────────────────────────────

def compute_metrics(pred, gt, Hs):
    err = pred - gt
    nRMSE = float(np.sqrt(np.mean(err**2)) / max(Hs, 1e-6))
    std_pred = float(np.std(pred))
    std_gt = float(np.std(gt))
    VR = (std_pred / max(std_gt, 1e-6))**2
    rho = float(np.corrcoef(pred.ravel(), gt.ravel())[0, 1]) if std_pred > 1e-8 and std_gt > 1e-8 else 0.0
    ss_res = np.sum(err**2)
    ss_tot = np.sum((gt - np.mean(gt))**2)
    skill = float(1.0 - ss_res / max(ss_tot, 1e-12))
    return dict(nRMSE=nRMSE, VR=VR, Pearson=rho, Skill=skill)


@torch.no_grad()
def evaluate_ddkl(model, test_episodes, device='cpu', n_frames=N_FRAMES_EVAL,
                  sigma_obs=SIGMA_OBS, also_eval_beamform=True):
    """Evaluate DDKL on test episodes. Optionally also eval raw beamform baseline."""
    model.eval()
    model.to(device)

    results = {}
    beam_results = {}  # beamform-only baseline
    for h5_name in test_episodes:
        h5_path = os.path.join(DATA_DIR, h5_name)
        if not os.path.exists(h5_path):
            print(f"  SKIP: {h5_name}")
            continue

        ep_key = h5_name.replace('episode_', '').replace('.h5', '')
        ep_data = load_episode_data(h5_path)

        init_xy_t = torch.tensor(ep_data['init_xy'], dtype=torch.float32, device=device)
        init_z_t = torch.tensor(ep_data['init_z'], dtype=torch.float32, device=device)
        beam_k_t = torch.tensor(ep_data['beam_k'], dtype=torch.float32, device=device)
        beam_theta_t = torch.tensor(ep_data['beam_theta'], dtype=torch.float32, device=device)
        beam_A_t = torch.tensor(ep_data['beam_A'], dtype=torch.float32, device=device)

        k_mag, theta, A = model(init_xy_t, init_z_t, beam_k_t, beam_theta_t, beam_A_t)

        # Prepare prediction grid
        gt_x = ep_data['gt_grid_x']
        gt_y = ep_data['gt_grid_y']
        Xg, Yg = np.meshgrid(gt_x, gt_y)
        X_pred = torch.tensor(
            np.column_stack([Xg.ravel(), Yg.ravel()]),
            dtype=torch.float32, device=device
        )

        nrmse_list = []
        metrics_list = []
        eval_frames = ep_data['eval_frames']

        for fi, frame in enumerate(eval_frames):
            if fi >= n_frames:
                break
            x_obs = frame.x_obs.cpu().numpy()
            y_obs = frame.y_obs.cpu().numpy()
            z_obs_np = frame.z_obs.cpu().numpy()
            gt = frame.gt_field.cpu().numpy()
            if len(x_obs) < 5:
                continue

            X_obs = torch.tensor(
                np.column_stack([x_obs, y_obs]),
                dtype=torch.float32, device=device
            )
            z_obs = torch.tensor(z_obs_np, dtype=torch.float32, device=device)

            K_oo = build_K_disp_torch(X_obs, X_obs, k_mag, theta, A)
            K_po = build_K_disp_torch(X_pred, X_obs, k_mag, theta, A)

            mu = gp_predict_torch(K_oo, K_po, z_obs, sigma_obs)
            pred = mu.cpu().numpy().reshape(len(gt_y), len(gt_x))

            m = compute_metrics(pred, gt, ep_data['Hs'])
            nrmse_list.append(m['nRMSE'])
            metrics_list.append(m)

        if nrmse_list:
            avg_metrics = {
                k: float(np.mean([m[k] for m in metrics_list]))
                for k in metrics_list[0].keys()
            }
            results[ep_key] = avg_metrics

            k_np = k_mag.cpu().numpy()
            A_np = A.cpu().numpy()
            print(f"  [DDKL {ep_key}] nRMSE={avg_metrics['nRMSE']:.4f}  "
                  f"Pearson={avg_metrics['Pearson']:.3f}  "
                  f"VR={avg_metrics['VR']:.3f}")

        # Also evaluate beamform-only baseline
        if also_eval_beamform:
            beam_metrics_list = []
            for fi, frame in enumerate(eval_frames):
                if fi >= n_frames:
                    break
                x_obs = frame.x_obs.cpu().numpy()
                y_obs = frame.y_obs.cpu().numpy()
                z_obs_np = frame.z_obs.cpu().numpy()
                gt = frame.gt_field.cpu().numpy()
                if len(x_obs) < 5:
                    continue

                X_obs = torch.tensor(
                    np.column_stack([x_obs, y_obs]),
                    dtype=torch.float32, device=device
                )
                z_obs = torch.tensor(z_obs_np, dtype=torch.float32, device=device)

                K_oo_b = build_K_disp_torch(X_obs, X_obs, beam_k_t, beam_theta_t, beam_A_t)
                K_po_b = build_K_disp_torch(X_pred, X_obs, beam_k_t, beam_theta_t, beam_A_t)
                mu_b = gp_predict_torch(K_oo_b, K_po_b, z_obs, sigma_obs)
                pred_b = mu_b.cpu().numpy().reshape(len(gt_y), len(gt_x))
                m_b = compute_metrics(pred_b, gt, ep_data['Hs'])
                beam_metrics_list.append(m_b)

            if beam_metrics_list:
                avg_beam = {
                    k: float(np.mean([m[k] for m in beam_metrics_list]))
                    for k in beam_metrics_list[0].keys()
                }
                beam_results[ep_key] = avg_beam
                print(f"  [BEAM {ep_key}] nRMSE={avg_beam['nRMSE']:.4f}  "
                      f"Pearson={avg_beam['Pearson']:.3f}  "
                      f"VR={avg_beam['VR']:.3f}")

    return results, beam_results


# ── Oracle baseline (for comparison) ─────────────────────────────────────────

def evaluate_oracle(test_episodes, n_frames=N_FRAMES_EVAL, sigma_obs=SIGMA_OBS):
    """Oracle K_disp: uses true wave components."""
    results = {}
    for h5_name in test_episodes:
        h5_path = os.path.join(DATA_DIR, h5_name)
        if not os.path.exists(h5_path):
            continue

        ep_key = h5_name.replace('episode_', '').replace('.h5', '')
        loader = EpisodeDataLoader(h5_path)
        wave_components = loader.wave_components
        gt_x = loader.gt_grid_x.cpu().numpy()
        gt_y = loader.gt_grid_y.cpu().numpy()
        Xg, Yg = np.meshgrid(gt_x, gt_y)
        X_pred = np.column_stack([Xg.ravel(), Yg.ravel()])

        eval_frames = loader.get_tracking_frames(tp_factor=2.0)
        metrics_list = []

        for fi, frame in enumerate(eval_frames):
            if fi >= n_frames:
                break
            x_obs = frame.x_obs.cpu().numpy()
            y_obs = frame.y_obs.cpu().numpy()
            z_obs = frame.z_obs.cpu().numpy()
            gt = frame.gt_field.cpu().numpy()
            if len(x_obs) < 5:
                continue

            X_obs = np.column_stack([x_obs, y_obs])

            # Build K with oracle components
            A = wave_components[:, 0]
            kx = wave_components[:, 2]
            ky = wave_components[:, 3]

            # K_oo
            dx_oo = X_obs[:, 0:1] - X_obs[:, 0][None, :]
            dy_oo = X_obs[:, 1:2] - X_obs[:, 1][None, :]
            K_oo = np.zeros((len(X_obs), len(X_obs)))
            for n in range(len(A)):
                K_oo += 0.5 * A[n]**2 * np.cos(kx[n] * dx_oo + ky[n] * dy_oo)

            # K_po
            dx_po = X_pred[:, 0:1] - X_obs[:, 0][None, :]
            dy_po = X_pred[:, 1:2] - X_obs[:, 1][None, :]
            K_po = np.zeros((len(X_pred), len(X_obs)))
            for n in range(len(A)):
                K_po += 0.5 * A[n]**2 * np.cos(kx[n] * dx_po + ky[n] * dy_po)

            # GP solve
            N = len(z_obs)
            K_noise = K_oo + sigma_obs**2 * np.eye(N) + 1e-6 * np.eye(N)
            try:
                alpha = np.linalg.solve(K_noise, z_obs)
            except np.linalg.LinAlgError:
                continue
            pred = (K_po @ alpha).reshape(len(gt_y), len(gt_x))

            m = compute_metrics(pred, gt, loader.Hs_true)
            metrics_list.append(m)

        if metrics_list:
            avg_metrics = {
                k: float(np.mean([m[k] for m in metrics_list]))
                for k in metrics_list[0].keys()
            }
            results[ep_key] = avg_metrics
            print(f"  [ORACLE {ep_key}] nRMSE={avg_metrics['nRMSE']:.4f}  "
                  f"Pearson={avg_metrics['Pearson']:.3f}")

    return results


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    refine_mode = sys.argv[1] if len(sys.argv) > 1 else 'full'
    assert refine_mode in ('full', 'amp'), f"bad refine_mode {refine_mode}"
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"DDKL Pilot — device={device}  refine_mode={refine_mode}")
    print(f"Train episodes: {len(TRAIN_EPISODES)}")
    print(f"Test episodes: {len(TEST_EPISODES)}")
    print(f"N_comp={N_COMP}, N_frames_train={N_FRAMES_TRAIN}")

    # Build model
    model = DDKLRefiner(n_comp=N_COMP, hidden_dim=64, refine_mode=refine_mode)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {n_params:,}")

    # Train
    t0 = time.time()
    history, best_loss = train_ddkl(
        model, TRAIN_EPISODES,
        n_epochs=150, lr=1e-3, device=device,
        n_grid_subsample=400, lambda_nml=0.1,
    )
    train_time = time.time() - t0
    print(f"\nTraining done: {train_time:.0f}s, best loss={best_loss:.4f}")

    # Save checkpoint
    os.makedirs('checkpoints', exist_ok=True)
    ckpt_path = f'checkpoints/ddkl_pilot_{refine_mode}.pt'
    torch.save(model.state_dict(), ckpt_path)
    print(f"Saved: {ckpt_path}")

    # Evaluate DDKL
    print("\n" + "=" * 60)
    print("DDKL Evaluation (test episodes)")
    print("=" * 60)
    t1 = time.time()
    ddkl_results, beam_results = evaluate_ddkl(model, TEST_EPISODES, device=device)
    eval_time = time.time() - t1

    # Oracle baseline
    print("\n" + "=" * 60)
    print("Oracle K_disp Evaluation")
    print("=" * 60)
    oracle_results = evaluate_oracle(TEST_EPISODES)

    # Summary
    print("\n" + "=" * 60)
    print("COMPARISON SUMMARY")
    print("=" * 60)

    def avg_metric(d, key):
        vals = [d[k][key] for k in d if key in d[k]]
        return float(np.mean(vals)) if vals else None

    print(f"\n  {'Method':<20} {'nRMSE':>8} {'Pearson':>8} {'VR':>8}")
    print("  " + "-" * 50)

    ddkl_nrmse = avg_metric(ddkl_results, 'nRMSE')
    ddkl_pearson = avg_metric(ddkl_results, 'Pearson')
    ddkl_vr = avg_metric(ddkl_results, 'VR')
    beam_nrmse = avg_metric(beam_results, 'nRMSE')
    beam_pearson = avg_metric(beam_results, 'Pearson')
    beam_vr = avg_metric(beam_results, 'VR')
    oracle_nrmse = avg_metric(oracle_results, 'nRMSE')
    oracle_pearson = avg_metric(oracle_results, 'Pearson')
    oracle_vr = avg_metric(oracle_results, 'VR')

    if ddkl_nrmse:
        print(f"  {'DDKL Refined':<20} {ddkl_nrmse:>8.4f} {ddkl_pearson:>8.3f} {ddkl_vr:>8.3f}")
    if beam_nrmse:
        print(f"  {'Beamform Only':<20} {beam_nrmse:>8.4f} {beam_pearson:>8.3f} {beam_vr:>8.3f}")
    if oracle_nrmse:
        print(f"  {'Oracle K_disp':<20} {oracle_nrmse:>8.4f} {oracle_pearson:>8.3f} {oracle_vr:>8.3f}")

    print("  --- cross-protocol refs (30-ep, NOT same 6 test eps) ---")
    print(f"  {'GP K_disp ms5*':<20} {'0.2232':>8} {'0.489':>8} {'0.783':>8}")
    print(f"  {'WaveSplat*':<20} {'0.2630':>8} {'0.346':>8} {'0.914':>8}")
    print(f"  {'WavePINO S4*':<20} {'0.2280':>8} {'0.376':>8} {'0.309':>8}")
    print("  (* = 30-episode protocol; same-protocol head-to-head is "
          "DDKL vs Beamform Only vs Oracle above)")

    if ddkl_nrmse and beam_nrmse:
        improvement = (beam_nrmse - ddkl_nrmse) / beam_nrmse * 100
        print(f"\n  DDKL vs Beamform: {improvement:+.1f}% nRMSE improvement")

    # Save results
    output = {
        'description': 'DDKL v3: neural refinement of beamforming K_disp estimates',
        'config': {
            'n_comp': N_COMP,
            'hidden_dim': 64,
            'n_epochs': 150,
            'lr': 1e-3,
            'lambda_nml': 0.1,
            'n_frames_train': N_FRAMES_TRAIN,
            'n_frames_eval': N_FRAMES_EVAL,
            'sigma_obs': SIGMA_OBS,
            'n_train_episodes': len(TRAIN_EPISODES),
            'n_params': n_params,
            'device': device,
        },
        'ddkl': ddkl_results,
        'beamform_only': beam_results,
        'oracle': oracle_results,
        'training': {
            'train_time_sec': train_time,
            'eval_time_sec': eval_time,
            'best_loss': best_loss,
            'loss_history': history,
        },
        'summary': {
            'ddkl_mean_nRMSE': ddkl_nrmse,
            'ddkl_mean_Pearson': ddkl_pearson,
            'beam_mean_nRMSE': beam_nrmse,
            'beam_mean_Pearson': beam_pearson,
            'oracle_mean_nRMSE': oracle_nrmse,
            'oracle_mean_Pearson': oracle_pearson,
        },
    }

    output['config']['refine_mode'] = refine_mode
    out_path = f'results/ddkl_pilot_{refine_mode}.json'
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == '__main__':
    main()
