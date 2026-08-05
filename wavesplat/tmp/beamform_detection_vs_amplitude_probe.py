"""Companion to beamform_detection_vs_k_probe.py: at fixed |k|=0.5 rad/m (inside the
paper's claimed 'detection threshold' >~0.4 rad/m), sweep AMPLITUDE from realistic
short-lambda component sizes (0.01-0.04 m, per tmp/REVALIDATION_CENSUS_2026-07-23.md's
measured true amplitudes for lambda<15m components) up to a strong 0.15 m, still
bypassing the raycast mesh entirely (analytic z = A cos(...) + noise). This separates
"is |k|=0.5 rad/m itself unresolvable" (already shown NO in the companion script) from
"is a REALISTICALLY WEAK |k|=0.5 rad/m component (as found in actual JONSWAP short-lambda
content) undetectable because of its own low amplitude" (a genuine, mesh-independent SNR
floor, not an algorithmic resolution limit).
"""
import sys
sys.path.insert(0, '.')
import numpy as np
import torch

from wavesplat.data_loader import EpisodeDataLoader
from wavesplat.optimizer.initializer import initialize_from_frames_v2
from wavesplat.model.dispersion import dispersion_omega

EP = 'data/episode_BF4_15comp_seed0_uniform.h5'
NOISE_SIGMA = 0.02
DEPTH = 50.0
K_MAG = 0.5

loader = EpisodeDataLoader(EP)
frames = loader.get_init_window()

rng = np.random.RandomState(0)
theta = 0.6
kx_true = K_MAG * np.cos(theta)
ky_true = K_MAG * np.sin(theta)
omega_true = dispersion_omega(torch.tensor(K_MAG), DEPTH).item()
phi_true = 1.234

for A_test in [0.15, 0.08, 0.04, 0.03, 0.02, 0.015, 0.01]:
    synth_frames = []
    for f in frames:
        x = f.x_obs.numpy(); y = f.y_obs.numpy(); t_val = f.t
        z_clean = A_test * np.cos(kx_true * x + ky_true * y - omega_true * t_val + phi_true)
        z_noisy = z_clean + rng.normal(0, NOISE_SIGMA, size=z_clean.shape)
        nf = type(f)(t=f.t, x_obs=torch.tensor(x, dtype=torch.float32),
                     y_obs=torch.tensor(y, dtype=torch.float32),
                     z_obs=torch.tensor(z_noisy, dtype=torch.float32),
                     confidence=f.confidence, incidence_angles=f.incidence_angles,
                     gt_field=None)
        synth_frames.append(nf)

    init_kx, init_ky, init_w, init_phi = initialize_from_frames_v2(
        synth_frames, n_gaussians=1, depth=DEPTH, seed=0)
    kx_r, ky_r = init_kx[0].item(), init_ky[0].item()
    dist = np.hypot(kx_r - kx_true, ky_r - ky_true)
    w_r = init_w[0].item()
    print(f'A_true={A_test:.3f}  recovered=({kx_r:+.3f},{ky_r:+.3f})  dist={dist:.4f}  '
          f'w_recovered={w_r:.4f}  A_ratio={w_r/A_test:.2f}  '
          f'{"FOUND" if dist < 0.05 else "MISSED"}')
