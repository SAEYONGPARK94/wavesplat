"""Cheap, mesh-independent probe: does the *beamforming initializer itself* fail to
detect components with |k| >~ 0.4 rad/m (paper/wavesplat.tex:1936), or is that number
actually the shipped mesh's own rendering cutoff dressed up as an algorithmic property?

Method: reuse a real episode's LiDAR sample geometry (x, y, t) -- so point count and
sparsity pattern are realistic -- but REPLACE z_obs with an analytically synthesized
single plane wave z = A cos(kx x + ky y - omega t + phi) + Gaussian noise (sigma=0.02 m,
matching wavesim/noise_model.py's range-noise floor), for |k| swept from 0.15 to 1.0
rad/m at fixed A=0.15 m (well above the noise floor, isolating resolution/detection from
amplitude/SNR). This never touches raycast_surface / the mesh, so if detection still
degrades above |k|~0.4 here, the 0.4 threshold is a genuine beamforming property; if it
does not, the paper's attribution is (at least partly) a mesh-rendering artifact restated
as an algorithm limitation.

Single seed (deterministic synthetic signal, no run-to-run estimator variance applies --
this is a one-shot beamform call, not an iterative fit), single amplitude, single
frequency-window duration matched to the episode's own T_init. Read-only w.r.t. any
production file.
"""
import sys
sys.path.insert(0, '.')
import numpy as np
import torch

from wavesplat.data_loader import EpisodeDataLoader
from wavesplat.optimizer.initializer import initialize_from_frames_v2
from wavesplat.model.dispersion import dispersion_omega

EP = 'data/episode_BF4_15comp_seed0_uniform.h5'
A_TEST = 0.15
NOISE_SIGMA = 0.02
DEPTH = 50.0

loader = EpisodeDataLoader(EP)
frames = loader.get_init_window()
print(f'episode={EP}  n_init_frames={len(frames)}  '
      f'pts/frame~{np.mean([len(f.x_obs) for f in frames]):.1f}')

rng = np.random.RandomState(0)
results = []
for k_mag in [0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.60, 0.75, 1.00]:
    theta = 0.6      # fixed direction, radians
    kx_true = k_mag * np.cos(theta)
    ky_true = k_mag * np.sin(theta)
    omega_true = dispersion_omega(torch.tensor(k_mag), DEPTH).item()
    phi_true = 1.234

    synth_frames = []
    for f in frames:
        x = f.x_obs.numpy()
        y = f.y_obs.numpy()
        t_val = f.t
        z_clean = A_TEST * np.cos(kx_true * x + ky_true * y - omega_true * t_val + phi_true)
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
    k_r = np.hypot(kx_r, ky_r)
    dist = np.hypot(kx_r - kx_true, ky_r - ky_true)
    results.append((k_mag, kx_true, ky_true, kx_r, ky_r, dist, init_w[0].item()))
    print(f'|k|_true={k_mag:.2f}  true=({kx_true:+.3f},{ky_true:+.3f})  '
          f'recovered=({kx_r:+.3f},{ky_r:+.3f})  dist={dist:.4f}  w_r={init_w[0].item():.4f}')

print()
print('Summary: dist should stay small (<~0.05, i.e. within one fine-grid bin) at every')
print('|k| if beamforming itself has no failure threshold near 0.4 rad/m in a')
print('mesh-independent, adequate-SNR synthetic test.')
