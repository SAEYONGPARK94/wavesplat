"""Observation-channel probe: how much of each TRUE component's amplitude
actually reaches the LiDAR returns, shipped mesh vs regenerated fine mesh.

Pre-registered read (stated before the numbers were produced):
  This is an ORACLE-k measurement — the design matrix is built at the episode's
  TRUE (kx, ky, omega), so it is an upper bound on what ANY estimator could
  recover. It answers one question only:

     "Is the short-wavelength content PRESENT in the observations at all?"

  * If the old/new amplitude ratios are similar, every past claim of the form
    "short-lambda components are unrecoverable" is about estimation, and the
    mesh regeneration cannot touch it.
  * If old is near zero and new is near one, those claims were measured on a
    channel that did not carry the signal, and are circular.

Both meshes are paired (same seed => bit-identical `config/wave_components`,
asserted below). No estimator is involved; no seed sensitivity; runtime ~10 s.

Usage:  python3 tmp/revalidation_shortlambda_probe.py
"""
import os
import sys

import h5py
import numpy as np

OLD = 'data/'
NEW = 'data_finemesh_2026-07-23/'

EPISODES = [
    # the exact episode the 2026-07-23 ANM gate called "weak short-lambda
    # unrecoverable" on (NOVELTY_ROUND3_execution, read 2)
    'episode_BF3_2comp_seed1_slope.h5',
    'episode_BF3_5comp_seed0_uniform.h5',
    'episode_BF4_15comp_seed0_uniform.h5',
    'episode_BF4_30comp_seed0_uniform.h5',
    'episode_BF5_30comp_seed0_uniform.h5',
]


def load(path):
    with h5py.File(path, 'r') as f:
        lp = f['frames/lidar_points'][:]
        vm = f['frames/lidar_valid_mask'][:].astype(bool)
        ts = f['frames/timestamps'][:]
        comp = f['config/wave_components'][:]      # [A, omega, kx, ky, phi, Q]
    X, Y, T, Z = [], [], [], []
    for i in range(lp.shape[0]):
        m = vm[i]
        X.append(lp[i][m][:, 0])
        Y.append(lp[i][m][:, 1])
        Z.append(lp[i][m][:, 2])
        T.append(np.full(int(m.sum()), ts[i]))
    return (np.concatenate(X), np.concatenate(Y),
            np.concatenate(T), np.concatenate(Z), comp)


def oracle_amplitudes(x, y, t, z, comp):
    """LS fit of cos/sin coefficients at the TRUE wavenumbers -> per-component
    recovered amplitude. Over-determined ~500:1 for the largest episode."""
    om, kx, ky = comp[:, 1], comp[:, 2], comp[:, 3]
    psi = (kx[None] * x[:, None] + ky[None] * y[:, None]
           - om[None] * t[:, None])
    D = np.concatenate([np.cos(psi), np.sin(psi)], axis=1)
    coef, *_ = np.linalg.lstsq(D, z, rcond=None)
    n = len(om)
    return np.hypot(coef[:n], coef[n:])


def main():
    print('component-amplitude transmission through the LiDAR observation '
          'channel (oracle-k LS)\n')
    for name in EPISODES:
        pa, pb = OLD + name, NEW + name
        if not (os.path.exists(pa) and os.path.exists(pb)):
            print(f'{name}: MISSING on one side, skipped')
            continue
        xa, ya, ta, za, ca = load(pa)
        xb, yb, tb, zb, cb = load(pb)
        assert np.array_equal(ca, cb), f'{name}: components differ -> not paired'

        A = ca[:, 0]
        lam = 2.0 * np.pi / np.hypot(ca[:, 2], ca[:, 3])
        ra = oracle_amplitudes(xa, ya, ta, za, ca) / A
        rb = oracle_amplitudes(xb, yb, tb, zb, cb) / A

        var = A ** 2 / 2.0
        short = lam < 14.0
        share = 100.0 * var[short].sum() / var.sum()
        print(f'{name}   n_obs old/new = {len(za)}/{len(zb)}   '
              f'variance share of lambda<14 m = {share:.1f}%')
        for i in np.argsort(lam)[:6]:
            print(f'    lambda={lam[i]:7.2f} m  A_true={A[i]:.4f}   '
                  f'A_rec/A_true  OLD={ra[i]:.2f}  NEW={rb[i]:.2f}')
        print()


if __name__ == '__main__':
    sys.exit(main())
