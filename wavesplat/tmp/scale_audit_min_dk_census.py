"""
Read-only diagnostic (SCALE_AUDIT, 2026-07-23): census of true wave-component wavenumber spacing
across every shipped episode, compared against the hardcoded `min_dk=0.15` [rad/m] suppression
radius used identically at `initializer.py:395`, `coarse_to_fine.py:190`, and
`sequential_update.py:515`. Reads only `config/wave_components` from each .h5 (ground truth
metadata) -- does not run the estimator, does not touch any production file.
"""
import glob
import h5py
import numpy as np
from scipy.spatial.distance import pdist

files = sorted(glob.glob('data/*.h5'))
n_total = len(files)
n_bad = 0
n_by_comp = {}
for fp in files:
    with h5py.File(fp, 'r') as f:
        wc = f['config/wave_components'][:]
    n = len(wc)
    n_by_comp.setdefault(n, [0, 0])
    n_by_comp[n][1] += 1
    if n < 2:
        continue
    kx, ky = wc[:, 2], wc[:, 3]
    pts = np.stack([kx, ky], axis=1)
    d = pdist(pts)
    if d.min() < 0.15:
        n_bad += 1
        n_by_comp[n][0] += 1

print(f'Total episodes: {n_total}')
print(f'Episodes with >=1 true-component pair closer than min_dk=0.15 rad/m: '
      f'{n_bad} ({100 * n_bad / n_total:.1f}%)')
print()
print('By component count (n_comp: bad/total):')
for k in sorted(n_by_comp):
    bad, tot = n_by_comp[k]
    print(f'  {k:3d} comp: {bad:3d}/{tot:3d}')

# Max |k| across the whole dataset, for the kx_range=(-4,4) inertness check.
maxk, maxfile = 0.0, None
for fp in files:
    with h5py.File(fp, 'r') as f:
        wc = f['config/wave_components'][:]
    k = np.sqrt(wc[:, 2] ** 2 + wc[:, 3] ** 2)
    if k.max() > maxk:
        maxk, maxfile = k.max(), fp
print(f'\nMax |k| across all {n_total} shipped episodes: {maxk:.4f} rad/m in {maxfile}')
