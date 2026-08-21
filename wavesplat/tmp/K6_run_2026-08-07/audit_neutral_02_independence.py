"""NEUTRAL re-run of the K6 independence gate (PREREG sec.3 / sec.6.1).

Written from scratch and DELIBERATELY DIFFERENT from independence_gate.py:
  - pairwise all-pairs distinctness (their n_draws() is a greedy first-match scan,
    which can undercount when the tolerance relation is not transitive),
  - two tolerances (1e-4 as pre-registered, and a loose 1e-2 stress test),
  - the gt-field axis is checked by max |corr| between episodes, not by allclose,
    because two independent fields can be near-duplicates without being byte-equal,
  - also reports the CONFIG fold structure (4 configs x 4 episode-seeds).
"""
import itertools
import os

import h5py
import numpy as np

REPO = '/home/vexit/wavesplat/wavesplat'
DIRS = {
    'K6 indep v2 (ra+rf+ds)': ('data_indep2_2026-08-07', True),
    'old finemesh 2026-07-23': ('data_finemesh_2026-07-23', False),
}
CROWDED16 = [f'episode_BF{bf}_{nc}comp_seed{s}_uniform.h5'
             for bf in (4, 5) for nc in (15, 30) for s in range(4)]


def load(d, indep):
    comps, flds = [], []
    for ep in CROWDED16:
        name = ep.replace('.h5', '_ra_rf_ds.h5') if indep else ep
        with h5py.File(os.path.join(REPO, d, name), 'r') as f:
            comps.append(np.array(f['config/wave_components']))
            flds.append(np.array(f['frames/gt_wave_field']))
    return comps, flds


def n_distinct_pairwise(vecs, tol):
    """Union-find over the 'indistinguishable' relation -> number of components."""
    n = len(vecs)
    par = list(range(n))

    def find(a):
        while par[a] != a:
            par[a] = par[par[a]]
            a = par[a]
        return a

    for i, j in itertools.combinations(range(n), 2):
        a, b = vecs[i], vecs[j]
        if len(a) == len(b) and np.allclose(a, b, atol=tol, rtol=0):
            ra, rb = find(i), find(j)
            if ra != rb:
                par[ra] = rb
    return len({find(i) for i in range(n)})


for label, (d, indep) in DIRS.items():
    print(f'\n=== {label}  ({d}) ===')
    comps, flds = load(d, indep)
    axes = {
        'amp   A': [c[:, 0] for c in comps],
        'omega  ': [c[:, 1] for c in comps],
        'phi    ': [c[:, 4] for c in comps],
        'theta  ': [np.arctan2(c[:, 3], c[:, 2]) for c in comps],
        '|k|    ': [np.hypot(c[:, 2], c[:, 3]) for c in comps],
        'gtfield': [f.ravel() for f in flds],
    }
    lo = 99
    for k, v in axes.items():
        d4 = n_distinct_pairwise(v, 1e-4)
        d2 = n_distinct_pairwise(v, 1e-2)
        lo = min(lo, d4)
        print(f'  {k}  distinct@1e-4 = {d4:2d}   distinct@1e-2 = {d2:2d}'
              f'   {"OK" if d4 == 16 else "<-- COLLAPSED"}')
    floor = min(1.0, 2.0 / 2 ** lo)
    print(f'  narrowest axis = {lo}  -> all-win floor p = {floor:.3e}'
          f'  {"alpha=0.05 reachable" if floor <= 0.05 else "alpha=0.05 UNREACHABLE"}')

    # gt-field near-duplication (correlation), the axis allclose cannot see
    F = np.array([f.ravel() for f in flds])
    F = (F - F.mean(1, keepdims=True))
    F /= (np.linalg.norm(F, axis=1, keepdims=True) + 1e-30)
    C = F @ F.T
    np.fill_diagonal(C, 0.0)
    i, j = np.unravel_index(np.argmax(np.abs(C)), C.shape)
    print(f'  max |corr| between distinct gt fields = {np.abs(C).max():.4f} '
          f'(ep{i} vs ep{j});  mean |corr| = {np.abs(C).sum()/(16*15):.4f}')

    # config folds
    print('  config folds (BFx_Ncomp): each has 4 episode-seeds -> n_folds = 4, '
          'sign-test floor p = 0.125')
