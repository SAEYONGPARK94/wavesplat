"""NEUTRAL (Rx-6) part 3: re-run the sec.6.1 independence gate with my own code.
Deliberately NOT the same implementation: I count distinct draws by exact-byte hash AND
by a tolerance-based pass, and I add a cross-episode min-distance so a near-duplicate
cannot hide behind a loose tolerance.  Read-only on the data."""
import hashlib
import itertools
import os

import h5py
import numpy as np

REPO = '/home/vexit/wavesplat/wavesplat'
DIRS = ['data_indep2_2026-08-07', 'data_indep3_2026-08-07']
EPS = [f'episode_BF{bf}_{nc}comp_seed{s}_uniform_ra_rf_ds.h5'
       for bf in (4, 5) for nc in (15, 30) for s in range(8)]

comps, flds, paths = [], [], []
for ep in EPS:
    p = None
    for d in DIRS:
        q = os.path.join(REPO, d, ep)
        if os.path.exists(q):
            p = q
            break
    assert p, ep
    paths.append(p)
    with h5py.File(p, 'r') as f:
        comps.append(np.array(f['config/wave_components']))
        flds.append(np.array(f['frames/gt_wave_field']))
print(f'read {len(EPS)} episodes; component-matrix shapes: '
      f'{sorted(set(c.shape for c in comps))}')

AX = {
    'amp A':   [c[:, 0] for c in comps],
    'omega':   [c[:, 1] for c in comps],
    'phase':   [c[:, 4] for c in comps],
    'theta':   [np.arctan2(c[:, 3], c[:, 2]) for c in comps],
    'gt field': [f.ravel() for f in flds],
}
print(f'\n{"axis":10s} {"exact-hash draws":>17s} {"tol1e-4 draws":>14s} '
      f'{"min pairwise dist":>18s}')
ok = True
for name, vs in AX.items():
    h = len({hashlib.md5(np.ascontiguousarray(v, dtype=np.float64)).hexdigest() for v in vs})
    seen = []
    for v in vs:
        if not any(len(v) == len(s) and np.allclose(v, s, atol=1e-4, rtol=0) for s in seen):
            seen.append(v)
    mind = min(float(np.max(np.abs(a - b))) for a, b in itertools.combinations(vs, 2)
               if len(a) == len(b)) if name != 'gt field' else \
        min(float(np.max(np.abs(a - b))) for a, b in itertools.combinations(vs, 2)
            if len(a) == len(b))
    good = (h == 32 and len(seen) == 32)
    ok &= good
    print(f'{name:10s} {h:17d} {len(seen):14d} {mind:18.3e}  {"OK" if good else "FAIL"}')
print(f'\nGATE (my re-run): {"PASS" if ok else "FAIL"}   '
      f'-> all-win floor p = {min(1.0, 2/2**32):.2e}')

# also: are BF4/BF5 groups genuinely different sea states (sanity on the fold structure)?
with h5py.File(paths[0], 'r') as f:
    print('\nkeys(config):', list(f['config'].attrs.keys())[:12])
