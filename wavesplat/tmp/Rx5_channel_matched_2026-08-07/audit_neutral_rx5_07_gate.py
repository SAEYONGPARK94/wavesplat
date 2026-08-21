"""NEUTRAL audit 07 -- PREREG sec.6.5 / sec.1: re-derive the independence gate MYSELF
from the HDF5 files.  Do not trust K6's saved independence_gate.txt.
5 axes must each show 16 distinct draws across the 16 episodes."""
import os, hashlib
import numpy as np
import h5py

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, '..', '..', 'data_indep2_2026-08-07')
EPS = [f'episode_BF{bf}_{nc}comp_seed{s}_uniform_ra_rf_ds.h5'
       for bf in (4, 5) for nc in (15, 30) for s in range(4)]

def h(a):
    return hashlib.md5(np.ascontiguousarray(np.asarray(a, dtype=np.float64)).tobytes()).hexdigest()

with h5py.File(os.path.join(DATA, EPS[0]), 'r') as f:
    wc = np.asarray(f['config/wave_components'])
    print(f'wave_components shape = {wc.shape}')
    keys = []
    f.visit(lambda n: keys.append(n))
    print('datasets:', keys[:20])

axes = {'A': [], 'omega': [], 'kx': [], 'ky': [], 'phi': [], 'theta': [], 'truth': []}
for ep in EPS:
    with h5py.File(os.path.join(DATA, ep), 'r') as f:
        wc = np.asarray(f['config/wave_components'])
        A, om, kx, ky, phi = (wc[:, i] for i in range(5))
        axes['A'].append(h(np.sort(A)))
        axes['omega'].append(h(np.sort(om)))
        axes['kx'].append(h(np.sort(kx)))
        axes['ky'].append(h(np.sort(ky)))
        axes['phi'].append(h(np.sort(phi)))
        axes['theta'].append(h(np.sort(np.arctan2(ky, kx))))
        axes['truth'].append(h(np.asarray(f['frames/gt_wave_field'])[:3]))

print(f'\n{"axis":8s} {"distinct draws / 16":>20}')
worst = 99
for k, v in axes.items():
    n = len(set(v))
    if k in ('A', 'omega', 'phi', 'theta', 'truth') and v[0] != 'n/a':
        worst = min(worst, n)
    print(f'  {k:8s} {n:18d}   {"OK" if n == 16 else "SHARED DRAWS"}')

floor_p = 2.0 / 2 ** worst
print(f'\n  narrowest pre-registered axis = {worst}  ->  all-win floor p = {floor_p:.3e}')
print(f'  PREREG sec.1 claims 16 on all five axes, floor p = 3.05e-05  ->  '
      f'{"CONFIRMED" if worst == 16 else "NOT CONFIRMED"}')
