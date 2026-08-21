"""CON (6) k-position -- attack 3: is "CRB 0.0285 < 0.0629, therefore no information wall"
a real bound, or a regularisation artifact?

Re-runs the SAME engine (scripts/d1_crb.py) at the SAME deployed init window as
tmp/rx2_init_2026-08-05/CON_crb.py, but sweeps the ridge that
`field_nrmse_crb(..., ridge_rel=1e-10)` silently adds before inverting F.

CPU ONLY (CUDA_VISIBLE_DEVICES="") -- the GPU is busy generating episodes.
Reads only existing episode files. Writes nothing but its own stdout.
"""
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..'))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, 'scripts'))

import d1_crb as C
from wavesplat.data_loader import EpisodeDataLoader

DATA = os.path.join(REPO, 'data_finemesh_2026-07-23')
EPS = ['episode_BF4_15comp_seed0_uniform.h5', 'episode_BF4_30comp_seed0_uniform.h5',
       'episode_BF5_15comp_seed0_uniform.h5', 'episode_BF5_30comp_seed0_uniform.h5']
SIGMA = 0.02
RIDGES = [1e-6, 1e-8, 1e-10, 1e-12, 1e-14, 0.0]

print('=' * 108)
print('CON-6 attack 3 -- ridge sensitivity of the "no information wall" CRB')
print(f'engine scripts/d1_crb.py  device={C.DEVICE}  sigma={SIGMA} m  '
      f'(rx2 CON used ridge_rel=1e-10, the function default)')
print('=' * 108)

crbs = {r: [] for r in RIDGES}
for e in EPS:
    path = os.path.join(DATA, e)
    ld = EpisodeDataLoader(path)
    depth = float(ld.depth)
    t_star = float(ld.get_init_window(tp_factor=2.0)[-1].t)
    ep = C.load_ep(path, depth)
    N = ep['theta'].shape[0]
    x, y, t = C.collect_obs(ep, t_star)
    F = C.fisher(ep['theta'], x, y, t, depth, SIGMA)
    Gm = C.field_design(ep['theta'], ep['gx'], ep['gy'], t_star, depth)

    ev = torch.linalg.eigvalsh(F).cpu().numpy()
    ev = np.sort(ev)
    n = F.shape[0]
    tr_over_n = float(torch.trace(F)) / n
    print(f'\n{e}')
    print(f'  N={N}  4N={n}  M={int(x.numel())}  depth={depth:.1f}  Hs={ep["Hs"]:.3f}')
    print(f'  eig(F): min={ev[0]:.4e}  max={ev[-1]:.4e}  cond={ev[-1] / ev[0]:.3e}  '
          f'tr(F)/n={tr_over_n:.4e}')
    neg = int((ev <= 0).sum())
    below = int((ev < ev[-1] * 2.2e-16 * n).sum())
    print(f'  eigenvalues <= 0 (numerically indefinite): {neg}/{n}   '
          f'below float64 noise floor eps*n*lmax: {below}/{n}')
    for r in RIDGES:
        lam = r * tr_over_n
        print(f'    ridge_rel={r:<8g} lambda={lam:12.4e}  lambda/eig_min={lam / ev[0]:10.3e}  ',
              end='')
        try:
            crb, _ = C.field_nrmse_crb(F, Gm, ep['Hs'], ridge_rel=r)
            crbs[r].append(crb)
            print(f'CRB nRMSE = {crb:.4f}')
        except Exception as exc:
            crbs[r].append(float('nan'))
            print(f'FAILED: {exc!r}')

print('\n' + '=' * 108)
print('MEAN CRB over the 4 crowded episodes vs the ridge that was never reported')
print('=' * 108)
print(f'{"ridge_rel":>12s} {"mean CRB":>10s}   verdict vs oracle init_nrmse 0.0629')
for r in RIDGES:
    m = float(np.mean(crbs[r]))
    print(f'{r:12g} {m:10.4f}   ' + ('CRB < 0.0629 -> "no wall"' if m < 0.0629
                                     else 'CRB > 0.0629 -> the bound FORBIDS the oracle level'))
print('\n  The quoted 0.0285 is the ridge_rel=1e-10 row. It is the DEFAULT argument of')
print('  field_nrmse_crb, was never swept, and never appears in the VERDICT sentence.')
