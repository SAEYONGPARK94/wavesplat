"""NEUTRAL (axis 6) — is "CRB 0.0285 < oracle 0.0629 => no information wall" a sound claim?

Re-runs the SAME engine (scripts/d1_crb.py) at the SAME window as
tmp/rx2_init_2026-08-05/CON_crb.py, then attacks the three places where the number
could be an artifact rather than a bound:

  A. RIDGE SENSITIVITY. field_nrmse_crb adds lam = 1e-10 * tr(F)/n to F before
     inverting. cond(F) was reported up to 8.5e13 in float64 (eps=2.2e-16). A ridge
     shrinks tr(F^-1 G) MONOTONICALLY DOWNWARD, so the printed CRB is a lower bound
     on a lower bound. Sweep the ridge and see how much of 0.0285 is ridge.
  B. AGGREGATION. 0.0285 is a MEAN over 4 episodes; the project's frozen convention
     everywhere else is a MEDIAN. Report both, and the per-episode comparison.
  C. WHAT IS BOUNDED. Print the parameter block the bound covers and the quantity it
     bounds, so the claim can be scoped correctly.

CPU-only (CUDA_VISIBLE_DEVICES='' is set before torch import) so the GPU episode
generation running in parallel is untouched.
"""
import os

os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ.setdefault('OMP_NUM_THREADS', '4')

import json
import math
import statistics as st
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..'))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, 'scripts'))

import d1_crb as C
from wavesplat.data_loader import EpisodeDataLoader

assert str(C.DEVICE) == 'cpu', f'expected cpu, got {C.DEVICE}'

DATA = os.path.join(REPO, 'data_finemesh_2026-07-23')
EPS = ['episode_BF4_15comp_seed0_uniform.h5', 'episode_BF4_30comp_seed0_uniform.h5',
       'episode_BF5_15comp_seed0_uniform.h5', 'episode_BF5_30comp_seed0_uniform.h5']
SIGMA = 0.02
RIDGES = [1e-6, 1e-8, 1e-10, 1e-12, 1e-14, 0.0]

OUT = []


def P(*a):
    s = ' '.join(str(x) for x in a)
    print(s, flush=True)
    OUT.append(s)


def crb_at_ridge(F, Gmat, Hs, ridge_rel):
    n = F.shape[0]
    if ridge_rel == 0.0:
        try:
            Finv = torch.linalg.inv(F)
        except RuntimeError:
            return float('nan')
    else:
        lam = ridge_rel * torch.trace(F) / n
        Finv = torch.linalg.inv(F + lam * torch.eye(n, dtype=C.DT))
    var = float(torch.trace(Finv @ Gmat))
    return math.sqrt(max(var, 0.0)) / Hs


def main():
    P('=' * 104)
    P('NEUTRAL axis-6 — audit of "CRB 0.0285 < oracle 0.0629 => the gap is reachable in principle"')
    P('=' * 104)
    P(f'engine: scripts/d1_crb.py (unmodified import)  device={C.DEVICE}  sigma={SIGMA} m  dtype={C.DT}')
    P(f'gradcheck (analytic Jacobian vs autograd): {C.gradcheck():.3e}')
    P('')

    rows = []
    for e in EPS:
        path = os.path.join(DATA, e)
        ld = EpisodeDataLoader(path)
        depth = float(ld.depth)
        Tp = float(ld.Tp_true)
        t_star = float(ld.get_init_window(tp_factor=2.0)[-1].t)
        ep = C.load_ep(path, depth)
        N = ep['theta'].shape[0]
        x, y, t = C.collect_obs(ep, t_star)
        F = C.fisher(ep['theta'], x, y, t, depth, SIGMA)
        Gm = C.field_design(ep['theta'], ep['gx'], ep['gy'], t_star, depth)
        cond = float(torch.linalg.cond(F))
        sv = torch.linalg.svdvals(F)
        # numerical rank at float64 machine precision
        tol = sv.max() * F.shape[0] * torch.finfo(C.DT).eps
        rank = int((sv > tol).sum())
        vals = {r: crb_at_ridge(F, Gm, ep['Hs'], r) for r in RIDGES}
        rows.append(dict(ep=e, N=N, M=int(x.numel()), Tp=Tp, t_star=t_star, depth=depth,
                         cond=cond, rank=rank, npar=4 * N, Hs=ep['Hs'], crb=vals))
        P(f'{e[8:-3]:28s} N={N:3d} 4N={4*N:4d} M={int(x.numel()):6d} Tp={Tp:.1f} '
          f't*={t_star:5.2f} cond(F)={cond:8.2e} numrank={rank}/{4*N}')

    P('')
    P('=' * 104)
    P('A. RIDGE SENSITIVITY of the field-nRMSE CRB   (ridge_rel = lam / (tr(F)/n))')
    P('=' * 104)
    P(f'{"episode":28s}' + ''.join(f'{r:>12.0e}' if r else f'{"exact(0)":>12s}' for r in RIDGES))
    for r in rows:
        P(f'{r["ep"][8:-3]:28s}' + ''.join(f'{r["crb"][k]:12.4f}' for k in RIDGES))
    P('-' * 104)
    P(f'{"MEAN (CON_crb.py rule)":28s}'
      + ''.join(f'{st.mean([r["crb"][k] for r in rows]):12.4f}' for k in RIDGES))
    P(f'{"MEDIAN (project rule)":28s}'
      + ''.join(f'{st.median([r["crb"][k] for r in rows]):12.4f}' for k in RIDGES))
    P('')
    d = rows[0]['crb']
    P('  Reading: the ridge is a DOWNWARD bias on tr(F^-1 G). If the column moves as the ridge')
    P('  shrinks, the printed 0.0285 is partly regularisation, not information.')

    P('')
    P('=' * 104)
    P('B. AGGREGATION — 0.0285 is a MEAN of 4; the frozen project convention is a MEDIAN')
    P('=' * 104)
    ref = [r['crb'][1e-10] for r in rows]
    P(f'  per-episode CRB @ridge 1e-10 : {["%.4f" % v for v in ref]}')
    P(f'  mean   = {st.mean(ref):.4f}   <- the quoted 0.0285')
    P(f'  median = {st.median(ref):.4f}')
    P(f'  max    = {max(ref):.4f}   (BF4_30comp; this single episode is 2.3x the mean)')
    P('  The oracle side (0.0629) is likewise a MEAN of 4 init_nrmse values')
    P('  [0.0279, 0.0918, 0.0290, 0.1029] read from tmp/s1s2_2026-08-05/results.jsonl.')

    P('')
    P('=' * 104)
    P('C. WHAT THE BOUND COVERS (read off scripts/d1_crb.py, not from any summary)')
    P('=' * 104)
    P('  parameter block : theta_i = (w_i, kx_i, ky_i, phi_i) for i=1..N  -> 4N free params')
    P('                    omega is NOT free (dispersion imposed as a hard constraint)')
    P('  bounded quantity: grid-averaged field reconstruction error at a SINGLE time t*,')
    P('                    normalised by Hs.  crb = sqrt(tr(F^-1 G)) / Hs,  G = <g g^T>_grid')
    P('  noise model     : eps ~ N(0, sigma^2) iid on ELEVATION, sigma = 0.02 m, sigma KNOWN')
    P('  regularity      : local bound evaluated AT THE TRUE theta, with N KNOWN = N_gt')
    P('  estimator class : UNBIASED only')
    P('')
    P('  Therefore the bound does NOT cover, and cannot be used to argue about:')
    P('   - model-order selection (adaptive_n): N is handed to the bound')
    P('   - any biased estimator: WaveSplat has L1 + energy pruning + a component budget')
    P('   - the tracked/forecast field: the bound is at ONE time, the deployed metric is a')
    P('     multi-frame average after tracking')
    P('   - real LiDAR noise: sigma=0.02 iid Gaussian on z, no dropout, no beam geometry')

    P('')
    P('=' * 104)
    P('D. DIRECTION-I ALREADY MEASURED THE BOUND TO BE VACUOUS HERE')
    P('=' * 104)
    P('  CON_crb.py lines 138-141 quote results/direction1/REPORT.md: at 1 s, 15c and 30c')
    P('  are both "CRB vacuous -- estimator beats the unbiased bound". A bound that the')
    P('  biased estimator already crosses cannot be used to certify that a target is')
    P('  reachable OR unreachable. CON_crb.py states this itself in its own docstring')
    P('  (lines 16-21) -- the limitation was declared, then dropped in the one-line summary.')

    json.dump({'rows': [{k: v for k, v in r.items() if k != 'crb'} | {
        'crb': {str(kk): vv for kk, vv in r['crb'].items()}} for r in rows]},
        open(os.path.join(HERE, 'neutral_crb_check.json'), 'w'), indent=1)
    open(os.path.join(HERE, 'neutral_crb_check.txt'), 'w').write('\n'.join(OUT) + '\n')


if __name__ == '__main__':
    main()
