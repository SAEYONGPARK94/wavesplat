"""CON attack 2 — where is the information-theoretic floor at the DEPLOYED geometry?

The oracle arm does not estimate (kx, ky); it is HANDED them. So `init_nrmse = 0.0629`
is not evidence that 0.0629 is reachable -- it is only evidence that the representation
can express the field. The question the target claim skips is: how much of the 0.1809
"init gap" lies BELOW the bound for any estimator that has to read (kx, ky) out of this
sensor?

This re-derives the Direction-I field-nRMSE CRB (scripts/d1_crb.py, engine unchanged) at
exactly the observation window the deployed initializer sees:
    T_obs = 2 * Tp   (wavesplat.data_loader.get_init_window(tp_factor=2.0))
    t*    = the last init frame -- the same frame `init_nrmse` is scored on
    depth = the episode's own depth (NOT d1_crb's hard-coded 50 m)
on the same 4 crowded episodes the prize is quoted on.

LIMITATION, stated up front: the CRB bounds UNBIASED estimators. WaveSplat is biased
(L1 + energy pruning + a component budget), and Direction I already measured that the
biased estimator BEATS the unbiased bound in this regime. So a CRB above 0.0629 does not
prove 0.0629 unreachable. What it does establish is the conditioning of the likelihood --
i.e. whether the DATA contain a gradient that could route a search to the oracle point.
The empirical counterpart is CON_probe.py's observation-residual comparison.
"""
import os, sys, json, math
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..'))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, 'scripts'))

import d1_crb as C          # engine verbatim (analytic Jacobian, autograd-checked 5.3e-15)
from wavesplat.data_loader import EpisodeDataLoader

DATA = os.path.join(REPO, 'data_finemesh_2026-07-23')
EPS = ['episode_BF4_15comp_seed0_uniform.h5', 'episode_BF4_30comp_seed0_uniform.h5',
       'episode_BF5_15comp_seed0_uniform.h5', 'episode_BF5_30comp_seed0_uniform.h5']
SIGMA = 0.02
OUT = []


def P(*a):
    s = ' '.join(str(x) for x in a); print(s, flush=True); OUT.append(s)


P('=' * 100)
P('CON attack 2 — CRB at the DEPLOYED initialisation window')
P('=' * 100)
P(f'engine: scripts/d1_crb.py  device={C.DEVICE}  sigma={SIGMA} m')
# integrity gate: the engine must still pass its own analytic-vs-autograd check
try:
    gc = C.gradcheck()
    P(f'gradcheck (analytic Jacobian vs autograd): {gc}')
except Exception as e:
    P(f'gradcheck raised: {e!r}')
P('')

rows = []
P(f'{"episode":38s} {"N":>3s} {"Tp":>5s} {"T_obs":>6s} {"M":>7s} {"depth":>6s} '
  f'{"cond(F)":>10s} {"CRB nRMSE":>10s} {"CRBdir deg":>10s} {"CRB|k| rel":>10s}')
for e in EPS:
    path = os.path.join(DATA, e)
    ld = EpisodeDataLoader(path)
    depth = float(ld.depth)
    Tp = float(ld.Tp_true)
    t_init = 2.0 * Tp
    init_frames = ld.get_init_window(tp_factor=2.0)
    t_star = float(init_frames[-1].t)

    ep = C.load_ep(path, depth)
    N = ep['theta'].shape[0]
    x, y, t = C.collect_obs(ep, t_star)
    F = C.fisher(ep['theta'], x, y, t, depth, SIGMA)
    Gm = C.field_design(ep['theta'], ep['gx'], ep['gy'], t_star, depth)
    crb, Finv = C.field_nrmse_crb(F, Gm, ep['Hs'])
    cond = float(torch.linalg.cond(F))
    dirs = C.direction_crb_deg(ep['theta'], Finv, N)
    kx, ky = ep['theta'][:, 1], ep['theta'][:, 2]
    kmag = torch.sqrt(kx ** 2 + ky ** 2)
    sk = []
    for i in range(N):
        g = torch.zeros(4 * N, dtype=C.DT, device=Finv.device)
        g[N + i] = kx[i] / kmag[i]; g[2 * N + i] = ky[i] / kmag[i]
        sk.append(float(torch.sqrt(torch.clamp(g @ Finv @ g, min=0))) / float(kmag[i]))
    rows.append(dict(ep=e, N=N, Tp=Tp, t_star=t_star, M=int(x.numel()), depth=depth,
                     cond=cond, crb=crb, dir_med=float(np.median(dirs)),
                     k_rel_med=float(np.median(sk)), Hs=ep['Hs']))
    P(f'{e:38s} {N:3d} {Tp:5.2f} {t_star:6.2f} {int(x.numel()):7d} {depth:6.1f} '
      f'{cond:10.2e} {crb:10.4f} {np.median(dirs):10.3f} {np.median(sk):10.4f}')

P('')
P('=' * 100)
P('COMPARISON with the measured arms (crowded 4, median-of-3 per episode)')
P('=' * 100)
s1s2 = []
for line in open(os.path.join(REPO, 'tmp', 's1s2_2026-08-05', 'results.jsonl')):
    try:
        r = json.loads(line)
    except json.JSONDecodeError:
        continue
    if 'error' not in r:
        s1s2.append(r)


def lvl(arm, ep, m='init_nrmse'):
    v = [r[m] for r in s1s2 if r['arm'] == arm and r['episode'] == ep
         and np.isfinite(r.get(m, np.nan))]
    return float(np.median(v)) if v else float('nan')


P(f'{"episode":38s} {"CRB":>8s} {"oracle+frz":>11s} {"oracle":>8s} {"shipped":>8s} '
  f'{"best deploy":>12s}')
for r in rows:
    e = r['ep']
    P(f'{e:38s} {r["crb"]:8.4f} {lvl("S2-T1f", e):11.4f} {lvl("S2-T1", e):8.4f} '
      f'{lvl("S1-A0", e):8.4f} {lvl("S1-A3", e):12.4f}')
P(f'{"MEAN":38s} {np.mean([r["crb"] for r in rows]):8.4f} '
  f'{np.mean([lvl("S2-T1f", r["ep"]) for r in rows]):11.4f} '
  f'{np.mean([lvl("S2-T1", r["ep"]) for r in rows]):8.4f} '
  f'{np.mean([lvl("S1-A0", r["ep"]) for r in rows]):8.4f} '
  f'{np.mean([lvl("S1-A3", r["ep"]) for r in rows]):12.4f}')

P('')
crbm = float(np.mean([r['crb'] for r in rows]))
orc = float(np.mean([lvl('S2-T1f', r['ep']) for r in rows]))
dep = float(np.mean([lvl('S1-A3', r['ep']) for r in rows]))
P(f'  quoted gap            = {dep - orc:.4f}  (best deployable init -> oracle+frozen)')
if crbm > orc:
    P(f'  portion BELOW the unbiased CRB ({crbm:.4f}) = {crbm - orc:.4f} '
      f'= {100*(crbm-orc)/(dep-orc):.1f}% of the quoted gap')
    P(f'  portion above the bound (recoverable by an unbiased estimator) = '
      f'{dep - crbm:.4f}')
else:
    P(f'  the CRB ({crbm:.4f}) sits BELOW the oracle arm -> the bound does not '
      f'forbid reaching the oracle; attack 2 does not land on this evidence.')

P('')
P('  Direction-I information wall for reference (results/direction1/REPORT.md, 2026-06-27,')
P('  BF4 seed0, d=50 m): T_wall(CRB<0.25) = 0.1 s (5c) / 4.3 s (15c) / 9.7 s (30c);')
P('  cond(F)@1s = 7.7e4 / 8.6e12 / 9.0e17. Estimator-vs-CRB at 1 s: 15c and 30c both')
P('  "CRB vacuous -- estimator beats the unbiased bound".')

json.dump({'rows': rows, 'crb_mean': crbm, 'oracle_mean': orc, 'best_deploy_mean': dep},
          open(os.path.join(HERE, 'CON_crb.json'), 'w'), indent=1)
open(os.path.join(HERE, 'CON_crb.txt'), 'w').write('\n'.join(OUT) + '\n')
