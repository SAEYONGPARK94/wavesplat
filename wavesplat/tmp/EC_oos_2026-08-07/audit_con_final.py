"""CON audit (E-C) part 5 -- (i) could PRIMARY-1 have returned the other branch at all?
(ii) sensor noise floor vs the fitted residual, (iii) NaN accounting.  Read-only."""
import json
import math
import os
import statistics as st
import sys
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
TMP = os.path.join(HERE, '..')
REPO = '/home/vexit/wavesplat/wavesplat'
R = [json.loads(l) for l in open(os.path.join(HERE, 'results.jsonl'))]
I = defaultdict(list)
for r in R:
    I[(r['cell'], r['episode'])].append(r)
NULL = {}
for nf in ('Rx6_channel_curve_2026-08-07/nulls_rx6.json',
           'Rx7_tracker_deploy_2026-08-07/nulls_rx7.json'):
    p = os.path.join(TMP, nf)
    if os.path.exists(p):
        for n in json.load(open(p)):
            NULL[n['episode']] = n['zero_null_tracking_nRMSE_mean']
EPS = sorted({r['episode'] for r in R})


def med(c, e, k):
    v = [x[k] for x in I[(c, e)] if x.get(k) is not None and x[k] == x[k]]
    return st.median(v) if v else None


KEPT = [e for e in EPS if med('AUTO', e, 'nRMSE') < NULL[e]]

print('=== [1] could PRIMARY-1 have returned OOS-NOT-LOWER? (design falsifiability) ===')
need = []
for e in KEPT:
    ai = med('AUTO', e, 'obs_resid_rms')
    bi = med('AUTO-NOTRACK', e, 'obs_resid_rms')
    need.append(bi / ai)          # optimism ratio AUTO would need for its oos to reach NOTRACK
meas = [med('AUTO', e, 'obs_resid_rms_oos') / med('AUTO', e, 'obs_resid_rms') for e in KEPT]
print(f'  AUTO oos/in ratio REQUIRED to flip the verdict: median={st.median(need):.2f}x  '
      f'min over 45 episodes={min(need):.2f}x')
print(f'  AUTO oos/in ratio MEASURED:                     median={st.median(meas):.3f}x  '
      f'max over 45 episodes={max(meas):.3f}x')
print(f'  episodes where the measured ratio even approached the required one: '
      f'{sum(1 for a, b in zip(meas, need) if a >= b)}/45')
print('  => with a 1-frame lead the OOS branch was unreachable by ~5x; the pre-registered')
print('     decision rule could only return OOS-ALSO-LOWER.')

print('\n=== [2] NaN / coverage accounting on the new metric ===')
n_nan = sum(1 for r in R if not (r.get('obs_resid_rms_oos') == r.get('obs_resid_rms_oos')))
n_nan2 = sum(1 for r in R if not (r.get('obs_resid_rms') == r.get('obs_resid_rms')))
print(f'  rows={len(R)}  NaN obs_resid_rms_oos={n_nan}  NaN obs_resid_rms={n_nan2}')
rad_nan = sum(1 for r in R for k, v in (r.get('rmse_by_radius') or {}).items() if v != v)
print(f'  NaN entries in rmse_by_radius across all rows = {rad_nan}')

print('\n=== [3] sensor noise floor vs the residual the tracker drives down ===')
sys.path.insert(0, REPO)
from wavesplat.data_loader import EpisodeDataLoader  # noqa: E402
from scipy.interpolate import RegularGridInterpolator  # noqa: E402

DIRS = [os.path.join(REPO, d) for d in ('data_indep2_2026-08-07',
                                        'data_indep3_2026-08-07',
                                        'data_indep4_2026-08-07')]
for ep in ('episode_BF4_15comp_seed0_uniform_ra_rf_ds.h5',
           'episode_BF5_30comp_seed11_uniform_ra_rf_ds.h5'):
    path = next(os.path.join(d, ep) for d in DIRS if os.path.exists(os.path.join(d, ep)))
    ld = EpisodeDataLoader(path)
    gx = ld.gt_grid_x.cpu().numpy()
    gy = ld.gt_grid_y.cpu().numpy()
    fr = ld.get_tracking_frames(tp_factor=2.0)
    err, dz1 = [], []
    for f in fr[:40]:
        gt = f.gt_field.cpu().numpy()
        it = RegularGridInterpolator((gy, gx), gt, bounds_error=False, fill_value=None)
        ox = f.x_obs.cpu().numpy()
        oy = f.y_obs.cpu().numpy()
        zz = f.z_obs.cpu().numpy()
        pred = it(np.stack([oy, ox], 1))
        err.append(float(np.sqrt(np.mean((zz - pred) ** 2))))
    for i in range(1, 40):
        a = fr[i - 1].gt_field.cpu().numpy()
        b = fr[i].gt_field.cpu().numpy()
        dz1.append(float(np.sqrt(np.mean((b - a) ** 2))))
    print(f'  {ep[8:-20]:22s} rms(z_obs - GT@obs) = {np.mean(err):.4f} m  '
          f'(sensor noise + interp + slope error)')
    print(f'  {"":22s} one-frame TRUE field change = {np.mean(dz1):.4f} m  '
          f'-> the entire dynamic range of a 1-frame OOS test')
print(f'  measured AUTO gap (oos-in) = '
      f'{st.median([med("AUTO",e,"obs_resid_rms_oos")-med("AUTO",e,"obs_resid_rms") for e in KEPT]):.5f} m')
print(f'  AUTO in-sample residual    = '
      f'{st.median([med("AUTO",e,"obs_resid_rms") for e in KEPT]):.5f} m')

print('\n=== [4] wavelength scale vs the OOS displacement ===')
for Tp in (7.0, 9.0):
    lam = 9.81 * Tp ** 2 / (2 * math.pi)
    print(f'  Tp={Tp}s -> deep-water lambda={lam:.1f} m; '
          f'median nearest-neighbour hop between consecutive frames ~2.0 m '
          f'= {100*2.0/lam:.1f}% of a wavelength; dt=0.1 s = {100*0.1/Tp:.1f}% of a period')
