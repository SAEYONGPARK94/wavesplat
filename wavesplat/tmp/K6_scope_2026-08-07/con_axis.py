"""CON (6) k-position -- attacks 3b and 4.

3b. The CRB assumed sigma_z = 0.02 m i.i.d. vertical. The simulator injects
    RANGE noise (0.02 m along the ray) + ANGULAR noise (0.05 deg). At grazing
    incidence those map to a very different effective VERTICAL error. Quantify it,
    because CRB scales linearly in sigma: the "no wall" margin is only 0.0311 -> 0.0629,
    i.e. a factor 2.02 in effective sigma would erase it exactly.

4.  "amplitude budget exhausted (Delta shrink saturates ~ -0.25), residual 94-96%
    decorrelated => next prize is phase/wavenumber". Which ARM, which EPISODES?
    Re-derive from the 12-episode B2 ledger, per cell, including the SHIPPED arm AUTO.

CPU only. Reads existing files. Writes nothing.
"""
import json
import os
from collections import defaultdict

import h5py
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..'))
DATA = os.path.join(REPO, 'data_finemesh_2026-07-23')
RES = os.path.join(REPO, 'tmp', 'B2_g1p1_ext_2026-08-06', 'results.jsonl')

CROWDED4 = ['episode_BF4_15comp_seed0_uniform.h5', 'episode_BF4_30comp_seed0_uniform.h5',
            'episode_BF5_15comp_seed0_uniform.h5', 'episode_BF5_30comp_seed0_uniform.h5']
SPARSE8 = ['episode_BF2_1comp_seed0_uniform.h5', 'episode_BF3_1comp_seed0_uniform.h5',
           'episode_BF3_2comp_seed0_uniform.h5', 'episode_BF3_2comp_seed0_slope.h5',
           'episode_BF3_5comp_seed0_uniform.h5', 'episode_BF3_5comp_seed0_slope.h5',
           'episode_BF4_1comp_seed0_uniform.h5', 'episode_BF4_5comp_seed0_uniform.h5']
P1_12 = SPARSE8 + CROWDED4
G = 9.81
SIG_R = 0.02          # wavesim/noise_model.add_noise default sigma_range [m]
SIG_ANG = np.deg2rad(0.05)   # default sigma_angle_deg

print('=' * 104)
print('CON-6 attack 3b -- what IS the effective vertical noise the CRB should have used?')
print('=' * 104)
print(f"{'episode':32s} {'Tobs':>5s} {'Mvalid':>7s} {'med range':>10s} {'med cos(inc)':>12s} "
      f"{'rms slope':>10s} {'s_range->z':>11s} {'s_ang->z':>10s} {'s_eff':>8s} {'/0.02':>6s}")
for e in CROWDED4:
    with h5py.File(os.path.join(DATA, e), 'r') as f:
        wc = np.array(f['config/wave_components'])       # A, w, kx, ky, phi, Q
        ts = np.array(f['frames/timestamps'])
        lp = np.array(f['frames/lidar_points'])
        vm = np.array(f['frames/lidar_valid_mask']).astype(bool)
        inc = np.array(f['frames/lidar_incidence_angles'])
        Tp = float(f['config'].attrs['Tp'])
    tmax = 2.0 * Tp
    fi = ts <= tmax + 1e-9
    pts = lp[fi][vm[fi]]
    incs = inc[fi][vm[fi]]
    x, y = pts[:, 0], pts[:, 1]
    t = np.repeat(ts[fi], vm[fi].shape[1])[vm[fi].reshape(-1)]
    # sensor at (ship_x, ship_y, 5). ship moves; use horizontal distance from the
    # per-frame minimum-x return as a lower bound on range -> range >= |x| roughly.
    rng = np.sqrt(x ** 2 + y ** 2 + 5.0 ** 2)
    A, w, kx, ky, phi = wc[:, 0], wc[:, 1], wc[:, 2], wc[:, 3], wc[:, 4]
    psi = np.outer(x, kx) + np.outer(y, ky) - np.outer(t, w) + phi[None, :]
    dzdx = -(A * kx)[None, :] * np.sin(psi)
    dzdy = -(A * ky)[None, :] * np.sin(psi)
    slope = np.sqrt(dzdx.sum(1) ** 2 + dzdy.sum(1) ** 2)
    cos_inc = np.cos(incs)
    s_r = SIG_R * np.median(cos_inc)                       # along-ray -> vertical
    s_a = np.median(rng) * SIG_ANG * np.sqrt(np.mean(slope ** 2))
    s_eff = np.sqrt(s_r ** 2 + s_a ** 2)
    print(f'{e[8:-3]:32s} {tmax:5.1f} {pts.shape[0]:7d} {np.median(rng):10.1f} '
          f'{np.median(cos_inc):12.4f} {np.sqrt(np.mean(slope ** 2)):10.4f} '
          f'{s_r:11.5f} {s_a:10.5f} {s_eff:8.5f} {s_eff / 0.02:6.2f}')
print('\n  CRB scales LINEARLY in sigma. A ratio < 1 means the true CRB is SMALLER than')
print('  the reported one, i.e. the "no information wall" conclusion gets STRONGER, not weaker.')

# ------------------------------------------------------------------ attack 4
rows = []
for line in open(RES):
    try:
        r = json.loads(line)
    except json.JSONDecodeError:
        continue
    if r.get('error') is None and np.isfinite(r.get('nRMSE', np.nan)):
        rows.append(r)
by = defaultdict(list)
for r in rows:
    by[(r['cell'], r['episode'])].append(r)


def med(cell, ep, key):
    v = [r[key] for r in by[(cell, ep)] if np.isfinite(r.get(key, np.nan))]
    return float(np.median(v)) if v else float('nan')


CELLS = ['X-BASE', 'G1', 'P1', 'G1P1', 'ORACLE-K', 'AUTO']
print('\n' + '=' * 104)
print('CON-6 attack 4 -- "amplitude budget exhausted, next is phase/wavenumber"')
print('  m = decorr + shrink,  decorr = 1-rho^2,  shrink = (sqrt(VR)-rho)^2   '
      '(run_rx2b.py:277-290)')
print('=' * 104)
for tag, eps in [('CROWDED4 (where the claim was made)', CROWDED4),
                 ('SPARSE8  (never checked)', SPARSE8),
                 ('ALL 12', P1_12)]:
    print(f'\n-- {tag} -- median over episodes of the per-episode median')
    print(f"{'cell':10s} {'m':>8s} {'decorr':>8s} {'shrink':>8s} {'shrink/m':>9s} "
          f"{'d shrink':>9s} {'d decorr':>9s}")
    base_s = np.median([med('X-BASE', e, 'shrink_mean') for e in eps])
    base_d = np.median([med('X-BASE', e, 'decorr_mean') for e in eps])
    for c in CELLS:
        m = np.median([med(c, e, 'm_mean') for e in eps])
        d = np.median([med(c, e, 'decorr_mean') for e in eps])
        s = np.median([med(c, e, 'shrink_mean') for e in eps])
        ds = np.median([med(c, e, 'shrink_mean') - med('X-BASE', e, 'shrink_mean') for e in eps])
        dd = np.median([med(c, e, 'decorr_mean') - med('X-BASE', e, 'decorr_mean') for e in eps])
        print(f'{c:10s} {m:8.4f} {d:8.4f} {s:8.4f} {s / m if m else np.nan:9.3f} '
              f'{ds:+9.4f} {dd:+9.4f}')

print('\n  Read the "d shrink" column: the claimed saturation "Delta shrink ~ -0.25 across')
print('  G1/AUTO/ORACLE-K/ORACLE" is a CROWDED4 statement. Check whether it holds on SPARSE8')
print('  and whether "residual is 94-96% decorrelated" is a FINDING or an IDENTITY of a')
print('  well-fit arm (decorr/m -> 1 whenever shrink -> 0, which is what a good fit does).')
