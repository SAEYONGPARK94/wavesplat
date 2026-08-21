"""CON (6) k-position -- attack 2c: in the regime that carries the "11/12" sign test,
is there any k-position deficit LEFT TO FIX?

If the shipped-fallback arm's matched-component wavenumber error is already ~0 on the
8 sparse episodes, then the ORACLE-K win there cannot be the k factor -- and the only
uncontrolled factors left in the ORACLE-K bundle are component count N and the stage
schedule.

Reads tmp/B2_g1p1_ext_2026-08-06/results.jsonl only. CPU, no model calls.
"""
import json
import os
from collections import defaultdict

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
RES = os.path.join(REPO, 'tmp', 'B2_g1p1_ext_2026-08-06', 'results.jsonl')

CROWDED4 = ['episode_BF4_15comp_seed0_uniform.h5', 'episode_BF4_30comp_seed0_uniform.h5',
            'episode_BF5_15comp_seed0_uniform.h5', 'episode_BF5_30comp_seed0_uniform.h5']
SPARSE8 = ['episode_BF2_1comp_seed0_uniform.h5', 'episode_BF3_1comp_seed0_uniform.h5',
           'episode_BF3_2comp_seed0_uniform.h5', 'episode_BF3_2comp_seed0_slope.h5',
           'episode_BF3_5comp_seed0_uniform.h5', 'episode_BF3_5comp_seed0_slope.h5',
           'episode_BF4_1comp_seed0_uniform.h5', 'episode_BF4_5comp_seed0_uniform.h5']

rows = []
for line in open(RES):
    try:
        r = json.loads(line)
    except json.JSONDecodeError:
        continue
    if r.get('error') is None and 'spec' in r:
        rows.append(r)
by = defaultdict(list)
for r in rows:
    by[(r['cell'], r['episode'])].append(r)


def sp(cell, ep, key):
    v = [r['spec'][key] for r in by[(cell, ep)]
         if key in r['spec'] and np.isfinite(r['spec'][key])]
    return float(np.median(v)) if v else float('nan')


print('=' * 100)
print('CON-6 attack 2c -- how much k-position error is there to fix, per regime?')
print('  k_mag_rel_error_mean / direction_error_deg_mean are computed over MATCHED')
print('  components only (Hungarian, cost dk/k_scale < 0.5). Both arms, per episode median-of-3.')
print('=' * 100)
for tag, eps in [('SPARSE8', SPARSE8), ('CROWDED4', CROWDED4)]:
    print(f'\n-- {tag} --')
    print(f"{'episode':32s} {'cell':10s} {'detect':>7s} {'|k| relerr':>11s} "
          f"{'dir err deg':>12s} {'n_est':>6s} {'n_gt':>5s}")
    for ep in eps:
        for c in ['X-BASE', 'AUTO', 'ORACLE-K']:
            print(f'{ep[8:-3]:32s} {c:10s} {sp(c, ep, "detection_rate"):7.3f} '
                  f'{sp(c, ep, "k_mag_rel_error_mean"):11.4f} '
                  f'{sp(c, ep, "direction_error_deg_mean"):12.3f} '
                  f'{sp(c, ep, "n_est"):6.1f} {sp(c, ep, "n_gt"):5.0f}')
        print()

print('=' * 100)
print('summary -- median over episodes')
print('=' * 100)
print(f"{'regime':10s} {'cell':10s} {'detect':>7s} {'|k| relerr':>11s} {'dir err deg':>12s}")
for tag, eps in [('SPARSE8', SPARSE8), ('CROWDED4', CROWDED4)]:
    for c in ['X-BASE', 'AUTO', 'ORACLE-K']:
        print(f'{tag:10s} {c:10s} '
              f'{np.median([sp(c, e, "detection_rate") for e in eps]):7.3f} '
              f'{np.median([sp(c, e, "k_mag_rel_error_mean") for e in eps]):11.4f} '
              f'{np.median([sp(c, e, "direction_error_deg_mean") for e in eps]):12.3f}')
