"""CON (6) k-position -- attack 6: the CROWDED4 control arm is BELOW the flat-sea null.

B3/rx1 was overturned yesterday for exactly this: "the four cells rx1 called deployment
were all m>1 = worse than flat sea, so it compared two zero-scoring answers."

X-BASE (v2_konly) on CROWDED4 has m_mean = 1.20 > 1. Every headline on that block --
G1 -0.0472/63.7%, G1P1 -0.0701/73.6%, AUTO -0.0456/70.6%, ORACLE-K -0.0789/100% --
is a delta and a ratio anchored on that arm.

Re-anchor the recovery ratio on the flat-sea null and see what survives.
Reads B2 results.jsonl + rx2 NEUTRAL_A3_nulls.json + B2 nulls12.json. CPU only.
"""
import json
import os
from collections import defaultdict

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
RES = os.path.join(REPO, 'tmp', 'B2_g1p1_ext_2026-08-06', 'results.jsonl')
NULLS = os.path.join(REPO, 'tmp', 'B2_g1p1_ext_2026-08-06', 'nulls12.json')

CROWDED4 = ['episode_BF4_15comp_seed0_uniform.h5', 'episode_BF4_30comp_seed0_uniform.h5',
            'episode_BF5_15comp_seed0_uniform.h5', 'episode_BF5_30comp_seed0_uniform.h5']
SPARSE8 = ['episode_BF2_1comp_seed0_uniform.h5', 'episode_BF3_1comp_seed0_uniform.h5',
           'episode_BF3_2comp_seed0_uniform.h5', 'episode_BF3_2comp_seed0_slope.h5',
           'episode_BF3_5comp_seed0_uniform.h5', 'episode_BF3_5comp_seed0_slope.h5',
           'episode_BF4_1comp_seed0_uniform.h5', 'episode_BF4_5comp_seed0_uniform.h5']
P1_12 = SPARSE8 + CROWDED4
CELLS = ['X-BASE', 'G1', 'P1', 'G1P1', 'ORACLE-K', 'AUTO']

nl = json.load(open(NULLS))
null = {r['episode']: float(r['zero_null_tracking_nRMSE_mean']) for r in nl}

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


def med(cell, ep, key='nRMSE'):
    v = [r[key] for r in by[(cell, ep)] if np.isfinite(r.get(key, np.nan))]
    return float(np.median(v)) if v else float('nan')


print('=' * 100)
print('CON-6 attack 6 -- the CROWDED4 control fails the flat-sea null in 3/4 cells')
print('=' * 100)
print(f"{'episode':30s} {'null':>8s} {'X-BASE':>8s} {'XB<null?':>9s} {'m(X-BASE)':>10s} "
      f"{'G1P1':>8s} {'ORACLE-K':>9s} {'AUTO':>8s}")
for ep in CROWDED4:
    n = null[ep]
    xb = med('X-BASE', ep)
    m = med('X-BASE', ep, 'm_mean')
    print(f'{ep[8:-3]:30s} {n:8.4f} {xb:8.4f} {"yes" if xb < n else "NO":>9s} {m:10.4f} '
          f'{med("G1P1", ep):8.4f} {med("ORACLE-K", ep):9.4f} {med("AUTO", ep):8.4f}')

print('\n  X-BASE beats the flat-sea null on 1/4 crowded episodes (m_mean 1.20 > 1).')
print('  Every crowded delta and every "% of ceiling recovered" is measured from that point.')

print('\n' + '=' * 100)
print('Re-anchored recovery: (ref - cell) / (ref - ORACLE-K), per-episode ratio, median')
print('=' * 100)
for tag, eps in [('CROWDED4', CROWDED4), ('SPARSE8', SPARSE8), ('ALL12', P1_12)]:
    print(f'\n-- {tag} --')
    print(f"{'cell':10s} {'ref = X-BASE (as published)':>28s} {'ref = flat-sea null':>22s}")
    for c in ['G1', 'P1', 'G1P1', 'AUTO']:
        r_xb, r_nl = [], []
        for e in eps:
            xb, ok, ce, nu = med('X-BASE', e), med('ORACLE-K', e), med(c, e), null[e]
            if abs(xb - ok) > 1e-12:
                r_xb.append((xb - ce) / (xb - ok))
            if abs(nu - ok) > 1e-12:
                r_nl.append((nu - ce) / (nu - ok))
        print(f'{c:10s} {100 * np.median(r_xb):27.1f}% {100 * np.median(r_nl):21.1f}%')

print('\n' + '=' * 100)
print('And how big is the "gap" if the reference is the null instead of the sub-null control?')
print('=' * 100)
for tag, eps in [('CROWDED4', CROWDED4), ('SPARSE8', SPARSE8)]:
    g_xb = np.median([med('X-BASE', e) - med('ORACLE-K', e) for e in eps])
    g_nl = np.median([null[e] - med('ORACLE-K', e) for e in eps])
    g_au = np.median([med('AUTO', e) - med('ORACLE-K', e) for e in eps])
    print(f'{tag:10s}  X-BASE -> ORACLE-K = {g_xb:+.4f} | '
          f'null -> ORACLE-K = {g_nl:+.4f} | SHIPPED AUTO -> ORACLE-K = {g_au:+.4f}')
print('\n  The prize that is actually on the table for a deployable change is the AUTO column.')
