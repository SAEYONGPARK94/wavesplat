"""Rx-4 INDEPENDENT RE-DERIVATION (PREREG sec.6.4).

Written from scratch against the frozen PREREG text, NOT by editing analyze_rx4.py:
different data structure (numpy record grouping instead of a dict index), different
median implementation (numpy.median instead of statistics.median), different sign test
(scipy.stats.binomtest instead of math.comb).  Any disagreement with analyze_rx4.py in
the last digit printed is reported rather than silently reconciled.
"""
import json
import os
from collections import defaultdict

import numpy as np
from scipy.stats import binomtest

HERE = os.path.dirname(os.path.abspath(__file__))

recs = []
for line in open(os.path.join(HERE, 'results.jsonl')):
    line = line.strip()
    if not line:
        continue
    d = json.loads(line)
    if 'error' in d:
        continue
    recs.append(d)

# 2026-08-07: see analyze_rx4.py -- nulls.json was missing the seed2/seed3 uniform
# crowded episodes, which this script would have SILENTLY intersected away (scoring
# crowded n=8 instead of 16).  nulls_full.json = frozen 24 reproduced bit-exactly + 8 new.
nulls = {d['episode']: d['zero_null_tracking_nRMSE_mean']
         for d in json.load(open(os.path.join(HERE, 'nulls_full.json')))}

# group: metric -> cell -> episode -> [values over run seeds]
G = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
for d in recs:
    for m in ('nRMSE', 'Pearson_rho', 'absVR_framewise', 'm_mean',
              'decorr_mean', 'shrink_mean'):
        if d.get(m) is not None and np.isfinite(d[m]):
            G[m][d['cell']][d['episode']].append(float(d[m]))

crowd = [e for e in nulls if 'comp_seed' in e and
         ('15comp' in e or '30comp' in e) and e.endswith('_uniform.h5')]
crowd = sorted(e for e in crowd if e in G['nRMSE']['AUTO'])
sparse = sorted(e for e in G['nRMSE']['AUTO'] if e not in crowd)
print(f'records={len(recs)}  crowded={len(crowd)}  sparse={len(sparse)}')


def per_ep(metric, cell, ep):
    v = G[metric][cell].get(ep)
    return float(np.median(v)) if v else np.nan


print('\n### null exclusion (control cell AUTO, episode median >= episode null) ###')
keep = [e for e in crowd if per_ep('nRMSE', 'AUTO', e) < nulls[e]]
drop = [e for e in crowd if e not in keep]
print(f'  kept {len(keep)}  dropped {len(drop)}: {[e[8:-11] for e in drop]}')

CELLS = sorted(G['nRMSE'].keys())
print('\n### BLOCK C (crowded, null-excluded): levels and paired deltas ###')
print(f"  {'cell':14s} {'medOfMed':>9s} | {'vs AUTO':>9s} {'k/n':>7s} {'p':>8s} "
      f"| {'vs FLOOR':>9s} {'k/n':>7s} {'p':>8s}")
res = {}
for c in CELLS:
    lvl = float(np.median([per_ep('nRMSE', c, e) for e in keep]))
    line = [lvl]
    for ref in ('AUTO', 'AUTO-NOTRACK'):
        dl = np.array([per_ep('nRMSE', c, e) - per_ep('nRMSE', ref, e) for e in keep])
        k = int((dl < 0).sum())
        p = binomtest(k, len(dl), 0.5, alternative='two-sided').pvalue
        line += [float(np.median(dl)), k, len(dl), float(p)]
    res[c] = line
    print(f'  {c:14s} {line[0]:9.4f} | {line[1]:+9.4f} {line[2]:3d}/{line[3]:<3d} '
          f'{line[4]:8.4f} | {line[5]:+9.4f} {line[6]:3d}/{line[7]:<3d} {line[8]:8.4f}')

fl = res['AUTO-NOTRACK'][1]
print('\n### recovery vs the NOTRACK floor (descriptive) ###')
for c in CELLS:
    if c in ('AUTO', 'AUTO-DUP'):
        continue
    print(f'  {c:14s} R = {100*res[c][1]/fl:6.1f}%')

print('\n### D1 head-to-head: T-PHIONLY vs T-WONLY (paired, crowded, null-excluded) ###')
dd = np.array([per_ep('nRMSE', 'T-PHIONLY', e) - per_ep('nRMSE', 'T-WONLY', e) for e in keep])
k = int((dd < 0).sum())
print(f'  medΔ={np.median(dd):+.4f}  T-PHIONLY wins {k}/{len(dd)}  '
      f'p={binomtest(k, len(dd), 0.5).pvalue:.4f}')

print('\n### BLOCK S (sparse/mid), null exclusion + paired deltas ###')
keeps = [e for e in sparse if per_ep('nRMSE', 'AUTO', e) < nulls[e]]
print(f'  kept {len(keeps)}/{len(sparse)}; dropped {[e[8:-3] for e in sparse if e not in keeps]}')
if keeps:
    print(f"  {'cell':14s} {'medOfMed':>9s} | {'vs AUTO':>9s} {'k/n':>7s} {'p':>8s}")
    for c in CELLS:
        lvl = float(np.median([per_ep('nRMSE', c, e) for e in keeps]))
        dl = np.array([per_ep('nRMSE', c, e) - per_ep('nRMSE', 'AUTO', e) for e in keeps])
        k = int((dl < 0).sum())
        print(f'  {c:14s} {lvl:9.4f} | {float(np.median(dl)):+9.4f} {k:3d}/{len(dl):<3d} '
              f'{binomtest(k, len(dl), 0.5).pvalue:8.4f}')

print('\n### panel (crowded, null-excluded, vs AUTO) ###')
for m, good in (('Pearson_rho', 1), ('absVR_framewise', -1), ('m_mean', -1),
                ('decorr_mean', -1), ('shrink_mean', -1)):
    print(f'  [{m}]')
    for c in CELLS:
        if c == 'AUTO':
            continue
        dl = np.array([per_ep(m, c, e) - per_ep(m, 'AUTO', e) for e in keep])
        k = int((dl * good > 0).sum())
        print(f'     {c:14s} medΔ={float(np.median(dl)):+8.4f} good={k:3d}/{len(dl):<3d} '
              f'p={binomtest(k, len(dl), 0.5).pvalue:.4f}')

print('\n### final-init (does the intervention still degrade the initialisation?) ###')
print(f"  {'cell':14s} {'med(final-init) crowded':>26s}")
for c in CELLS:
    v = []
    for e in keep:
        d = [x['nRMSE'] - x['init_nrmse'] for x in recs
             if x['cell'] == c and x['episode'] == e]
        if d:
            v.append(float(np.median(d)))
    print(f'  {c:14s} {float(np.median(v)):26.4f}')
