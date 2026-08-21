"""PRO (6): is the crowded error simply the ENERGY OF THE COMPONENTS YOU FAILED TO PLACE?
Test nRMSE  vs  sqrt(1 - energy_detection_rate).  If it holds, the k-position axis owns
the crowded error by construction."""
import json, statistics as st
from collections import defaultdict
import numpy as np

B2 = '/home/vexit/wavesplat/wavesplat/tmp/B2_g1p1_ext_2026-08-06/results.jsonl'
rows = [json.loads(l) for l in open(B2)]
CR = {'episode_BF4_15comp_seed0_uniform.h5', 'episode_BF4_30comp_seed0_uniform.h5',
      'episode_BF5_15comp_seed0_uniform.h5', 'episode_BF5_30comp_seed0_uniform.h5'}
pts = []
for r in rows:
    e = r['spec'].get('energy_detection_rate')
    if e is None or not np.isfinite(e): continue
    pts.append((r['episode'] in CR, r['cell'], float(e), float(r['nRMSE'])))

for tag, sel in (('CROWDED', True), ('SPARSE', False)):
    P = [(c, e, n) for cr, c, e, n in pts if cr == sel]
    E = np.array([e for _, e, _ in P]); N = np.array([n for _, _, n in P])
    pred = np.sqrt(np.clip(1 - E, 0, None))
    r = np.corrcoef(N, pred)[0, 1]
    from scipy.stats import spearmanr
    print(f'{tag}: n={len(P)}  corr(nRMSE, sqrt(1-Edet)) = {r:+.3f}  '
          f'spearman {spearmanr(N, pred).statistic:+.3f}')
    print(f'   median nRMSE {np.median(N):.4f}  median sqrt(1-Edet) {np.median(pred):.4f}  '
          f'ratio {np.median(N)/max(np.median(pred),1e-9):.3f}')
    cells = sorted({c for c, _, _ in P})
    print(f'   {"cell":10s}{"Edet":>8s}{"sqrt(1-Edet)":>14s}{"nRMSE":>9s}{"nRMSE/pred":>12s}')
    for c in cells:
        S = [(e, n) for cc, e, n in P if cc == c]
        me = np.median([e for e, _ in S]); mn = np.median([n for _, n in S])
        pr = np.sqrt(max(1 - me, 0))
        print(f'   {c:10s}{me:8.4f}{pr:14.4f}{mn:9.4f}{mn/max(pr,1e-9):12.3f}')
    print()
