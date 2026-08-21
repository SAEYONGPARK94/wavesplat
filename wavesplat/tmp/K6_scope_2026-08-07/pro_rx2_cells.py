"""PRO (6): full rx2 cell table on CROWDED4 incl. the W (phase-seeding) axis,
to see which single-factor cells were never crossed with the k-grid factor."""
import json, statistics as st
from collections import defaultdict
import numpy as np

RX2 = '/home/vexit/wavesplat/wavesplat/tmp/rx2_init_2026-08-05/results.jsonl'
rows = []
for l in open(RX2):
    try: r = json.loads(l)
    except Exception: continue
    if 'nRMSE' in r and 'error' not in r: rows.append(r)
by = defaultdict(list)
for r in rows: by[(r['cell'], r['episode'])].append(r)
eps = sorted({r['episode'] for r in rows})
cells = sorted({r['cell'] for r in rows})
print('episodes:', [e[8:-3] for e in eps])
print('cells   :', cells)

def M(c, e, f):
    v = [f(r) for r in by[(c, e)] if f(r) == f(r)]
    return float(np.median(v)) if v else float('nan')

F = {'nRMSE': lambda r: r['nRMSE'],
     'init_nR': lambda r: r.get('init_nrmse', np.nan),
     'det%': lambda r: r['spec'].get('detection_rate', np.nan) * 100,
     'Edet%': lambda r: r['spec'].get('energy_detection_rate', np.nan) * 100,
     'dir_deg': lambda r: r['spec'].get('direction_error_deg_mean', np.nan),
     '|k|rel': lambda r: r['spec'].get('k_mag_rel_error_mean', np.nan),
     'VR': lambda r: r['Variance_Ratio'],
     'N': lambda r: r['N']}

print('\n' + '=' * 118)
print('rx2 CROWDED4 block medians + paired delta vs X-BASE   (median of 3 seeds per ep)')
print('=' * 118)
print(f'{"cell":10s}' + ''.join(f'{k:>10s}' for k in F) + '  | ' +
      ''.join(f'{"d"+k:>10s}' for k in ('nRMSE', 'det%', 'dir_deg', 'VR')))
order = ['X-BASE', 'G1', 'P1', 'G1P1', 'G1P3', 'W1', 'W2', 'W3', 'P2', 'P3',
         'S1', 'S2', 'AUTO', 'ORACLE-K', 'ORACLE']
for c in order:
    if c not in cells: continue
    vals = [st.median([M(c, e, f) for e in eps]) for f in F.values()]
    ds = []
    for k in ('nRMSE', 'det%', 'dir_deg', 'VR'):
        d = [M(c, e, F[k]) - M('X-BASE', e, F[k]) for e in eps]
        d = [x for x in d if x == x]
        w = sum(1 for x in d if x < 0)
        ds.append(f'{st.median(d):+.4f}' if k != 'det%' else f'{st.median(d):+.1f}')
    print(f'{c:10s}' + ''.join(f'{v:10.4f}' for v in vals) + '  | ' +
          ''.join(f'{x:>10s}' for x in ds))

print('\n>> CELLS PRESENT:', sorted(cells))
print('>> Is there any cell crossing the k-grid factor (G1) with the phase factor (W1)? ',
      'G1W1' in cells or 'W1G1' in cells)
