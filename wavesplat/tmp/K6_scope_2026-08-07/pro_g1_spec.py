"""PRO (6): does the k-position intervention (G1 = narrow coarse range) actually move the
k-ACCURACY diagnostics, not just nRMSE?  Direct causal read on the shipped-fallback arm."""
import json, re, statistics as st
from collections import defaultdict
import numpy as np

B2 = '/home/vexit/wavesplat/wavesplat/tmp/B2_g1p1_ext_2026-08-06/results.jsonl'
rows = [json.loads(l) for l in open(B2)]
by = defaultdict(list)
for r in rows:
    by[(r['cell'], r['episode'])].append(r)

CROWDED4 = ['episode_BF4_15comp_seed0_uniform.h5', 'episode_BF4_30comp_seed0_uniform.h5',
            'episode_BF5_15comp_seed0_uniform.h5', 'episode_BF5_30comp_seed0_uniform.h5']
NEW8 = ['episode_BF2_1comp_seed0_uniform.h5', 'episode_BF3_1comp_seed0_uniform.h5',
        'episode_BF3_2comp_seed0_uniform.h5', 'episode_BF3_2comp_seed0_slope.h5',
        'episode_BF3_5comp_seed0_uniform.h5', 'episode_BF3_5comp_seed0_slope.h5',
        'episode_BF4_1comp_seed0_uniform.h5', 'episode_BF4_5comp_seed0_uniform.h5']
CELLS = ['X-BASE', 'G1', 'P1', 'G1P1', 'AUTO', 'ORACLE-K']

def M(c, e, f):
    v = [f(r) for r in by[(c, e)]]
    v = [x for x in v if x == x]
    return float(np.median(v)) if v else float('nan')

METRICS = [('det%', lambda r: r['spec'].get('detection_rate', np.nan) * 100),
           ('Edet%', lambda r: r['spec'].get('energy_detection_rate', np.nan) * 100),
           ('dir_deg', lambda r: r['spec'].get('direction_error_deg_mean', np.nan)),
           ('|k|rel', lambda r: r['spec'].get('k_mag_rel_error_mean', np.nan)),
           ('N', lambda r: r['N']),
           ('init_nRMSE', lambda r: r['init_nrmse']),
           ('nRMSE', lambda r: r['nRMSE'])]

for blk, eps in [('CROWDED 4', CROWDED4), ('SPARSE/MODERATE 8', NEW8)]:
    print('=' * 104)
    print(f'{blk}  —  block level = MEDIAN over episodes of (median over 3 seeds)')
    print('=' * 104)
    print(f'{"cell":10s}' + ''.join(f'{m:>13s}' for m, _ in METRICS))
    for c in CELLS:
        vals = [st.median([M(c, e, f) for e in eps]) for _, f in METRICS]
        print(f'{c:10s}' + ''.join(f'{v:13.4f}' for v in vals))
    # paired deltas vs X-BASE on the k-accuracy metrics
    print(f'\n  paired delta vs X-BASE (median over episodes), + = worse for det/Edet reversed')
    print(f'  {"cell":10s}' + ''.join(f'{m:>13s}' for m, _ in METRICS))
    for c in CELLS[1:]:
        ds = []
        for mi, (_, f) in enumerate(METRICS):
            d = [M(c, e, f) - M('X-BASE', e, f) for e in eps]
            d = [x for x in d if x == x]
            ds.append(st.median(d) if d else float('nan'))
        print(f'  {c:10s}' + ''.join(f'{v:+13.4f}' for v in ds))
    print()

print('=' * 104)
print('PER-EPISODE, CROWDED ONLY: detection & direction error, all cells')
print('=' * 104)
for e in CROWDED4:
    ngt = int(re.search(r'_(\d+)comp', e).group(1))
    print(f'\n{e[8:-3]} (Ngt={ngt})')
    print(f'  {"cell":10s}{"det%":>8s}{"Edet%":>8s}{"dir_deg":>9s}{"|k|rel":>9s}{"N":>5s}'
          f'{"init_nR":>9s}{"nRMSE":>9s}')
    for c in CELLS:
        v = [M(c, e, f) for _, f in METRICS]
        print(f'  {c:10s}{v[0]:8.1f}{v[1]:8.1f}{v[2]:9.2f}{v[3]:9.4f}{v[4]:5.0f}'
              f'{v[5]:9.4f}{v[6]:9.4f}')
