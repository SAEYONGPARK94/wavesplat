"""PRO (6): decompose the k error of the SHIPPED-fallback initializer (X-BASE) and AUTO
into (a) count N, (b) position (direction / |k|), (c) near-component merging.
Uses only the recorded per-run `spec` block + dumped atom coordinates. Read-only."""
import json, os, re, statistics as st
from collections import defaultdict
import numpy as np
import h5py

B2 = '/home/vexit/wavesplat/wavesplat/tmp/B2_g1p1_ext_2026-08-06/results.jsonl'
DATA = '/home/vexit/wavesplat/wavesplat/data_finemesh_2026-07-23'

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
ALL12 = NEW8 + CROWDED4

def m(vals): return float(np.median([v for v in vals if v == v])) if vals else float('nan')

print('='*118)
print('A. SPECTRAL DIAGNOSTICS at the END of estimation (median of 3 seeds)')
print('   det = fraction of TRUE components matched (<0.5 k_scale) | Edet = energy-weighted')
print('='*118)
hdr = f'{"episode":30s}{"Ngt":>4s} | ' + ' | '.join(
    f'{c:>34s}' for c in ['X-BASE  det  Edet dirdeg  N', 'ORACLE-K det Edet dirdeg  N'])
print(hdr)
tab = {}
for e in ALL12:
    ngt = int(re.search(r'_(\d+)comp', e).group(1))
    line = f'{e[8:-3]:30s}{ngt:4d} | '
    cells = []
    for c in ('X-BASE', 'ORACLE-K'):
        rs = by[(c, e)]
        det = m([r['spec'].get('detection_rate', float('nan')) for r in rs])
        ed = m([r['spec'].get('energy_detection_rate', float('nan')) for r in rs])
        dd = m([r['spec'].get('direction_error_deg_mean', float('nan')) for r in rs])
        km = m([r['spec'].get('k_mag_rel_error_mean', float('nan')) for r in rs])
        n = m([r['N'] for r in rs])
        tab[(c, e)] = dict(det=det, ed=ed, dd=dd, km=km, n=n)
        cells.append(f'{det:6.3f}{ed:6.3f}{dd:7.2f}{n:4.0f}')
    print(line + ' | '.join(f'{x:>34s}' for x in cells))

print('\n' + '='*118)
print('B. WHERE DOES THE MISS COME FROM?  (X-BASE, crowded vs sparse)')
print('   n_est vs Ngt  |  matched  |  UNMATCHED-GT (missed true comps)  |  '
      'UNMATCHED-EST (spurious atoms)')
print('='*118)
print(f'{"episode":30s}{"Ngt":>4s}{"n_est":>7s}{"match":>7s}{"missGT":>8s}{"spurEST":>9s}'
      f'{"det%":>7s}{"Edet%":>7s}{"dirdeg":>8s}{"|k|rel":>8s}')
for e in ALL12:
    ngt = int(re.search(r'_(\d+)comp', e).group(1))
    rs = by[('X-BASE', e)]
    ne = m([r['spec']['n_est'] for r in rs]); nm = m([r['spec']['n_matched'] for r in rs])
    t = tab[('X-BASE', e)]
    print(f'{e[8:-3]:30s}{ngt:4d}{ne:7.0f}{nm:7.0f}{ngt-nm:8.0f}{ne-nm:9.0f}'
          f'{t["det"]*100:7.1f}{t["ed"]*100:7.1f}{t["dd"]:8.2f}{t["km"]:8.4f}')

print('\n' + '='*118)
print('C. THE SAME FOR THE SHIPPED ARM (AUTO) — this is what actually deploys')
print('='*118)
print(f'{"episode":30s}{"Ngt":>4s}{"n_est":>7s}{"match":>7s}{"missGT":>8s}{"spurEST":>9s}'
      f'{"det%":>7s}{"Edet%":>7s}{"dirdeg":>8s}{"nRMSE":>8s}')
for e in ALL12:
    ngt = int(re.search(r'_(\d+)comp', e).group(1))
    rs = by[('AUTO', e)]
    ne = m([r['spec']['n_est'] for r in rs]); nm = m([r['spec']['n_matched'] for r in rs])
    det = m([r['spec'].get('detection_rate', float('nan')) for r in rs])
    ed = m([r['spec'].get('energy_detection_rate', float('nan')) for r in rs])
    dd = m([r['spec'].get('direction_error_deg_mean', float('nan')) for r in rs])
    nr = m([r['nRMSE'] for r in rs])
    print(f'{e[8:-3]:30s}{ngt:4d}{ne:7.0f}{nm:7.0f}{ngt-nm:8.0f}{ne-nm:9.0f}'
          f'{det*100:7.1f}{ed*100:7.1f}{dd:8.2f}{nr:8.4f}')

print('\n' + '='*118)
print('D. GROUND-TRUTH k GEOMETRY — is the miss explained by NEAR-COMPONENT crowding?')
print('   min pairwise |dk| between true components vs min_dk=0.15 merge scale')
print('='*118)
print(f'{"episode":30s}{"Ngt":>4s}{"min|dk|":>9s}{"med|dk|nn":>11s}{"pairs<0.15":>12s}'
      f'{"kmag rng":>18s}{"det%":>7s}')
for e in ALL12:
    with h5py.File(os.path.join(DATA, e), 'r') as f:
        wc = np.array(f['config/wave_components'])
    kx, ky = wc[:, 2], wc[:, 3]
    n = len(kx)
    P = np.stack([kx, ky], 1)
    D = np.sqrt(((P[:, None, :] - P[None, :, :]) ** 2).sum(-1))
    np.fill_diagonal(D, np.inf)
    nn = D.min(1)
    npairs = int(((D < 0.15).sum() - 0) / 2) if n > 1 else 0
    kmag = np.hypot(kx, ky)
    print(f'{e[8:-3]:30s}{n:4d}{(nn.min() if n>1 else float("nan")):9.4f}'
          f'{(np.median(nn) if n>1 else float("nan")):11.4f}{npairs:12d}'
          f'{f"{kmag.min():.3f}-{kmag.max():.3f}":>18s}{tab[("X-BASE",e)]["det"]*100:7.1f}')
