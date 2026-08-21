"""PRO (6): the deployment-relevant headroom is ORACLE-K vs AUTO (the SHIPPED arm),
not vs X-BASE (the shipping FALLBACK). Re-derive both, with cluster correction."""
import json, math, re, statistics as st
from collections import defaultdict
import numpy as np

B2 = '/home/vexit/wavesplat/wavesplat/tmp/B2_g1p1_ext_2026-08-06/results.jsonl'
rows = [json.loads(l) for l in open(B2)]
by = defaultdict(dict)
for r in rows: by[(r['cell'], r['episode'])][r['seed']] = r
CROWDED4 = ['episode_BF4_15comp_seed0_uniform.h5', 'episode_BF4_30comp_seed0_uniform.h5',
            'episode_BF5_15comp_seed0_uniform.h5', 'episode_BF5_30comp_seed0_uniform.h5']
NEW8 = ['episode_BF2_1comp_seed0_uniform.h5', 'episode_BF3_1comp_seed0_uniform.h5',
        'episode_BF3_2comp_seed0_uniform.h5', 'episode_BF3_2comp_seed0_slope.h5',
        'episode_BF3_5comp_seed0_uniform.h5', 'episode_BF3_5comp_seed0_slope.h5',
        'episode_BF4_1comp_seed0_uniform.h5', 'episode_BF4_5comp_seed0_uniform.h5']
ALL12 = NEW8 + CROWDED4
def med(c, e): return st.median([v['nRMSE'] for v in by[(c, e)].values()])
def sign_p(k, n):
    if n == 0: return 1.0
    kk = min(k, n - k)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(kk + 1)) / 2 ** n)

print('=' * 108)
print('HEADROOM ABOVE THE ACTUALLY-SHIPPED ARM (AUTO), not above the fallback (X-BASE)')
print('=' * 108)
print(f'{"episode":30s}{"Ngt":>4s}{"X-BASE":>9s}{"AUTO":>9s}{"G1P1":>9s}{"ORACLE-K":>10s}'
      f'{"OK-AUTO":>10s}{"AUTOrecov":>11s}')
for e in ALL12:
    ngt = int(re.search(r'_(\d+)comp', e).group(1))
    xb, au, gp, ok = med('X-BASE', e), med('AUTO', e), med('G1P1', e), med('ORACLE-K', e)
    rec = (xb - au) / (xb - ok) * 100 if xb != ok else float('nan')
    print(f'{e[8:-3]:30s}{ngt:4d}{xb:9.4f}{au:9.4f}{gp:9.4f}{ok:10.4f}{ok-au:+10.4f}{rec:10.1f}%')

for nm, eps in [('ALL12', ALL12), ('NEW8', NEW8), ('CROWDED4', CROWDED4)]:
    ds = [med('ORACLE-K', e) - med('AUTO', e) for e in eps]
    w = sum(1 for x in ds if x < 0)
    rec = st.median([(med('X-BASE', e) - med('AUTO', e)) /
                     (med('X-BASE', e) - med('ORACLE-K', e)) for e in eps]) * 100
    recg = st.median([(med('X-BASE', e) - med('G1P1', e)) /
                      (med('X-BASE', e) - med('ORACLE-K', e)) for e in eps]) * 100
    print(f'\n{nm:9s} ORACLE-K - AUTO : median {st.median(ds):+.4f}  wins {w}/{len(ds)}  '
          f'naive p {sign_p(w,len(ds)):.4f}   | AUTO recovery {rec:5.1f}%  G1P1 recovery {recg:5.1f}%')

# cluster (5 phase draws = ncomp groups)
clus = defaultdict(list)
for e in ALL12:
    clus[int(re.search(r'_(\d+)comp', e).group(1))].append(med('ORACLE-K', e) - med('AUTO', e))
cm = [st.median(v) for k, v in sorted(clus.items())]
print(f'\nCLUSTER-CORRECTED (C=5 phase draws): medians '
      f'{[f"{x:+.4f}" for x in cm]}  neg {sum(1 for x in cm if x<0)}/5  '
      f'p={sign_p(sum(1 for x in cm if x<0),5):.4f}')

# what fraction of the total ORACLE-K prize is still unclaimed by the shipped arm?
tot = [med('X-BASE', e) - med('ORACLE-K', e) for e in ALL12]
left = [med('AUTO', e) - med('ORACLE-K', e) for e in ALL12]
print(f'\nUNCLAIMED PRIZE (AUTO -> ORACLE-K), median over 12 = {st.median(left):+.4f} nRMSE')
print(f'   crowded4 = {st.median([med("AUTO",e)-med("ORACLE-K",e) for e in CROWDED4]):+.4f}'
      f'   sparse8 = {st.median([med("AUTO",e)-med("ORACLE-K",e) for e in NEW8]):+.4f}')
