"""CON — MDE of the Rx-1 design as pre-registered (4 crowded episodes, median-of-3,
two-sided sign test on paired episode deltas)."""
import json, itertools
import numpy as np
from collections import defaultdict
from math import comb

rows=[json.loads(l) for l in open('tmp/s1s2_2026-08-05/results.jsonl')]
c=defaultdict(dict)
for r in rows: c[(r['arm'],r['episode'])][r['seed']]=r
eps=sorted(set(r['episode'] for r in rows)); seeds=[42,123,456]

print('='*96); print('MDE-1 — the design can never reach alpha=0.05'); print('='*96)
for n in (4,6,8,12):
    p=2*sum(comb(n,i) for i in range(1))/2**n
    print(f'  n={n:2d} episodes, two-sided sign test, best case k=n/n -> floor p = {p:.4f}'
          f'   {"(alpha=0.05 UNREACHABLE)" if p>0.05 else ""}')

print()
print('='*96)
print('MDE-2 — seed-induced noise on the PAIRED delta (arms share a bit-identical init)')
print('='*96)
for on,off in [('S2-T1f','X-T1f-notrack'), ('S1-A0','X-A0-notrack')]:
    print(f'\n  {on} - {off}:')
    allsd=[]
    for e in eps:
        d=[c[(on,e)][s]['nRMSE']-c[(off,e)][s]['nRMSE'] for s in seeds]
        allsd.append(np.std(d,ddof=1))
        print(f'    {e[8:28]:22s} per-seed deltas={[round(x,4) for x in d]}  sd={np.std(d,ddof=1):.4f}')
    sd=float(np.mean(allsd))
    # sd of a median-of-3 ~ 1.16 * sd/sqrt(3) for gaussian
    sd_m3 = 1.16*sd/np.sqrt(3)
    print(f'    mean within-episode sd of the delta = {sd:.4f};  sd(median-of-3) ~ {sd_m3:.4f}')
    print(f'    => for a 4/4 sign result the true per-episode effect must exceed ~{2*sd_m3:.4f} '
          f'(2 sigma) on EVERY episode')
    print(f'    MDE (4/4 at p=0.125, ~80% power) ~ {2.0*sd_m3:.4f}')

print()
print('='*96)
print('MDE-3 — what the K1-K5 knobs must beat to be readable at all')
print('='*96)
on,off='S2-T1f','X-T1f-notrack'
allsd=np.mean([np.std([c[(on,e)][s]['nRMSE']-c[(off,e)][s]['nRMSE'] for s in seeds],ddof=1) for e in eps])
mde=2.0*1.16*allsd/np.sqrt(3)
harm=np.mean([np.median([c[(on,e)][s]['nRMSE'] for s in seeds])-
              np.median([c[(off,e)][s]['nRMSE'] for s in seeds]) for e in eps])
print(f'  total tracker-attributable error at oracle = {harm:.4f}')
print(f'  MDE of the design                          = {mde:.4f}')
print(f'  => a knob must remove >= {mde/harm*100:.0f}% of the tracker term to be distinguishable '
      f'from seed noise at 4/4.')
print(f'  A NULL on this design therefore only excludes knobs worth more than {mde:.4f} nRMSE;')
print(f'  it does NOT license "regularisation was not causal".')
