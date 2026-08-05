import json, os
import numpy as np
from scipy import stats
from collections import defaultdict

D = '/home/vexit/wavesplat/wavesplat/tmp/wp_integration_2026-07-25'
rows = [json.loads(l) for l in open(os.path.join(D, 'arm_runs.jsonl'))]
print('rows', len(rows), 'errs', sum(1 for r in rows if r.get('err')), 'flags', sum(1 for r in rows if r.get('flag')))

g = defaultdict(lambda: defaultdict(list))
for r in rows:
    g[r['episode']][r['arm']].append(r)

prim = ['episode_BF%d_%dcomp_seed%d_%s.h5' % (bf, nc, s, a)
        for bf in (4, 5) for nc in (15, 30) for s in range(5) for a in ('uniform', 'slope')]
prim = [e for e in prim if e in g]
print('primary episodes present:', len(prim))
ks = set()
for e in prim:
    for a in ('learned', 'v2_konly', 'learned_konly'):
        ks.add((a, len(g[e][a])))
print('seeds per (arm,ep):', sorted(ks))

def med(e, arm, key='avg_nRMSE'):
    return float(np.median([r[key] for r in g[e][arm]]))

def pair(a, b, key='avg_nRMSE', eps=prim):
    d = np.array([med(e, a, key) - med(e, b, key) for e in eps])
    p = stats.wilcoxon(d, alternative='less')[1]
    return d, p

for lab, a, b in (('learned vs v2_konly (PRIMARY, both tracked)', 'learned', 'v2_konly'),
                  ('learned_konly vs v2_konly', 'learned_konly', 'v2_konly'),
                  ('learned_konly vs learned', 'learned_konly', 'learned')):
    d, p = pair(a, b)
    print('\n--- %s ---' % lab)
    print(' n=%d medD=%.6f meanD=%.6f wins(<0)=%d ties(==0)=%d p1=%.4g' %
          (len(d), np.median(d), d.mean(), int((d < 0).sum()), int((d == 0).sum()), p))
    print(' arm means: %s=%.4f  %s=%.4f   (mean-of-arms diff=%.4f)' %
          (a, np.mean([med(e, a) for e in prim]), b, np.mean([med(e, b) for e in prim]),
           np.mean([med(e, a) for e in prim]) - np.mean([med(e, b) for e in prim])))
    print(' near-ties |d|<0.005:', int((np.abs(d) < 0.005).sum()), ' |d|<0.01:', int((np.abs(d) < 0.01).sum()))

# ws_best reference (min of init/tracked for v2 arm)
d = []
for e in prim:
    v2b = min(med(e, 'v2_konly', 'init_nrmse'), med(e, 'v2_konly', 'avg_nRMSE'))
    d.append(med(e, 'learned') - v2b)
d = np.array(d)
print('\n--- learned(tracked) vs v2 ws_best=min(init,tracked) ---')
print(' n=%d medD=%.6f wins=%d p1=%.4g' % (len(d), np.median(d), int((d < 0).sum()), stats.wilcoxon(d, alternative='less')[1]))

# retention: crowded head-to-head cond gain was -0.1052 (median vs ws_best)
print('\n retention = medD(learned vs v2_konly tracked) / 0.1052 = %.3f' %
      (np.median(pair('learned', 'v2_konly')[0]) / -0.1051647255818049))

# decomposition
print('\n--- decomposition (means over 40 eps of per-ep medians) ---')
for lab, arm, key in (('learned raw_init', 'learned', 'raw_init_nrmse'),
                      ('learned post_init', 'learned', 'init_nrmse'),
                      ('learned tracked', 'learned', 'avg_nRMSE'),
                      ('v2 init', 'v2_konly', 'init_nrmse'),
                      ('v2 tracked', 'v2_konly', 'avg_nRMSE')):
    print('  %-20s %.4f' % (lab, np.mean([med(e, arm, key) for e in prim])))
print('  learned final_n median %.1f  v2 final_n median %.1f' %
      (np.median([med(e, 'learned', 'final_n') for e in prim]),
       np.median([med(e, 'v2_konly', 'final_n') for e in prim])))

# TRACKER TAX: does the tracker give back the init gain?
gi = np.array([med(e, 'learned', 'init_nrmse') - med(e, 'v2_konly', 'init_nrmse') for e in prim])
gt = np.array([med(e, 'learned', 'avg_nRMSE') - med(e, 'v2_konly', 'avg_nRMSE') for e in prim])
print('\n TRACKER GIVE-BACK: median init gain %.4f -> median tracked gain %.4f  (retained %.0f%%)' %
      (np.median(gi), np.median(gt), 100 * np.median(gt) / np.median(gi)))

# sparse control
sparse_eps = [e for e in g if e not in prim]
print('\n non-primary episodes in file (%d):' % len(sparse_eps))
d = []
for e in sorted(sparse_eps):
    d.append(med(e, 'learned') - med(e, 'v2_konly'))
d = np.array(d)
print(' sparse-ish n=%d medD=%+.4f learned worse on %d/%d' % (len(d), np.median(d), int((d > 0).sum()), len(d)))
