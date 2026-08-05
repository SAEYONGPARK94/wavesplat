import json, re
import numpy as np
from scipy import stats

BATCHC = '/home/vexit/wavesplat/wavesplat/tmp/rerun_2026-07-25/batchC_ws_phasefix.jsonl'
S1 = '/home/vexit/wavesplat/wavesplat/tmp/wp_integration_2026-07-25/arm_runs.jsonl'
X = '/home/vexit/wavesplat/wavesplat/tmp/unified_2026-07-26/arm_runs_extra.jsonl'
LANES = [(42, 0), (1, 1), (2, 2), (3, 3), (4, 4)]
def comp(ep): return int(re.search(r'_(\d+)comp', ep).group(1))

v2 = {}; eps = set()
for l in open(BATCHC):
    d = json.loads(l); v2[(d['episode'], d['seed'])] = d['nRMSE']; eps.add(d['episode'])
learned = {}
for src in (S1, X):
    for l in open(src):
        d = json.loads(l)
        if d.get('arm') == 'learned' and d.get('err') is None:
            learned[(d['episode'], d['est_seed'], d['diff_seed'])] = d['avg_nRMSE']

SP = sorted([e for e in eps if comp(e) in (1, 2)])
print('sparse episodes', len(SP))
V = np.array([np.median([v2[(e, E)] for E, D in LANES]) for e in SP])
L = np.array([np.median([learned[(e, E, D)] for E, D in LANES]) for e in SP])
d = L - V
print('\npure-v2 sparse MEAN %.5f   pure-learned sparse MEAN %.5f  (bar = v2+0.005 = %.5f)'
      % (V.mean(), L.mean(), V.mean()+0.005))
print('pure-v2 sparse MEDIAN %.5f  pure-learned sparse MEDIAN %.5f' % (np.median(V), np.median(L)))
print('paired: medianD %+.5f  meanD %+.5f  learned-better %d/%d  Wilcoxon2 p=%.4f  sign p=%.4f'
      % (np.median(d), d.mean(), int((d < 0).sum()), len(d), stats.wilcoxon(d)[1],
         stats.binomtest(int((d < 0).sum()), len(d)).pvalue))
print('  >>> SIGN CONFLICT: mean says learned WORSE by %+.4f ; median-of-pairs says learned BETTER by %+.4f'
      % (L.mean()-V.mean(), np.median(d)))

# catastrophes
cat = [(e, l, v) for e, l, v in zip(SP, L, V) if l > 0.10]
print('\ncatastrophic learned episodes (learned>0.10): %d' % len(cat))
for e, l, v in cat: print('   %-42s learned %.4f  v2 %.4f' % (e[8:-3], l, v))
noncat = [i for i, l in enumerate(L) if l <= 0.10]
print('\nNON-catastrophic %d/%d: learned mean %.5f  v2 mean %.5f  (diff %+.5f)'
      % (len(noncat), len(SP), L[noncat].mean(), V[noncat].mean(), L[noncat].mean()-V[noncat].mean()))
dn = (L-V)[noncat]
print('   paired medianD %+.5f  learned-better %d/%d  Wilcoxon2 p=%.4f'
      % (np.median(dn), int((dn < 0).sum()), len(dn), stats.wilcoxon(dn)[1]))
print('   are all catastrophes BF2_1comp? ->', all('BF2_1comp' in e for e, _, _ in cat))

# MDE for the FAIL bars
for lab, arr in (('pure-learned', L),):
    sd = (arr-V).std(ddof=1)
    print('\nMDE for the mean-bar test on n=40 sparse (paired, 2-sided 80%%): %.5f  (sd=%.5f)'
          % ((stats.norm.ppf(.975)+stats.norm.ppf(.8))*sd/np.sqrt(40), sd))
    sdn = dn.std(ddof=1)
    print('MDE excluding the 5 catastrophes (n=35): %.5f  (sd=%.5f)'
          % ((stats.norm.ppf(.975)+stats.norm.ppf(.8))*sdn/np.sqrt(len(dn)), sdn))

# anchor / merge variants: mean bar vs paired median
for name, path, fld, vs in (('peak-anchor', '/home/vexit/wavesplat/wavesplat/tmp/learned_only_2026-07-26/anchor_arms.jsonl', 'anchor_frac', [0.375, 0.6]),
                            ('kp-merge', '/home/vexit/wavesplat/wavesplat/tmp/unified_guard_2026-07-26/merge_arms.jsonl', 'merge_frac', [0.25, 0.375])):
    rows = [json.loads(l) for l in open(path)]
    dd = {}
    for r in rows:
        if r.get('err') is None:
            dd[(r['episode'], r['est_seed'], r['diff_seed'], r[fld])] = r['avg_nRMSE']
    print('\n--- %s variants ---' % name)
    for f in vs:
        A = np.array([np.median([dd[(e, E, D, f)] for E, D in LANES]) for e in SP])
        p = A - V
        print('  frac=%-6s mean %.5f (bar %.5f -> %s) | paired medianD vs v2 %+.5f  wins %d/40  p=%.3f'
              % (f, A.mean(), V.mean()+0.005, 'PASS' if A.mean() <= V.mean()+0.005 else 'FAIL',
                 np.median(p), int((p < 0).sum()), stats.wilcoxon(p)[1]))
        nc = [i for i in range(len(SP)) if A[i] <= 0.10]
        print('         non-catastrophic n=%d: mean %.5f vs v2 %.5f -> %s'
              % (len(nc), A[nc].mean(), V[nc].mean(), 'PASS' if A[nc].mean() <= V[nc].mean()+0.005 else 'FAIL'))
