"""Audit re-derivation of Conclusion 1 (wp_crowded 2026-07-25) from raw JSONL only.
No new runs. Reads cond_runs.jsonl / ws_runs.jsonl.
"""
import json, os, itertools
import numpy as np
from scipy import stats

D = '/home/vexit/wavesplat/wavesplat/tmp/wp_crowded_2026-07-25'

cond = [json.loads(l) for l in open(os.path.join(D, 'cond_runs.jsonl'))]
ws = [json.loads(l) for l in open(os.path.join(D, 'ws_runs.jsonl'))]
print('raw lines: cond=%d ws=%d' % (len(cond), len(ws)))
print('cond errs:', sum(1 for r in cond if r.get('err')), ' ws errs:', sum(1 for r in ws if r.get('err')))
print('ws flags true:', sum(1 for r in ws if r.get('flag')))

# group
from collections import defaultdict
cg = defaultdict(list); wg_i = defaultdict(list); wg_t = defaultdict(list)
for r in cond:
    cg[r['episode']].append(r['cond_polish_nrmse_Hs'])
for r in ws:
    wg_i[r['episode']].append(r['init_nrmse']); wg_t[r['episode']].append(r['avg_nRMSE'])

# frozen primary set from PREREG sec 6
prim = ['episode_BF%d_%dcomp_seed%d_%s.h5' % (bf, nc, s, a)
        for bf in (4, 5) for nc in (15, 30) for s in range(5) for a in ('uniform', 'slope')]
sparse = ['episode_BF2_1comp_seed0_uniform.h5','episode_BF2_1comp_seed0_slope.h5',
 'episode_BF3_1comp_seed0_uniform.h5','episode_BF3_2comp_seed0_uniform.h5',
 'episode_BF3_2comp_seed0_slope.h5','episode_BF3_5comp_seed0_uniform.h5',
 'episode_BF3_5comp_seed1_uniform.h5','episode_BF3_5comp_seed0_slope.h5',
 'episode_BF4_1comp_seed0_uniform.h5','episode_BF4_5comp_seed0_uniform.h5',
 'episode_BF4_5comp_seed0_slope.h5','episode_BF4_5comp_seed1_uniform.h5']

def build(eps):
    rows = []
    for e in eps:
        assert e in cg, ('missing cond', e)
        assert e in wg_i, ('missing ws', e)
        c = float(np.median(cg[e])); ii = float(np.median(wg_i[e])); tt = float(np.median(wg_t[e]))
        rows.append(dict(ep=e, nc=int(e.split('_')[2].replace('comp','')), c=c, i=ii, t=tt,
                         best=min(ii, tt), kc=len(cg[e]), kw=len(wg_i[e])))
    return rows

rows = build(prim)
print('\nk per episode: cond', sorted(set(r['kc'] for r in rows)), 'ws', sorted(set(r['kw'] for r in rows)))

def wtest(dd, alt='less'):
    st, p = stats.wilcoxon(dd, alternative=alt)
    return st, p

for label, ref in (('ws_best (PRIMARY)', 'best'), ('ws_tracked (secondary)', 't'), ('ws_init', 'i')):
    d = np.array([r['c'] - r[ref] for r in rows])
    st, p = wtest(d)
    wins = int((d < 0).sum())
    print('\n--- %s ---' % label)
    print(' n=%d  median pairedD=%.6f  mean=%.6f  wins=%d/%d  Wilcoxon one-sided p=%.4g' %
          (len(d), np.median(d), d.mean(), wins, len(d), p))
    refv = np.array([r[ref] for r in rows]); cv = np.array([r['c'] for r in rows])
    print(' AGGREGATION VARIANTS for the "%%" figure:')
    print('   (a) medianD / mean(ref)             = %.1f%%' % (100*np.median(d)/refv.mean()))
    print('   (b) medianD / median(ref)           = %.1f%%' % (100*np.median(d)/np.median(refv)))
    print('   (c) median of per-ep ratios (c/ref-1)= %.1f%%' % (100*np.median(cv/refv - 1)))
    print('   (d) mean(c)/mean(ref) - 1           = %.1f%%' % (100*(cv.mean()/refv.mean()-1)))
    print('   (e) median(c)/median(ref) - 1       = %.1f%%' % (100*(np.median(cv)/np.median(refv)-1)))
    # rank biserial
    dnz = d[d != 0]
    r_ = stats.rankdata(np.abs(dnz))
    Wp = r_[dnz > 0].sum(); Wm = r_[dnz < 0].sum()
    print('   rank-biserial r = 1-2W-/(W+ + W-) = %.4f  (report sign convention!)' % (1 - 2*Wm/(Wp+Wm)))

# by ncomp
print('\n--- split by n_comp (STRATIFYING VARIABLE CHECK) ---')
for nc in (15, 30):
    sub = [r for r in rows if r['nc'] == nc]
    d = np.array([r['c'] - r['best'] for r in sub])
    st, p = wtest(d)
    print(' %dcomp n=%d medD=%.4f wins=%d p=%.3g  cond_med=%.4f ws_best_med=%.4f' %
          (nc, len(d), np.median(d), int((d < 0).sum()), p, np.median([r['c'] for r in sub]), np.median([r['best'] for r in sub])))

# sparse control
srows = build(sparse)
d = np.array([r['c'] - r['best'] for r in srows])
print('\n--- SPARSE control n=%d ---' % len(d))
print(' cond worse (WS wins) on %d/%d ; median D=%+.4f ; two-sided p=%.4g' %
      (int((d > 0).sum()), len(d), np.median(d), stats.wilcoxon(d)[1]))
print(' per-ep: ' + ', '.join('%s c=%.3f best=%.3f' % (r['ep'][8:-3], r['c'], r['best']) for r in srows))

# the one loss in primary
loss = [r for r in rows if r['c'] - r['best'] >= 0]
print('\n primary non-wins:', [(r['ep'], round(r['c'],4), round(r['best'],4)) for r in loss])
