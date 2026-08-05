import json, os, re
import numpy as np
from scipy import stats
from collections import defaultdict

D = '/home/vexit/wavesplat/wavesplat/tmp/heldout40_2026-07-26'
def load(f):
    return [json.loads(l) for l in open(os.path.join(D, f))]

L = load('arm_runs.jsonl'); V = load('arm_runs_v2_control.jsonl'); A = load('arm_runs_auto_live.jsonl')
print('errs L/V/A:', sum(1 for r in L if r.get('err')), sum(1 for r in V if r.get('err')), sum(1 for r in A if r.get('err')))
print('learned flag=True:', sum(1 for r in L if r.get('flag')))

def comp(ep): return int(re.search(r'_(\d+)comp', ep).group(1))
def slc(ep):
    c = comp(ep); return 'sparse' if c in (1, 2) else ('moderate' if c == 5 else 'crowded')

lanes = sorted({(r['est_seed'], r['diff_seed']) for r in L})
print('lanes', lanes)

def bylane(rows):
    d = defaultdict(dict)
    for r in rows: d[r['episode']][(r['est_seed'], r['diff_seed'])] = r['avg_nRMSE']
    return d
lv, ll, la = bylane(V), bylane(L), bylane(A)
eps = sorted(ll)
print('episodes', len(eps), {k: sum(1 for e in eps if slc(e) == k) for k in ('sparse','moderate','crowded')})

def med(d, e): return float(np.median([d[e][k] for k in lanes]))

# auto-with-guard reassembly is in score_results.json; use live auto arm as 'auto'
tab = {}
for name, d in (('pure_v2', lv), ('pure_learned', ll), ('auto_live', la)):
    tab[name] = {e: med(d, e) for e in eps}
# oracle lane-consistent
tab['oracle_lane'] = {e: float(np.median([min(lv[e][k], ll[e][k]) for k in lanes])) for e in eps}

print('\n slice means (median-of-5 per ep, then mean over eps):')
print(' %-10s %5s %9s %9s %9s %9s' % ('slice','n','pure_v2','pure_lrn','auto_live','oracle'))
for s in ('all','sparse','moderate','crowded'):
    ee = [e for e in eps if s == 'all' or slc(e) == s]
    print(' %-10s %5d %9.5f %9.5f %9.5f %9.5f' % (s, len(ee),
          np.mean([tab['pure_v2'][e] for e in ee]), np.mean([tab['pure_learned'][e] for e in ee]),
          np.mean([tab['auto_live'][e] for e in ee]), np.mean([tab['oracle_lane'][e] for e in ee])))

# crowded learned gain
ee = [e for e in eps if slc(e) == 'crowded']
v = np.array([tab['pure_v2'][e] for e in ee]); l = np.array([tab['pure_learned'][e] for e in ee])
d = l - v
print('\n CROWDED (n=%d) learned vs v2:' % len(ee))
print('  mean-of-arms: %.5f vs %.5f  -> ratio-of-means %+.1f%%' % (l.mean(), v.mean(), 100*(l.mean()/v.mean()-1)))
print('  median paired D = %+.5f ; median of per-ep ratios = %+.1f%% ; wins %d/%d ; p1(less)=%.4g'
      % (np.median(d), 100*np.median(l/v-1), int((d<0).sum()), len(d), stats.wilcoxon(d, alternative='less')[1]))
print('  median-of-medians: learned %.5f v2 %.5f -> %+.1f%%' % (np.median(l), np.median(v), 100*(np.median(l)/np.median(v)-1)))

# per-slice paired tests + MDE
def mde(n, sd, alpha=0.05, power=0.8, onesided=True):
    z_a = stats.norm.ppf(1-alpha) if onesided else stats.norm.ppf(1-alpha/2)
    return (z_a + stats.norm.ppf(power)) * sd / np.sqrt(n)

print('\n per-slice paired (learned - v2) with normal-approx MDE @80%% power:')
for s in ('sparse','moderate','crowded','all'):
    ee = [e for e in eps if s == 'all' or slc(e) == s]
    d = np.array([tab['pure_learned'][e] - tab['pure_v2'][e] for e in ee])
    p2 = stats.wilcoxon(d)[1] if len(d) > 5 else float('nan')
    print('  %-9s n=%2d medD=%+.5f meanD=%+.5f sd=%.5f p2=%.4g  MDE(paired t,1-sided)=%.5f  signtest floor p=%.4g'
          % (s, len(d), np.median(d), d.mean(), d.std(ddof=1), p2, mde(len(d), d.std(ddof=1)), 0.5**len(d)*2))
