"""Items 3 (stratifying variable) and 4 (MDE table). Existing raw only."""
import json, re
import numpy as np
from scipy import stats
from collections import defaultdict

NEFF = {1: 1.000, 2: 1.063, 5: 2.557, 15: 4.555, 30: 7.820}   # tmp/depth3_ceiling_2026-08-05/CON_followups.json
print('N_eff(n_comp) from today: ', NEFF, '  (6->1.97, 7->1.62, 8->1.87 = NON-MONOTONE just above 5)')

def comp(ep): return int(re.search(r'_(\d+)comp', ep).group(1))
def mde(n, sd, one=True, alpha=.05, power=.8):
    z = stats.norm.ppf(1-alpha) if one else stats.norm.ppf(1-alpha/2)
    return (z + stats.norm.ppf(power)) * sd / np.sqrt(n)

# ---------- C1 crowded: 15 vs 30 ----------
D = '/home/vexit/wavesplat/wavesplat/tmp/wp_crowded_2026-07-25'
cg = defaultdict(list); wi = defaultdict(list); wt = defaultdict(list)
for l in open(D+'/cond_runs.jsonl'):
    r = json.loads(l)
    if r['cond_polish_nrmse_Hs'] is not None: cg[r['episode']].append(r['cond_polish_nrmse_Hs'])
for l in open(D+'/ws_runs.jsonl'):
    r = json.loads(l); wi[r['episode']].append(r['init_nrmse']); wt[r['episode']].append(r['avg_nRMSE'])
prim = [e for e in cg if comp(e) in (15, 30) and e in wi]
print('\n=== C1 head-to-head: does the gain grow with crowding? ===')
for nc in (15, 30):
    ee = [e for e in prim if comp(e) == nc]
    d = np.array([np.median(cg[e]) - min(np.median(wi[e]), np.median(wt[e])) for e in ee])
    print('  %2dcomp (N_eff %.2f) n=%d medD %+.4f  rel %+.1f%%  wins %d'
          % (nc, NEFF[nc], len(ee), np.median(d),
             100*np.median([np.median(cg[e])/min(np.median(wi[e]), np.median(wt[e]))-1 for e in ee]),
             int((d < 0).sum())))
print('  >>> gain SHRINKS from 15comp to 30comp (both absolute and relative).')

# ---------- C2 integration 15 vs 30 ----------
I = '/home/vexit/wavesplat/wavesplat/tmp/wp_integration_2026-07-25/arm_runs.jsonl'
g = defaultdict(lambda: defaultdict(list))
for l in open(I):
    r = json.loads(l); g[r['episode']][r['arm']].append(r['avg_nRMSE'])
print('\n=== C2 integration: 15 vs 30 ===')
for nc in (15, 30):
    ee = [e for e in g if comp(e) == nc]
    d = np.array([np.median(g[e]['learned']) - np.median(g[e]['v2_konly']) for e in ee])
    print('  %2dcomp n=%d medD %+.4f wins %d/%d p1=%.3g  MDE=%.4f' %
          (nc, len(ee), np.median(d), int((d < 0).sum()), len(d),
           stats.wilcoxon(d, alternative='less')[1], mde(len(d), d.std(ddof=1))))

# ---------- sparse: 1comp vs 2comp inside the "sparse" label ----------
BATCHC = '/home/vexit/wavesplat/wavesplat/tmp/rerun_2026-07-25/batchC_ws_phasefix.jsonl'
X = '/home/vexit/wavesplat/wavesplat/tmp/unified_2026-07-26/arm_runs_extra.jsonl'
LANES = [(42, 0), (1, 1), (2, 2), (3, 3), (4, 4)]
v2 = defaultdict(dict); eps = set()
for l in open(BATCHC):
    d = json.loads(l); v2[d['episode']][d['seed']] = d['nRMSE']; eps.add(d['episode'])
lr = defaultdict(dict)
for src in (I, X):
    for l in open(src):
        d = json.loads(l)
        if d.get('arm') == 'learned' and d.get('err') is None:
            lr[d['episode']][(d['est_seed'], d['diff_seed'])] = d['avg_nRMSE']
print('\n=== "sparse" label (n_comp in {1,2}) is NOT homogeneous ===')
for nc in (1, 2):
    ee = sorted([e for e in eps if comp(e) == nc and e in lr])
    if not ee: continue
    L = np.array([np.median([lr[e][k] for k in LANES]) for e in ee])
    V = np.array([np.median([v2[e][E] for E, _ in LANES]) for e in ee])
    d = L - V
    print('  %dcomp (N_eff %.3f) n=%d  learned mean %.4f v2 mean %.4f | medD %+.5f  learned-better %d/%d  #blowups(>0.1) %d'
          % (nc, NEFF[nc], len(ee), L.mean(), V.mean(), np.median(d), int((d < 0).sum()), len(d), int((L > 0.1).sum())))

# ---------- MDE TABLE ----------
print('\n=== MDE TABLE (paired, 80% power, alpha .05) ===')
print(' %-42s %4s %10s %10s %10s %s' % ('conclusion / test', 'n', 'obs eff', 'sd', 'MDE', 'verdict'))
rows = []
# C1
d = np.array([np.median(cg[e]) - min(np.median(wi[e]), np.median(wt[e])) for e in prim])
rows.append(('C1 cond vs ws_best (crowded 40)', d, True, 'PASS'))
# C2
ee = sorted(g)
ee = [e for e in ee if comp(e) in (15, 30)]
d = np.array([np.median(g[e]['learned']) - np.median(g[e]['v2_konly']) for e in ee])
rows.append(('C2 learned vs v2_konly (crowded 40)', d, True, 'PASS'))
# C3 heldout crowded / sparse / moderate
H = '/home/vexit/wavesplat/wavesplat/tmp/heldout40_2026-07-26/'
def hl(f, key='avg_nRMSE'):
    out = defaultdict(dict)
    for l in open(H+f):
        r = json.loads(l); out[r['episode']][(r['est_seed'], r['diff_seed'])] = r[key]
    return out
hL, hV, hA = hl('arm_runs.jsonl'), hl('arm_runs_v2_control.jsonl'), hl('arm_runs_auto_live.jsonl')
lanes = sorted(hL[list(hL)[0]])
def m(o, e): return float(np.median([o[e][k] for k in lanes]))
for lab, sel in (('C3 heldout crowded learned vs v2', lambda e: comp(e) in (15, 30)),
                 ('C3 heldout sparse learned vs v2', lambda e: comp(e) in (1, 2)),
                 ('C3 heldout moderate learned vs v2', lambda e: comp(e) == 5),
                 ('C3 heldout S1: auto+live vs pure_v2 (all)', lambda e: True)):
    ee = [e for e in hL if sel(e)]
    src = hA if 'auto' in lab else hL
    d = np.array([m(src, e) - m(hV, e) for e in ee])
    rows.append((lab, d, False, ''))
# C4 e2e
ft = defaultdict(list)
for l in open('/home/vexit/wavesplat/wavesplat/tmp/wp_e2e_2026-07-25/ft_cond_runs.jsonl'):
    r = json.loads(l); ft[r['episode']].append(r['cond_polish_nrmse_Hs'])
d = np.array([np.median(ft[e]) - np.median(cg[e]) for e in prim])
rows.append(('C4 e2e finetune vs current (crowded 40)', d, True, 'FAIL'))
# C5 real
R = json.load(open('/home/vexit/wavesplat/wavesplat/tmp/wp_real_2026-07-25/wp_real.json'))['results']
d = np.array([r['learned_now'] - r['hand_now'] for r in R if not r['collapse']])
rows.append(('C5 real normal-27 learned vs hand', d, False, 'LOSS-sig'))
# C6 G1
txt = open('/home/vexit/wavesplat/wavesplat/tmp/G1_waveprior_power/eval_fan_prereg.log').read()
mm = re.findall(r'hand ([\d.]+)\s+fanTF ([\d.]+)\s+oldTF ([\d.]+)', txt)
d = np.array([float(a)-float(b) for a, b, _ in mm])
rows.append(('C6 G1 fan PRIMARY fanTF vs hand (n=30)', -d, True, 'FAIL'))
# C7 learned-only sparse
ee = sorted([e for e in eps if comp(e) in (1, 2) and e in lr])
d = np.array([np.median([lr[e][k] for k in LANES]) - np.median([v2[e][E] for E, _ in LANES]) for e in ee])
rows.append(('C7 pure-learned vs v2 (sparse 40)', d, False, 'FAIL(mean bar)'))
for lab, d, one, verd in rows:
    print(' %-42s %4d %10.5f %10.5f %10.5f %s' %
          (lab, len(d), np.median(d), d.std(ddof=1), mde(len(d), d.std(ddof=1), one), verd))
