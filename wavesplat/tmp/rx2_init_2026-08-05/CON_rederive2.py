"""CON re-derivation, part 2 — sharpening attacks 1, 3, 5.

Adds: tie-safe sign test, a proper alpha=0.05 MDE for n>=6, the crowded-only E2 subset,
the panel-level size of the prize, and the tautology check on init_nrmse -> nRMSE.
Pure re-analysis; no runs.
"""
import json, os, itertools
import numpy as np
from math import comb
from scipy.stats import norm

TMP = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
OUT = []


def P(*a):
    s = ' '.join(str(x) for x in a); print(s); OUT.append(s)


def load(path, key):
    rows = []
    for line in open(path):
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if 'error' in r:
            continue
        r['_cell'] = r[key]; rows.append(r)
    return rows


def sign_p(k, n):
    if n == 0:
        return float('nan')
    return min(1.0, 2 * sum(comb(n, i) for i in range(0, min(k, n - k) + 1)) / 2 ** n)


def signtest(deltas, tol=1e-12):
    """Tie-safe: ties are dropped (standard), reported separately."""
    d = [v for v in deltas if abs(v) > tol]
    ties = len(deltas) - len(d)
    k = sum(1 for v in d if v < 0)
    return k, len(d), ties, sign_p(k, len(d))


def crit_k(n, alpha=0.05):
    """smallest k such that the two-sided sign test with k of n reaches alpha."""
    for k in range(0, n // 2 + 1):
        if sign_p(k, n) <= alpha:
            return k
    return None


def mde_alpha(sd, n, alpha=0.05, power=0.80):
    """Effect needed for `power` chance of reaching alpha with the two-sided sign test.
    Returns None where alpha is unreachable at this n (n<6)."""
    k = crit_k(n, alpha)
    if k is None:
        return None
    # need at least n-k successes; P(X >= n-k) with X~Bin(n, p), p = Phi(mu/sd)
    lo, hi = 0.5, 1 - 1e-9
    for _ in range(200):
        mid = (lo + hi) / 2
        pw = sum(comb(n, i) * mid ** i * (1 - mid) ** (n - i) for i in range(n - k, n + 1))
        if pw < power:
            lo = mid
        else:
            hi = mid
    return float(norm.ppf(hi) * sd)


def mde_allsame(sd, n, power=0.80):
    return float(norm.ppf(power ** (1.0 / n)) * sd)


def levels(rows, metric):
    d = {}
    for r in rows:
        v = r.get(metric)
        if v is None or not np.isfinite(v):
            continue
        d.setdefault((r['_cell'], r['episode']), []).append(float(v))
    return {ce: float(np.median(v)) for ce, v in d.items()}, d


def paired(rows, a, b, metric, eps=None):
    lv, _ = levels(rows, metric)
    E = sorted({e for (c, e) in lv if c == a} & {e for (c, e) in lv if c == b})
    if eps is not None:
        E = [e for e in E if e in eps]
    return [lv[(b, e)] - lv[(a, e)] for e in E], E


s1s2 = load(os.path.join(TMP, 's1s2_2026-08-05', 'results.jsonl'), 'arm')
rx1 = load(os.path.join(TMP, 'rx1_tracker_2026-08-05', 'results2.jsonl'), 'cell')
e2 = load(os.path.join(TMP, 'e2_grid_sep_2026-08-05', 'results.jsonl'), 'cell')
e1 = load(os.path.join(TMP, 'e1_grid_2026-08-05', 'results.jsonl'), 'cell')

CROWD = ['episode_BF4_15comp_seed0_uniform.h5', 'episode_BF4_30comp_seed0_uniform.h5',
         'episode_BF5_15comp_seed0_uniform.h5', 'episode_BF5_30comp_seed0_uniform.h5']

PRIZE = 0.0974   # median paired delta nRMSE, S1-A0 -> S2-T1f (re-derived in part 1)
PRIZE_HONEST = None  # filled below

P('=' * 100)
P('ATTACK 1 (sharpened) — MDE ledger, tie-safe, with an alpha=0.05 MDE where n allows')
P('=' * 100)
dl_h, _ = paired(s1s2, 'S1-A0', 'S2-T1', 'nRMSE')
PRIZE_HONEST = abs(float(np.median(dl_h)))
P(f'  prize as quoted   (S1-A0 -> S2-T1f, oracle AND k frozen) = {PRIZE:.4f} nRMSE')
P(f'  prize, placement only (S1-A0 -> S2-T1, same machinery)   = {PRIZE_HONEST:.4f} nRMSE')
P('')
CAND = [
    ('tracker OFF @deployment',   rx1,  'D-BASE', 'D-NOTRACK', None),
    ('tracker lambda=0 @deploy',  rx1,  'D-BASE', 'D-K12',     None),
    ('tracker steps=1 @deploy',   rx1,  'D-BASE', 'D-K6',      None),
    ('budget 15->22 (S1-A1)',     s1s2, 'S1-A0',  'S1-A1',     None),
    ('budget 15->30 (S1-A2)',     s1s2, 'S1-A0',  'S1-A2',     None),
    ('budget 15->40 (S1-A3)',     s1s2, 'S1-A0',  'S1-A3',     None),
    ('prune OFF @oracle',         s1s2, 'S2-T1',  'S2-T2',     None),
    ('adaptive-N OFF @oracle',    s1s2, 'S2-T1f', 'S2-T4',     None),
    ('grid 81->401 (12 ep)',      e2,   'A0',     'U1',        None),
    ('grid 81->401 (crowded 4)',  e2,   'A0',     'U1',        CROWD),
    ('grid+minsep (12 ep)',       e2,   'A0',     'U1M2',      None),
    ('grid+minsep (crowded 4)',   e2,   'A0',     'U1M2',      CROWD),
    ('E1 suppression D (12 ep)',  e1,   'A',      'D',         None),
    ('E1 suppression D (crowd 4)', e1,  'A',      'D',         CROWD),
]
P(f'{"candidate":28s} {"n":>3s} {"ties":>4s} {"PRIMARY":>9s} {"k/n":>7s} {"p":>7s} '
  f'{"sd":>7s} {"MDE(n/n)":>9s} {"MDE(a=.05)":>11s} {"MDE/prize":>10s}')
ledger = {}
for name, rows, a, b, eps in CAND:
    dl, E = paired(rows, a, b, 'nRMSE', eps)
    if not dl:
        P(f'{name:28s}  -- missing'); continue
    k, nn, ties, p = signtest(dl)
    sd = float(np.std(dl, ddof=1)) if len(dl) > 1 else float('nan')
    m1 = mde_allsame(sd, len(dl))
    m2 = mde_alpha(sd, len(dl))
    ledger[name] = dict(n=len(dl), primary=float(np.median(dl)), k=k, nn=nn, ties=ties,
                        p=p, sd=sd, mde_allsame=m1, mde_alpha=m2,
                        frac_prize=m1 / PRIZE, deltas=dl)
    P(f'{name:28s} {len(dl):3d} {ties:4d} {np.median(dl):+9.4f} {k:3d}/{nn:<3d} {p:7.4f} '
      f'{sd:7.4f} {m1:9.4f} '
      f'{("%.4f" % m2) if m2 else "unreach.":>11s} {m1/PRIZE:9.0%}')
P('')
P('  Reading: "MDE(n/n)" is the effect that would be found 80% of the time by the ONLY')
P('  outcome that reaches the p-floor at n=4 (all four deltas one way). An intervention')
P('  whose MDE/prize is large came back NULL from a test that could not have seen an')
P('  effect that big. It is not evidence of absence at that scale.')

P('')
P('=' * 100)
P('ATTACK 1b — seed noise vs. the levers')
P('=' * 100)
_, raw = levels(s1s2, 'nRMSE')
for arm in ['S1-A0', 'S1-A3', 'S2-C1', 'S2-T1', 'S2-T1f']:
    rg = [max(v) - min(v) for (c, e), v in raw.items() if c == arm]
    P(f'  {arm:8s} within-cell nRMSE range across 3 seeds: median {np.median(rg):.4f}  '
      f'max {max(rg):.4f}')
P(f'  every measured DEPLOYABLE intervention moves the median by <= 0.015;')
P(f'  the seed range of the shipped arm alone is ~0.038 on the same episodes.')

P('')
P('=' * 100)
P('ATTACK 3 (sharpened) — is "initialisation dominates" a finding or a restatement?')
P('=' * 100)
li, _ = levels(e2, 'init_nrmse'); ln, _ = levels(e2, 'nRMSE')
for arm in ['A0', 'U1', 'U2']:
    xs = np.array([li[(arm, e)] for (c, e) in li if c == arm])
    ys = np.array([ln[(arm, e)] for (c, e) in ln if c == arm])
    P(f'  E2 {arm:5s}: across 12 episodes r(init_nrmse, nRMSE) = '
      f'{np.corrcoef(xs, ys)[0,1]:+.4f}   R^2 = {np.corrcoef(xs, ys)[0,1]**2:.4f}')
li2, _ = levels(s1s2, 'init_nrmse'); ln2, _ = levels(s1s2, 'nRMSE')
xs = np.array([v for (c, e), v in li2.items() if c.startswith('S1')])
ys = np.array([ln2[(c, e)] for (c, e) in li2 if c.startswith('S1')])
P(f'  s1s2 deployable arms (16 cells): r = {np.corrcoef(xs, ys)[0,1]:+.4f}')
P('')
P('  Combined with the measured fact that turning the tracker OFF at deployment changes')
P('  nRMSE by +0.0000 (rx1 D-NOTRACK), "the deployment field error IS the initialisation')
P('  field error" is close to an identity, not a causal discovery. What is NOT an identity,')
P('  and is the thing that actually needs testing, is whether any DEPLOYABLE knob can move')
P('  init_nrmse by an amount worth having:')
best = {}
for a, b in [('S1-A0', 'S1-A1'), ('S1-A0', 'S1-A2'), ('S1-A0', 'S1-A3'),
             ('S1-A0', 'S2-C1')]:
    di, _ = paired(s1s2, a, b, 'init_nrmse')
    dn, _ = paired(s1s2, a, b, 'nRMSE')
    ki, ni, _, pi = signtest(di)
    P(f'    {a}->{b:8s}  med d(init_nrmse) {np.median(di):+.4f} ({ki}/{ni}, p={pi:.3f})'
      f'   med d(nRMSE) {np.median(dn):+.4f}')
    best[f'{a}->{b}'] = (float(np.median(di)), float(np.median(dn)))
P(f'    best deployable init gain measured anywhere = '
  f'{min(v[0] for v in best.values()):+.4f}  '
  f'= {abs(min(v[0] for v in best.values()))/0.1809:.1%} of the quoted 0.1809 init gap')

P('')
P('=' * 100)
P('ATTACK 5 (sharpened) — how big is the prize on the panel the paper actually reports?')
P('=' * 100)
rows = sorted([(e, li[('A0', e)], ln[('A0', e)]) for (c, e) in li if c == 'A0'])
arr_n = np.array([r[2] for r in rows])
oracle_n, _ = paired(s1s2, 'S1-A0', 'S2-T1f', 'nRMSE')
lvn2, _ = levels(s1s2, 'nRMSE')
orc = {e: lvn2[('S2-T1f', e)] for e in CROWD}
new = np.array([orc[e] if e in orc else v for (e, _, v) in rows])
P(f'  12-episode panel, shipped arm:      mean nRMSE {arr_n.mean():.4f}  '
  f'median {np.median(arr_n):.4f}')
P(f'  same panel with the 4 crowded episodes replaced by the ORACLE arm (unattainable):')
P(f'                                      mean nRMSE {new.mean():.4f}  '
  f'median {np.median(new):.4f}')
P(f'  -> perfect, oracle-grade initialisation on every crowded episode moves the panel')
P(f'     mean by {arr_n.mean()-new.mean():.4f} and the panel MEDIAN by '
  f'{np.median(arr_n)-np.median(new):.4f}.')
P('')
P('  Where the deployment error lives, by regime (shipped arm, 12-ep panel):')
crowd = [r for r in rows if '15comp' in r[0] or '30comp' in r[0]]
rest = [r for r in rows if r not in crowd]
P(f'    crowded  n={len(crowd):2d}  init {np.median([r[1] for r in crowd]):.4f}  '
  f'nRMSE {np.median([r[2] for r in crowd]):.4f}')
P(f'    other    n={len(rest):2d}  init {np.median([r[1] for r in rest]):.4f}  '
  f'nRMSE {np.median([r[2] for r in rest]):.4f}   '
  f'ratio crowded/other = {np.median([r[2] for r in crowd])/np.median([r[2] for r in rest]):.1f}x')
P('')
P('  And what fraction of the 100-episode deployment distribution is "crowded"?')
P('   (dataset census, data_finemesh_2026-07-23, uniform+slope):')
import collections
DATA = os.path.join(TMP, '..', 'data_finemesh_2026-07-23')
cnt = collections.Counter()
for f in os.listdir(DATA):
    if not f.endswith('.h5'):
        continue
    nc = f.split('_')[2].replace('comp', '')
    cnt[int(nc)] += 1
tot = sum(cnt.values())
for k in sorted(cnt):
    P(f'    {k:3d}comp : {cnt[k]:3d}  ({cnt[k]/tot:5.1%})')
P(f'    crowded (15/30comp) = {(cnt[15]+cnt[30])/tot:.1%} of the corpus')

json.dump({'ledger': ledger, 'prize_quoted': PRIZE, 'prize_placement_only': PRIZE_HONEST},
          open(os.path.join(os.path.dirname(__file__), 'CON_rederive2.json'), 'w'), indent=1)
open(os.path.join(os.path.dirname(__file__), 'CON_rederive2.txt'), 'w').write('\n'.join(OUT) + '\n')
