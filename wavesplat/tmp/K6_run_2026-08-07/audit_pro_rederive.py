#!/usr/bin/env python3
"""PRO audit of K6 -- INDEPENDENT re-derivation written from scratch from results.jsonl.
Reads only.  PREREG sec.4 protocol applied literally:
  nRMSE worse = larger; aggregate seed-median -> median of paired episode deltas;
  paired two-sided SIGN test; ties (delta exactly 0) dropped from n and counted;
  frozen n->required-wins table; null-failing control episodes excluded from PRIMARY
  with the non-excluded result reported alongside.
"""
import json, os, math, itertools
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, 'results.jsonl')
NUL = os.path.join(HERE, 'nulls_k6.json')

# ---------- load ----------
rows = []
with open(RES) as f:
    for ln in f:
        ln = ln.strip()
        if ln:
            rows.append(json.loads(ln))

nulls = {d['episode']: d for d in json.load(open(NUL))}

CELLS = sorted({r['cell'] for r in rows})
EPS = sorted({r['episode'] for r in rows})
SEEDS = sorted({r['seed'] for r in rows})

print('=' * 78)
print('0. INVENTORY')
print('=' * 78)
print(f'runs={len(rows)}  cells={len(CELLS)}  episodes={len(EPS)}  seeds={SEEDS}')
print('cells:', CELLS)
cnt = defaultdict(int)
for r in rows:
    cnt[(r['cell'], r['episode'], r['seed'])] += 1
dups = {k: v for k, v in cnt.items() if v > 1}
missing = [(c, e, s) for c in CELLS for e in EPS for s in SEEDS if (c, e, s) not in cnt]
print(f'duplicate (cell,ep,seed) keys: {len(dups)}   missing: {len(missing)}')
bad = [r for r in rows if r.get('nRMSE') is None or (isinstance(r['nRMSE'], float) and math.isnan(r['nRMSE']))]
print(f'rows with null/NaN nRMSE: {len(bad)}')
err = [r for r in rows if 'error' in r]
print(f'rows carrying an error field: {len(err)}')

# router composition of the AUTO arm
rt = defaultdict(int)
for r in rows:
    if r['cell'] == 'AUTO':
        rt[r.get('auto_selected')] += 1
print('AUTO router composition (48 runs):', dict(rt))
rt2 = defaultdict(int)
for r in rows:
    if r['cell'] == 'AUTO-DUP':
        rt2[r.get('auto_selected')] += 1
print('AUTO-DUP router composition:', dict(rt2))
# which episodes fell back
fb = sorted({(r['episode'], r['seed']) for r in rows
             if r['cell'] == 'AUTO' and r.get('auto_selected') != 'learned'})
print('AUTO non-learned (episode,seed):', fb)

# ---------- integrity: AUTO vs AUTO-DUP bit identity ----------
print()
print('=' * 78)
print('1. INTEGRITY -- worker-leak probe (PREREG sec.6.3)')
print('=' * 78)
IDX = {(r['cell'], r['episode'], r['seed']): r for r in rows}
FIELDS = ['nRMSE', 'init_nrmse', 'Pearson_rho', 'absVR_framewise',
          'decorr_mean', 'shrink_mean', 'm_mean', 'Variance_Ratio', 'Hs_error']
nid, ndiff, worst = 0, 0, (0.0, None, None)
for e in EPS:
    for s in SEEDS:
        a, b = IDX[('AUTO', e, s)], IDX[('AUTO-DUP', e, s)]
        same = all(a[f] == b[f] for f in FIELDS)
        nid += same
        if not same:
            ndiff += 1
            d = max(abs(a[f] - b[f]) for f in FIELDS)
            if d > worst[0]:
                worst = (d, e, s)
print(f'AUTO == AUTO-DUP exactly, on all {len(FIELDS)} panel fields: {nid}/48   differing: {ndiff}')
if ndiff:
    print('  worst abs diff:', worst)

# ---------- aggregation ----------
def seed_median(cell, ep, field):
    v = [IDX[(cell, ep, s)][field] for s in SEEDS]
    v = [x for x in v if x is not None]
    v.sort()
    n = len(v)
    return None if n == 0 else (v[n // 2] if n % 2 else 0.5 * (v[n // 2 - 1] + v[n // 2]))

def spec_seed_median(cell, ep, key):
    v = []
    for s in SEEDS:
        sp = IDX[(cell, ep, s)].get('spec') or {}
        if sp.get(key) is not None:
            v.append(sp[key])
    v.sort(); n = len(v)
    return None if n == 0 else (v[n // 2] if n % 2 else 0.5 * (v[n // 2 - 1] + v[n // 2]))

def med(v):
    v = sorted(v); n = len(v)
    return v[n // 2] if n % 2 else 0.5 * (v[n // 2 - 1] + v[n // 2])

def binom_two_sided(k, n):
    """exact two-sided sign-test p (doubling the smaller tail, capped at 1)."""
    C = math.comb
    lo = sum(C(n, i) for i in range(0, min(k, n - k) + 1)) / 2 ** n
    return min(1.0, 2 * lo)

REQ = {16: 13, 15: 12, 14: 12, 13: 11, 12: 10, 11: 10, 10: 9}   # PREREG sec.4, frozen

# ---------- null screen (control = AUTO, metric nRMSE) ----------
print()
print('=' * 78)
print('2. NULL SCREEN (flat-sea null, control cell = AUTO)')
print('=' * 78)
keep, drop = [], []
for e in EPS:
    a = seed_median('AUTO', e, 'nRMSE')
    z = nulls[e]['zero_null_tracking_nRMSE_mean']
    (keep if a < z else drop).append(e)
    print(f'  {e[:52]:54s} AUTO {a:.4f}  null {z:.4f}  {"beats" if a < z else "*** FAILS ***"}')
print(f'\nkept n={len(keep)}   dropped n={len(drop)} -> {drop}')

# how many cells beat the null per episode (context)
print('\nper-cell null-beating count over 16 episodes:')
for c in CELLS:
    n_ok = sum(1 for e in EPS if seed_median(c, e, 'nRMSE') < nulls[e]['zero_null_tracking_nRMSE_mean'])
    print(f'  {c:12s} {n_ok:2d}/16')

# ---------- contrast machinery ----------
def contrast(a, b, block, field='nRMSE', lower_is_better=True):
    """signed paired delta  a - b  (PREREG sec.4: never absolute values)."""
    d = []
    for e in block:
        va, vb = seed_median(a, e, field), seed_median(b, e, field)
        d.append((e, va - vb))
    deltas = [x[1] for x in d]
    ties = sum(1 for x in deltas if x == 0.0)
    nz = [x for x in deltas if x != 0.0]
    n = len(nz)
    # "win" = a is better than b
    wins = sum(1 for x in nz if (x < 0 if lower_is_better else x > 0))
    p = binom_two_sided(wins, n) if n else 1.0
    return dict(a=a, b=b, field=field, n=n, ties=ties, wins=wins,
                median=med(deltas) if deltas else float('nan'),
                median_nz=med(nz) if nz else float('nan'),
                p=p, req=REQ.get(n), per_ep=d)

def line(c):
    ok = (c['req'] is not None and c['wins'] >= c['req'] and
          ((c['median'] < 0) if c['field'] != 'Pearson_rho' else (c['median'] > 0)))
    return (f"{c['a']:>10s} - {c['b']:<10s} {c['field']:>16s}  "
            f"median {c['median']:+.4f}  {c['wins']:2d}/{c['n']:<2d}  p={c['p']:.4f}  "
            f"req={c['req']}  ties={c['ties']}  {'PASS' if ok else 'fail'}")

BLOCK_KEEP, BLOCK_ALL = keep, EPS

print()
print('=' * 78)
print('3. HEADLINE CONTRASTS -- nRMSE (PRIMARY metric)')
print('=' * 78)
PAIRS = [
    ('K6-KLR01', 'AUTO'),      # PRIMARY
    ('K6-KLR03', 'AUTO'),
    ('K6-KLR00', 'AUTO'),
    ('ORACLE', 'AUTO'),
    ('ORACLE-K', 'AUTO'),
    ('K6-NSCHED', 'AUTO'),
    ('ORACLE-K', 'K6-NSCHED'),  # H2: k-placement factor (1 knob)
    ('ORACLE', 'ORACLE-K'),     # H3: k trust region at oracle placement (1 knob)
    ('ORACLE', 'K6-NSCHED'),
    ('AUTO-DUP', 'AUTO'),       # leak probe
    ('K6-KLR01', 'K6-KLR03'),
    ('K6-KLR00', 'K6-KLR01'),
]
print(f'--- null-EXCLUDED block (n={len(BLOCK_KEEP)}) ---')
res_keep = {}
for a, b in PAIRS:
    c = contrast(a, b, BLOCK_KEEP)
    res_keep[(a, b)] = c
    print('  ' + line(c))
print(f'--- FULL block, no null exclusion (n={len(BLOCK_ALL)}) [co-reported per sec.4] ---')
res_all = {}
for a, b in PAIRS:
    c = contrast(a, b, BLOCK_ALL)
    res_all[(a, b)] = c
    print('  ' + line(c))

# ---------- panels ----------
print()
print('=' * 78)
print('4. PRE-REGISTERED PANELS (null-excluded block)')
print('=' * 78)
PANEL = [('init_nrmse', True), ('Pearson_rho', False), ('absVR_framewise', True),
         ('m_mean', True), ('decorr_mean', True), ('shrink_mean', True)]
for a, b in [('K6-KLR01', 'AUTO'), ('K6-KLR00', 'AUTO'), ('ORACLE', 'AUTO'),
             ('ORACLE-K', 'K6-NSCHED'), ('ORACLE', 'ORACLE-K'), ('K6-NSCHED', 'AUTO')]:
    print(f'  --- {a} - {b} ---')
    for fl, lib in PANEL:
        c = contrast(a, b, BLOCK_KEEP, field=fl, lower_is_better=lib)
        print('    ' + line(c))

# ---------- block medians (budget split) ----------
print()
print('=' * 78)
print('5. BLOCK MEDIANS (levels, not deltas)')
print('=' * 78)
for f in ['nRMSE', 'init_nrmse', 'Pearson_rho', 'm_mean', 'decorr_mean', 'shrink_mean']:
    print(f'  {f}:')
    for c in ['AUTO', 'AUTO-DUP', 'K6-KLR03', 'K6-KLR01', 'K6-KLR00',
              'K6-NSCHED', 'ORACLE-K', 'ORACLE']:
        vk = med([seed_median(c, e, f) for e in BLOCK_KEEP])
        va = med([seed_median(c, e, f) for e in BLOCK_ALL])
        print(f'    {c:12s} keep14 {vk:+.4f}   all16 {va:+.4f}')

# ---------- config folds ----------
print()
print('=' * 78)
print('6. CONFIG FOLDS (n=4, floor p=0.125 -- NO "significant" allowed, PREREG sec.4/7)')
print('=' * 78)
def cfg_of(e):
    return e.split('_seed')[0]   # episode_BF4_15comp
CFGS = sorted({cfg_of(e) for e in EPS})
for a, b in [('K6-KLR01', 'AUTO'), ('K6-KLR00', 'AUTO'), ('ORACLE', 'AUTO'),
             ('ORACLE-K', 'K6-NSCHED'), ('ORACLE', 'ORACLE-K')]:
    fold = []
    for g in CFGS:
        block = [e for e in BLOCK_KEEP if cfg_of(e) == g]
        if not block:
            fold.append((g, None)); continue
        d = med([seed_median(a, e, 'nRMSE') - seed_median(b, e, 'nRMSE') for e in block])
        fold.append((g, d))
    w = sum(1 for _, d in fold if d is not None and d < 0)
    n = sum(1 for _, d in fold if d is not None)
    print(f'  {a:>10s} - {b:<10s}  ' +
          '  '.join(f'{g.replace("episode_","")}: {d:+.4f}' for g, d in fold if d is not None) +
          f'   -> {w}/{n}')

# ---------- seed noise floor ----------
print()
print('=' * 78)
print('7. SEED NOISE FLOOR (within-cell, within-episode spread of nRMSE over 3 seeds)')
print('=' * 78)
for c in ['AUTO', 'K6-KLR01', 'K6-KLR00', 'ORACLE-K', 'ORACLE', 'K6-NSCHED']:
    rng = [max(IDX[(c, e, s)]['nRMSE'] for s in SEEDS) - min(IDX[(c, e, s)]['nRMSE'] for s in SEEDS)
           for e in BLOCK_KEEP]
    print(f'  {c:12s} median seed-range {med(rng):.4f}   max {max(rng):.4f}')
