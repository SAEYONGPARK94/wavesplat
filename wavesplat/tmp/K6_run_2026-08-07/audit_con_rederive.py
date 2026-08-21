#!/usr/bin/env python3
"""CON audit: independent re-derivation of every K6 headline from results.jsonl.
Written from scratch (no import of run_k6 or of any scorer)."""
import json, os, math, itertools
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(HERE, 'results.jsonl')
NUL = os.path.join(HERE, 'nulls_k6.json')

rows = [json.loads(l) for l in open(R) if l.strip()]
print(f"rows = {len(rows)}")

cells = sorted({r['cell'] for r in rows})
eps = sorted({r['episode'] for r in rows})
seeds = sorted({r['seed'] for r in rows})
print(f"cells({len(cells)}) = {cells}")
print(f"episodes = {len(eps)}   seeds = {seeds}")

# ---- median of 3 seeds ------------------------------------------------------
def med(v):
    v = sorted(v); n = len(v)
    return v[n//2] if n % 2 else 0.5*(v[n//2-1]+v[n//2])

M = defaultdict(dict)          # M[cell][ep][metric] = seed-median
raw = defaultdict(lambda: defaultdict(list))
METRICS = ['nRMSE','init_nrmse','Pearson_rho','absVR_framewise',
           'm_mean','decorr_mean','shrink_mean','avg_RMSE_unobserved']
for r in rows:
    raw[(r['cell'], r['episode'])]['_rows'].append(r)
for (c, e), d in raw.items():
    out = {}
    for m in METRICS:
        vals = [x[m] for x in d['_rows'] if x.get(m) is not None]
        if len(vals) == 3:
            out[m] = med(vals)
        elif vals:
            out[m] = med(vals); out['_nseed_'+m] = len(vals)
    out['_nseed'] = len(d['_rows'])
    M[c][e] = out

for c in cells:
    ns = {M[c][e]['_nseed'] for e in M[c]}
    print(f"  {c:12s} episodes={len(M[c])}  seeds/ep={sorted(ns)}")

# ---- nulls ------------------------------------------------------------------
nulls = {n['episode']: n for n in json.load(open(NUL))}
auto_beats_null = {}
for e in eps:
    a = M['AUTO'][e]['nRMSE']; z = nulls[e]['zero_null_tracking_nRMSE_mean']
    auto_beats_null[e] = a < z
excl = [e for e in eps if not auto_beats_null[e]]
print(f"\nAUTO fails the flat-sea null on {len(excl)}: {excl}")
KEEP = [e for e in eps if auto_beats_null[e]]

# ---- sign test --------------------------------------------------------------
def C(n,k):
    return math.comb(n,k)
def signtest(deltas, tol=0.0):
    """two-sided sign test; ties (|d|<=tol) dropped."""
    d = [x for x in deltas if abs(x) > tol]
    ties = len(deltas)-len(d)
    n = len(d)
    if n == 0: return dict(n=0, wins=0, p=1.0, ties=ties, med=float('nan'))
    wins = sum(1 for x in d if x < 0)          # negative delta = improvement
    k = max(wins, n-wins)
    p = 2*sum(C(n,i) for i in range(k, n+1))/2**n
    return dict(n=n, wins=wins, p=min(1.0,p), ties=ties, med=med(deltas))

def contrast(a, b, block, metric='nRMSE', tol=0.0):
    ds = [M[a][e][metric]-M[b][e][metric] for e in block]
    r = signtest(ds, tol)
    r['a'], r['b'], r['metric'], r['block_n'] = a, b, metric, len(block)
    return r

def show(r, label=''):
    print(f"  {label:34s} median D = {r['med']:+.4f}  {r['wins']}/{r['n']}"
          f"  p={r['p']:.5g}  ties={r['ties']}  (block {r['block_n']})")

print("\n" + "="*78)
print("A.  PRIMARY + all K6-KLR* vs AUTO   (nRMSE, lower better)")
print("="*78)
for blk, nm in ((KEEP, f'null-excluded n={len(KEEP)}'), (eps, f'ALL n={len(eps)}')):
    print(f"-- block: {nm}")
    for c in ['K6-KLR03','K6-KLR01','K6-KLR00']:
        show(contrast(c,'AUTO',blk), f'{c} - AUTO')

print("\n" + "="*78)
print("B.  headline contrasts")
print("="*78)
for blk, nm in ((KEEP, f'null-excluded n={len(KEEP)}'), (eps, f'ALL n={len(eps)}')):
    print(f"-- block: {nm}")
    show(contrast('ORACLE','ORACLE-K',blk),  'ORACLE - ORACLE-K  (k freeze)')
    show(contrast('ORACLE-K','K6-NSCHED',blk),'ORACLE-K - K6-NSCHED (placement)')
    show(contrast('ORACLE','AUTO',blk),       'ORACLE - AUTO')
    show(contrast('ORACLE-K','AUTO',blk),     'ORACLE-K - AUTO')
    show(contrast('K6-NSCHED','AUTO',blk),    'K6-NSCHED - AUTO  (harm?)')
    show(contrast('AUTO-DUP','AUTO',blk),     'AUTO-DUP - AUTO  (leak probe)')

# ---- AUTO-DUP bit identity --------------------------------------------------
print("\n" + "="*78); print("C.  AUTO-DUP bit-identity (per run, not per median)"); print("="*78)
byjob = {}
for r in rows:
    byjob[(r['cell'], r['episode'], r['seed'])] = r
nd = 0; tot = 0
for e in eps:
    for s in seeds:
        a = byjob.get(('AUTO', e, s)); b = byjob.get(('AUTO-DUP', e, s))
        if a and b:
            tot += 1
            if a['nRMSE'] != b['nRMSE'] or a['init_nrmse'] != b['init_nrmse']:
                nd += 1
print(f"  compared {tot} (ep,seed) pairs; NOT bit-identical: {nd}")

# ---- init_nrmse: is K6-NSCHED just under-converged? -------------------------
print("\n" + "="*78); print("D.  init_nrmse (end of Phase 1) by cell -- convergence check"); print("="*78)
print(f"  {'cell':12s} {'med init_nrmse':>15s} {'med nRMSE':>11s} {'med m_mean':>11s}"
      f" {'med decorr':>11s} {'med shrink':>11s}")
for c in ['AUTO','AUTO-DUP','K6-KLR03','K6-KLR01','K6-KLR00',
          'K6-NSCHED','ORACLE-K','ORACLE']:
    f = lambda m: med([M[c][e][m] for e in eps])
    print(f"  {c:12s} {f('init_nrmse'):15.4f} {f('nRMSE'):11.4f} {f('m_mean'):11.4f}"
          f" {f('decorr_mean'):11.4f} {f('shrink_mean'):11.4f}")
zn = med([nulls[e]['zero_null_tracking_nRMSE_mean'] for e in eps])
zi = med([nulls[e]['zero_null_init_nrmse'] for e in eps])
print(f"  {'FLAT NULL':12s} {zi:15.4f} {zn:11.4f}")

# ---- per-cell null margin ---------------------------------------------------
print("\n" + "="*78); print("E.  how many of 16 episodes beat the flat-sea null, per cell"); print("="*78)
for c in ['AUTO','K6-KLR03','K6-KLR01','K6-KLR00','K6-NSCHED','ORACLE-K','ORACLE']:
    w = sum(1 for e in eps if M[c][e]['nRMSE'] < nulls[e]['zero_null_tracking_nRMSE_mean'])
    wi = sum(1 for e in eps if M[c][e]['init_nrmse'] < nulls[e]['zero_null_init_nrmse'])
    print(f"  {c:12s} nRMSE beats null {w:2d}/16   init_nrmse beats null {wi:2d}/16")

# ---- routing ----------------------------------------------------------------
print("\n" + "="*78); print("F.  router selection per cell (run level)"); print("="*78)
sel = defaultdict(lambda: defaultdict(int))
for r in rows:
    sel[r['cell']][r.get('auto_selected')] += 1
for c in cells:
    print(f"  {c:12s} {dict(sel[c])}")

# ---- panel metrics on the headline contrasts --------------------------------
print("\n" + "="*78); print("G.  panel metrics on ORACLE - ORACLE-K and ORACLE-K - K6-NSCHED"); print("="*78)
for a,b in [('ORACLE','ORACLE-K'), ('ORACLE-K','K6-NSCHED'), ('K6-KLR01','AUTO')]:
    print(f"-- {a} - {b}  (null-excluded n={len(KEEP)})")
    for m in ['nRMSE','init_nrmse','absVR_framewise','m_mean','decorr_mean','shrink_mean']:
        r = contrast(a,b,KEEP,m); show(r, m)
    # Pearson_rho: larger is better -> flip sign so 'wins' still means improvement
    ds = [-(M[a][e]['Pearson_rho']-M[b][e]['Pearson_rho']) for e in KEEP]
    r = signtest(ds); r['block_n']=len(KEEP); show(r, 'Pearson_rho (sign-flipped)')
