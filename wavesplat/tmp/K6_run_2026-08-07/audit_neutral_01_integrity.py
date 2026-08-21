"""NEUTRAL audit of K6 -- integrity checks (PREREG sec.6), written from scratch.
Reads results.jsonl / nulls_k6.json only.  Touches nothing.
"""
import hashlib
import json
import os
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
R = [json.loads(l) for l in open(os.path.join(HERE, 'results.jsonl'))]
print(f'records: {len(R)}')

cells = sorted({r['cell'] for r in R})
eps = sorted({r['episode'] for r in R})
seeds = sorted({r['seed'] for r in R})
print(f'cells  ({len(cells)}): {cells}')
print(f'episodes({len(eps)})')
for e in eps:
    print('   ', e)
print(f'seeds  ({len(seeds)}): {seeds}')
print(f'expected runs = {len(cells)}x{len(eps)}x{len(seeds)} = '
      f'{len(cells)*len(eps)*len(seeds)}')

# completeness / duplicates
key = defaultdict(list)
for r in R:
    key[(r['cell'], r['episode'], r['seed'])].append(r)
dups = {k: len(v) for k, v in key.items() if len(v) > 1}
missing = [(c, e, s) for c in cells for e in eps for s in seeds
           if (c, e, s) not in key]
print(f'duplicate (cell,ep,seed) keys: {len(dups)}   missing: {len(missing)}')
if missing:
    print('  MISSING:', missing[:20])

# ---------------- check 3: worker leak, AUTO vs AUTO-DUP bit identity ----------
print('\n=== CHECK 3: AUTO vs AUTO-DUP bit identity ===')
FLOATKEYS = ['nRMSE', 'Variance_Ratio', 'Pearson_rho', 'Skill_Score',
             'Directional_R2', 'Hs_error', 'absVR_framewise', 'decorr_mean',
             'shrink_mean', 'm_mean', 'avg_RMSE_unobserved', 'init_nrmse',
             'init_rmse']
nbit, nne = 0, 0
worst = ('', 0.0, '')
for e in eps:
    for s in seeds:
        a = key[('AUTO', e, s)][0]
        d = key[('AUTO-DUP', e, s)][0]
        same = True
        for k in FLOATKEYS:
            if a[k] != d[k]:
                same = False
                dd = abs(a[k] - d[k])
                if dd > worst[1]:
                    worst = (k, dd, f'{e[8:-3]}|{s}')
        # vector fields too
        for k in ('kx', 'ky', 'w'):
            if a[k] != d[k]:
                same = False
        for k in ('vr_frames', 'rho_frames'):
            if a[k] != d[k]:
                same = False
        if a['spec'] != d['spec']:
            same = False
        if same:
            nbit += 1
        else:
            nne += 1
print(f'bit-identical (cell,ep,seed) pairs: {nbit}/{nbit+nne}   differing: {nne}')
if nne:
    print(f'  worst float mismatch: {worst}')
    for e in eps:
        for s in seeds:
            a = key[('AUTO', e, s)][0]
            d = key[('AUTO-DUP', e, s)][0]
            if a['nRMSE'] != d['nRMSE']:
                print(f'   {e[8:-3]:34s} seed{s} AUTO={a["nRMSE"]:.10f} '
                      f'DUP={d["nRMSE"]:.10f} d={d["nRMSE"]-a["nRMSE"]:+.3e}')

# ---------------- check 2: knobs per cell, from the RECORDED cfg --------------
print('\n=== CHECK 2: knobs per cell, from the recorded cfg dumps ===')


def cfgsig(r):
    return json.dumps(r['cfg'], sort_keys=True)


# cfg must be constant within a cell (over episodes it may differ via n_gaussians=[N_gt])
for c in cells:
    sigs = {}
    for e in eps:
        for s in seeds:
            sigs.setdefault(cfgsig(key[(c, e, s)][0]), []).append((e, s))
    print(f'  {c:11s} distinct cfg dumps = {len(sigs)}')

# canonical cfg per (cell, episode) -- must be seed-invariant
canon = {}
for c in cells:
    for e in eps:
        ss = {cfgsig(key[(c, e, s)][0]) for s in seeds}
        assert len(ss) == 1, (c, e, ss)
        canon[(c, e)] = key[(c, e, seeds[0])][0]['cfg']


def diff(c1, c2, e):
    a, b = canon[(c1, e)], canon[(c2, e)]
    out = {}
    for k in set(a) | set(b):
        if a.get(k, '<absent>') != b.get(k, '<absent>'):
            out[k] = (a.get(k, '<absent>'), b.get(k, '<absent>'))
    return out


print('\n--- core contrast 1: ORACLE-K vs K6-NSCHED (must be init_method ONLY) ---')
for e in eps[:3]:
    print(f'  {e[8:-3]:34s} {diff("ORACLE-K", "K6-NSCHED", e)}')
nk = {len(diff('ORACLE-K', 'K6-NSCHED', e)) for e in eps}
kk = set()
for e in eps:
    kk |= set(diff('ORACLE-K', 'K6-NSCHED', e))
print(f'  => n_knobs over all 16 eps = {nk}   keys = {sorted(kk)}')

print('\n--- core contrast 2: ORACLE vs ORACLE-K (must be init_k_lr_scale ONLY) ---')
for e in eps[:3]:
    print(f'  {e[8:-3]:34s} {diff("ORACLE", "ORACLE-K", e)}')
nk = {len(diff('ORACLE', 'ORACLE-K', e)) for e in eps}
kk = set()
for e in eps:
    kk |= set(diff('ORACLE', 'ORACLE-K', e))
print(f'  => n_knobs over all 16 eps = {nk}   keys = {sorted(kk)}')

print('\n--- core contrast 3: AUTO vs K6-KLR* (must be init_k_lr_scale ONLY) ---')
for c in ('K6-KLR03', 'K6-KLR01', 'K6-KLR00'):
    nk = set()
    kk = set()
    for e in eps:
        d = diff('AUTO', c, e)
        nk.add(len(d))
        kk |= set(d)
    print(f'  AUTO vs {c:9s} n_knobs={nk}  keys={sorted(kk)}  '
          f'ex={diff("AUTO", c, eps[0])}')

print('\n--- (descriptive) AUTO vs the ceiling cells: NOT one knob ---')
for c in ('ORACLE-K', 'ORACLE', 'K6-NSCHED', 'AUTO-DUP'):
    d = diff('AUTO', c, eps[0])
    print(f'  AUTO vs {c:10s} n_knobs={len(d)}  keys={sorted(d)}')

# ---------------- check 4: AUTO / AUTO-DUP Phase-1 invariance -----------------
print('\n=== CHECK 4: Phase-1 invariance required ONLY between AUTO and AUTO-DUP ===')
bad = [(e, s) for e in eps for s in seeds
       if key[('AUTO', e, s)][0]['init_nrmse'] != key[('AUTO-DUP', e, s)][0]['init_nrmse']]
print(f'  init_nrmse mismatches: {len(bad)}  {bad[:5]}')

# router composition
print('\n=== router composition (init_method="auto" cells) ===')
for c in cells:
    sel = defaultdict(int)
    for e in eps:
        for s in seeds:
            sel[key[(c, e, s)][0].get('auto_selected')] += 1
    print(f'  {c:11s} {dict(sel)}')

# nulls present
N = {r['episode']: r for r in json.load(open(os.path.join(HERE, 'nulls_k6.json')))}
print(f'\nnulls: {len(N)} episodes; missing for run set = '
      f'{[e for e in eps if e not in N]}')

# file hashes for the record
for f in ('PREREG.md', 'run_k6.py', 'results.jsonl', 'nulls_k6.json'):
    h = hashlib.md5(open(os.path.join(HERE, f), 'rb').read()).hexdigest()
    print(f'md5 {f:16s} {h}')
