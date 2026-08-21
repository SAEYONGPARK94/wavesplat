"""NEUTRAL audit 01 -- PREREG sec.6 integrity, re-run by hand from raw results.jsonl.

Written independently of run_rx5.py's own bookkeeping.  Reads ONLY results.jsonl.
"""
import json, os, sys, hashlib, collections

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, 'results.jsonl')

recs = [json.loads(l) for l in open(RES) if l.strip()]
print(f'records = {len(recs)}')

# ---- 0. shape -------------------------------------------------------------------
cells = sorted({r['cell'] for r in recs})
eps = sorted({r['episode'] for r in recs})
seeds = sorted({r['seed'] for r in recs})
print(f'cells  = {len(cells)}  {cells}')
print(f'eps    = {len(eps)}')
print(f'seeds  = {seeds}')

key = lambda r: (r['cell'], r['episode'], r['seed'])
cnt = collections.Counter(key(r) for r in recs)
dupes = [k for k, v in cnt.items() if v > 1]
missing = [(c, e, s) for c in cells for e in eps for s in seeds if (c, e, s) not in cnt]
print(f'duplicate (cell,ep,seed) = {len(dupes)}   missing = {len(missing)}')
if dupes: print('  DUP', dupes[:5])
if missing: print('  MISS', missing[:5])

BY = {key(r): r for r in recs}

# ---- number of tracked frames (PREREG sec.1 says 160) ---------------------------
nfr = collections.Counter(len(r.get('vr_frames') or []) for r in recs)
print(f'\nlen(vr_frames) histogram = {dict(nfr)}')

# ---- 6.1 worker leak: AUTO-DUP bit-identical to AUTO ----------------------------
print('\n=== 6.1 WORKER LEAK (AUTO-DUP vs AUTO, bit-identical) ===')
SCALARS = ['nRMSE', 'Variance_Ratio', 'Pearson_rho', 'Skill_Score', 'Directional_R2',
           'Hs_error', 'absVR_framewise', 'decorr_mean', 'shrink_mean', 'm_mean',
           'avg_RMSE_unobserved', 'N', 'init_nrmse', 'init_rmse', 'auto_selected',
           'auto_n_selected', 'ensemble_k']
bad = []
for e in eps:
    for s in seeds:
        a, d = BY[('AUTO', e, s)], BY[('AUTO-DUP', e, s)]
        for f in SCALARS:
            if a.get(f) != d.get(f):
                bad.append((e, s, f, a.get(f), d.get(f)))
        # also the full frame vectors
        for f in ('vr_frames', 'rho_frames', 'kx', 'ky', 'w'):
            if a.get(f) != d.get(f):
                bad.append((e, s, f, 'vec-mismatch', ''))
print(f'  mismatching (ep,seed,field) = {len(bad)} / {len(eps)*len(seeds)} pairs')
for b in bad[:10]: print('   ', b)
print('  6.1 ->', 'PASS' if not bad else 'FAIL')

# ---- 6.2 Phase-1 invariance across ALL 14 cells ---------------------------------
print('\n=== 6.2 PHASE-1 INVARIANCE (all 14 cells, same (ep,seed)) ===')
P1 = ['init_nrmse', 'auto_selected', 'N', 'ensemble_k', 'auto_n_selected', 'init_rmse']
viol = collections.defaultdict(list)   # field -> list of (ep,seed,cell,val,ref)
for e in eps:
    for s in seeds:
        ref = BY[('AUTO', e, s)]
        for c in cells:
            r = BY[(c, e, s)]
            for f in P1:
                if r.get(f) != ref.get(f):
                    viol[f].append((e, s, c, r.get(f), ref.get(f)))
for f in P1:
    v = viol[f]
    print(f'  {f:16s} mismatches = {len(v):4d} / {len(eps)*len(seeds)*(len(cells)-1)}'
          f'   {"PASS" if not v else "FAIL"}')
    for x in v[:4]: print('      ', x)
percell = collections.Counter(x[2] for f in P1 for x in viol[f])
if percell: print('  per-cell violation counts:', dict(percell))
print('  6.2 ->', 'PASS' if not any(viol.values()) else 'FAIL')

# also: kx/ky/w at initialisation are not recorded, but N and init_nrmse are the
# pre-registered witnesses.  Report initial-condition identity strength:
print(f'  (witness strength: {len(P1)} Phase-1 fields x {len(eps)*len(seeds)} (ep,seed)'
      f' x {len(cells)-1} non-AUTO cells)')

# ---- 6.3 one knob per cell ------------------------------------------------------
print('\n=== 6.3 ONE KNOB PER CELL (from per-run cfg dumps) ===')
def cfgof(cell):
    cs = [json.dumps(BY[(cell, e, s)]['cfg'], sort_keys=True) for e in eps for s in seeds]
    assert len(set(cs)) == 1, f'{cell}: cfg varies across runs!'
    return json.loads(cs[0])

CFG = {c: cfgof(c) for c in cells}
def diff(a, b):
    ka, kb = CFG[a], CFG[b]
    out = {}
    for k in set(ka) | set(kb):
        if ka.get(k, '<absent>') != kb.get(k, '<absent>'):
            out[k] = (ka.get(k, '<absent>'), kb.get(k, '<absent>'))
    return out

PAIRS = [('W-5e3', 'W-5e4', 1), ('W-5e3', 'W-MATCH', 1), ('W-5e3', 'W-5e6', 1),
         ('AUTO', 'W-5e3', 2), ('AUTO', 'PHI-5e3', 2),
         ('AUTO', 'E-2', 1), ('AUTO', 'E-5', 1), ('AUTO', 'E-10', 1),
         ('AUTO', 'E-20', 1), ('AUTO', 'E-40', 1), ('AUTO', 'E-80', 1),
         ('AUTO', 'AUTO-NOTRACK', 1), ('AUTO', 'AUTO-DUP', 0)]
ok = True
for a, b, want in PAIRS:
    d = diff(a, b)
    good = len(d) == want
    ok &= good
    print(f'  {a:6s} vs {b:12s} n={len(d)} (want {want}) {"ok" if good else "MISMATCH"}  {d}')
print('  6.3 ->', 'PASS' if ok else 'FAIL')

# W-MATCH lr must equal the frozen constant
wm = CFG['W-MATCH'].get('tracking_lr')
ws = json.load(open(os.path.join(HERE, 'wscale.json')))
print(f"  W-MATCH tracking_lr = {wm!r}   wscale.json RX5_LR_W_MATCHED = "
      f"{ws['RX5_LR_W_MATCHED']!r}   identical={wm == ws['RX5_LR_W_MATCHED']}")

# relative steps
print(f"  relative step w   = {wm/ws['median_abs_w']*100:.4f} %   "
      f"(shipped 0.005 -> {0.005/ws['median_abs_w']*100:.1f} %)")
print(f"  relative step phi = {0.005/(2*3.141592653589793)*100:.4f} %")
