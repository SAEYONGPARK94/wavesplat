#!/usr/bin/env python3
"""CON reviewer independent re-derivation of Rx-4.  Written from scratch from PREREG.md
sections 3/4/5/6.  Reads only results.jsonl / repro.jsonl / nulls_full.json.
Nothing here is imported from analyze_rx4.py or rederive_rx4.py.
"""
import json, itertools, math, os
from collections import defaultdict

D = os.path.dirname(os.path.abspath(__file__))

def load(p):
    return [json.loads(l) for l in open(os.path.join(D, p)) if l.strip()]

ROWS = load('results.jsonl')
NULLS = {r['episode']: r['zero_null_tracking_nRMSE_mean']
         for r in json.load(open(os.path.join(D, 'nulls_full.json')))}

CROWDED = [f'episode_BF{b}_{c}comp_seed{s}_uniform.h5'
           for b in (4, 5) for c in (15, 30) for s in range(4)]
ALL_EP = sorted(set(r['episode'] for r in ROWS))
SPARSE = [e for e in ALL_EP if e not in CROWDED]
CELLS = ['AUTO', 'AUTO-DUP', 'AUTO-NOTRACK', 'T-PHIONLY', 'T-WONLY', 'T-EARLY20', 'T-K4REF']

# ---------- median ----------
def med(xs):
    xs = sorted(xs)
    n = len(xs)
    if n == 0: return float('nan')
    return xs[n // 2] if n % 2 else 0.5 * (xs[n // 2 - 1] + xs[n // 2])

# ---------- seed -> episode median, per (cell, episode, metric) ----------
BY = defaultdict(list)          # (cell, ep) -> list of rows
for r in ROWS:
    BY[(r['cell'], r['episode'])].append(r)

def epmed(cell, ep, metric):
    rs = BY[(cell, ep)]
    return med([x[metric] for x in rs])

# ---------- sign test, two-sided, exact binomial ----------
def C(n, k):
    return math.comb(n, k)

def signtest(k, n):
    """two-sided exact binomial p at p0=0.5 for k successes out of n"""
    if n == 0: return float('nan')
    def pmf(i): return C(n, i) / 2.0 ** n
    obs = pmf(k)
    return min(1.0, sum(pmf(i) for i in range(n + 1) if pmf(i) <= obs * (1 + 1e-12)))

# ---------- paired delta over an episode list ----------
BETTER_NEG = {'nRMSE', 'absVR_framewise', 'm_mean', 'decorr_mean', 'shrink_mean',
              'avg_RMSE_unobserved', 'Hs_error'}
BETTER_POS = {'Pearson_rho', 'Skill_Score', 'Directional_R2', 'Variance_Ratio'}

def paired(cell, ref, eps, metric='nRMSE'):
    ds = []
    for e in eps:
        ds.append(epmed(cell, e, metric) - epmed(ref, e, metric))
    n = len(ds)
    if metric in BETTER_NEG:
        wins = sum(1 for d in ds if d < 0)          # ties counted UNFAVOURABLY (prereg 4.5)
    else:
        wins = sum(1 for d in ds if d > 0)
    return med(ds), wins, n, signtest(wins, n), ds

# ================= null exclusion (prereg 4a) =================
def excluded(eps):
    return [e for e in eps if epmed('AUTO', e, 'nRMSE') >= NULLS[e]]

EXC_C = excluded(CROWDED)
EXC_S = excluded(SPARSE)
C_KEEP = [e for e in CROWDED if e not in EXC_C]
S_KEEP = [e for e in SPARSE if e not in EXC_S]

print('=' * 100)
print('CON independent re-derivation — Rx-4')
print('=' * 100)
print(f'rows={len(ROWS)}  cells={len(set(r["cell"] for r in ROWS))}  eps={len(ALL_EP)}  '
      f'seeds={sorted(set(r["seed"] for r in ROWS))}')
print(f'block C n={len(CROWDED)}  excluded(AUTO>=null)={EXC_C}  -> keep {len(C_KEEP)}')
print(f'block S n={len(SPARSE)}  excluded={EXC_S}  -> keep {len(S_KEEP)}')
print()
print('per-episode AUTO median nRMSE vs null (block C):')
for e in CROWDED:
    a = epmed('AUTO', e, 'nRMSE'); nl = NULLS[e]
    print(f'  {e:42s} AUTO={a:.4f} null={nl:.4f} margin={nl-a:+.4f} '
          f'{"EXCLUDED" if a>=nl else ""}')
print()

# ================= headline table =================
def table(eps, label, metric='nRMSE'):
    print('-' * 100)
    print(f'{label}  (n={len(eps)})  metric={metric}')
    print(f'{"cell":14s} {"medOfMed":>9s} {"medD vs AUTO":>13s} {"w/n":>7s} {"p":>8s} '
          f'{"medD vs NOTR":>13s} {"w/n":>7s} {"p":>8s} {"medNull":>8s} {"nullfail":>8s}')
    nullmed = med([NULLS[e] for e in eps])
    out = {}
    for c in CELLS:
        lv = med([epmed(c, e, metric) for e in eps])
        dA, wA, nA, pA, _ = paired(c, 'AUTO', eps, metric)
        dN, wN, nN, pN, _ = paired(c, 'AUTO-NOTRACK', eps, metric)
        nf = 'NULL-FAIL' if (metric == 'nRMSE' and lv >= nullmed) else ''
        print(f'{c:14s} {lv:9.4f} {dA:+13.4f} {wA:3d}/{nA:<3d} {pA:8.4f} '
              f'{dN:+13.4f} {wN:3d}/{nN:<3d} {pN:8.4f} {nullmed:8.4f} {nf:>8s}')
        out[c] = dict(level=lv, dA=dA, wA=wA, nA=nA, pA=pA, dN=dN, wN=wN, nN=nN, pN=pN)
    print()
    return out

TC = table(C_KEEP, 'BLOCK C — crowded, null-excluded (PRIMARY)')
TC16 = table(CROWDED, 'BLOCK C — crowded, ALL 16 (no null exclusion)')
TS = table(S_KEEP, 'BLOCK S — sparse, null-excluded')
TS8 = table(SPARSE, 'BLOCK S — sparse, ALL 8')

# ================= recovery ratio =================
print('-' * 100)
print('recovery ratio R = medD(cell-AUTO) / medD(NOTRACK-AUTO)   [descriptive only]')
base = TC['AUTO-NOTRACK']['dA']
for c in CELLS:
    print(f'  {c:14s} R = {100*TC[c]["dA"]/base:7.1f}%   (medD {TC[c]["dA"]:+.4f} / base {base:+.4f})')
print()

# ================= n=4 config-level fold =================
print('-' * 100)
print('n=4 CONFIG-LEVEL fold (4 episode-seeds folded by median, then 4 configs)')
CFG = {'BF4_15': [e for e in C_KEEP if 'BF4_15' in e],
       'BF4_30': [e for e in C_KEEP if 'BF4_30' in e],
       'BF5_15': [e for e in C_KEEP if 'BF5_15' in e],
       'BF5_30': [e for e in C_KEEP if 'BF5_30' in e]}
for k, v in CFG.items():
    print(f'  {k}: {len(v)} eps kept')
for c in CELLS:
    ds = []
    for k, v in CFG.items():
        if not v: continue
        dc = med([epmed(c, e, 'nRMSE') for e in v])
        da = med([epmed('AUTO', e, 'nRMSE') for e in v])
        ds.append(dc - da)
    w = sum(1 for d in ds if d < 0)
    print(f'  {c:14s} medD={med(ds):+.4f}  {w}/{len(ds)}  p={signtest(w,len(ds)):.4f}   ds={[round(d,4) for d in ds]}')
print()

# ================= AUTO router distribution =================
print('-' * 100)
print('AUTO router: auto_selected / auto_n_selected / ensemble_k distribution')
cnt = defaultdict(int)
for r in ROWS:
    if r['cell'] == 'AUTO':
        cnt[r['auto_selected']] += 1
print('  AUTO cell run counts by auto_selected:', dict(cnt))
print('  per-episode (AUTO cell), seed->selected:')
for e in ALL_EP:
    rs = sorted(BY[('AUTO', e)], key=lambda x: x['seed'])
    tag = 'C' if e in CROWDED else 'S'
    print(f'   [{tag}] {e:42s} ' + ' '.join(f'{x["seed"]}:{x["auto_selected"]}/N{x["auto_n_selected"]}/k{x["ensemble_k"]}' for x in rs))
print()

# ================= integrity 6.2 Phase-1 invariance =================
print('-' * 100)
print('INTEGRITY 6.2 — Phase-1 invariance across the 7 cells')
FIELDS = ['init_nrmse', 'auto_selected', 'auto_n_selected', 'ensemble_k', 'N']
bad = 0; checked = 0
for e in ALL_EP:
    for s in (1524, 1525, 1526):
        vals = {}
        for c in CELLS:
            rs = [x for x in BY[(c, e)] if x['seed'] == s]
            if not rs: continue
            vals[c] = tuple(rs[0][f] for f in FIELDS)
        checked += 1
        if len(set(vals.values())) > 1:
            bad += 1
            print(f'  MISMATCH {e} seed{s}: {vals}')
print(f'  (episode,seed) groups checked={checked}  mismatches={bad}')
print()

# ================= integrity 6.3 AUTO-DUP bit identity =================
print('-' * 100)
print('INTEGRITY 6.3 — AUTO-DUP vs AUTO bit identity')
NUMF = ['nRMSE', 'Variance_Ratio', 'Pearson_rho', 'Skill_Score', 'Directional_R2', 'Hs_error',
        'absVR_framewise', 'decorr_mean', 'shrink_mean', 'm_mean', 'avg_RMSE_unobserved',
        'init_nrmse', 'init_rmse', 'N']
nmis = 0; ncmp = 0
for e in ALL_EP:
    for s in (1524, 1525, 1526):
        a = [x for x in BY[('AUTO', e)] if x['seed'] == s]
        d = [x for x in BY[('AUTO-DUP', e)] if x['seed'] == s]
        if not a or not d: continue
        for f in NUMF:
            ncmp += 1
            if a[0][f] != d[0][f]:
                nmis += 1
                print(f'  DIFF {e} s{s} {f}: {a[0][f]} vs {d[0][f]}')
print(f'  comparisons={ncmp}  mismatches={nmis}')
# the sign test on AUTO-DUP is a pure tie-convention artifact:
dD, wD, nD, pD, dsD = paired('AUTO-DUP', 'AUTO', C_KEEP, 'nRMSE')
print(f'  AUTO-DUP paired vs AUTO on C_KEEP: medD={dD:+.6g} wins={wD}/{nD} p={pD:.4f}  '
      f'(all deltas exactly 0: {all(d==0 for d in dsD)})')
print()

# ================= integrity 6.1 repro vs B3 =================
print('-' * 100)
print('INTEGRITY 6.1 — repro.jsonl vs B3 results.jsonl (AUTO, crowded4, seeds 1521-1523)')
try:
    B3 = load('../B3_rx3_2026-08-06/results.jsonl')
    RP = load('repro.jsonl')
    b3i = {(r['cell'], r['episode'], r['seed']): r for r in B3}
    nm = 0; nc = 0; miss = 0
    for r in RP:
        key = (r['cell'], r['episode'], r['seed'])
        if key not in b3i:
            miss += 1; print('  MISSING in B3:', key); continue
        o = b3i[key]
        for f in NUMF:
            if f in o and f in r:
                nc += 1
                if o[f] != r[f]:
                    nm += 1; print(f'  DIFF {key} {f}: {o[f]} vs {r[f]}')
    print(f'  repro rows={len(RP)} comparisons={nc} mismatches={nm} missing={miss}')
except Exception as ex:
    print('  ERROR:', ex)
print()

# ================= T-EARLY20 frame-fraction test =================
print('-' * 100)
print('T-EARLY20 — is delta explained by the 20/T frame fraction?')
nf = [len(r['vr_frames']) for r in ROWS]
print(f'  frames per run: min={min(nf)} max={max(nf)} (unique={sorted(set(nf))})')
T = max(nf)
print(f'  fraction of frames still tracked in T-EARLY20 = 20/{T} = {20/T:.4f}')
print(f'  => naive "linear damage accumulation" prediction: '
      f'EARLY20 recovers (1 - 20/{T}) = {1-20/T:.4f} of the floor')
print(f'  observed R(T-EARLY20) = {TC["T-EARLY20"]["dA"]/base:.4f}')
print()

# ================= multiple comparisons =================
print('-' * 100)
print('MULTIPLICITY — every (cell x metric) paired test on block C keep, vs AUTO')
METRICS = ['nRMSE', 'Pearson_rho', 'absVR_framewise', 'm_mean', 'decorr_mean', 'shrink_mean']
pv = []
for c in CELLS:
    if c in ('AUTO', 'AUTO-DUP'): continue
    for m in METRICS:
        d, w, n, p, _ = paired(c, 'AUTO', C_KEEP, m)
        pv.append((p, c, m, d, w, n))
pv.sort()
K = len(pv)
print(f'  family size K={K} (5 intervention cells x 6 metrics, block C only)')
print(f'  {"rank":>4s} {"p":>8s} {"BH q":>8s} {"Bonf":>8s}  cell / metric / medD / w/n')
for i, (p, c, m, d, w, n) in enumerate(pv, 1):
    print(f'  {i:4d} {p:8.4f} {min(1,p*K/i):8.4f} {min(1,p*K):8.4f}  {c:11s} {m:16s} {d:+.4f} {w}/{n}')
print()

# ================= per-episode delta dump for the key cells =================
print('-' * 100)
print('per-episode paired deltas vs AUTO (block C, all 16; * = null-excluded)')
hdr = f'{"episode":42s}' + ''.join(f'{c:>14s}' for c in ['AUTO-NOTRACK','T-PHIONLY','T-WONLY','T-EARLY20','T-K4REF'])
print(hdr)
for e in CROWDED:
    mark = '*' if e in EXC_C else ' '
    line = f'{mark}{e:41s}'
    for c in ['AUTO-NOTRACK','T-PHIONLY','T-WONLY','T-EARLY20','T-K4REF']:
        line += f'{epmed(c,e,"nRMSE")-epmed("AUTO",e,"nRMSE"):+14.4f}'
    print(line)
print()

print('-' * 100)
print('per-episode paired deltas vs AUTO-NOTRACK (block C keep)')
print(hdr)
for e in C_KEEP:
    line = f' {e:41s}'
    for c in ['AUTO-NOTRACK','T-PHIONLY','T-WONLY','T-EARLY20','T-K4REF']:
        line += f'{epmed(c,e,"nRMSE")-epmed("AUTO-NOTRACK",e,"nRMSE"):+14.4f}'
    print(line)
print()

# ================= seed-level noise vs effect size (MDE) =================
print('-' * 100)
print('seed dispersion vs effect size')
import statistics as st
for c in ['AUTO', 'T-PHIONLY', 'T-EARLY20', 'AUTO-NOTRACK']:
    sds = []
    for e in C_KEEP:
        v = [x['nRMSE'] for x in BY[(c, e)]]
        sds.append(st.pstdev(v))
    print(f'  {c:14s} median across-seed sd of nRMSE = {med(sds):.4f} '
          f'(range {min(sds):.4f}-{max(sds):.4f})')
print()
