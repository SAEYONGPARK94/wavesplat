"""CON audit — independent re-derivation of every Rx-7 headline from results.jsonl.

Written from scratch: does NOT import analyze_rx7.py.  Read-only.
"""
import json, math, os, itertools
import statistics as st
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
NEED = {16: 13, 15: 12, 14: 12, 13: 11, 12: 10}
CELLS = ['AUTO', 'AUTO-DUP', 'AUTO-LR10', 'AUTO-LR100', 'AUTO-LR1000', 'AUTO-NOTRACK']
CAND = ['AUTO-LR10', 'AUTO-LR100', 'AUTO-LR1000']
PANEL = ['nRMSE', 'avg_RMSE_observed', 'avg_RMSE_unobserved', 'Pearson_rho',
         'absVR_framewise', 'm_mean', 'decorr_mean', 'shrink_mean']
BIG_IS_BAD = {k: True for k in PANEL}
BIG_IS_BAD['Pearson_rho'] = False


def sgn_p(k, n):
    if n == 0:
        return 1.0
    lo = min(k, n - k)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(lo + 1)) / 2.0 ** n)


rows = [json.loads(l) for l in open(f'{HERE}/results.jsonl') if l.strip()]
rows = [r for r in rows if 'error' not in r]
NULL = {n['episode']: n['zero_null_tracking_nRMSE_mean']
        for n in json.load(open(f'{HERE}/nulls_rx7.json'))}
HS = {n['episode']: n['Hs_true'] for n in json.load(open(f'{HERE}/nulls_rx7.json'))}

D = defaultdict(list)
for r in rows:
    D[(r['cell'], r['episode'], r['seed'])].append(r)
EPS = sorted({r['episode'] for r in rows})
SEEDS = sorted({r['seed'] for r in rows})


def val(c, e, s, k):
    v = D[(c, e, s)]
    return v[0][k] if v else None


def med(c, e, k='nRMSE'):
    v = [val(c, e, s, k) for s in SEEDS]
    v = [x for x in v if x is not None and x == x]
    return st.median(v) if v else None


KEPT = [e for e in EPS if med('AUTO', e) < NULL[e]]
EXCL = [e for e in EPS if e not in KEPT]


def paired(a, b, key='nRMSE', blk=None):
    blk = blk if blk is not None else KEPT
    raw = [med(a, e, key) - med(b, e, key) for e in blk]
    d = [x for x in raw if x != 0]
    wins = sum(1 for x in d if (x < 0) == BIG_IS_BAD[key])
    m = st.median(d) if d else None
    return m, wins, len(d), sgn_p(wins, len(d)), raw


print('=' * 78)
print('1. RE-DERIVATION vs analysis.txt')
print('=' * 78)
print(f'rows ok = {len(rows)}; eps={len(EPS)} seeds={SEEDS}')
print(f'null-excluded ({len(EXCL)}): {[e[8:-25] for e in EXCL]}   n={len(KEPT)} need={NEED[len(KEPT)]}')
print(f'\n{"cell":14s} {"nRMSE medOfMed":>15s} {"obsRMSE":>10s} {"unobsRMSE":>10s} {"nullpass":>9s}')
for c in CELLS:
    lv = st.median([med(c, e) for e in KEPT])
    lo = st.median([med(c, e, 'avg_RMSE_observed') for e in KEPT])
    lu = st.median([med(c, e, 'avg_RMSE_unobserved') for e in KEPT])
    npass = sum(1 for e in KEPT if med(c, e) < NULL[e])
    print(f'{c:14s} {lv:15.5f} {lo:10.5f} {lu:10.5f} {npass:6d}/{len(KEPT)}')

print('\n--- PRIMARY-A (candidate - AUTO-NOTRACK, nRMSE) ---')
resA = {}
for c in CAND:
    m, w, n, p, raw = paired(c, 'AUTO-NOTRACK')
    resA[c] = (m, w, n, p, raw)
    print(f'  {c:13s} medD={m:+.5f} win {w:2d}/{n} p={p:.4f}')

print('\n--- PRIMARY-B (AUTO - AUTO-NOTRACK) ---')
for k in ('avg_RMSE_observed', 'avg_RMSE_unobserved'):
    m, w, n, p, _ = paired('AUTO', 'AUTO-NOTRACK', k)
    print(f'  {k:22s} medD={m:+.5f} improve {w:2d}/{n} p={p:.4f}')

print('\n' + '=' * 78)
print('2. ATTACK: is the OBSERVED-region effect scale-comparable? (units)')
print('=' * 78)
# avg_RMSE_observed/unobserved are in METRES; nRMSE is RMSE/Hs (dimensionless).
for e in KEPT[:0]:
    pass
print('Hs in block:', sorted(set(HS[e] for e in KEPT)))
print('\nNormalised by Hs (per-episode) so obs/unobs are on the nRMSE scale:')
for c in CELLS:
    lo = st.median([med(c, e, 'avg_RMSE_observed') / HS[e] for e in KEPT])
    lu = st.median([med(c, e, 'avg_RMSE_unobserved') / HS[e] for e in KEPT])
    lv = st.median([med(c, e) for e in KEPT])
    print(f'  {c:14s} nRMSE={lv:.4f}  obs/Hs={lo:.4f}  unobs/Hs={lu:.4f}')
# paired deltas normalised
for k, lbl in (('avg_RMSE_observed', 'obs'), ('avg_RMSE_unobserved', 'unobs')):
    raw = [(med('AUTO', e, k) - med('AUTO-NOTRACK', e, k)) / HS[e] for e in KEPT]
    d = [x for x in raw if x != 0]
    w = sum(1 for x in d if x < 0)
    print(f'  paired D({lbl})/Hs  med={st.median(d):+.5f}  improve {w}/{len(d)} p={sgn_p(w,len(d)):.4f}')

print('\n' + '=' * 78)
print('3. ATTACK: PRIMARY-A LR1000 fragility')
print('=' * 78)
m, w, n, p, raw = resA['AUTO-LR1000']
print('per-episode paired deltas (LR1000 - NOTRACK), sorted:')
pairs = sorted(zip(raw, [e[8:-25] for e in KEPT]))
for v, e in pairs:
    print(f'   {v:+.6f}  {e}')
print(f'\n  sum(neg)={sum(x for x in raw if x<0):+.6f}  sum(pos)={sum(x for x in raw if x>0):+.6f}')
print(f'  mean={st.mean(raw):+.6f}  median={st.median(raw):+.6f}')
print('\nLOO (drop one episode):')
flips = 0
for i in range(len(KEPT)):
    blk = KEPT[:i] + KEPT[i + 1:]
    m2, w2, n2, p2, _ = paired('AUTO-LR1000', 'AUTO-NOTRACK', blk=blk)
    ok = (m2 < 0) and (w2 >= NEED[n2]) and (p2 <= 0.025)
    if not ok:
        flips += 1
    print(f'   drop {KEPT[i][8:-25]:22s} medD={m2:+.6f} {w2:2d}/{n2} p={p2:.4f} Holm(.025) {"PASS" if ok else "FAIL"}')
print(f'  LOO failures at Holm alpha=0.025: {flips}/{len(KEPT)}')

print('\nPer-seed (no median-of-3; each estimator seed alone):')
for s in SEEDS:
    raws = [val('AUTO-LR1000', e, s, 'nRMSE') - val('AUTO-NOTRACK', e, s, 'nRMSE') for e in KEPT]
    d = [x for x in raws if x != 0]
    ww = sum(1 for x in d if x < 0)
    print(f'   seed {s}: medD={st.median(d):+.6f} {ww:2d}/{len(d)} p={sgn_p(ww,len(d)):.4f}')

print('\nMDE (80% power, paired sign test, n=13, alpha=0.025 two-sided):')
sd = st.pstdev(raw)
print(f'  sd of paired deltas = {sd:.6f}; |medD| = {abs(m):.6f}; |mean| = {abs(st.mean(raw)):.6f}')
# t-test based MDE approximation
import statistics
n13 = len(raw)
mde_t = 3.13 * sd / math.sqrt(n13)   # (z_{.9875}+z_{.8}) = 2.24+0.84 ~ 3.08; use 3.13 conservative
print(f'  approx paired-t MDE(80%,alpha=.025) = {mde_t:.6f}  -> observed/MDE = {abs(st.mean(raw))/mde_t:.3f}')
print(f'  tracker slack (AUTO - NOTRACK) = {paired("AUTO","AUTO-NOTRACK")[0]:+.5f}')
print(f'  |medD(LR1000)| / slack = {abs(m)/abs(paired("AUTO","AUTO-NOTRACK")[0])*100:.1f}%')

print('\nlr->0 degeneracy check: how often is LR1000 BIT-IDENTICAL to NOTRACK?')
nb = ident = 0
for e in EPS:
    for s in SEEDS:
        a = val('AUTO-LR1000', e, s, 'nRMSE'); b = val('AUTO-NOTRACK', e, s, 'nRMSE')
        nb += 1
        if a == b:
            ident += 1
print(f'  identical nRMSE in {ident}/{nb} (ep,seed)')
allr = [abs(val('AUTO-LR1000', e, s, 'nRMSE') - val('AUTO-NOTRACK', e, s, 'nRMSE'))
        for e in EPS for s in SEEDS]
print(f'  |run-level diff|: median={st.median(allr):.2e} max={max(allr):.2e}')
for c in CAND + ['AUTO']:
    dd = [abs(val(c, e, s, 'nRMSE') - val('AUTO-NOTRACK', e, s, 'nRMSE')) for e in EPS for s in SEEDS]
    print(f'  {c:13s} median|run diff vs NOTRACK| = {st.median(dd):.5f}')

print('\n' + '=' * 78)
print('4. ATTACK: did the 3 null exclusions make the verdict?')
print('=' * 78)
for blk, lbl in ((KEPT, f'kept n={len(KEPT)}'), (EPS, f'ALL n={len(EPS)}')):
    print(f'-- {lbl} --')
    for c in CAND:
        m2, w2, n2, p2, _ = paired(c, 'AUTO-NOTRACK', blk=blk)
        need = NEED.get(n2, 99)
        print(f'   {c:13s} medD={m2:+.6f} {w2:2d}/{n2} p={p2:.4f} need={need} '
              f'{"THRESH-OK" if (m2<0 and w2>=need) else "thresh-fail"}')
    for k in ('avg_RMSE_observed', 'avg_RMSE_unobserved'):
        m2, w2, n2, p2, _ = paired('AUTO', 'AUTO-NOTRACK', k, blk=blk)
        print(f'   B {k:22s} medD={m2:+.5f} {w2:2d}/{n2} p={p2:.4f}')
print('\nExcluded episodes: direction of the LR1000 effect there')
for e in EXCL:
    d = med('AUTO-LR1000', e) - med('AUTO-NOTRACK', e)
    print(f'   {e[8:-25]:24s} AUTO={med("AUTO",e):.4f} null={NULL[e]:.4f} '
          f'D(LR1000-NOTRACK)={d:+.6f}')

print('\n' + '=' * 78)
print('5. MULTIPLICITY: 5 cells x 8 metrics vs AUTO-NOTRACK')
print('=' * 78)
tests = []
for c in CELLS[:-1]:
    for k in PANEL:
        m2, w2, n2, p2, _ = paired(c, 'AUTO-NOTRACK', k)
        if n2 == 0:
            continue
        tests.append((p2, c, k, m2, w2, n2))
tests.sort()
M = len(tests)
print(f'  family size M={M}')
print(f'  {"p":>9s} {"cell":13s} {"metric":22s} {"medD":>11s} {"win":>7s} {"Bonf":>6s} {"BH":>4s}')
bh_thresh = 0
for i, (p2, c, k, m2, w2, n2) in enumerate(tests, 1):
    if p2 <= 0.05 * i / M:
        bh_thresh = i
for i, (p2, c, k, m2, w2, n2) in enumerate(tests, 1):
    bonf = 'PASS' if p2 <= 0.05 / M else '-'
    bh = 'PASS' if i <= bh_thresh else '-'
    print(f'  {p2:9.5f} {c:13s} {k:22s} {m2:+11.5f} {w2:3d}/{n2:<3d} {bonf:>6s} {bh:>4s}')

print('\n  --- PRIMARY-A family only (3 candidates x nRMSE) with BH ---')
sub = sorted([(resA[c][3], c) for c in CAND])
for i, (p2, c) in enumerate(sub, 1):
    print(f'   {c:13s} p={p2:.4f} BH thr={0.05*i/3:.4f} {"PASS" if p2 <= 0.05*i/3 else "-"} '
          f'(dir {"OK" if resA[c][0]<0 else "WRONG-SIGN"})')

print('\n' + '=' * 78)
print('6. BLOCK DIFFICULTY + seed variability')
print('=' * 78)
print('per-episode AUTO / NOTRACK / null:')
for e in EPS:
    tag = '' if e in KEPT else '  <-EXCLUDED'
    print(f'  {e[8:-25]:24s} null={NULL[e]:.4f} AUTO={med("AUTO",e):.4f} '
          f'NOTRACK={med("AUTO-NOTRACK",e):.4f} init={med("AUTO",e,"init_nrmse"):.4f}{tag}')
print(f'\n  block median null = {st.median([NULL[e] for e in KEPT]):.4f}')
print('\nSeed spread per cell (across the 3 estimator seeds, per episode):')
for c in CELLS:
    sp = []
    for e in KEPT:
        v = [val(c, e, s, 'nRMSE') for s in SEEDS]
        sp.append(max(v) - min(v))
    print(f'  {c:14s} median seed range = {st.median(sp):.5f}  max = {max(sp):.5f}')
print(f'\n  compare: |medD(LR1000-NOTRACK)| = {abs(m):.5f}')

print('\nHow often does the LR1000 vs NOTRACK sign flip across seeds (per episode)?')
consist = 0
for e in KEPT:
    sg = [1 if val('AUTO-LR1000', e, s, 'nRMSE') < val('AUTO-NOTRACK', e, s, 'nRMSE') else 0
          for s in SEEDS]
    if sum(sg) in (0, 3):
        consist += 1
    print(f'   {e[8:-25]:24s} per-seed better? {sg}  ({sum(sg)}/3)')
print(f'  episodes with unanimous 3-seed sign: {consist}/{len(KEPT)}')
