"""CON audit of E-B.  Independent re-derivation from raw jsonl.  Reads only."""
import json
import math
import os
import statistics as st
from itertools import permutations

HERE = os.path.dirname(os.path.abspath(__file__))
L = ['AUTO-NOTRACK', 'AUTO-LR1000', 'AUTO-LR100', 'AUTO-LR10', 'AUTO']  # lr ladder asc


def sgn_p(k, n):
    if n == 0:
        return 1.0
    lo = min(k, n - k)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(lo + 1)) / 2 ** n)


def load(f):
    return [json.loads(l) for l in open(os.path.join(HERE, f)) if l.strip()]


def build(rows):
    IDX = {}
    for r in rows:
        if 'error' in r:
            continue
        IDX.setdefault((r['cell'], r['episode']), []).append(r)
    return IDX


def med(IDX, c, e, k):
    v = []
    for x in IDX.get((c, e), []):
        y = x.get(k)
        if isinstance(y, dict):
            continue
        if y is not None and y == y:
            v.append(y)
    return st.median(v) if v else None


def medrad(IDX, c, e, rk):
    v = [x['rmse_by_radius'][rk] for x in IDX.get((c, e), [])
         if rk in x.get('rmse_by_radius', {}) and
         x['rmse_by_radius'][rk] == x['rmse_by_radius'][rk]]
    return st.median(v) if v else None


def spear(a, b):
    n = len(a)
    ra = [sorted(a).index(x) for x in a]
    rb = [sorted(b).index(x) for x in b]
    ma, mb = sum(ra) / n, sum(rb) / n
    num = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    den = math.sqrt(sum((x - ma) ** 2 for x in ra) * sum((y - mb) ** 2 for y in rb))
    return num / den if den else float('nan')


def paired(IDX, blk, a, b, k, better_low=True):
    d = [med(IDX, a, e, k) - med(IDX, b, e, k) for e in blk
         if med(IDX, a, e, k) is not None and med(IDX, b, e, k) is not None]
    d = [x for x in d if x != 0]
    w = sum(1 for x in d if (x < 0) == better_low)
    return st.median(d), w, len(d), sgn_p(w, len(d))


rows1 = load('results_lane1.jsonl')
rows2 = load('results_lane2.jsonl')
I1, I2 = build(rows1), build(rows2)
EPS1 = sorted({r['episode'] for r in rows1})
EPS2 = sorted({r['episode'] for r in rows2})
NULL = {}
for nf in ('Rx7_tracker_deploy_2026-08-07/nulls_rx7.json',
           'Rx6_channel_curve_2026-08-07/nulls_rx6.json'):
    p = os.path.join(HERE, '..', nf)
    for n in json.load(open(p)):
        NULL[n['episode']] = n['zero_null_tracking_nRMSE_mean']
KEPT1 = [e for e in EPS1 if med(I1, 'AUTO', e, 'nRMSE') < NULL[e]]
EXCL1 = [e for e in EPS1 if e not in KEPT1]
KEPT2 = [e for e in EPS2 if med(I2, 'AUTO', e, 'nRMSE') < NULL[e]]

print('=' * 78)
print('A. INDEPENDENT RE-DERIVATION OF PRIMARY-1 (lane1)')
print(f'   n_kept={len(KEPT1)}  excluded={[e[8:-17] for e in EXCL1]}')
for k, lbl in (('avg_obs_loss', 'objective'), ('obs_resid_rms', 'GT-free resid'),
               ('nRMSE', 'field nRMSE'), ('avg_RMSE_observed', 'RMSE_obs(10m)'),
               ('avg_RMSE_unobserved', 'RMSE_unobs(10m)')):
    m, w, n, p = paired(I1, KEPT1, 'AUTO', 'AUTO-NOTRACK', k)
    m2, w2, n2, p2 = paired(I1, EPS1, 'AUTO', 'AUTO-NOTRACK', k)
    print(f'   {lbl:16s} kept: medD={m:+.6f} {w:2d}/{n:<2d} p={p:.2e} | '
          f'ALL16: medD={m2:+.6f} {w2:2d}/{n2:<2d} p={p2:.2e}')
print(f'   [floor] exact two-sided sign-test minimum at n=14 is p={sgn_p(14,14):.2e}; '
      f'at n=16 p={sgn_p(16,16):.2e}')

print('\nB. IS PRIMARY-1 TAUTOLOGICAL?  run-level and ladder monotonicity')
# run-level (ep,seed) pairs
for k in ('avg_obs_loss', 'obs_resid_rms', 'nRMSE'):
    tot = down = 0
    for e in EPS1:
        for s in (1539, 1540, 1541):
            a = [x for x in I1.get(('AUTO', e), []) if x['seed'] == s]
            b = [x for x in I1.get(('AUTO-NOTRACK', e), []) if x['seed'] == s]
            if a and b:
                tot += 1
                down += (a[0][k] < b[0][k])
    print(f'   run-level AUTO<NOTRACK  {k:16s} {down}/{tot}')
# monotonicity along the 5-rung lr ladder
print('   strict monotone decrease along lr ladder (NOTRACK->AUTO), per episode:')
for k in ('avg_obs_loss', 'obs_resid_rms', 'avg_RMSE_observed',
          'avg_RMSE_unobserved', 'nRMSE'):
    mono = 0
    for e in EPS1:
        v = [med(I1, c, e, k) for c in L]
        if all(v[i] > v[i + 1] for i in range(4)):
            mono += 1
    inc = 0
    for e in EPS1:
        v = [med(I1, c, e, k) for c in L]
        if all(v[i] < v[i + 1] for i in range(4)):
            inc += 1
    print(f'     {k:22s} strictly-DOWN {mono:2d}/16   strictly-UP {inc:2d}/16')
# any adjacent-rung violation at run level for the objective?
viol = tot = 0
for e in EPS1:
    for s in (1539, 1540, 1541):
        v = []
        for c in L:
            r = [x for x in I1.get((c, e), []) if x['seed'] == s]
            v.append(r[0]['avg_obs_loss'] if r else None)
        if None in v:
            continue
        for i in range(4):
            tot += 1
            viol += (v[i + 1] >= v[i])
print(f'   objective adjacent-rung NON-decreases (more lr did not lower it): '
      f'{viol}/{tot} run-level comparisons')

print('\nC. PRIMARY-2: is Spearman=1.0 automatic?')
rh_obs, rh_nr, rh_unobs = [], [], []
for e in KEPT1:
    a = [med(I1, c, e, 'obs_resid_rms') for c in L]
    rh_obs.append(spear(a, [med(I1, c, e, 'avg_RMSE_observed') for c in L]))
    rh_nr.append(spear(a, [med(I1, c, e, 'nRMSE') for c in L]))
    rh_unobs.append(spear(a, [med(I1, c, e, 'avg_RMSE_unobserved') for c in L]))
print(f'   rho(obs_resid, RMSE_observed)   median={st.median(rh_obs):+.4f}  '
      f'>=0.9: {sum(1 for r in rh_obs if r >= .9)}/{len(rh_obs)}')
print(f'   rho(obs_resid, nRMSE)           median={st.median(rh_nr):+.4f}  '
      f'<=-0.9: {sum(1 for r in rh_nr if r <= -.9)}/{len(rh_nr)}')
print(f'   rho(obs_resid, RMSE_unobserved) median={st.median(rh_unobs):+.4f}  '
      f'<=-0.9: {sum(1 for r in rh_unobs if r <= -.9)}/{len(rh_unobs)}')
# what a permutation null gives
cnt = sum(1 for p in permutations(range(5)) if spear(list(range(5)), list(p)) >= 0.9)
print(f'   permutation null for n=5: P(rho>=0.9) = {cnt}/120 = {cnt/120:.4f}')
print('   -> but the relevant null is "both series monotone in lr", see B above.')
# harder surrogate test 1: across seeds within (cell, episode) -- the multi-restart use case
for c in ('AUTO', 'AUTO-LR10'):
    hit_o = hit_n = tot = 0
    for e in EPS1:
        rr = sorted(I1.get((c, e), []), key=lambda x: x['seed'])
        if len(rr) < 3:
            continue
        tot += 1
        pick = min(rr, key=lambda x: x['obs_resid_rms'])
        hit_o += (pick['avg_RMSE_observed'] == min(x['avg_RMSE_observed'] for x in rr))
        hit_n += (pick['nRMSE'] == min(x['nRMSE'] for x in rr))
    print(f'   [restart selection, cell={c}] argmin obs_resid over 3 seeds also argmin '
          f'RMSE_observed {hit_o}/{tot} | argmin nRMSE {hit_n}/{tot}  (chance 1/3)')
# harder surrogate test 2: across episodes at fixed cell
for c in ('AUTO', 'AUTO-NOTRACK'):
    a = [med(I1, c, e, 'obs_resid_rms') for e in KEPT1]
    b = [med(I1, c, e, 'avg_RMSE_observed') for e in KEPT1]
    d = [med(I1, c, e, 'nRMSE') for e in KEPT1]
    print(f'   [across 14 episodes, cell={c}] rho(resid,RMSE_obs)={spear(a,b):+.3f}  '
          f'rho(resid,nRMSE)={spear(a,d):+.3f}')
# pooled across all 5 cells x 14 eps
pa, pb, pd = [], [], []
for c in L:
    for e in KEPT1:
        pa.append(med(I1, c, e, 'obs_resid_rms'))
        pb.append(med(I1, c, e, 'avg_RMSE_observed'))
        pd.append(med(I1, c, e, 'nRMSE'))
print(f'   [pooled 5x14=70] rho(resid,RMSE_obs)={spear(pa,pb):+.3f}  '
      f'rho(resid,nRMSE)={spear(pa,pd):+.3f}')

print('\nD. CROSSOVER: observed AND unobserved radius profiles (analyzer showed only obs)')
print(f'   {"r":>4s} {"medD_obs":>10s} {"win":>7s} {"p":>9s} | {"medD_unobs":>11s} '
      f'{"win":>7s} {"p":>9s}')
for rk in ('5', '10', '20', '30', '50'):
    out = []
    for pref in ('', 'u'):
        d = [medrad(I1, 'AUTO', e, pref + rk) - medrad(I1, 'AUTO-NOTRACK', e, pref + rk)
             for e in KEPT1]
        d = [x for x in d if x == x and x != 0]
        w = sum(1 for x in d if x < 0)
        out.append((st.median(d), w, len(d), sgn_p(w, len(d))))
    (m1, w1, n1, p1), (m2, w2, n2, p2) = out
    print(f'   {rk:>4s} {m1:+10.5f} {w1:3d}/{n1:<3d} {p1:9.5f} | {m2:+11.5f} '
          f'{w2:3d}/{n2:<3d} {p2:9.5f}')
m, w, n, p = paired(I1, KEPT1, 'AUTO', 'AUTO-NOTRACK', 'nRMSE')
print(f'   full-grid nRMSE (DIMENSIONLESS, not metres like the rows above): '
      f'{m:+.5f} {w}/{n} p={p:.5f}')
print('   levels [m], block median over 14 kept episodes:')
print(f'   {"cell":14s}' + ''.join(f'{"r"+r:>9s}' for r in ('5', '10', '20', '30', '50'))
      + ''.join(f'{"u"+r:>9s}' for r in ('5', '10', '20', '30', '50')))
for c in L[::-1]:
    v = [st.median([medrad(I1, c, e, rk) for e in KEPT1])
         for rk in ('5', '10', '20', '30', '50', 'u5', 'u10', 'u20', 'u30', 'u50')]
    print(f'   {c:14s}' + ''.join(f'{x:9.4f}' for x in v))

print('\nE. NULL EXCLUSION: what the 2 dropped episodes would have contributed')
for e in EXCL1:
    print(f'   {e[8:-17]:22s} AUTO_nRMSE={med(I1,"AUTO",e,"nRMSE"):.4f} '
          f'null={NULL[e]:.4f}  D(AUTO-NOTRACK): '
          f'nRMSE={med(I1,"AUTO",e,"nRMSE")-med(I1,"AUTO-NOTRACK",e,"nRMSE"):+.4f} '
          f'obj={med(I1,"AUTO",e,"avg_obs_loss")-med(I1,"AUTO-NOTRACK",e,"avg_obs_loss"):+.5f} '
          f'resid={med(I1,"AUTO",e,"obs_resid_rms")-med(I1,"AUTO-NOTRACK",e,"obs_resid_rms"):+.5f}')
print('   NOTE the null filter conditions on the TREATMENT arm (AUTO) beating the '
      'flat-sea null.')

print('\nF. MULTIPLE COMPARISON (BH over every p printed in analysis.txt)')
fam = [('L1 obj', 1.22e-4), ('L1 resid', 1.22e-4), ('L1 nRMSE', 1.22e-4),
       ('L1 r5', 1.22e-4), ('L1 r10', 1.22e-4), ('L1 r20', 1.83e-3),
       ('L1 r30', 1.83e-3), ('L1 r50', 0.17957), ('L1 LR10 keep', 0.0018),
       ('L1 LR100 keep', 1.0), ('L1 LR1000 keep', 0.0574),
       ('L1 LR10 all', 0.0042), ('L1 LR100 all', 0.8036), ('L1 LR1000 all', 0.0768),
       ('L2 nRMSE', 3.4e-5), ('L2 obs', 1e-9), ('L2 unobs', 5e-6), ('L2 obj', 1e-9)]
fam.sort(key=lambda x: x[1])
m = len(fam)
print(f'   family size m={m}; Bonferroni alpha=0.05/{m}={0.05/m:.2e}')
for i, (nm, p) in enumerate(fam, 1):
    print(f'   {i:2d} {nm:16s} p={p:9.2e}  BH_crit={0.05*i/m:.5f}  '
          f'{"survive" if p <= 0.05*i/m else "DIE"}  '
          f'Bonf={"survive" if p <= 0.05/m else "DIE"}')

print('\nG. LANE1 vs LANE2 LEVEL DIFFERENCE')
a = [med(I1, 'AUTO', e, 'nRMSE') for e in KEPT1]
b = [med(I2, 'AUTO', e, 'nRMSE') for e in KEPT2]
print(f'   lane1 AUTO nRMSE  median={st.median(a):.4f} min={min(a):.4f} max={max(a):.4f}')
print(f'   lane2 AUTO nRMSE  median={st.median(b):.4f} min={min(b):.4f} max={max(b):.4f}')
# rank-sum
allv = sorted(a + b)
Ra = sum(allv.index(x) + 1 for x in a)
n1, n2 = len(a), len(b)
U = Ra - n1 * (n1 + 1) / 2
mu = n1 * n2 / 2
sd = math.sqrt(n1 * n2 * (n1 + n2 + 1) / 12)
z = (U - mu) / sd
print(f'   Mann-Whitney U={U:.0f} z={z:+.2f} (|z|<1.96 => no detectable block shift)')
print('   per-config medians (lane1 | lane2):')
for cfgname in ('BF4_15', 'BF4_30', 'BF5_15', 'BF5_30'):
    x = [med(I1, 'AUTO', e, 'nRMSE') for e in KEPT1 if cfgname in e]
    y = [med(I2, 'AUTO', e, 'nRMSE') for e in KEPT2 if cfgname in e]
    print(f'     {cfgname}: n={len(x)} {st.median(x):.4f} | n={len(y)} {st.median(y):.4f}')
print(f'   episode overlap lane1/lane2: {len(set(EPS1)&set(EPS2))}')

print('\nH. INTEGRITY, MY OWN VERSION (all scalar fields, not the 9 the scorer used)')
skip = {'job_id', 'cell', 'wall_s', 'elapsed_ms', 'cfg', 'spec'}
for nm, I, eps in (('lane1', I1, EPS1), ('lane2', I2, EPS2)):
    nb = nc = 0
    for e in eps:
        for s in (1539, 1540, 1541):
            A = [x for x in I.get(('AUTO', e), []) if x['seed'] == s]
            D = [x for x in I.get(('AUTO-DUP', e), []) if x['seed'] == s]
            if A and D:
                for k in A[0]:
                    if k in skip:
                        continue
                    nc += 1
                    nb += (json.dumps(A[0][k], sort_keys=True)
                           != json.dumps(D[0].get(k), sort_keys=True))
    print(f'   {nm}: AUTO vs AUTO-DUP {nc} field comparisons, {nb} mismatches')
# complement check
bad = 0
for I, eps in ((I1, EPS1), (I2, EPS2)):
    for c in L:
        for e in eps:
            for x in I.get((c, e), []):
                p = x['rmse_by_radius']
                bad += abs(p['10'] - x['avg_RMSE_observed']) > 1e-9
                bad += abs(p['u10'] - x['avg_RMSE_unobserved']) > 1e-9
print(f'   r10/u10 vs avg_RMSE_observed/unobserved mismatches: {bad}  (PREREG sec.5.6)')
# cfg check: is init_method really 'auto' everywhere, one knob only?
cfgs = {}
for r in rows1 + rows2:
    cfgs.setdefault(r['cell'], set()).add(json.dumps(r['cfg'], sort_keys=True))
for c in L:
    d = json.loads(list(cfgs[c])[0])
    base = json.loads(list(cfgs['AUTO'])[0])
    diff = {k: (base.get(k), d.get(k)) for k in set(base) | set(d)
            if base.get(k) != d.get(k)}
    print(f'   cfg {c:14s} variants={len(cfgs[c])} init_method={d["init_method"]} '
          f'diff_vs_AUTO={diff}')
