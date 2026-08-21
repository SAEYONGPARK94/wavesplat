#!/usr/bin/env python3
"""PRO audit part 3 -- the mechanism behind the sign flip, robustness of the ORACLE
contrast, and gap-closure accounting.  Reads only."""
import json, os, math, statistics as st

HERE = os.path.dirname(os.path.abspath(__file__))
rows = [json.loads(l) for l in open(os.path.join(HERE, 'results.jsonl')) if l.strip()]
nulls = {d['episode']: d for d in json.load(open(os.path.join(HERE, 'nulls_k6.json')))}
IDX = {(r['cell'], r['episode'], r['seed']): r for r in rows}
EPS = sorted({r['episode'] for r in rows}); SEEDS = sorted({r['seed'] for r in rows})

def med(v):
    v = sorted(v); n = len(v)
    return v[n // 2] if n % 2 else 0.5 * (v[n // 2 - 1] + v[n // 2])
def sm(c, e, f): return med([IDX[(c, e, s)][f] for s in SEEDS])
def sp(c, e, k): return med([IDX[(c, e, s)]['spec'][k] for s in SEEDS])
def bts(k, n):
    C = math.comb
    return min(1.0, 2 * sum(C(n, i) for i in range(0, min(k, n - k) + 1)) / 2 ** n)

KEEP = [e for e in EPS if sm('AUTO', e, 'nRMSE') < nulls[e]['zero_null_tracking_nRMSE_mean']]
DROP = [e for e in EPS if e not in KEEP]

print('=' * 78)
print('F. MECHANISM -- k MOTION IS REPAIR AT BLIND PLACEMENT, DAMAGE AT ORACLE PLACEMENT')
print('=' * 78)
print('  (final matched k error; init_k_lr_scale=0 cells report the INITIAL k unchanged)')
print('                     dir_err[deg]   k_mag_rel   [DESCRIPTIVE ONLY, PREREG sec.7]')
for c in ['K6-KLR00', 'K6-KLR01', 'K6-KLR03', 'AUTO', 'ORACLE', 'ORACLE-K']:
    de = [sp(c, e, 'direction_error_deg_mean') for e in KEEP]
    km = [sp(c, e, 'k_mag_rel_error_mean') for e in KEEP]
    tag = ('  <- k FROZEN  == the placement Phase 1 was handed' if c in ('K6-KLR00', 'ORACLE')
           else '  <- k TRAINED')
    print(f'  {c:10s}  {med(de):6.2f}  [{min(de):5.2f},{max(de):5.2f}]   {med(km):.4f}{tag}')
print()
print(f'  BLIND arm : frozen(KLR00) {med([sp("K6-KLR00",e,"direction_error_deg_mean") for e in KEEP]):.2f} deg'
      f' -> trained(AUTO) {med([sp("AUTO",e,"direction_error_deg_mean") for e in KEEP]):.2f} deg'
      f'   => Phase-1 k training REPAIRS')
print(f'  ORACLE arm: frozen(ORACLE) {med([sp("ORACLE",e,"direction_error_deg_mean") for e in KEEP]):.2f} deg'
      f' -> trained(ORACLE-K) {med([sp("ORACLE-K",e,"direction_error_deg_mean") for e in KEEP]):.2f} deg'
      f'  => Phase-1 k training DAMAGES')
# paired sign tests on the direction error (descriptive)
for a, b, lab in [('AUTO', 'K6-KLR00', 'trained better than frozen, BLIND arm'),
                  ('ORACLE-K', 'ORACLE', 'trained better than frozen, ORACLE arm')]:
    d = [sp(a, e, 'direction_error_deg_mean') - sp(b, e, 'direction_error_deg_mean') for e in KEEP]
    w = sum(1 for x in d if x < 0)
    print(f'  sign test [{lab}]: median {med(d):+.2f} deg  {w}/{len(d)}  p={bts(w,len(d)):.4f}')
print('  ALSO on the pre-registered FIELD endpoint init_nrmse (allowed as evidence):')
for a, b in [('K6-KLR00', 'AUTO'), ('ORACLE', 'ORACLE-K')]:
    d = [sm(a, e, 'init_nrmse') - sm(b, e, 'init_nrmse') for e in KEEP]
    w = sum(1 for x in d if x < 0)
    print(f'    {a:9s} - {b:9s} init_nrmse median {med(d):+.4f}  {w}/{len(d)}  p={bts(w,len(d)):.4f}')

print()
print('=' * 78)
print('G. ROBUSTNESS OF THE ORACLE CONTRAST')
print('=' * 78)
for a, b in [('ORACLE', 'AUTO'), ('ORACLE-K', 'K6-NSCHED'), ('ORACLE', 'ORACLE-K')]:
    print(f'  --- {a} - {b} ---')
    # per-seed (no median-of-3): three separate 14-episode sign tests
    for s in SEEDS:
        d = [IDX[(a, e, s)]['nRMSE'] - IDX[(b, e, s)]['nRMSE'] for e in KEEP]
        w = sum(1 for x in d if x < 0)
        print(f'    seed {s} alone: median {med(d):+.4f}  {w}/{len(d)}  p={bts(w,len(d)):.4f}')
    # per-run (42 runs, NOT independent -- reported for completeness only)
    d = [IDX[(a, e, s)]['nRMSE'] - IDX[(b, e, s)]['nRMSE'] for e in KEEP for s in SEEDS]
    w = sum(1 for x in d if x < 0)
    print(f'    all 42 runs (pseudo-replicate, NOT a valid n): {w}/{len(d)}')
    # seed-mean instead of seed-median (aggregation robustness)
    d = [st.mean([IDX[(a, e, s)]['nRMSE'] for s in SEEDS]) -
         st.mean([IDX[(b, e, s)]['nRMSE'] for s in SEEDS]) for e in KEEP]
    w = sum(1 for x in d if x < 0)
    print(f'    seed-MEAN aggregation: median {med(d):+.4f}  {w}/{len(d)}  p={bts(w,len(d)):.4f}')
    # effect size vs seed noise
    rng = med([max(IDX[(b, e, s)]['nRMSE'] for s in SEEDS) -
               min(IDX[(b, e, s)]['nRMSE'] for s in SEEDS) for e in KEEP])
    dd = [sm(a, e, 'nRMSE') - sm(b, e, 'nRMSE') for e in KEEP]
    print(f'    |effect| {abs(med(dd)):.4f} = {abs(med(dd))/rng:.1f}x the control seed-range '
          f'({rng:.4f}); SD of paired delta {st.pstdev(dd):.4f}; '
          f'|median|/SD = {abs(med(dd))/st.pstdev(dd):.2f}')

print()
print('=' * 78)
print('H. NULL MARGIN m = arm / flat-sea null   (m>1 == worse than a flat sea)')
print('=' * 78)
for c in ['AUTO', 'K6-KLR03', 'K6-KLR01', 'K6-KLR00', 'K6-NSCHED', 'ORACLE-K', 'ORACLE']:
    m = [sm(c, e, 'nRMSE') / nulls[e]['zero_null_tracking_nRMSE_mean'] for e in EPS]
    nb = sum(1 for x in m if x < 1)
    mk = [sm(c, e, 'nRMSE') / nulls[e]['zero_null_tracking_nRMSE_mean'] for e in KEEP]
    print(f'  {c:10s} m(all16) median {med(m):.3f}  beats-null {nb}/16   '
          f'm(keep14) {med(mk):.3f}   panel m_mean {med([sm(c,e,"m_mean") for e in KEEP]):.3f}')

print()
print('=' * 78)
print('I. GAP-CLOSURE ACCOUNTING vs the ORACLE ceiling, on the shipping control')
print('=' * 78)
ceil = med([sm('ORACLE', e, 'nRMSE') - sm('AUTO', e, 'nRMSE') for e in KEEP])
print(f'  headroom AUTO -> ORACLE = {ceil:+.4f}  (14/14)')
for c in ['K6-KLR03', 'K6-KLR01', 'K6-KLR00', 'ORACLE-K']:
    d = med([sm(c, e, 'nRMSE') - sm('AUTO', e, 'nRMSE') for e in KEEP])
    print(f'  {c:10s}: {d:+.4f}  = {100*d/ceil:5.1f}% of the headroom '
          f'({"NULL, not a recovery" if abs(d) < 0.019 else ""})')
print(f'  reference: MDE at 80% power (n=14, req 12) = 0.0190 nRMSE '
      f'=> {100*0.0190/abs(ceil):.1f}% of the headroom is the smallest slice this batch could see')

print()
print('=' * 78)
print('J. THE TWO NULL-FAILING EPISODES (screen report only -- NOT citable as results)')
print('=' * 78)
for e in DROP:
    print(f'  {e[8:44]}  null {nulls[e]["zero_null_tracking_nRMSE_mean"]:.4f}')
    for c in ['AUTO', 'K6-KLR03', 'K6-KLR01', 'K6-KLR00', 'K6-NSCHED', 'ORACLE-K', 'ORACLE']:
        v = sm(c, e, 'nRMSE')
        print(f'    {c:10s} {v:.4f}  m={v/nulls[e]["zero_null_tracking_nRMSE_mean"]:.3f}'
              f'  {"beats null" if v < nulls[e]["zero_null_tracking_nRMSE_mean"] else "FAILS null"}')
print('  (PREREG sec.4: these episodes are excluded from PRIMARY; their deltas are not results.)')

print()
print('=' * 78)
print('K. HOW FAR IS THE SHIPPING ARM FROM THE ORACLE END OF THE INTERACTION CURVE?')
print('=' * 78)
de_auto = sorted(sp('AUTO', e, 'direction_error_deg_mean') for e in KEEP)
de_ok = sorted(sp('ORACLE-K', e, 'direction_error_deg_mean') for e in KEEP)
print(f'  AUTO     final dir_err over 14 eps: {de_auto[0]:.2f} .. {de_auto[-1]:.2f} deg (median {med(de_auto):.2f})')
print(f'  ORACLE-K final dir_err over 14 eps: {de_ok[0]:.2f} .. {de_ok[-1]:.2f} deg (median {med(de_ok):.2f})')
print(f'  ORACLE   final dir_err: 0.00 deg by construction (k frozen at truth)')
print(f'  => the BEST shipping episode ({de_auto[0]:.2f} deg) is still worse than the MEDIAN')
print(f'     oracle-placement episode ({med(de_ok):.2f} deg): the shipping arm never enters the')
print('     regime where freezing k was measured to pay.  [DESCRIPTIVE metric, sec.7]')
