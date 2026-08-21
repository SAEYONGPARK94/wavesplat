"""CON audit (E-C) part 1 -- independent re-derivation from results.jsonl.

Own median / sign-test code.  Nothing imported from analyze_ec.py.
Read-only, CPU only.
"""
import json
import math
import os
import statistics as st
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
TMP = os.path.join(HERE, '..')
CELLS = ['AUTO', 'AUTO-DUP', 'AUTO-LR10', 'AUTO-LR100', 'AUTO-LR1000', 'AUTO-NOTRACK']


def sgn_p(k, n):
    if n == 0:
        return 1.0
    lo = min(k, n - k)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(lo + 1)) / 2 ** n)


def load(p):
    return [json.loads(l) for l in open(p) if l.strip()]


R = load(os.path.join(HERE, 'results.jsonl'))
NULL = {}
for nf in ('Rx6_channel_curve_2026-08-07/nulls_rx6.json',
           'Rx7_tracker_deploy_2026-08-07/nulls_rx7.json'):
    p = os.path.join(TMP, nf)
    if os.path.exists(p):
        for n in json.load(open(p)):
            NULL[n['episode']] = n['zero_null_tracking_nRMSE_mean']

IDX = defaultdict(list)
for r in R:
    IDX[(r['cell'], r['episode'])].append(r)
EPS = sorted({r['episode'] for r in R})


def med(c, e, k):
    v = [x[k] for x in IDX[(c, e)] if x.get(k) is not None and x[k] == x[k]]
    return st.median(v) if v else None


KEPT = [e for e in EPS if med('AUTO', e, 'nRMSE') < NULL[e]]
print(f'# rows={len(R)}  eps={len(EPS)}  kept={len(KEPT)}  excl={len(EPS)-len(KEPT)}')


def paired(a, b, k, blk=None, good_neg=True):
    blk = blk or KEPT
    d = [med(a, e, k) - med(b, e, k) for e in blk
         if med(a, e, k) is not None and med(b, e, k) is not None]
    d = [x for x in d if x != 0]
    w = sum(1 for x in d if (x < 0) == good_neg)
    return st.median(d), w, len(d), sgn_p(w, len(d))


print('\n=== [1] PRIMARY-1 re-derivation (my code) ===')
for k in ('avg_obs_loss', 'obs_resid_rms', 'obs_resid_rms_oos', 'nRMSE'):
    m, w, n, p = paired('AUTO', 'AUTO-NOTRACK', k)
    print(f'  {k:20s} medΔ={m:+.6f}  lower {w}/{n}  p={p:.3g}')

print('\n=== [2] the quantity the prereg did NOT test: generalisation GAP per cell ===')
print('  (overfitting = oos - in WITHIN a cell; the prereg compared oos ACROSS cells)')
print(f'  {"cell":14s} {"in-sample":>10s} {"oos":>10s} {"gap[m]":>10s} {"oos/in":>8s} '
      f'{"gap>0 eps":>10s} {"p":>9s}')
gaps = {}
for c in CELLS:
    gi = [med(c, e, 'obs_resid_rms') for e in KEPT]
    go = [med(c, e, 'obs_resid_rms_oos') for e in KEPT]
    d = [b - a for a, b in zip(gi, go) if a is not None and b is not None]
    w = sum(1 for x in d if x > 0)
    ratio = [b / a for a, b in zip(gi, go) if a]
    gaps[c] = (st.median(d), st.median(ratio))
    print(f'  {c:14s} {st.median(gi):10.5f} {st.median(go):10.5f} {st.median(d):+10.5f} '
          f'{st.median(ratio):8.4f} {w:6d}/{len(d):<3d} {sgn_p(w, len(d)):9.3g}')

print('\n  per-episode PAIRED gap contrast AUTO vs AUTO-NOTRACK:')
for k, lbl in (('abs', 'gap_AUTO - gap_NOTRACK [m]'), ('rel', 'ratio_AUTO - ratio_NOTRACK')):
    d = []
    for e in KEPT:
        ai, ao = med('AUTO', e, 'obs_resid_rms'), med('AUTO', e, 'obs_resid_rms_oos')
        bi, bo = med('AUTO-NOTRACK', e, 'obs_resid_rms'), med('AUTO-NOTRACK', e, 'obs_resid_rms_oos')
        if None in (ai, ao, bi, bo) or not ai or not bi:
            continue
        d.append(((ao - ai) - (bo - bi)) if k == 'abs' else ((ao / ai) - (bo / bi)))
    w = sum(1 for x in d if x > 0)
    print(f'    {lbl:32s} med={st.median(d):+.5f}  AUTO worse {w}/{len(d)}  p={sgn_p(w,len(d)):.3g}')

print('\n=== [3] ratio 0.973 -- ratio of medians vs median of ratios ===')
mo = paired('AUTO', 'AUTO-NOTRACK', 'obs_resid_rms_oos')[0]
mi = paired('AUTO', 'AUTO-NOTRACK', 'obs_resid_rms')[0]
print(f'  ratio-of-medians (reported) = {mo/mi:.4f}')
per = []
for e in KEPT:
    a = med('AUTO', e, 'obs_resid_rms_oos') - med('AUTO-NOTRACK', e, 'obs_resid_rms_oos')
    b = med('AUTO', e, 'obs_resid_rms') - med('AUTO-NOTRACK', e, 'obs_resid_rms')
    if b:
        per.append(a / b)
per.sort()
print(f'  median-of-ratios            = {st.median(per):.4f}   '
      f'IQR=[{per[len(per)//4]:.4f},{per[3*len(per)//4]:.4f}]  '
      f'min={per[0]:.4f} max={per[-1]:.4f}')

print('\n=== [4] PREREG §5.6 complement check (scorer never ran it) ===')
bad = worst = 0
for r in R:
    pr = r.get('rmse_by_radius') or {}
    for a, b in (('10', 'avg_RMSE_observed'), ('u10', 'avg_RMSE_unobserved')):
        if a in pr and r.get(b) is not None and pr[a] == pr[a] and r[b] == r[b]:
            dd = abs(pr[a] - r[b])
            worst = max(worst, dd)
            bad += dd > 1e-12
print(f'  rows checked={len(R)*2}  mismatches(>1e-12)={bad}  worst|Δ|={worst:.3e}  '
      f'-> {"PASS" if bad == 0 else "FAIL"}')

print('\n=== [5] PREREG §5.5 ensemble check (scorer never ran it) ===')
ek = defaultdict(int)
for r in R:
    ek[r.get('ensemble_k')] += 1
print(f'  ensemble_k distribution: {dict(ek)}')
print('  -> jsonl alone CANNOT verify the member-median wiring (no per-member residual '
      'stored). Code-level check is in audit_con_leak.py.')

print('\n=== [6] crossover: PREREG §4 literal criterion (adjacent pair, both pass) ===')
RADS = ['5', '10', '20', '30', '50', '75', '100', '150']
NEED = {45: 30, 44: 29, 43: 29, 48: 31}
prof = {}
for rad in RADS:
    d = []
    for e in KEPT:
        va = [x['rmse_by_radius'][rad] for x in IDX[('AUTO', e)]
              if x['rmse_by_radius'].get(rad) == x['rmse_by_radius'].get(rad)]
        vb = [x['rmse_by_radius'][rad] for x in IDX[('AUTO-NOTRACK', e)]
              if x['rmse_by_radius'].get(rad) == x['rmse_by_radius'].get(rad)]
        if va and vb:
            d.append(st.median(va) - st.median(vb))
    d = [x for x in d if x != 0]
    w = sum(1 for x in d if x < 0)
    need = NEED.get(len(d), int(0.5 * len(d) + 1.5 * math.sqrt(len(d))) + 1)
    sd = st.stdev(d)
    mde = 2.49 * sd / math.sqrt(len(d))
    prof[rad] = (st.median(d), w, len(d), sgn_p(w, len(d)), need, mde)
    verdict = ('BETTER pass' if w >= need else
               'WORSE pass' if (len(d) - w) >= need else 'NEITHER (null)')
    print(f'  r={rad:>4s}m  medΔ={st.median(d):+9.5f}  better {w:2d}/{len(d)}  '
          f'p={sgn_p(w,len(d)):8.4g}  need={need}  {verdict:14s} MDE={mde:.5f} '
          f'|med|/MDE={abs(st.median(d))/mde:5.2f}')
adj = []
for a, b in zip(RADS, RADS[1:]):
    ma, wa, na, _, nea, _ = prof[a]
    mb, wb, nb, _, neb, _ = prof[b]
    pa = wa >= nea or (na - wa) >= nea
    pb = wb >= neb or (nb - wb) >= neb
    if ma * mb < 0 and pa and pb:
        adj.append((a, b))
print(f'  adjacent pairs with sign flip AND both sides passing threshold: {adj or "NONE"}')

print('\n=== [7] AUTO-LR1000 ===')
for c in ('AUTO-LR10', 'AUTO-LR100', 'AUTO-LR1000'):
    m, w, n, p = paired(c, 'AUTO-NOTRACK', 'nRMSE')
    raw = [med(c, e, 'nRMSE') - med('AUTO-NOTRACK', e, 'nRMSE') for e in KEPT]
    sd = st.stdev(raw)
    print(f'  {c:12s} medΔ={m:+.6f} {w}/{n} p={p:.3g}  sd={sd:.6f}  '
          f'mean={st.mean(raw):+.6f}  |max|={max(abs(x) for x in raw):.5f}')
# how big is the LR1000 effect next to run-to-run seed noise within one cell?
noise = []
for e in KEPT:
    v = [x['nRMSE'] for x in IDX[('AUTO-LR1000', e)]]
    if len(v) > 1:
        noise.append(max(v) - min(v))
print(f'  seed spread (max-min over 3 seeds) of AUTO-LR1000 nRMSE: median={st.median(noise):.5f}')
print(f'  ratio |medΔ| / seed-spread = {abs(paired("AUTO-LR1000","AUTO-NOTRACK","nRMSE")[0])/st.median(noise):.4f}')
# lr magnitude
print('  tracking_lr: AUTO=5e-3, LR10=5e-4, LR100=5e-5, LR1000=5e-6, NOTRACK=0')
for c in ('AUTO', 'AUTO-LR1000', 'AUTO-NOTRACK'):
    print(f'    {c:14s} resid_rms(med)={st.median([med(c,e,"obs_resid_rms") for e in KEPT]):.5f}  '
          f'obs_loss(med)={st.median([med(c,e,"avg_obs_loss") for e in KEPT]):.6f}  '
          f'RMSE_obs={st.median([med(c,e,"avg_RMSE_observed") for e in KEPT]):.5f}')

print('\n=== [8] null exclusion: does it move anything? ===')
for k in ('avg_obs_loss', 'obs_resid_rms', 'obs_resid_rms_oos', 'nRMSE'):
    m1, w1, n1, p1 = paired('AUTO', 'AUTO-NOTRACK', k)
    m2, w2, n2, p2 = paired('AUTO', 'AUTO-NOTRACK', k, blk=EPS)
    print(f'  {k:20s} kept {m1:+.6f} {w1}/{n1} | ALL48 {m2:+.6f} {w2}/{n2}')
excl = [e for e in EPS if e not in KEPT]
print(f'  excluded: {[e[8:-20] for e in excl]}')
for e in excl:
    print(f'    {e[8:-20]:22s} AUTO={med("AUTO",e,"nRMSE"):.4f} null={NULL[e]:.4f} '
          f'NOTRACK={med("AUTO-NOTRACK",e,"nRMSE"):.4f} '
          f'Δ(AUTO-NOTRACK)={med("AUTO",e,"nRMSE")-med("AUTO-NOTRACK",e,"nRMSE"):+.4f}')

print('\n=== [9] multiplicity (Holm over the whole reported family) ===')
fam = []
for k in ('avg_obs_loss', 'obs_resid_rms', 'obs_resid_rms_oos', 'nRMSE'):
    fam.append((f'P1:{k}', paired('AUTO', 'AUTO-NOTRACK', k)[3]))
for rad in RADS:
    fam.append((f'rad:{rad}', prof[rad][3]))
for c in ('AUTO-LR10', 'AUTO-LR100', 'AUTO-LR1000'):
    fam.append((f'lr:{c}', paired(c, 'AUTO-NOTRACK', 'nRMSE')[3]))
fam.sort(key=lambda x: x[1])
mm = len(fam)
prev = 0.0
for i, (nm, p) in enumerate(fam):
    thr = 0.05 / (mm - i)
    prev = max(prev, p)
    print(f'  {nm:22s} p={p:11.4g}  Holm thr={thr:.5f}  '
          f'{"survives" if prev <= thr else "FAILS"}')
