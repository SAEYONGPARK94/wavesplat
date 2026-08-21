"""CON audit (E-C) part 4 -- is the generalisation gap the classical optimism of a
2N-parameter fit, does it scale with the knob, and how uncertain is the crossover radius?
Read-only."""
import json
import math
import os
import random
import statistics as st
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
TMP = os.path.join(HERE, '..')
CELLS = ['AUTO', 'AUTO-LR10', 'AUTO-LR100', 'AUTO-LR1000', 'AUTO-NOTRACK']
LR = {'AUTO': 5e-3, 'AUTO-LR10': 5e-4, 'AUTO-LR100': 5e-5, 'AUTO-LR1000': 5e-6,
      'AUTO-NOTRACK': 0.0}
R = [json.loads(l) for l in open(os.path.join(HERE, 'results.jsonl'))]
I = defaultdict(list)
for r in R:
    I[(r['cell'], r['episode'])].append(r)
NULL = {}
for nf in ('Rx6_channel_curve_2026-08-07/nulls_rx6.json',
           'Rx7_tracker_deploy_2026-08-07/nulls_rx7.json'):
    p = os.path.join(TMP, nf)
    if os.path.exists(p):
        for n in json.load(open(p)):
            NULL[n['episode']] = n['zero_null_tracking_nRMSE_mean']
EPS = sorted({r['episode'] for r in R})


def med(c, e, k):
    v = [x[k] for x in I[(c, e)] if x.get(k) is not None and x[k] == x[k]]
    return st.median(v) if v else None


KEPT = [e for e in EPS if med('AUTO', e, 'nRMSE') < NULL[e]]


def spear(a, b):
    n = len(a)

    def rk(v):
        s = sorted(range(n), key=lambda i: v[i])
        r = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j + 1 < n and v[s[j + 1]] == v[s[i]]:
                j += 1
            avg = (i + j) / 2.0
            for t in range(i, j + 1):
                r[s[t]] = avg
            i = j + 1
        return r
    ra, rb = rk(a), rk(b)
    ma, mb = sum(ra) / n, sum(rb) / n
    num = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    den = math.sqrt(sum((x - ma) ** 2 for x in ra) * sum((y - mb) ** 2 for y in rb))
    return num / den if den else float('nan')


print('=== [1] gap ladder vs field-damage ladder (knob = tracking_lr) ===')
print(f'  {"cell":14s} {"lr":>8s} {"gap[m]":>9s} {"oos/in":>8s} {"ΔnRMSE":>9s} '
      f'{"Δ nRMSE/gap":>12s} {"N(med)":>7s}')
for c in CELLS:
    gap = st.median([med(c, e, 'obs_resid_rms_oos') - med(c, e, 'obs_resid_rms') for e in KEPT])
    ratio = st.median([med(c, e, 'obs_resid_rms_oos') / med(c, e, 'obs_resid_rms') for e in KEPT])
    dn = st.median([med(c, e, 'nRMSE') - med('AUTO-NOTRACK', e, 'nRMSE') for e in KEPT])
    nn = st.median([med(c, e, 'N') for e in KEPT])
    print(f'  {c:14s} {LR[c]:8.0e} {gap:9.5f} {ratio:8.4f} {dn:+9.5f} '
          f'{(dn/gap if gap else float("nan")):12.2f} {nn:7.1f}')

print('\n=== [2] is the gap the classical optimism of fitting 2N params to 3xM points? ===')
print('  tracking_mode=w_phi -> p = 2N free params; window=3 frames x ~49 pts = M~147')
print('  classical: RMSE_oos/RMSE_in ~ sqrt((1+p/M)/(1-p/M))')
M = 147.0
pr, ms = [], []
for e in KEPT:
    n = med('AUTO', e, 'N')
    p = 2 * n
    if p / M >= 1:
        continue
    pred = math.sqrt((1 + p / M) / (1 - p / M))
    meas = med('AUTO', e, 'obs_resid_rms_oos') / med('AUTO', e, 'obs_resid_rms')
    pr.append(pred)
    ms.append(meas)
print(f'  N median={st.median([med("AUTO",e,"N") for e in KEPT]):.0f}  '
      f'predicted ratio median={st.median(pr):.4f}   measured median={st.median(ms):.4f}')
print(f'  Spearman(predicted, measured) over {len(pr)} episodes = {spear(pr, ms):+.3f}')
print(f'  Spearman(N, measured ratio) = '
      f'{spear([med("AUTO",e,"N") for e in KEPT], ms):+.3f}')

print('\n=== [3] does the gap EXPLAIN the field damage, episode by episode? ===')
g = [med('AUTO', e, 'obs_resid_rms_oos') - med('AUTO', e, 'obs_resid_rms') for e in KEPT]
f = [med('AUTO', e, 'nRMSE') - med('AUTO-NOTRACK', e, 'nRMSE') for e in KEPT]
u = [med('AUTO', e, 'avg_RMSE_unobserved') - med('AUTO-NOTRACK', e, 'avg_RMSE_unobserved')
     for e in KEPT]
print(f'  Spearman(gap_AUTO, ΔnRMSE)            = {spear(g, f):+.3f}')
print(f'  Spearman(gap_AUTO, ΔRMSE_unobserved)  = {spear(g, u):+.3f}')
print(f'  Spearman(Δ_in-sample resid, ΔnRMSE)   = '
      f'{spear([med("AUTO",e,"obs_resid_rms")-med("AUTO-NOTRACK",e,"obs_resid_rms") for e in KEPT], f):+.3f}')
print('  (a gap that explains the damage would show a strong positive rank tie)')

print('\n=== [4] crossover radius: bootstrap interval over episodes ===')
RADS = [5, 10, 20, 30, 50, 75, 100, 150]
per_ep = {}
for e in KEPT:
    row = []
    for rad in RADS:
        va = [x['rmse_by_radius'][str(rad)] for x in I[('AUTO', e)]]
        vb = [x['rmse_by_radius'][str(rad)] for x in I[('AUTO-NOTRACK', e)]]
        row.append(st.median(va) - st.median(vb))
    per_ep[e] = row


def crossing(meds):
    for i in range(len(RADS) - 1):
        if meds[i] < 0 <= meds[i + 1]:
            a, b = meds[i], meds[i + 1]
            return RADS[i] + (RADS[i + 1] - RADS[i]) * (-a) / (b - a)
    return float('inf') if meds[-1] < 0 else float('-inf')


base = [st.median([per_ep[e][i] for e in KEPT]) for i in range(len(RADS))]
print(f'  point estimate (linear interp on the median profile): {crossing(base):.1f} m')
random.seed(0)
bs = []
for _ in range(4000):
    s = [random.choice(KEPT) for _ in KEPT]
    m = [st.median([per_ep[e][i] for e in s]) for i in range(len(RADS))]
    bs.append(crossing(m))
bs = [x for x in bs if math.isfinite(x)]
bs.sort()
print(f'  bootstrap 4000: finite crossings={len(bs)}  '
      f'95% interval = [{bs[int(0.025*len(bs))]:.1f}, {bs[int(0.975*len(bs))]:.1f}] m  '
      f'median={bs[len(bs)//2]:.1f} m')
print(f'  fraction of resamples whose crossing falls in the audited bracket 50-100 m: '
      f'{100*sum(1 for x in bs if 50 <= x <= 100)/len(bs):.1f}%')

print('\n=== [5] the same crossover on the UNOBSERVED complement (sanity) ===')
for rad in RADS:
    d = []
    for e in KEPT:
        va = [x['rmse_by_radius']['u' + str(rad)] for x in I[('AUTO', e)]]
        vb = [x['rmse_by_radius']['u' + str(rad)] for x in I[('AUTO-NOTRACK', e)]]
        if va and vb and va[0] == va[0] and vb[0] == vb[0]:
            d.append(st.median(va) - st.median(vb))
    d = [x for x in d if x == x and x != 0]
    w = sum(1 for x in d if x < 0)
    print(f'  complement r>{rad:>4d} m: medΔ={st.median(d):+.5f}  better {w:2d}/{len(d)}')
