"""CON audit (E-C) part 3 -- arithmetic decomposition, annulus (shell) reconstruction,
and the episode-reuse information content.  Read-only."""
import json
import math
import os
import statistics as st
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
TMP = os.path.join(HERE, '..')


def sgn_p(k, n):
    lo = min(k, n - k)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(lo + 1)) / 2 ** n) if n else 1.0


def load(p):
    return [json.loads(l) for l in open(p) if l.strip()]


def index(rows):
    d = defaultdict(list)
    for r in rows:
        d[(r['cell'], r['episode'])].append(r)
    return d


EC = load(os.path.join(HERE, 'results.jsonl'))
IEC = index(EC)
NULL = {}
for nf in ('Rx6_channel_curve_2026-08-07/nulls_rx6.json',
           'Rx7_tracker_deploy_2026-08-07/nulls_rx7.json'):
    p = os.path.join(TMP, nf)
    if os.path.exists(p):
        for n in json.load(open(p)):
            NULL[n['episode']] = n['zero_null_tracking_nRMSE_mean']
EPS = sorted({r['episode'] for r in EC})


def med(I, c, e, k):
    v = [x[k] for x in I[(c, e)] if x.get(k) is not None and x[k] == x[k]]
    return st.median(v) if v else None


KEPT = [e for e in EPS if med(IEC, 'AUTO', e, 'nRMSE') < NULL[e]]

print('=== [A] AUTO-NOTRACK: is oos EXACTLY equal to in-sample? (lr=0 => frozen model) ===')
worst = 0.0
nrow = 0
for r in EC:
    if r['cell'] == 'AUTO-NOTRACK':
        nrow += 1
        worst = max(worst, abs(r['obs_resid_rms'] - r['obs_resid_rms_oos']))
print(f'  rows={nrow}  max|oos - in| = {worst:.3e}  -> the baseline arm does NOT move '
      f'during Phase 2 (TRACK_PHASE_PREDICT=False, tracking_lr=0)')

print('\n=== [B] the headline is arithmetic: Δ_oos = Δ_in + gap_AUTO ===')
res = []
for e in KEPT:
    ai = med(IEC, 'AUTO', e, 'obs_resid_rms')
    ao = med(IEC, 'AUTO', e, 'obs_resid_rms_oos')
    bi = med(IEC, 'AUTO-NOTRACK', e, 'obs_resid_rms')
    bo = med(IEC, 'AUTO-NOTRACK', e, 'obs_resid_rms_oos')
    res.append((ao - bo, ai - bi, ao - ai, bo - bi))
d_oos = st.median([x[0] for x in res])
d_in = st.median([x[1] for x in res])
gap_a = st.median([x[2] for x in res])
gap_b = st.median([x[3] for x in res])
print(f'  med Δ_oos = {d_oos:+.6f}   med Δ_in = {d_in:+.6f}   '
      f'med gap_AUTO = {gap_a:+.6f}   med gap_NOTRACK = {gap_b:+.6f}')
print(f'  Δ_in + gap_AUTO - gap_NOTRACK = {d_in + gap_a - gap_b:+.6f}  (vs Δ_oos {d_oos:+.6f})')
print(f'  reported ratio 0.973 == 1 - gap_AUTO/|Δ_in| = {1 - gap_a/abs(d_in):.4f}')
print('  => the only NEW number in the headline is gap_AUTO = '
      f'{gap_a*1000:.2f} mm; Δ_in ({d_in:+.4f}) is the E-B result re-run.')

print('\n=== [C] annulus (shell) reconstruction from the cumulative-disk profile ===')
print('  approximate: MSE_disk(r)*cov(r) = sum of shell MSE*area; levels are block medians')
COV = {'5': 0.029, '10': 0.055, '20': 0.109, '30': 0.166, '50': 0.282,
       '75': 0.445, '100': 0.578, '150': 0.808, 'ALL': 1.000}
RADS = ['5', '10', '20', '30', '50', '75', '100', '150']


def lvl(cell, rad):
    if rad == 'ALL':
        # full-grid RMSE in metres: nRMSE * Hs.  Hs is per episode; use RMSE_z proxy
        return None
    return st.median([med(IEC, cell, e, 'rmse_by_radius')[rad]
                      if isinstance(med(IEC, cell, e, 'rmse_by_radius'), dict) else None
                      for e in KEPT])


def lvl2(cell, rad):
    v = []
    for e in KEPT:
        vv = [x['rmse_by_radius'][rad] for x in IEC[(cell, e)]
              if x['rmse_by_radius'].get(rad) == x['rmse_by_radius'].get(rad)]
        if vv:
            v.append(st.median(vv))
    return st.median(v)


print(f'  {"shell":>12s} {"AUTO":>9s} {"NOTRACK":>9s} {"Δ shell":>10s} {"Δ disk":>10s}')
prev = None
for rad in RADS:
    a, b = lvl2('AUTO', rad), lvl2('AUTO-NOTRACK', rad)
    if prev is None:
        sa, sb, lab = a, b, f'0-{rad}'
    else:
        pr, pa, pb = prev
        ca, cp = COV[rad], COV[pr]
        na = (a * a * ca - pa * pa * cp) / (ca - cp)
        nb = (b * b * ca - pb * pb * cp) / (ca - cp)
        sa, sb = math.sqrt(max(na, 0)), math.sqrt(max(nb, 0))
        lab = f'{pr}-{rad}'
    print(f'  {lab:>12s} {sa:9.4f} {sb:9.4f} {sa-sb:+10.4f} {a-b:+10.4f}')
    prev = (rad, a, b)

print('\n=== [D] episode reuse: how much of E-C is predicted by E-B (same 48 episodes)? ===')
EB = load(os.path.join(TMP, 'EB_instrument_2026-08-07/results_lane1.jsonl')) + \
    load(os.path.join(TMP, 'EB_instrument_2026-08-07/results_lane2.jsonl'))
IEB = index(EB)
EPS_EB = sorted({r['episode'] for r in EB})
print(f'  E-B episodes={len(EPS_EB)}  E-C episodes={len(EPS)}  identical set: {EPS_EB == EPS}')
print(f'  E-B seeds={sorted({r["seed"] for r in EB})}  E-C seeds={sorted({r["seed"] for r in EC})}')


def pear(a, b):
    n = len(a)
    ma, mb = sum(a) / n, sum(b) / n
    num = sum((x - ma) * (y - mb) for x, y in zip(a, b))
    den = math.sqrt(sum((x - ma) ** 2 for x in a) * sum((y - mb) ** 2 for y in b))
    return num / den if den else float('nan')


for key in ('nRMSE', 'obs_resid_rms', 'avg_obs_loss'):
    xa, xb = [], []
    for e in EPS:
        a = med(IEB, 'AUTO', e, key)
        b = med(IEB, 'AUTO-NOTRACK', e, key)
        c = med(IEC, 'AUTO', e, key)
        d = med(IEC, 'AUTO-NOTRACK', e, key)
        if None not in (a, b, c, d):
            xa.append(a - b)
            xb.append(c - d)
    agree = sum(1 for u, v in zip(xa, xb) if (u < 0) == (v < 0))
    print(f'  Δ(AUTO-NOTRACK) per episode, E-B vs E-C: {key:16s} '
          f'r={pear(xa, xb):+.3f}  sign agreement {agree}/{len(xa)}')
xa = [med(IEB, 'AUTO', e, 'nRMSE') for e in EPS]
xb = [med(IEC, 'AUTO', e, 'nRMSE') for e in EPS]
print(f'  AUTO level per episode, E-B vs E-C: r={pear(xa, xb):+.3f}  '
      f'(episode identity, not estimator seed, drives the level)')

print('\n=== [E] LR1000: did E-B see it too? (same episodes, different seeds) ===')
for lane, I, blk in (('E-B(1539-41)', IEB, EPS), ('E-C(1542-44)', IEC, EPS)):
    d = [med(I, 'AUTO-LR1000', e, 'nRMSE') - med(I, 'AUTO-NOTRACK', e, 'nRMSE') for e in blk]
    d = [x for x in d if x == x and x != 0]
    w = sum(1 for x in d if x < 0)
    print(f'  {lane}  medΔ={st.median(d):+.6f}  better {w}/{len(d)}  p={sgn_p(w,len(d)):.3g}')
# joint: is the per-episode LR1000 delta reproducible?
xa = [med(IEB, 'AUTO-LR1000', e, 'nRMSE') - med(IEB, 'AUTO-NOTRACK', e, 'nRMSE') for e in EPS]
xb = [med(IEC, 'AUTO-LR1000', e, 'nRMSE') - med(IEC, 'AUTO-NOTRACK', e, 'nRMSE') for e in EPS]
agree = sum(1 for u, v in zip(xa, xb) if (u < 0) == (v < 0))
print(f'  per-episode LR1000 delta: r={pear(xa, xb):+.3f}  sign agreement {agree}/{len(xa)}')
