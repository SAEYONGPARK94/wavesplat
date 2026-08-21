"""CON audit part 5: dose-response shape, where the anti-relation lives, process gaps."""
import json
import math
import os
import statistics as st
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
L = ['AUTO-NOTRACK', 'AUTO-LR1000', 'AUTO-LR100', 'AUTO-LR10', 'AUTO']
LR = [0.0, 5e-6, 5e-5, 5e-4, 5e-3]


def sgn_p(k, n):
    lo = min(k, n - k)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(lo + 1)) / 2 ** n) if n else 1.0


rows = [json.loads(l) for l in open(os.path.join(HERE, 'results_lane1.jsonl')) if l.strip()]
I = {}
for r in rows:
    I.setdefault((r['cell'], r['episode']), []).append(r)
EPS = sorted({r['episode'] for r in rows})
NULL = {n['episode']: n['zero_null_tracking_nRMSE_mean'] for n in json.load(
    open(os.path.join(HERE, '..', 'Rx7_tracker_deploy_2026-08-07', 'nulls_rx7.json')))}


def med(c, e, k):
    return st.median([x[k] for x in I[(c, e)]])


KEPT = [e for e in EPS if med('AUTO', e, 'nRMSE') < NULL[e]]

print('1. DOSE-RESPONSE SHAPE: is nRMSE monotone in lr?  (analysis.txt never asks)')
print(f'   {"pair":26s} {"D_obj":>10s} {"D_resid":>9s} {"D_nRMSE":>9s} {"win":>7s} {"p":>8s}')
for i in range(4):
    a, b = L[i + 1], L[i]
    do = st.median([med(a, e, 'avg_obs_loss') - med(b, e, 'avg_obs_loss') for e in KEPT])
    dr = st.median([med(a, e, 'obs_resid_rms') - med(b, e, 'obs_resid_rms') for e in KEPT])
    d = [med(a, e, 'nRMSE') - med(b, e, 'nRMSE') for e in KEPT]
    d = [x for x in d if x != 0]
    w = sum(1 for x in d if x > 0)      # w = # episodes where MORE lr made the field WORSE
    print(f'   {b+"->"+a:26s} {do:+10.5f} {dr:+9.5f} {st.median(d):+9.5f} '
          f'{w:3d}/{len(d):<3d} {sgn_p(w,len(d)):8.4f}')
print('   levels: nRMSE  ' + '  '.join(
    f'{c.replace("AUTO-","").replace("AUTO","AUTO"):>10s}={st.median([med(c,e,"nRMSE") for e in KEPT]):.4f}'
    for c in L))
print('   => over the bottom three rungs the objective improves 2.4x and the field is FLAT.')
print('      The anti-relation exists only on the top two rungs.')

print('\n2. RANGE COMPARISON')
o = [st.median([med(c, e, 'avg_obs_loss') for e in KEPT]) for c in L]
r = [st.median([med(c, e, 'obs_resid_rms') for e in KEPT]) for c in L]
n = [st.median([med(c, e, 'nRMSE') for e in KEPT]) for c in L]
print(f'   objective spans {o[0]/o[-1]:.1f}x ; residual {r[0]/r[-1]:.1f}x ; '
      f'nRMSE spans {max(n)/min(n):.2f}x (i.e. {100*(max(n)/min(n)-1):.0f}%)')

print('\n3. WHICH EPISODE FAILS PRIMARY-2, and what is the target region worth?')


def spear(a, b):
    ra = [sorted(a).index(x) for x in a]
    rb = [sorted(b).index(x) for x in b]
    ma, mb = sum(ra) / 5, sum(rb) / 5
    num = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    den = math.sqrt(sum((x - ma) ** 2 for x in ra) * sum((y - mb) ** 2 for y in rb))
    return num / den if den else float('nan')


for e in KEPT:
    rho = spear([med(c, e, 'obs_resid_rms') for c in L],
                [med(c, e, 'avg_RMSE_observed') for c in L])
    if rho < 0.9:
        print(f'   FAIL  {e[8:-17]:22s} rho={rho:+.3f}  '
              f'RMSE_obs ladder=' + ' '.join(f'{med(c,e,"avg_RMSE_observed"):.4f}' for c in L))
print('   NOTE: the PRIMARY-2 target avg_RMSE_observed uses min_dist=10 m, which the')
print('   geometry audit measures at 3.5-6.7% of the grid.  SURROGATE-VALID therefore')
print('   reads: the GT-free residual ranks the GT error inside ~5% of the grid.')

print('\n4. PROCESS: what was frozen and what was not')
for f in ('PREREG.md', 'PREREG_frozen_md5.txt', 'run_eb.py', 'results_lane1.jsonl',
          'analyze_eb.py', 'results_lane2.jsonl', 'analysis.txt'):
    p = os.path.join(HERE, f)
    ts = subprocess.run(['stat', '-c', '%y', p], capture_output=True,
                        text=True).stdout.strip()[:19]
    md5 = subprocess.run(['md5sum', p], capture_output=True,
                         text=True).stdout.split()[0]
    frozen = md5 in open(os.path.join(HERE, 'PREREG_frozen_md5.txt')).read()
    print(f'   {f:24s} {ts}  frozen={frozen}')
print('   => PREREG.md and run_eb.py are md5-frozen.  analyze_eb.py is NOT, and it was')
print('      written 41 min AFTER results_lane1.jsonl existed.  The decision rules it')
print('      implements are in the frozen PREREG and my independent re-derivation')
print('      reproduces every headline number, so the freedom was not exercised --')
print('      but it was available.')

print('\n5. THRESHOLD TABLE IS NOT GERRYMANDERED')
for n in (12, 13, 14, 15, 16, 30, 31, 32):
    need = {32: 22, 31: 22, 30: 21, 16: 13, 15: 12, 14: 12, 13: 11, 12: 10}[n]
    print(f'   n={n:2d} need={need}  p(need)={sgn_p(need,n):.4f}  '
          f'p(need-1)={sgn_p(need-1,n):.4f}   (need = exact alpha=0.05 boundary: '
          f'{sgn_p(need,n)<=0.05 and sgn_p(need-1,n)>0.05})')
