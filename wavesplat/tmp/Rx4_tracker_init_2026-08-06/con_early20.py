#!/usr/bin/env python3
"""CON: is T-EARLY20 a distinct mechanism, or a strictly-dominated DOSE of the floor knob?"""
import json, math, os, statistics as st
from collections import defaultdict
D = os.path.dirname(os.path.abspath(__file__))
ROWS = [json.loads(l) for l in open(os.path.join(D, 'results.jsonl')) if l.strip()]
NULLS = {r['episode']: r['zero_null_tracking_nRMSE_mean']
         for r in json.load(open(os.path.join(D, 'nulls_full.json')))}
CROWDED = [f'episode_BF{b}_{c}comp_seed{s}_uniform.h5'
           for b in (4, 5) for c in (15, 30) for s in range(4)]
BY = defaultdict(list)
for r in ROWS: BY[(r['cell'], r['episode'])].append(r)
def med(xs):
    xs = sorted(xs); n = len(xs)
    return xs[n//2] if n % 2 else .5*(xs[n//2-1]+xs[n//2])
def epmed(c, e, m='nRMSE'): return med([x[m] for x in BY[(c, e)]])
def signp(k, n):
    pm = lambda i: math.comb(n, i)/2.0**n
    o = pm(k); return min(1.0, sum(pm(i) for i in range(n+1) if pm(i) <= o*(1+1e-12)))
KEEP = [e for e in CROWDED if epmed('AUTO', e) < NULLS[e]]

print('MECHANISM: run_rx4.py:266-275 -- rx4_track_frames sets EVERY Adam param-group lr to 0.0')
print('from update call #20 onward.  AUTO-NOTRACK sets tracking_lr=0.0, i.e. lr 0 from call #0.')
print('=> T-EARLY20 and AUTO-NOTRACK are THE SAME KNOB at two doses (freeze-at-0 vs freeze-at-20).')
print('   PREREG sec.2 says this in as many words.  T-EARLY20 is therefore not a new mechanism;')
print('   it is a partial application of the floor, and the floor is the full application.')
print()
print('DOSE-RESPONSE: constant-rate prediction  D_pred(ep) = (1 - 20/T(ep)) * D_floor(ep)')
print(f'{"episode":24s} {"T":>4s} {"D_floor":>9s} {"D_pred":>9s} {"D_obs":>9s} {"obs-pred":>9s}')
res = []
for e in KEEP:
    T = len(BY[('AUTO', e)][0]['vr_frames'])
    df = epmed('AUTO-NOTRACK', e) - epmed('AUTO', e)
    dp = (1 - 20.0/T) * df
    do = epmed('T-EARLY20', e) - epmed('AUTO', e)
    res.append((do - dp, df, dp, do, T, e))
    print(f'{e.replace("episode_","").replace("_uniform.h5",""):24s} {T:4d} {df:+9.4f} {dp:+9.4f} {do:+9.4f} {do-dp:+9.4f}')
diffs = [r[0] for r in res]
w = sum(1 for d in diffs if d > 0)     # obs WORSE (less negative) than constant-rate prediction
print(f'\nmed(obs - pred) = {med(diffs):+.4f}; obs is worse than the constant-rate model in {w}/14, '
      f'p={signp(w,14):.4f}')
print('=> the dose-response is CONVEX/front-loaded: the first 20 frames (12.5-16.7% of the')
print('   window) carry ~40% of the damage.  diag_analysis.txt sec.2 described the AUTO error')
print('   curve as a near-constant slope (1.8-3.9e-4/frame, "no jumps"); a constant slope')
print('   predicts 83-88% recovery, and 60% was observed.  So the time-structure diagnosis')
print('   under-describes the intervention too -- not only the channel decomposition (D1).')
print('   CAVEAT: freezing at frame 20 does not stop the field from evolving, so part of the')
print('   shortfall may be post-freeze model error rather than front-loaded damage.  These two')
print('   are NOT separated by this experiment.')
print()
print('DOMINANCE: is there ANY episode where EARLY20 beats the floor by more than seed noise?')
for e in KEEP:
    d = epmed('T-EARLY20', e) - epmed('AUTO-NOTRACK', e)
    sd = st.pstdev([x['nRMSE'] for x in BY[('T-EARLY20', e)]] + [x['nRMSE'] for x in BY[('AUTO-NOTRACK', e)]])
    print(f'  {e.replace("episode_","").replace("_uniform.h5",""):24s} EARLY20-FLOOR = {d:+.4f}  '
          f'pooled seed sd = {sd:.4f}   {"EARLY20 better" if d<0 else "floor better"}'
          f'{"  (|d| > sd)" if abs(d) > sd else ""}')
print()
print('OTHER METRICS: does EARLY20 beat the floor anywhere in the frozen panel?')
NEG = {'nRMSE', 'absVR_framewise', 'm_mean', 'decorr_mean', 'shrink_mean'}
for m in ['nRMSE', 'Pearson_rho', 'absVR_framewise', 'm_mean', 'decorr_mean', 'shrink_mean']:
    ds = [epmed('T-EARLY20', e, m) - epmed('AUTO-NOTRACK', e, m) for e in KEEP]
    w = sum(1 for d in ds if (d < 0 if m in NEG else d > 0))
    print(f'  {m:17s} medD(EARLY20 - FLOOR) = {med(ds):+.4f}  EARLY20 better {w}/14  p={signp(w,14):.4f}')
