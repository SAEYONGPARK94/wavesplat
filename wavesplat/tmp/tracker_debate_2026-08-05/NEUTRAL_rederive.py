"""NEUTRAL re-derivation for proposition G (tracker = net loss?).

No new wave runs. Reads only:
  tmp/s1s2_2026-08-05/results.jsonl      (132 runs, 11 arms x 4 ep x 3 seeds)
  tmp/rerun_2026-08-03/{ablation_seeded_rerun,p1_comparison_rerun}.json
Run from the repo root:  python3 tmp/tracker_debate_2026-08-05/NEUTRAL_rederive.py
"""
import json, re, statistics as st
from pathlib import Path
import numpy as np

S1S2 = Path('tmp/s1s2_2026-08-05/results.jsonl')
R83 = Path('tmp/rerun_2026-08-03')
EPS = ['episode_BF4_15comp_seed0_uniform.h5', 'episode_BF4_30comp_seed0_uniform.h5',
       'episode_BF5_15comp_seed0_uniform.h5', 'episode_BF5_30comp_seed0_uniform.h5']
short = lambda e: e.split('_')[1] + '_' + e.split('_')[2]

rs = [json.loads(l) for l in open(S1S2)]
med = lambda a, e, k: float(np.median([r[k] for r in rs
                                       if r['arm'] == a and r['episode'] == e and k in r]))
TRACK_ARMS = sorted({r['arm'] for r in rs if 'notrack' not in r['arm']})


def sec(t):
    print('\n' + '=' * 78 + f'\n{t}\n' + '=' * 78)


# --- 1. the notrack arm reproduces its own init score -------------------------
sec('1. notrack arm score == its own init score (the arm adds no new column)')
for e in EPS:
    for a in ('X-A0-notrack', 'X-T1f-notrack'):
        i, t = med(a, e, 'init_nrmse'), med(a, e, 'nRMSE')
        print(f'  {short(e):14s} {a:15s} init {i:.4f} -> tracked {t:.4f} ({(t-i)/i*100:+5.1f}%)')

# --- 2. the tracker is an affine contraction ---------------------------------
sec('2. tracked = a + b*init over the 9 tracking-ON arms, per episode')
print(f"{'episode':14s} {'b':>7s} {'a':>7s} {'fixed pt':>9s} {'r':>7s} {'atten':>7s}")
bs = []
for e in EPS:
    xs = np.array([med(a, e, 'init_nrmse') for a in TRACK_ARMS])
    ys = np.array([med(a, e, 'nRMSE') for a in TRACK_ARMS])
    b, a0 = np.polyfit(xs, ys, 1)
    r = np.corrcoef(xs, ys)[0, 1]
    within = np.median([np.std([q['init_nrmse'] for q in rs
                                if q['arm'] == a and q['episode'] == e], ddof=1)
                        for a in TRACK_ARMS])
    atten = xs.std(ddof=1) ** 2 / (xs.std(ddof=1) ** 2 + within ** 2)
    bs.append(b)
    print(f'  {short(e):14s} {b:7.3f} {a0:7.3f} {a0/(1-b):9.3f} {r:7.3f} {atten:7.3f}')
print(f'  median b = {np.median(bs):.3f}  (attenuation-corrected ~ {np.median(bs)/0.92:.3f})')

sec('3. contraction model predicts the measured tracker harm')
for e in EPS:
    xs = np.array([med(a, e, 'init_nrmse') for a in TRACK_ARMS])
    ys = np.array([med(a, e, 'nRMSE') for a in TRACK_ARMS])
    b, a0 = np.polyfit(xs, ys, 1); fp = a0 / (1 - b)
    for arm, ctrl in (('S2-T1f', 'X-T1f-notrack'), ('S1-A0', 'X-A0-notrack')):
        i = med(arm, e, 'init_nrmse')
        obs = med(arm, e, 'nRMSE') - med(ctrl, e, 'nRMSE')
        print(f'  {short(e):14s} {arm:7s} init {i:.4f}  predicted {(1-b)*(fp-i):+.4f}  measured {obs:+.4f}')

# --- 4. panel + paired deltas + seed noise -----------------------------------
sec('4. the 4 cells, paired deltas, seed range')
for k in ('nRMSE', 'Variance_Ratio', 'Hs_error', 'Pearson_rho'):
    for ctrl, trt in (('S2-T1f', 'X-T1f-notrack'), ('S1-A0', 'X-A0-notrack')):
        d = [med(trt, e, k) - med(ctrl, e, k) for e in EPS]
        print(f'  {trt:15s} - {ctrl:7s} {k:15s} med {np.median(d):+.4f}  '
              f'neg {sum(x < 0 for x in d)}/4  ' + ' '.join(f'{x:+.4f}' for x in d))
print()
for a in ('S1-A0', 'X-A0-notrack', 'S2-T1f', 'X-T1f-notrack'):
    rng = [max(v) - min(v) for v in ([r['nRMSE'] for r in rs
           if r['arm'] == a and r['episode'] == e] for e in EPS)]
    print(f'  {a:15s} nRMSE seed range median {np.median(rng):.4f}')

# --- 5. regime dependence from the 2026-08-03 rerun --------------------------
sec('5. init -> tracked degradation across regimes (no new runs)')
rows = [('ABL-Full', r['episode'], r['init_nrmse']['median'], r['avg_nRMSE']['median'])
        for r in json.load(open(R83 / 'ablation_seeded_rerun.json')) if r['config'] == 'Full']
for ep, v in json.load(open(R83 / 'p1_comparison_rerun.json')).items():
    for arm in ('WS-P0', 'WS-P1'):
        sr = v[arm]['seed_results']
        rows.append((arm, ep, st.median([s['init_nrmse'] for s in sr]),
                     st.median([s['avg_nRMSE'] for s in sr])))
nc = lambda e: int(re.search(r'_(\d+)comp', e).group(1))
buckets = {}
for a, ep, i, t in rows:
    buckets.setdefault('sparse (<=5comp)' if nc(ep) <= 5 else 'crowded (15/30comp)',
                       []).append((t - i, (t - i) / i * 100, i))
for k, v in buckets.items():
    print(f'  {k:22s} n={len(v):2d}  median abs {np.median([x[0] for x in v]):+.4f}  '
          f'median rel {np.median([x[1] for x in v]):+6.1f}%  '
          f'worse {sum(x[0] > 0 for x in v)}/{len(v)}  median init {np.median([x[2] for x in v]):.4f}')
ii = np.array([x[2] for v in buckets.values() for x in v])
rr = np.array([x[1] for v in buckets.values() for x in v])
from scipy.stats import spearmanr
print(f'  Spearman(init_nrmse, relative damage) = {spearmanr(ii, rr).correlation:+.3f}  n={len(ii)}')
