"""Analysis of CON_probe.jsonl (attacks 2-empirical, 4-lever, 5-regime)."""
import json, os
import numpy as np
from math import comb

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = []


def P(*a):
    s = ' '.join(str(x) for x in a); print(s); OUT.append(s)


R = [json.loads(l) for l in open(os.path.join(HERE, 'CON_probe.jsonl'))]
ok = [r for r in R if 'error' not in r]
bad = [r for r in R if 'error' in r]
P(f'runs {len(R)}  ok {len(ok)}  errors {len(bad)}')


def sp(k, n):
    return min(1.0, 2 * sum(comb(n, i) for i in range(0, min(k, n - k) + 1)) / 2 ** n) if n else float('nan')


def med(cell, ep, m):
    v = [r[m] for r in ok if r['cell'] == cell and r['episode'] == ep
         and np.isfinite(r.get(m, np.nan))]
    return float(np.median(v)) if v else float('nan')


def raww(cell, ep, m):
    return [r[m] for r in ok if r['cell'] == cell and r['episode'] == ep
            and np.isfinite(r.get(m, np.nan))]


def contrast(a, b, m, eps, nb=6000, seed=11):
    d = [med(b, e, m) - med(a, e, m) for e in eps]
    k = sum(1 for v in d if v < 0)
    rng = np.random.default_rng(seed)
    bs = np.empty(nb)
    for i in range(nb):
        bs[i] = np.median([np.median(rng.choice(raww(b, e, m), len(raww(b, e, m))))
                           - np.median(rng.choice(raww(a, e, m), len(raww(a, e, m))))
                           for e in eps])
    return np.median(d), k, len(eps), sp(k, len(eps)), \
        (float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))), d


CROWD = ['episode_BF4_15comp_seed0_uniform.h5', 'episode_BF4_30comp_seed0_uniform.h5',
         'episode_BF5_15comp_seed0_uniform.h5', 'episode_BF5_30comp_seed0_uniform.h5']
REG = [e for e in ['episode_BF2_1comp_seed0_uniform.h5',
                   'episode_BF4_1comp_seed0_uniform.h5',
                   'episode_BF3_2comp_seed0_uniform.h5',
                   'episode_BF3_5comp_seed0_uniform.h5',
                   'episode_BF4_5comp_seed0_uniform.h5']
       if any(r['episode'] == e for r in ok)]

P('')
P('=' * 104)
P('BIT-IDENTITY vs tmp/s1s2_2026-08-05 (my harness must reproduce it exactly)')
P('=' * 104)
s = []
for line in open(os.path.join(HERE, '..', 's1s2_2026-08-05', 'results.jsonl')):
    try:
        r = json.loads(line)
    except json.JSONDecodeError:
        continue
    if 'error' not in r:
        s.append(r)
MAP = {'Q-DEP': 'S1-A0', 'Q-ORC': 'S2-T1', 'Q-ORCFRZ': 'S2-T1f'}
nmatch = ndiff = 0
worst = 0.0
for r in ok:
    if r['cell'] not in MAP:
        continue
    ref = [x for x in s if x['arm'] == MAP[r['cell']] and x['episode'] == r['episode']
           and x['seed'] == r['seed']]
    if not ref:
        continue
    d = abs(ref[0]['nRMSE'] - r['nRMSE'])
    worst = max(worst, d)
    if d < 1e-9:
        nmatch += 1
    else:
        ndiff += 1
        if ndiff <= 5:
            P(f'  MISMATCH {r["job_id"]}  mine {r["nRMSE"]:.6f} vs s1s2 {ref[0]["nRMSE"]:.6f}')
P(f'  matched {nmatch}  mismatched {ndiff}  max|diff| {worst:.3e}')

P('')
P('=' * 104)
P('ATTACK 4 (lever OUTSIDE the search space) — freeze k at the BEAMFORM estimate')
P('=' * 104)
P('Q-DEPFRZ = shipped ladder [3,8,15] + init_k_lr_scale=0.0. Never run before.')
P(f'{"metric":18s} {"medD":>9s} {"k/n":>6s} {"p":>7s} {"95% CI(seed)":>21s}')
for m in ['init_nrmse', 'nRMSE', 'Pearson_rho', 'Hs_error']:
    md, k, n, p, ci, d = contrast('Q-DEP', 'Q-DEPFRZ', m, CROWD)
    P(f'{m:18s} {md:+9.4f} {k:3d}/{n:<2d} {p:7.4f} [{ci[0]:+8.4f},{ci[1]:+8.4f}]   '
      f'per-ep {[round(v,4) for v in d]}')
P('')
P(f'{"cell":12s} {"init_nrmse":>11s} {"nRMSE":>8s} {"rho":>7s} {"|VR-1|":>8s} {"Hs_err":>8s} {"N":>5s}')
for c in ['Q-DEP', 'Q-DEPFRZ', 'Q-ORC', 'Q-ORCFRZ']:
    vi = np.median([med(c, e, 'init_nrmse') for e in CROWD])
    vn = np.median([med(c, e, 'nRMSE') for e in CROWD])
    vr = np.median([med(c, e, 'Pearson_rho') for e in CROWD])
    vv = np.median([abs(med(c, e, 'Variance_Ratio') - 1) for e in CROWD])
    vh = np.median([med(c, e, 'Hs_error') for e in CROWD])
    vN = np.median([med(c, e, 'N') for e in CROWD])
    P(f'{c:12s} {vi:11.4f} {vn:8.4f} {vr:7.4f} {vv:8.4f} {vh:8.4f} {vN:5.1f}')

P('')
P('=' * 104)
P('ATTACK 2 (empirical) — can the OBSERVATIONS tell the placements apart?')
P('=' * 104)
P('obs residual RMS [m]. `heldout` = model evaluated on a frame BEFORE it is used.')
P('Hs of these episodes: '
  + ', '.join(f'{e.split("_")[1]}_{e.split("_")[2]}={med(c,e,"Hs_true"):.2f}m'
              for c in ['Q-DEP'] for e in CROWD))
P('')
P(f'{"cell":12s} {"init_obs_in":>12s} {"init_obs_out":>13s} {"trk_obs_in":>11s} '
  f'{"trk_obs_out":>12s} | {"init_nrmse":>10s} {"nRMSE":>8s}')
for c in ['Q-DEP', 'Q-DEPFRZ', 'Q-ORC', 'Q-ORCFRZ']:
    row = [np.median([med(c, e, k) for e in CROWD]) for k in
           ('init_obs_rms_insample', 'init_obs_rms_heldout1',
            'track_obs_rms_insample', 'track_obs_rms_heldout',
            'init_nrmse', 'nRMSE')]
    P(f'{c:12s} {row[0]:12.5f} {row[1]:13.5f} {row[2]:11.5f} {row[3]:12.5f} | '
      f'{row[4]:10.4f} {row[5]:8.4f}')
P('')
for m in ['init_obs_rms_insample', 'init_obs_rms_heldout1', 'track_obs_rms_heldout']:
    md, k, n, p, ci, d = contrast('Q-DEP', 'Q-ORCFRZ', m, CROWD)
    P(f'  Q-DEP -> Q-ORCFRZ  {m:26s} {md:+.5f}  {k}/{n}  p={p:.3f}  '
      f'CI [{ci[0]:+.5f},{ci[1]:+.5f}]')
P('')
P('  ratio (oracle obs residual) / (deployed obs residual), and the same ratio in FIELD error:')
for m, lab in [('init_obs_rms_insample', 'obs in-sample'),
               ('init_obs_rms_heldout1', 'obs held-out'),
               ('init_nrmse', 'FIELD init_nrmse')]:
    a = np.median([med('Q-DEP', e, m) for e in CROWD])
    b = np.median([med('Q-ORCFRZ', e, m) for e in CROWD])
    P(f'    {lab:20s} deployed {a:.5f}  oracle {b:.5f}  ratio {a/b:6.2f}x')

P('')
P('=' * 104)
P('ATTACK 5 — the same contrast at n_comp = 1 and 5 (BF4 and BF5, sea state held)')
P('=' * 104)
P(f'{"episode":40s} {"n_gt":>4s} {"R-DEP":>8s} {"R-DEPFRZ":>9s} {"R-ORCFRZ":>9s} '
  f'{"gap DEP-ORC":>12s}')
for e in REG:
    n_gt = int(med('R-DEP', e, 'n_gt'))
    a = med('R-DEP', e, 'nRMSE'); b = med('R-DEPFRZ', e, 'nRMSE')
    c = med('R-ORCFRZ', e, 'nRMSE')
    P(f'{e:40s} {n_gt:4d} {a:8.4f} {b:9.4f} {c:9.4f} {a-c:12.4f}')
P('')
P(f'{"episode":40s} {"init R-DEP":>11s} {"init R-ORCFRZ":>14s} {"init gap":>9s} {"ratio":>7s}')
for e in REG:
    a = med('R-DEP', e, 'init_nrmse'); c = med('R-ORCFRZ', e, 'init_nrmse')
    P(f'{e:40s} {a:11.4f} {c:14.4f} {a-c:9.4f} {a/c if c else float("nan"):7.2f}x')
P('')
gaps_r = [med('R-DEP', e, 'nRMSE') - med('R-ORCFRZ', e, 'nRMSE') for e in REG]
P(f'  sparse/moderate (n_comp 1 and 5): median oracle prize in nRMSE = '
  f'{np.median(gaps_r):+.4f}')
P(f'  crowded (n_comp 15 and 30)      : median oracle prize in nRMSE = '
  f'{np.median([med("Q-DEP", e, "nRMSE") - med("Q-ORCFRZ", e, "nRMSE") for e in CROWD]):+.4f}')
P(f'  ratio = '
  f'{np.median([med("Q-DEP", e, "nRMSE") - med("Q-ORCFRZ", e, "nRMSE") for e in CROWD])/np.median(gaps_r):.1f}x')
P('')
P('=' * 104)
P('OBS-SPACE vs FIELD-SPACE coupling across the 4 crowded cells')
P('=' * 104)
xs, ys, labs = [], [], []
for c in ['Q-DEP', 'Q-DEPFRZ', 'Q-ORC', 'Q-ORCFRZ']:
    for e in CROWD:
        a = med(c, e, 'init_obs_rms_heldout1'); b = med(c, e, 'init_nrmse')
        if np.isfinite(a) and np.isfinite(b) and a > 0 and b > 0:
            xs.append(np.log(a)); ys.append(np.log(b)); labs.append((c, e))
xs, ys = np.array(xs), np.array(ys)
sl, ic = np.polyfit(xs, ys, 1)
P(f'  log(field init_nrmse) = {sl:+.3f} * log(held-out obs residual) {ic:+.3f}   '
  f'r = {np.corrcoef(xs, ys)[0,1]:+.4f}  n={len(xs)}')
P('  Spearman rank agreement between the GT-free obs residual and the GT field error:')
from scipy.stats import spearmanr
P(f'    rho_s = {spearmanr(np.exp(xs), np.exp(ys)).statistic:+.4f}')
P('  (LiDAR elevation noise floor sigma = 0.02 m -- the oracle arm sits ON it, the')
P('   deployed arm sits 3.9-5.2x above it, i.e. the data still contain unexplained signal.)')
P('')
P('  and the k-freezing lever in the non-crowded regime:')
for m in ['init_nrmse', 'nRMSE']:
    md, k, n, p, ci, d = contrast('R-DEP', 'R-DEPFRZ', m, REG)
    P(f'    R-DEP -> R-DEPFRZ  {m:12s} {md:+.4f}  {k}/{n}  p={p:.3f}  '
      f'CI [{ci[0]:+.4f},{ci[1]:+.4f}]  per-ep {[round(v,4) for v in d]}')

open(os.path.join(HERE, 'CON_probe_analysis.txt'), 'w').write('\n'.join(OUT) + '\n')
