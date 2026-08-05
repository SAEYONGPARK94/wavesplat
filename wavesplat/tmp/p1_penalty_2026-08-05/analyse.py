"""Analysis of tmp/p1_penalty_2026-08-05/results.jsonl against PREREG.md (frozen)."""
import json
import math
import os
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
R = [json.loads(l) for l in open(os.path.join(HERE, 'results.jsonl'))]
OLD = os.path.join(HERE, '..', 'rerun_2026-08-03', 'progress.jsonl')

EPS = ['BF2_1comp_seed0_uniform', 'BF3_1comp_seed0_uniform',
       'BF3_2comp_seed0_uniform', 'BF3_2comp_seed0_slope',
       'BF3_5comp_seed0_uniform', 'BF3_5comp_seed0_slope',
       'BF4_1comp_seed0_uniform', 'BF4_5comp_seed0_uniform',
       'BF4_15comp_seed0_uniform', 'BF4_30comp_seed0_uniform',
       'BF5_15comp_seed0_uniform', 'BF5_30comp_seed0_uniform']
SPARSE = EPS[:8]
CROWD = EPS[8:]
SEEDS = [42, 123, 456]

cell = {(r['arm'], r['episode'], r['seed']): r for r in R}


def med(arm, ep, key):
    v = [cell[(arm, ep, s)][key] for s in SEEDS]
    v = [x for x in v if x is not None and np.isfinite(x)]
    return float(np.median(v)) if v else float('nan')


def sign_test(deltas, tol=0.0):
    """two-sided exact sign test; ties dropped."""
    d = [x for x in deltas if abs(x) > tol]
    n = len(d)
    k = sum(1 for x in d if x > 0)
    if n == 0:
        return 0, 0, 1.0
    m = max(k, n - k)
    p = 2 * sum(math.comb(n, i) for i in range(m, n + 1)) / 2 ** n
    return k, n, min(1.0, p)


out = []
W = out.append

# ------------------------------------------------------------------ gate 1+2
W('=' * 92)
W('GATE 1 -- bit-identity of CTRL vs tmp/rerun_2026-08-03 (WS-P1, same config)')
W('=' * 92)
old = {}
for line in open(OLD):
    o = json.loads(line)
    if o['tag'].startswith('p1/') and o['tag'].endswith('WS-P1') and 'error' not in o:
        old[(o['episode'], o['seed'])] = o
worst = 0.0
worst_i = 0.0
nchk = 0
for ep in EPS:
    for s in SEEDS:
        o = old.get((ep, s))
        if not o:
            continue
        nchk += 1
        worst = max(worst, abs(cell[('CTRL', ep, s)]['avg_nRMSE'] - o['avg_nRMSE']))
        worst_i = max(worst_i, abs(cell[('CTRL', ep, s)]['init_nrmse'] - o['init_nrmse']))
W(f'  checked {nchk}/36 cells   max |d avg_nRMSE| = {worst:.3e}   '
  f'max |d init_nrmse| = {worst_i:.3e}')
W(f'  -> {"PASS" if worst < 1e-6 else "FAIL"} (log stores 6 dp; init_nrmse is full precision)')

W('')
W('GATE 2 -- Phase 1 untouched: init_nrmse must be IDENTICAL for CTRL and P1')
d_init = max(abs(cell[('CTRL', e, s)]['init_nrmse'] - cell[('P1', e, s)]['init_nrmse'])
             for e in EPS for s in SEEDS)
W(f'  max |init_nrmse(CTRL) - init_nrmse(P1)| = {d_init:.3e}  '
  f'-> {"PASS" if d_init == 0.0 else "FAIL"}')
d_initb = max(abs(cell[('CTRL', e, s)]['init_nrmse'] - cell[('P1-both', e, s)]['init_nrmse'])
              for e in EPS for s in SEEDS)
W(f'  (P1-both, expected to DIFFER since it changes Phase 1 too): '
  f'max diff = {d_initb:.3e}')

# ------------------------------------------------------------------ primary
W('')
W('=' * 92)
W('PRIMARY -- avg_nRMSE, median-of-3, paired  Delta = CTRL - P1  (Delta>0 = P1 better)')
W('=' * 92)
W(f'{"#":<3}{"episode":<26}{"init":>8}{"CTRL":>9}{"P1":>9}{"Delta":>9}'
  f'{"P-A pred":>10}{"P-B pred":>10}{"|D|>MDE?":>10}')
rows = []
for i, ep in enumerate(EPS, 1):
    ini = med('CTRL', ep, 'init_nrmse')
    c = med('CTRL', ep, 'avg_nRMSE')
    p = med('P1', ep, 'avg_nRMSE')
    d = c - p
    pa = 0.1248 - 0.4911 * ini
    gb_ctrl = c - ini
    pb = gb_ctrl - (-0.0018 + 0.0550 * ini)
    rows.append(dict(ep=ep, init=ini, ctrl=c, p1=p, d=d, pa=pa, pb=pb, gb=gb_ctrl))
    W(f'{i:<3}{ep:<26}{ini:>8.4f}{c:>9.4f}{p:>9.4f}{d:>+9.4f}{pa:>+10.4f}{pb:>+10.4f}'
      f'{("yes" if abs(d) > 0.0131 else "no"):>10}')
D = np.array([r['d'] for r in rows])
k, n, pv = sign_test(list(D))
W(f'  median Delta = {np.median(D):+.4f}   mean = {D.mean():+.4f}   '
  f'sign {k}/{n}   two-sided p = {pv:.4f}   '
  f'-> {"SIGNIFICANT (>=10/12)" if k >= 10 and pv <= 0.05 else "NOT significant at a=0.05"}')
try:
    from scipy.stats import wilcoxon
    W(f'  Wilcoxon signed-rank (secondary): p = {wilcoxon(D).pvalue:.4f}')
except Exception:
    pass

W('')
W('  per-seed pairs (n=36, not the pre-registered unit -- reported for spread only):')
Ds = np.array([cell[('CTRL', e, s)]['avg_nRMSE'] - cell[('P1', e, s)]['avg_nRMSE']
               for e in EPS for s in SEEDS])
k2, n2, p2 = sign_test(list(Ds))
W(f'    median {np.median(Ds):+.4f}   sign {k2}/{n2}   p = {p2:.2e}')

# --------------------------------------------------- give-back law verdict
W('')
W('=' * 92)
W('PREDICTION CHECK -- give-back law (P-A) vs law-calibrated (P-B)')
W('=' * 92)
for tag, sub in (('all 12', rows), ('sparse 8', [r for r in rows if r['ep'] in SPARSE]),
                 ('crowded 4', [r for r in rows if r['ep'] in CROWD])):
    d = np.array([r['d'] for r in sub])
    pa = np.array([r['pa'] for r in sub])
    pb = np.array([r['pb'] for r in sub])
    W(f'  {tag:<11} observed median Delta {np.median(d):+.4f} | '
      f'P-A {np.median(pa):+.4f} | P-B {np.median(pb):+.4f} | '
      f'MAE vs P-A {np.abs(d - pa).mean():.4f}  vs P-B {np.abs(d - pb).mean():.4f}')
sp = np.median([r['d'] for r in rows if r['ep'] in SPARSE])
W(f'  pre-registered rule on SPARSE median Delta: >=+0.060 -> P-A ; <=+0.013 -> P-B')
W(f'  observed sparse median = {sp:+.4f}  ==>  '
  f'{"P-A" if sp >= 0.060 else ("P-B" if sp <= 0.013 else "UNDECIDED")}')
gb_obs = np.array([r['gb'] for r in rows])
gb_law = np.array([0.1230 - 0.4361 * r['init'] for r in rows])
W(f'  CTRL give-back: observed median {np.median(gb_obs):+.4f} vs law {np.median(gb_law):+.4f} '
  f'(law over-predicts {np.mean(gb_law) / max(np.mean(gb_obs), 1e-9):.1f}x on the mean)')
ii = np.array([r['init'] for r in rows])
W(f'  refit on this arm/set: gb = {np.polyfit(ii, gb_obs, 1)[1]:+.4f} '
  f'{np.polyfit(ii, gb_obs, 1)[0]:+.4f}*init   r = {np.corrcoef(ii, gb_obs)[0, 1]:+.3f}'
  f'   (law slope was -0.4361, r=-0.896)')
# does the law hold for P1's own give-back?
gbp = np.array([med('P1', e, 'avg_nRMSE') - med('P1', e, 'init_nrmse') for e in EPS])
W(f'  P1 give-back: median {np.median(gbp):+.4f}  (frozen-arm line predicts '
  f'{np.median([-0.0018 + 0.0550 * r["init"] for r in rows]):+.4f})')

# ------------------------------------------------------------------ safety
W('')
W('SAFETY (pre-registered): episodes where P1 is worse than CTRL by more than MDE 0.0131')
bad = [r for r in rows if r['d'] < -0.0131]
W(f'  n = {len(bad)}  -> {"SAFE" if not bad else "NOT_SAFE"}')
for r in bad:
    W(f'    {r["ep"]}  Delta {r["d"]:+.4f}')
W(f'  worst single episode: {min(rows, key=lambda r: r["d"])["ep"]} '
  f'{min(r["d"] for r in rows):+.4f}')

# ------------------------------------------------------------------ panel
W('')
W('=' * 92)
W('FULL PANEL -- median-of-3 per episode, paired sign test over 12 episodes')
W('=' * 92)
PANEL = [('avg_nRMSE', 'lower', -1), ('rho', 'higher', +1), ('SS', 'higher', +1),
         ('DirR2', 'higher', +1), ('avg_Hs_error', 'lower', -1),
         ('det_rate', 'higher', +1), ('energy_det_rate', 'higher', +1),
         ('RMSE_unobs', 'lower', -1), ('obs_loss', 'lower', -1)]
W(f'{"metric":<16}{"better":<8}{"CTRL med":>10}{"P1 med":>10}{"med Delta":>11}'
  f'{"sign":>8}{"p":>9}')
for key, better, sgn in PANEL:
    c = np.array([med('CTRL', e, key) for e in EPS])
    p = np.array([med('P1', e, key) for e in EPS])
    d = sgn * (c - p) * -1 if False else (p - c) * sgn      # >0 = P1 better
    k, n, pv = sign_test(list(d))
    W(f'{key:<16}{better:<8}{np.median(c):>10.4f}{np.median(p):>10.4f}'
      f'{np.median(d):>+11.4f}{f"{k}/{n}":>8}{pv:>9.4f}')
# VR and sum|w| use distance-to-target
for key, target, lab in (('VR', 1.0, '|VR-1|'), ('sum_w_ratio', 1.0, '|sum|w|/sumA - 1|')):
    c = np.array([abs(med('CTRL', e, key) - target) for e in EPS])
    p = np.array([abs(med('P1', e, key) - target) for e in EPS])
    d = c - p
    k, n, pv = sign_test(list(d))
    W(f'{lab:<16}{"lower":<8}{np.median(c):>10.4f}{np.median(p):>10.4f}'
      f'{np.median(d):>+11.4f}{f"{k}/{n}":>8}{pv:>9.4f}')
W(f'{"final_n":<16}{"n/a":<8}{np.median([med("CTRL", e, "final_n") for e in EPS]):>10.1f}'
  f'{np.median([med("P1", e, "final_n") for e in EPS]):>10.1f}')

W('')
W('  raw VR and sum|w|/sum|A| per episode (mechanism):')
W(f'{"episode":<26}{"VR CTRL":>9}{"VR P1":>9}{"VR both":>9}'
  f'{"sw CTRL":>9}{"sw P1":>9}{"sw both":>9}')
for ep in EPS:
    W(f'{ep:<26}{med("CTRL", ep, "VR"):>9.3f}{med("P1", ep, "VR"):>9.3f}'
      f'{med("P1-both", ep, "VR"):>9.3f}'
      f'{med("CTRL", ep, "sum_w_ratio"):>9.3f}{med("P1", ep, "sum_w_ratio"):>9.3f}'
      f'{med("P1-both", ep, "sum_w_ratio"):>9.3f}')
swc = np.array([med('CTRL', e, 'sum_w_ratio') for e in EPS])
swp = np.array([med('P1', e, 'sum_w_ratio') for e in EPS])
k, n, pv = sign_test(list(swp - swc))
W(f'  pre-registered #3: sum|w| ratio P1 > CTRL in {k}/{n}  (needed >=10/12)  p={pv:.4f}')

# ------------------------------------------------------------------ P1-both
W('')
W('=' * 92)
W('NEGATIVE CONTROL P1-both (penalties off in BOTH phases)')
W('=' * 92)
W(f'{"episode":<26}{"init C":>9}{"init both":>10}{"CTRL":>9}{"both":>9}'
  f'{"D=C-both":>10}{"VR both":>9}')
Db = []
for ep in EPS:
    c = med('CTRL', ep, 'avg_nRMSE')
    b = med('P1-both', ep, 'avg_nRMSE')
    Db.append(c - b)
    W(f'{ep:<26}{med("CTRL", ep, "init_nrmse"):>9.4f}'
      f'{med("P1-both", ep, "init_nrmse"):>10.4f}{c:>9.4f}{b:>9.4f}{c - b:>+10.4f}'
      f'{med("P1-both", ep, "VR"):>9.3f}')
Db = np.array(Db)
k, n, pv = sign_test(list(Db))
W(f'  median Delta(CTRL-both) = {np.median(Db):+.4f}   sign {k}/{n} positive   p={pv:.4f}')
W(f'  pre-registered #2: expect Delta<0 in >=9/12 -> observed {n - k}/12 negative')
vrb = np.array([med('P1-both', e, 'VR') for e in EPS])
W(f'  pre-registered #2b: expect VR median > 1.2 -> observed {np.median(vrb):.3f}')
Dpb = np.array([med('P1', e, 'avg_nRMSE') - med('P1-both', e, 'avg_nRMSE') for e in EPS])
k, n, pv = sign_test(list(Dpb))
W(f'  P1 vs P1-both: median {np.median(-Dpb):+.4f} in favour of P1, '
  f'P1 better in {n - k}/{n}, p={pv:.4f}')

# ------------------------------------------------------------------ regime
W('')
W('=' * 92)
W('REGIME SPLIT (pre-registered: sparse should be the biggest beneficiary if P-A holds)')
W('=' * 92)
for lab, sub in (('sparse (1/2/5comp, n=8)', SPARSE), ('crowded (15/30comp, n=4)', CROWD)):
    d = np.array([r['d'] for r in rows if r['ep'] in sub])
    c = np.array([med('CTRL', e, 'avg_nRMSE') for e in sub])
    rel = 100 * d / c
    k, n, pv = sign_test(list(d))
    W(f'  {lab:<26} median Delta {np.median(d):+.4f}  '
      f'({np.median(rel):+.1f}% relative)  P1 better {k}/{n}  p={pv:.3f}')
    W(f'  {"":<26} CTRL med {np.median(c):.4f} -> P1 med '
      f'{np.median([med("P1", e, "avg_nRMSE") for e in sub]):.4f}')

# ------------------------------------------------------------------ mechanism
W('')
W('=' * 92)
W('MECHANISM -- PRO_BRIEF nRMSE identity  nRMSE = sqrt(1 + v - 2*rho*sqrt(v))/4,')
W('            MSE-optimal variance ratio v* = rho^2.  Shrinkage helps iff VR > v*.')
W('=' * 92)
W(f'{"episode":<26}{"rho C":>7}{"rho P1":>8}{"v*=rho2":>9}{"VR C":>7}{"VR P1":>7}'
  f'{"nR C":>8}{"nR P1":>8}{"idty C":>8}{"idty P1":>8}{"det C":>7}')


def idty(r, v):
    return float(np.sqrt(max(1 + v - 2 * r * np.sqrt(v), 0.0)) / 4)


for ep in EPS:
    rc, rp = med('CTRL', ep, 'rho'), med('P1', ep, 'rho')
    vc, vp = med('CTRL', ep, 'VR'), med('P1', ep, 'VR')
    W(f'{ep:<26}{rc:>7.3f}{rp:>8.3f}{rc ** 2:>9.3f}{vc:>7.3f}{vp:>7.3f}'
      f'{med("CTRL", ep, "avg_nRMSE"):>8.4f}{med("P1", ep, "avg_nRMSE"):>8.4f}'
      f'{idty(rc, vc):>8.4f}{idty(rp, vp):>8.4f}{med("CTRL", ep, "det_rate"):>7.2f}')
above = [ep for ep in EPS if med('CTRL', ep, 'VR') > med('CTRL', ep, 'rho') ** 2]
W(f'  episodes where CTRL VR is ALREADY ABOVE v* (shrinkage still pays): '
  f'{len(above)}/12 -> {[e.split("_")[0] + "_" + e.split("_")[1] for e in above]}')
sgn_ok = sum(1 for ep in EPS
             if np.sign(med('CTRL', ep, 'avg_nRMSE') - med('P1', ep, 'avg_nRMSE'))
             == np.sign(idty(med('CTRL', ep, 'rho'), med('CTRL', ep, 'VR'))
                        - idty(med('P1', ep, 'rho'), med('P1', ep, 'VR'))))
W(f'  identity reproduces the SIGN of Delta in {sgn_ok}/12 episodes')

# post-hoc noise / achieved MDE
W('')
W('POST-HOC noise (the PREREG could only bound this):')
pair_sd = []
for ep in EPS:
    v = [cell[('CTRL', ep, s)]['avg_nRMSE'] - cell[('P1', ep, s)]['avg_nRMSE'] for s in SEEDS]
    pair_sd.append(np.std(v, ddof=1))
W(f'  paired-difference seed sd, median over episodes = {np.median(pair_sd):.4f}')
W(f'  (PREREG bound was 0.0052-0.0116 -> achieved MDE '
  f'~{1.13 * 0.67 * np.median(pair_sd):.4f})')
csd = np.median([np.std([cell[("CTRL", ep, s)]["avg_nRMSE"] for s in SEEDS], ddof=1)
                 for ep in EPS])
W(f'  CTRL arm seed sd (median over episodes) = {csd:.4f}  '
  f'(2026-08-03 same arm: 0.0123)')

txt = '\n'.join(out)
print(txt)
open(os.path.join(HERE, 'RESULTS.txt'), 'w').write(txt + '\n')
