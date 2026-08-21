#!/usr/bin/env python3
"""PRO audit part 2 -- (a) knob-count audit of every contrast, (b) additivity /
median-artifact check on the ORACLE-K decomposition, (c) THE k-quality x trust-region
interaction test, (d) MDE / power.  Reads only."""
import json, os, math
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
rows = [json.loads(l) for l in open(os.path.join(HERE, 'results.jsonl')) if l.strip()]
nulls = {d['episode']: d for d in json.load(open(os.path.join(HERE, 'nulls_k6.json')))}
IDX = {(r['cell'], r['episode'], r['seed']): r for r in rows}
EPS = sorted({r['episode'] for r in rows})
SEEDS = sorted({r['seed'] for r in rows})
CELLS = sorted({r['cell'] for r in rows})

def med(v):
    v = sorted(v); n = len(v)
    return v[n // 2] if n % 2 else 0.5 * (v[n // 2 - 1] + v[n // 2])

def sm(cell, ep, field):
    return med([IDX[(cell, ep, s)][field] for s in SEEDS])

def smspec(cell, ep, key):
    v = [IDX[(cell, ep, s)]['spec'][key] for s in SEEDS
         if (IDX[(cell, ep, s)].get('spec') or {}).get(key) is not None]
    return med(v) if v else float('nan')

def binom_two_sided(k, n):
    C = math.comb
    lo = sum(C(n, i) for i in range(0, min(k, n - k) + 1)) / 2 ** n
    return min(1.0, 2 * lo)

KEEP = [e for e in EPS if sm('AUTO', e, 'nRMSE') < nulls[e]['zero_null_tracking_nRMSE_mean']]

print('=' * 78)
print('A. KNOB-COUNT AUDIT  (what actually differs between the cells being contrasted)')
print('=' * 78)
BASE = dict(init_method='v2_konly', n_gaussians=[3, 8, 15],
            loss_thresholds=[0.1, 0.05, None], max_iters_per_stage=[200, 200, 500],
            analytic_ensemble_k=None, init_k_lr_scale=None)
CFG = {
    'AUTO':      dict(BASE, init_method='auto', analytic_ensemble_k=3),
    'AUTO-DUP':  dict(BASE, init_method='auto', analytic_ensemble_k=3),
    'K6-KLR03':  dict(BASE, init_method='auto', analytic_ensemble_k=3, init_k_lr_scale=0.3),
    'K6-KLR01':  dict(BASE, init_method='auto', analytic_ensemble_k=3, init_k_lr_scale=0.1),
    'K6-KLR00':  dict(BASE, init_method='auto', analytic_ensemble_k=3, init_k_lr_scale=0.0),
    'K6-NSCHED': dict(BASE, init_method='v2_konly', n_gaussians='[N_gt]',
                      loss_thresholds=[None], max_iters_per_stage=[900]),
    'ORACLE-K':  dict(BASE, init_method='oracle', n_gaussians='[N_gt]',
                      loss_thresholds=[None], max_iters_per_stage=[900]),
    'ORACLE':    dict(BASE, init_method='oracle', n_gaussians='[N_gt]',
                      loss_thresholds=[None], max_iters_per_stage=[900], init_k_lr_scale=0.0),
}
for a, b in [('K6-KLR01', 'AUTO'), ('K6-KLR00', 'AUTO'), ('ORACLE-K', 'K6-NSCHED'),
             ('ORACLE', 'ORACLE-K'), ('K6-NSCHED', 'AUTO'), ('ORACLE', 'AUTO'),
             ('ORACLE-K', 'AUTO')]:
    diff = [k for k in CFG[a] if CFG[a][k] != CFG[b][k]]
    print(f'  {a:>10s} vs {b:<10s}  knobs={len(diff)}  {diff}')
# empirical confirmation of the ensemble column
print('\n  recorded ensemble_k per cell:',
      {c: sorted({r['ensemble_k'] for r in rows if r['cell'] == c}) for c in CELLS})
print('  recorded auto_selected per cell:',
      {c: dict((k, sum(1 for r in rows if r['cell'] == c and r.get('auto_selected') == k))
               for k in {r.get('auto_selected') for r in rows if r['cell'] == c})
       for c in CELLS})

print()
print('=' * 78)
print('B. ADDITIVITY / MEDIAN ARTIFACT on the ORACLE-K decomposition (block = keep14)')
print('=' * 78)
d_place = [sm('ORACLE-K', e, 'nRMSE') - sm('K6-NSCHED', e, 'nRMSE') for e in KEEP]
d_rest  = [sm('K6-NSCHED', e, 'nRMSE') - sm('AUTO', e, 'nRMSE') for e in KEEP]
d_net   = [sm('ORACLE-K', e, 'nRMSE') - sm('AUTO', e, 'nRMSE') for e in KEEP]
print(f'  median(placement factor)              = {med(d_place):+.4f}')
print(f'  median(rest-of-bundle factor)         = {med(d_rest):+.4f}')
print(f'  sum of the two medians               -> {med(d_place)+med(d_rest):+.4f}')
print(f'  median(net, ORACLE-K - AUTO)          = {med(d_net):+.4f}   <-- EXACT, per-episode')
print(f'  max |per-episode identity residual|   = '
      f'{max(abs((p+r)-n) for p,r,n in zip(d_place,d_rest,d_net)):.2e}  (must be 0)')
print(f'  => medians do NOT add ({med(d_place)+med(d_rest):+.4f} vs {med(d_net):+.4f}); '
      f'report the exact net, not the sum.')
print(f'  |placement| / |net| = {abs(med(d_place))/abs(med(d_net)):.1f}x')

print()
print('=' * 78)
print('C. K-QUALITY x TRUST-REGION INTERACTION')
print('=' * 78)
print('C1. Difference-in-differences across the two placement qualities (same knob,')
print('    init_k_lr_scale 1.0 -> 0.0), on the SAME 14 episodes')
dd_or = [sm('ORACLE', e, 'nRMSE') - sm('ORACLE-K', e, 'nRMSE') for e in KEEP]     # oracle k
dd_bl = [sm('K6-KLR00', e, 'nRMSE') - sm('AUTO', e, 'nRMSE') for e in KEEP]       # blind k
inter = [a - b for a, b in zip(dd_or, dd_bl)]
w = sum(1 for x in inter if x < 0)
print(f'  freeze k at ORACLE placement : median {med(dd_or):+.4f}  '
      f'{sum(1 for x in dd_or if x<0)}/{len(dd_or)}  p={binom_two_sided(sum(1 for x in dd_or if x<0), len(dd_or)):.4f}')
print(f'  freeze k at BLIND  placement : median {med(dd_bl):+.4f}  '
      f'{sum(1 for x in dd_bl if x<0)}/{len(dd_bl)}  p={binom_two_sided(sum(1 for x in dd_bl if x<0), len(dd_bl)):.4f}')
print(f'  INTERACTION (diff-in-diff)   : median {med(inter):+.4f}  '
      f'{w}/{len(inter)}  p={binom_two_sided(w, len(inter)):.4f}')
print('  CAVEAT: the two arms also differ in N / schedule / ensemble, so this is')
print('          "k quality + operating point", not k quality alone.')

print()
print('C2. Dose-response of the trust region at BLIND placement (shipping arm, 1 knob)')
for c in ['AUTO', 'K6-KLR03', 'K6-KLR01', 'K6-KLR00']:
    lv = med([sm(c, e, 'nRMSE') for e in KEEP])
    iv = med([sm(c, e, 'init_nrmse') for e in KEEP])
    print(f'  init_k_lr_scale {"1.0" if c=="AUTO" else c[-2]+"."+c[-1]:>3s}  '
          f'nRMSE {lv:.4f}   init_nrmse {iv:.4f}')
print('  at ORACLE placement:  klr 1.0 -> 0.0 :  '
      f'{med([sm("ORACLE-K",e,"nRMSE") for e in KEEP]):.4f} -> '
      f'{med([sm("ORACLE",e,"nRMSE") for e in KEEP]):.4f}')

print()
print('C3. WITHIN the shipping arm: does the trust region pay off more where AUTO`s')
print('    own k is better?  (stratify by AUTO`s Phase-1 quality)')
def spearman(x, y):
    def rank(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v); i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r
    rx, ry = rank(x), rank(y)
    n = len(x)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = math.sqrt(sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry))
    rho = num / den if den else float('nan')
    t = rho * math.sqrt(max(n - 2, 1) / max(1e-12, 1 - rho ** 2))
    return rho, t

# k-quality proxies measured ON THE CONTROL ARM (AUTO), so they are not caused by the knob
QUAL = {
    'AUTO init_nrmse (lower=better k)':        [sm('AUTO', e, 'init_nrmse') for e in KEEP],
    'AUTO dir_err_deg (lower=better)':         [smspec('AUTO', e, 'direction_error_deg_mean') for e in KEEP],
    'AUTO k_mag_rel_err (lower=better) [desc]':[smspec('AUTO', e, 'k_mag_rel_error_mean') for e in KEEP],
    'AUTO detection_rate (higher=better)[desc]':[smspec('AUTO', e, 'detection_rate') for e in KEEP],
    'AUTO energy_det_rate (higher=better)[desc]':[smspec('AUTO', e, 'energy_detection_rate') for e in KEEP],
    'AUTO nRMSE itself (lower=better)':        [sm('AUTO', e, 'nRMSE') for e in KEEP],
}
for name, q in QUAL.items():
    for lab, d in [('KLR01-AUTO', [sm('K6-KLR01', e, 'nRMSE') - sm('AUTO', e, 'nRMSE') for e in KEEP]),
                   ('KLR03-AUTO', [sm('K6-KLR03', e, 'nRMSE') - sm('AUTO', e, 'nRMSE') for e in KEEP]),
                   ('KLR00-AUTO', [sm('K6-KLR00', e, 'nRMSE') - sm('AUTO', e, 'nRMSE') for e in KEEP])]:
        rho, t = spearman(q, d)
        print(f'  rho( {name:44s}, {lab} ) = {rho:+.3f}   t={t:+.2f}')

print()
print('C4. median split on AUTO init_nrmse (better-k half vs worse-k half)')
order = sorted(range(len(KEEP)), key=lambda i: QUAL['AUTO init_nrmse (lower=better k)'][i])
good = [KEEP[i] for i in order[:7]]
bad = [KEEP[i] for i in order[7:]]
print(f'  better-k half (AUTO init_nrmse <= '
      f'{QUAL["AUTO init_nrmse (lower=better k)"][order[6]]:.4f}) : {len(good)} episodes')
for lab, cell in [('KLR03', 'K6-KLR03'), ('KLR01', 'K6-KLR01'), ('KLR00', 'K6-KLR00')]:
    for hl, blk in [('better-k', good), ('worse-k ', bad)]:
        d = [sm(cell, e, 'nRMSE') - sm('AUTO', e, 'nRMSE') for e in blk]
        wn = sum(1 for x in d if x < 0)
        print(f'    {lab} - AUTO  {hl}: median {med(d):+.4f}  {wn}/{len(d)}  '
              f'p={binom_two_sided(wn, len(d)):.3f}   [n=7, floor p=0.016, DESCRIPTIVE]')

print()
print('C5. how much k damage is there to prevent, per arm?  Phase-1 -> field drift')
for c in ['AUTO', 'K6-KLR01', 'K6-KLR00', 'K6-NSCHED', 'ORACLE-K', 'ORACLE']:
    de = med([smspec(c, e, 'direction_error_deg_mean') for e in KEEP])
    km = med([smspec(c, e, 'k_mag_rel_error_mean') for e in KEEP])
    dr = med([smspec(c, e, 'detection_rate') for e in KEEP])
    print(f'  {c:12s} dir_err {de:6.2f} deg   k_mag_rel {km:.4f}   det {dr:.3f}  [DESCRIPTIVE, sec.7]')

print()
print('=' * 78)
print('D. MDE / POWER  (sign test, two-sided, PREREG frozen thresholds)')
print('=' * 78)
def power(n, req, p):
    C = math.comb
    hi = sum(C(n, i) * p ** i * (1 - p) ** (n - i) for i in range(req, n + 1))
    lo = sum(C(n, i) * p ** i * (1 - p) ** (n - i) for i in range(0, n - req + 1))
    return hi + lo
for n, req in [(14, 12), (16, 13)]:
    print(f'  n={n}, required wins={req}, exact two-sided alpha='
          f'{binom_two_sided(req, n):.4f}')
    for p in (0.70, 0.75, 0.80, 0.85, 0.90, 0.95):
        print(f'    p_win={p:.2f} -> power {power(n, req, p):.3f}')
# translate p_win into an nRMSE effect via the observed paired-delta scale
import statistics as st
d01 = [sm('K6-KLR01', e, 'nRMSE') - sm('AUTO', e, 'nRMSE') for e in KEEP]
sd = st.pstdev(d01)
print(f'\n  observed SD of paired delta (KLR01-AUTO) = {sd:.4f}')
print(f'  observed mean = {st.mean(d01):+.4f}   median = {med(d01):+.4f}')
for p in (0.80, 0.85, 0.90):
    # p_win = Phi(delta/sd) under a symmetric location shift
    from math import erf, sqrt
    z = 0.0; lo_, hi_ = 0.0, 5.0
    for _ in range(200):
        z = 0.5 * (lo_ + hi_)
        cdf = 0.5 * (1 + erf(z / sqrt(2)))
        if cdf < p: lo_ = z
        else: hi_ = z
    print(f'  p_win={p:.2f} corresponds to a shift of {z*sd:.4f} nRMSE '
          f'(power at n=14/req12 = {power(14,12,p):.3f})')
print('\n  => MDE at 80% power, n=14, req 12:  p_win ~ 0.90  ->  ~'
      f'{1.2816*sd:.4f} nRMSE   (observed KLR01 effect: {med(d01):+.4f})')
print(f'  observed |effect| / MDE = {abs(med(d01))/(1.2816*sd):.2f}  '
      '<-- underpowered vs. no effect?')

print()
print('=' * 78)
print('E. WHERE DOES THE ORACLE PRIZE LIVE? (per-episode ORACLE-AUTO vs episode traits)')
print('=' * 78)
d_or = [sm('ORACLE', e, 'nRMSE') - sm('AUTO', e, 'nRMSE') for e in KEEP]
for name, q in QUAL.items():
    rho, t = spearman(q, d_or)
    print(f'  rho( {name:44s}, ORACLE-AUTO ) = {rho:+.3f}  t={t:+.2f}')
print('\n  per-episode ORACLE - AUTO (all 14 negative):')
for e, d in sorted(zip(KEEP, d_or), key=lambda x: x[1]):
    print(f'    {e[8:40]:34s} {d:+.4f}   AUTO {sm("AUTO",e,"nRMSE"):.4f} -> '
          f'ORACLE {sm("ORACLE",e,"nRMSE"):.4f}')
