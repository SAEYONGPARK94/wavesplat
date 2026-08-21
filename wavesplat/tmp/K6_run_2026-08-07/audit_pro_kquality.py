#!/usr/bin/env python3
"""PRO audit part 4 -- MATCHING-FREE k-quality (COV / PREC, the K6_scope NEUTRAL protocol),
used to (a) re-test the repair-vs-damage mechanism without the sec.7-banned metrics and
(b) stratify the shipping-arm trust-region effect by the quality of the k that gets frozen.

kx/ky in results.jsonl are the END-OF-RUN wavenumbers, and tracking_mode='w_phi' freezes
kx/ky through all of Phase 2, so END-OF-RUN kx/ky == END-OF-PHASE-1 kx/ky.  For the
init_k_lr_scale=0 cells (K6-KLR00, ORACLE) they are additionally == the placement Phase 1
was handed, i.e. exactly the k that the trust region would freeze.  Read-only, CPU only.
"""
import json, os, math, statistics as st
import numpy as np
import h5py

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..'))
DATA = os.path.join(REPO, 'data_indep2_2026-08-07')

rows = [json.loads(l) for l in open(os.path.join(HERE, 'results.jsonl')) if l.strip()]
nulls = {d['episode']: d for d in json.load(open(os.path.join(HERE, 'nulls_k6.json')))}
IDX = {(r['cell'], r['episode'], r['seed']): r for r in rows}
EPS = sorted({r['episode'] for r in rows}); SEEDS = sorted({r['seed'] for r in rows})

def med(v):
    v = sorted(v); n = len(v)
    return v[n // 2] if n % 2 else 0.5 * (v[n // 2 - 1] + v[n // 2])
def sm(c, e, f): return med([IDX[(c, e, s)][f] for s in SEEDS])
def bts(k, n):
    C = math.comb
    return min(1.0, 2 * sum(C(n, i) for i in range(0, min(k, n - k) + 1)) / 2 ** n)

GT = {}
for e in EPS:
    with h5py.File(os.path.join(DATA, e), 'r') as f:
        wc = np.asarray(f['config/wave_components'])   # [A, omega, kx, ky, phi, Q]
    GT[e] = dict(A=wc[:, 0], kx=wc[:, 2], ky=wc[:, 3])

def cov_prec(run, ep):
    """Matching-free, acceptance-radius-free.  Distances in rad/m, energy-weighted.
    COV  = how far truth's energy is from the nearest ESTIMATED atom  (misses)
    PREC = how far the estimate's energy is from the nearest TRUE atom (spurious atoms)
    """
    g = GT[ep]
    K = np.stack([g['kx'], g['ky']], 1)            # (Ngt,2)
    wt = g['A'] ** 2
    ex, ey = np.asarray(run['kx']), np.asarray(run['ky'])
    if ex.size == 0:
        return float('nan'), float('nan')
    E = np.stack([ex, ey], 1)
    we = np.asarray(run['w']) ** 2
    D = np.linalg.norm(K[:, None, :] - E[None, :, :], axis=2)   # (Ngt, Nest)
    cov = float((wt * D.min(1)).sum() / wt.sum())
    prec = float((we * D.min(0)).sum() / max(we.sum(), 1e-12))
    return cov, prec

def sm_cp(cell, ep, which):
    v = [cov_prec(IDX[(cell, ep, s)], ep)[0 if which == 'COV' else 1] for s in SEEDS]
    return med(v)

KEEP = [e for e in EPS if sm('AUTO', e, 'nRMSE') < nulls[e]['zero_null_tracking_nRMSE_mean']]

# sanity: true nearest-neighbour spacing, so COV/PREC have a scale
sp_all = []
for e in EPS:
    K = np.stack([GT[e]['kx'], GT[e]['ky']], 1)
    D = np.linalg.norm(K[:, None] - K[None, :], axis=2) + np.eye(len(K)) * 1e9
    sp_all.append(float(np.median(D.min(1))))
print('=' * 78)
print('L. MATCHING-FREE k QUALITY (COV / PREC, rad/m)  -- NOT a sec.7-banned metric')
print('=' * 78)
print(f'  true nearest-neighbour k spacing, median over 16 episodes: {med(sp_all):.4f} rad/m')
print()
print(f'  {"cell":11s} {"COV":>8s} {"PREC":>8s}   note')
NOTE = {'K6-KLR00': 'k FROZEN -> this IS the k the trust region would freeze',
        'ORACLE': 'k FROZEN at truth -> COV=PREC=0 by construction',
        'AUTO': 'k TRAINED (shipping)', 'ORACLE-K': 'k TRAINED from truth',
        'K6-KLR01': 'k trained at 0.1x lr', 'K6-KLR03': 'k trained at 0.3x lr',
        'K6-NSCHED': 'k TRAINED from blind beamformer, N_gt/900'}
for c in ['K6-KLR00', 'K6-KLR01', 'K6-KLR03', 'AUTO', 'K6-NSCHED', 'ORACLE', 'ORACLE-K']:
    print(f'  {c:11s} {med([sm_cp(c,e,"COV") for e in KEEP]):8.4f} '
          f'{med([sm_cp(c,e,"PREC") for e in KEEP]):8.4f}   {NOTE[c]}')

print()
print('  paired sign tests on COV (lower = better), block = keep14:')
for a, b, lab in [('AUTO', 'K6-KLR00', 'BLIND arm : trained vs frozen'),
                  ('ORACLE-K', 'ORACLE', 'ORACLE arm: trained vs frozen'),
                  ('K6-KLR01', 'K6-KLR00', 'BLIND arm : 0.1x vs frozen'),
                  ('AUTO', 'K6-KLR01', 'BLIND arm : 1.0x vs 0.1x')]:
    d = [sm_cp(a, e, 'COV') - sm_cp(b, e, 'COV') for e in KEEP]
    w = sum(1 for x in d if x < 0)
    print(f'    {a:9s} - {b:9s}  median {med(d):+.5f}  {w:2d}/{len(d)}  '
          f'p={bts(w,len(d)):.4f}   [{lab}]')
print('  paired sign tests on PREC (lower = better):')
for a, b in [('AUTO', 'K6-KLR00'), ('ORACLE-K', 'ORACLE')]:
    d = [sm_cp(a, e, 'PREC') - sm_cp(b, e, 'PREC') for e in KEEP]
    w = sum(1 for x in d if x < 0)
    print(f'    {a:9s} - {b:9s}  median {med(d):+.5f}  {w:2d}/{len(d)}  p={bts(w,len(d)):.4f}')

print()
print('=' * 78)
print('M. INTERACTION TEST, RE-RUN ON THE MATCHING-FREE QUALITY OF THE FROZEN k')
print('=' * 78)
q = [sm_cp('K6-KLR00', e, 'COV') for e in KEEP]     # quality of the k that gets frozen
def spearman(x, y):
    def rank(v):
        o = sorted(range(len(v)), key=lambda i: v[i]); r = [0.0]*len(v); i = 0
        while i < len(o):
            j = i
            while j+1 < len(o) and v[o[j+1]] == v[o[i]]: j += 1
            for k in range(i, j+1): r[o[k]] = (i+j)/2.0 + 1
            i = j + 1
        return r
    rx, ry = rank(x), rank(y); n = len(x)
    mx, my = sum(rx)/n, sum(ry)/n
    num = sum((a-mx)*(b-my) for a, b in zip(rx, ry))
    den = math.sqrt(sum((a-mx)**2 for a in rx)*sum((b-my)**2 for b in ry))
    rho = num/den if den else float('nan')
    return rho, rho*math.sqrt(max(n-2,1)/max(1e-12, 1-rho**2))
print('  H: the better the k you would freeze, the more freezing pays')
print('     => predicts rho(COV of frozen k, delta) > 0  (worse k -> delta more positive)')
for lab, cell in [('KLR03-AUTO', 'K6-KLR03'), ('KLR01-AUTO', 'K6-KLR01'),
                  ('KLR00-AUTO', 'K6-KLR00')]:
    d = [sm(cell, e, 'nRMSE') - sm('AUTO', e, 'nRMSE') for e in KEEP]
    rho, t = spearman(q, d)
    print(f'    rho( COV_frozen , {lab} ) = {rho:+.3f}  t={t:+.2f}  '
          f'(n=14, |rho| needed for p<.05 ~ 0.53)')
print()
print('  tertile split on COV of the frozen k (DESCRIPTIVE, n<=5 per bin):')
order = sorted(range(len(KEEP)), key=lambda i: q[i])
bins = [order[:5], order[5:9], order[9:]]
for bi, b in enumerate(bins):
    lab = ['best-k', 'mid-k ', 'worst-k'][bi]
    covs = [q[i] for i in b]
    line = f'    {lab} (COV {min(covs):.4f}-{max(covs):.4f}, n={len(b)}): '
    for cell, nm in [('K6-KLR03', 'KLR03'), ('K6-KLR01', 'KLR01'), ('K6-KLR00', 'KLR00')]:
        d = [sm(cell, KEEP[i], 'nRMSE') - sm('AUTO', KEEP[i], 'nRMSE') for i in b]
        line += f'{nm} {med(d):+.4f}({sum(1 for x in d if x<0)}/{len(d)})  '
    print(line)
print()
print('  the ORACLE cell freezes a k with COV = 0.0000; the SHIPPING arm would freeze')
print(f'  a k with COV in [{min(q):.4f}, {max(q):.4f}] -- i.e. the shipping arm never supplies')
print('  a k anywhere near the quality at which freezing was measured to pay.')

print()
print('=' * 78)
print('N. IS THE k -> FIELD PAYOFF CURVE FLAT IN THE SHIPPING BAND?')
print('=' * 78)
print('  per-cell block medians (COV of the k actually used  ->  field nRMSE):')
for c in ['ORACLE', 'ORACLE-K', 'AUTO', 'K6-KLR03', 'K6-KLR01', 'K6-KLR00', 'K6-NSCHED']:
    print(f'    {c:11s} COV {med([sm_cp(c,e,"COV") for e in KEEP]):.4f} -> '
          f'nRMSE {med([sm(c,e,"nRMSE") for e in KEEP]):.4f}   '
          f'init_nrmse {med([sm(c,e,"init_nrmse") for e in KEEP]):.4f}')
# pooled within-shipping-band relation
xs, ys = [], []
for c in ['AUTO', 'K6-KLR03', 'K6-KLR01', 'K6-KLR00']:
    for e in KEEP:
        xs.append(sm_cp(c, e, 'COV')); ys.append(sm(c, e, 'nRMSE'))
rho, t = spearman(xs, ys)
print(f'\n  pooled over the 4 SHIPPING-arm cells x 14 episodes (n={len(xs)}, NOT independent):')
print(f'    rho(COV, nRMSE) = {rho:+.3f}  t={t:+.2f}')
xs2, ys2 = [], []
for c in ['ORACLE', 'ORACLE-K']:
    for e in KEEP:
        xs2.append(sm_cp(c, e, 'COV')); ys2.append(sm(c, e, 'nRMSE'))
rho2, t2 = spearman(xs2, ys2)
print(f'  pooled over the 2 ORACLE-arm cells x 14 episodes (n={len(xs2)}, NOT independent):')
print(f'    rho(COV, nRMSE) = {rho2:+.3f}  t={t2:+.2f}')
print('  [pooling across cells mixes the intervention with the covariate -- DESCRIPTIVE only,')
print('   feedback_control_stratifying_variable applies.]')
