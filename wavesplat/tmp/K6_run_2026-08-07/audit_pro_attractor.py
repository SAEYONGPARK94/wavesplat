#!/usr/bin/env python3
"""PRO audit part 5 -- the attractor claim and the harm channel of init_k_lr_scale=0."""
import json, os, math
import numpy as np, h5py

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.abspath(os.path.join(HERE, '..', '..', 'data_indep2_2026-08-07'))
rows = [json.loads(l) for l in open(os.path.join(HERE, 'results.jsonl')) if l.strip()]
nulls = {d['episode']: d for d in json.load(open(os.path.join(HERE, 'nulls_k6.json')))}
IDX = {(r['cell'], r['episode'], r['seed']): r for r in rows}
EPS = sorted({r['episode'] for r in rows}); SEEDS = sorted({r['seed'] for r in rows})
def med(v):
    v = sorted(v); n = len(v)
    return v[n//2] if n % 2 else 0.5*(v[n//2-1]+v[n//2])
def sm(c, e, f): return med([IDX[(c, e, s)][f] for s in SEEDS])
def bts(k, n):
    C = math.comb
    return min(1.0, 2*sum(C(n, i) for i in range(0, min(k, n-k)+1))/2**n)
GT = {}
for e in EPS:
    with h5py.File(os.path.join(DATA, e), 'r') as f:
        wc = np.asarray(f['config/wave_components'])
    GT[e] = (wc[:, 0], wc[:, 2], wc[:, 3])
def cov(run, ep):
    A, gx, gy = GT[ep]
    K = np.stack([gx, gy], 1); wt = A**2
    E = np.stack([np.asarray(run['kx']), np.asarray(run['ky'])], 1)
    D = np.linalg.norm(K[:, None]-E[None, :], axis=2)
    return float((wt*D.min(1)).sum()/wt.sum())
def smcov(c, e): return med([cov(IDX[(c, e, s)], e) for s in SEEDS])
KEEP = [e for e in EPS if sm('AUTO', e, 'nRMSE') < nulls[e]['zero_null_tracking_nRMSE_mean']]

print('=' * 78)
print('O. THE ATTRACTOR: does Phase 1 land in the same k-accuracy regardless of start?')
print('=' * 78)
sp = med([float(np.median((np.linalg.norm(
        np.stack([GT[e][1], GT[e][2]], 1)[:, None] - np.stack([GT[e][1], GT[e][2]], 1)[None, :], axis=2)
        + np.eye(len(GT[e][0]))*1e9).min(1))) for e in EPS])
print(f'  true component spacing (median NN, rad/m) = {sp:.4f}')
for c in ['ORACLE', 'ORACLE-K', 'AUTO', 'K6-KLR00', 'K6-NSCHED']:
    v = [smcov(c, e) for e in KEEP]
    print(f'  COV {c:10s} median {med(v):.4f}  = {100*med(v)/sp:5.1f}% of the true spacing '
          f'  [{min(v):.4f},{max(v):.4f}]')
print()
for a, b, lab in [('ORACLE-K', 'ORACLE', 'k started AT TRUTH and was trained'),
                  ('ORACLE-K', 'AUTO',   'trained-from-truth vs trained-from-blind'),
                  ('ORACLE-K', 'K6-KLR00', 'trained-from-truth vs the RAW blind proposal'),
                  ('AUTO', 'K6-NSCHED', 'shipping router vs blind beamformer @N_gt/900')]:
    d = [smcov(a, e) - smcov(b, e) for e in KEEP]
    w = sum(1 for x in d if x < 0)
    print(f'  COV  {a:9s} - {b:9s}  median {med(d):+.5f}  {w:2d}/14  p={bts(w,14):.4f}  [{lab}]')
print()
print('  => the optimiser, handed PERFECT k, walks it out to COV '
      f'{med([smcov("ORACLE-K",e) for e in KEEP]):.4f} rad/m,')
print(f'     which is {100*med([smcov("ORACLE-K",e) for e in KEEP])/sp:.0f}% of the true component '
      'spacing and is NOT distinguishable (above)')
print('     from where the blind shipping router already lands.  Truth is not a resting')
print('     point of the Phase-1 objective.')
print('  FIELD consequence, both directions:')
for a, b in [('ORACLE-K', 'AUTO'), ('ORACLE', 'AUTO'), ('ORACLE', 'ORACLE-K')]:
    d = [sm(a, e, 'nRMSE') - sm(b, e, 'nRMSE') for e in KEEP]
    w = sum(1 for x in d if x < 0)
    print(f'    nRMSE {a:9s} - {b:9s} median {med(d):+.4f}  {w:2d}/14  p={bts(w,14):.4f}')

print()
print('=' * 78)
print('P. HARM CHANNEL of init_k_lr_scale=0 on the shipping arm (it is NOT k quality)')
print('=' * 78)
d = [smcov('K6-KLR00', e) - smcov('AUTO', e) for e in KEEP]
w = sum(1 for x in d if x < 0)
print(f'  COV        KLR00 - AUTO median {med(d):+.5f}  {w:2d}/14  p={bts(w,14):.4f}  -> NULL')
for f in ['init_nrmse', 'nRMSE', 'Pearson_rho', 'm_mean', 'shrink_mean']:
    d = [sm('K6-KLR00', e, f) - sm('AUTO', e, f) for e in KEEP]
    w = sum(1 for x in d if (x > 0 if f == 'Pearson_rho' else x < 0))
    print(f'  {f:11s} KLR00 - AUTO median {med(d):+.4f}  {w:2d}/14  p={bts(w,14):.4f}')
print()
print('  atom count actually used (adaptive N reacts to the frozen k):')
for c in ['AUTO', 'K6-KLR03', 'K6-KLR01', 'K6-KLR00', 'ORACLE', 'ORACLE-K', 'K6-NSCHED']:
    ns = [len(IDX[(c, e, s)]['kx']) for e in KEEP for s in SEEDS]
    an = [IDX[(c, e, s)].get('auto_n_selected') for e in KEEP for s in SEEDS]
    an = [x for x in an if x is not None]
    print(f'    {c:11s} N(final atoms) median {med(ns):.1f}  range [{min(ns)},{max(ns)}]'
          + (f'   auto_n_selected median {med(an):.1f}' if an else ''))
