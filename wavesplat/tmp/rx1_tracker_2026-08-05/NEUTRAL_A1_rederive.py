"""A1: re-derive the 0.1602 vs 0.0621 headline from raw s1s2 artifacts.

No numbers are trusted from VERDICT.md / PRO_FINDINGS.md. Everything recomputed
from tmp/s1s2_2026-08-05/results.jsonl.
"""
import json, os, itertools
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, '..', 's1s2_2026-08-05', 'results.jsonl')
rows = [json.loads(l) for l in open(SRC) if l.strip()]
rows = [r for r in rows if 'error' not in r]
EPS = sorted({r['episode'] for r in rows})
SEEDS = sorted({r['seed'] for r in rows})

def get(arm, ep, seed, key):
    v = [r for r in rows if r['arm'] == arm and r['episode'] == ep and r['seed'] == seed]
    assert len(v) == 1, (arm, ep, seed, len(v))
    return v[0].get(key)

def med3(arm, ep, key):
    return float(np.median([get(arm, ep, s, key) for s in SEEDS]))

def flat(arm, key):
    return [get(arm, ep, s, key) for ep in EPS for s in SEEDS]

print('=' * 78)
print('A1.0  ARM COMPARABILITY (cfg diff, N, frames, iters)')
print('=' * 78)
A, B = 'S2-T1f', 'X-T1f-notrack'
for ep in EPS:
    for s in SEEDS:
        ca = get(A, ep, s, 'cfg'); cb = get(B, ep, s, 'cfg')
        d = {k: (ca.get(k), cb.get(k)) for k in set(ca) | set(cb) if ca.get(k) != cb.get(k)}
        na, nb = get(A, ep, s, 'N'), get(B, ep, s, 'N')
        ia, ib = get(A, ep, s, 'init_nrmse'), get(B, ep, s, 'init_nrmse')
        print(f'{ep[8:-3]:<24s} seed{s:<4d} cfgdiff={d} N={na}/{nb} '
              f'init_nrmse={ia:.6f}/{ib:.6f} identical={abs(ia-ib)<1e-12}')

print()
print('spec (component-space) identity check, per (ep,seed):')
for ep in EPS:
    for s in SEEDS:
        sa, sb = get(A, ep, s, 'spec'), get(B, ep, s, 'spec')
        same = {k for k in sa if isinstance(sa[k], float) and abs(sa[k] - sb.get(k, np.nan)) < 1e-15}
        diff = {k: (round(sa[k], 5), round(sb[k], 5)) for k in sa
                if isinstance(sa[k], (int, float)) and k not in same}
        print(f'  {ep[8:-3]:<24s} seed{s:<4d} bit-identical={sorted(same)} differs={diff}')

print()
print('=' * 78)
print('A1.1  RE-DERIVED HEADLINE (three aggregations)')
print('=' * 78)
for key in ['nRMSE', 'init_nrmse', 'Pearson_rho', 'Variance_Ratio', 'Hs_error',
            'Skill_Score', 'avg_RMSE_unobserved']:
    a_all, b_all = flat(A, key), flat(B, key)
    a_m3 = [med3(A, ep, key) for ep in EPS]
    b_m3 = [med3(B, ep, key) for ep in EPS]
    print(f'{key:<22s} mean12 {np.mean(a_all):.4f} -> {np.mean(b_all):.4f} | '
          f'mean(med3) {np.mean(a_m3):.4f} -> {np.mean(b_m3):.4f} | '
          f'median(med3) {np.median(a_m3):.4f} -> {np.median(b_m3):.4f}')

print()
print('PREREG primary statistic = median of PAIRED deltas (med3 per episode):')
for key in ['nRMSE', 'Pearson_rho', 'Hs_error', 'Skill_Score']:
    d = [med3(B, ep, key) - med3(A, ep, key) for ep in EPS]
    print(f'  {key:<18s} paired deltas {[round(x,4) for x in d]}  median {np.median(d):+.4f}')
d = [abs(med3(B, ep, 'Variance_Ratio') - 1) - abs(med3(A, ep, 'Variance_Ratio') - 1) for ep in EPS]
print(f'  {"|VR-1|":<18s} paired deltas {[round(x,4) for x in d]}  median {np.median(d):+.4f}')

print()
print('per-episode nRMSE (med3):')
for ep in EPS:
    print(f'  {ep[8:-3]:<24s} ON {med3(A, ep, "nRMSE"):.4f}  OFF {med3(B, ep, "nRMSE"):.4f}  '
          f'delta {med3(B, ep, "nRMSE")-med3(A, ep, "nRMSE"):+.4f}  '
          f'ratio_of_attribution {(med3(A,ep,"nRMSE")-med3(B,ep,"nRMSE"))/med3(A,ep,"nRMSE"):.3f}')

print()
print('=' * 78)
print('A1.2  seed spread inside each cell (is the effect > seed noise?)')
print('=' * 78)
for arm in [A, B, 'S1-A0', 'X-A0-notrack']:
    for ep in EPS:
        v = [get(arm, ep, s, 'nRMSE') for s in SEEDS]
        print(f'  {arm:<15s} {ep[8:-3]:<24s} nRMSE {[round(x,4) for x in v]} '
              f'range {max(v)-min(v):.4f}')
