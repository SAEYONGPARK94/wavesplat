"""PRO (6) k-position: independent re-derivation of the ORACLE-K headline from raw jsonl.
Written from scratch (does not import analyze_b2). Read-only."""
import json, math, os, statistics as st, re
from collections import defaultdict

B2 = '/home/vexit/wavesplat/wavesplat/tmp/B2_g1p1_ext_2026-08-06/results.jsonl'

CROWDED4 = ['episode_BF4_15comp_seed0_uniform.h5', 'episode_BF4_30comp_seed0_uniform.h5',
            'episode_BF5_15comp_seed0_uniform.h5', 'episode_BF5_30comp_seed0_uniform.h5']
NEW8 = ['episode_BF2_1comp_seed0_uniform.h5', 'episode_BF3_1comp_seed0_uniform.h5',
        'episode_BF3_2comp_seed0_uniform.h5', 'episode_BF3_2comp_seed0_slope.h5',
        'episode_BF3_5comp_seed0_uniform.h5', 'episode_BF3_5comp_seed0_slope.h5',
        'episode_BF4_1comp_seed0_uniform.h5', 'episode_BF4_5comp_seed0_uniform.h5']
ALL12 = NEW8 + CROWDED4

raw = defaultdict(dict)
extra = defaultdict(dict)
for line in open(B2):
    r = json.loads(line)
    raw[(r['cell'], r['episode'])][r['seed']] = r['nRMSE']
    extra[(r['cell'], r['episode'])][r['seed']] = r

def med(c, e): return st.median(list(raw[(c, e)].values()))

def sign_p(k, n):
    if n == 0: return 1.0
    kk = min(k, n - k)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(kk + 1)) / 2 ** n)

print('='*100)
print('A. ORACLE-K vs X-BASE, per episode (median of 3 seeds), independently recomputed')
print('='*100)
print(f'{"episode":36s}{"Ngt":>4s}{"X-BASE":>9s}{"ORACLE-K":>10s}{"delta":>9s}{"N_xb":>6s}{"N_ok":>6s}')
deltas = []
for e in ALL12:
    ngt = int(re.search(r'_(\d+)comp', e).group(1))
    xb, ok = med('X-BASE', e), med('ORACLE-K', e)
    d = ok - xb
    deltas.append((e, ngt, xb, ok, d))
    nxb = st.median([extra[('X-BASE', e)][s]['N'] for s in extra[('X-BASE', e)]])
    nok = st.median([extra[('ORACLE-K', e)][s]['N'] for s in extra[('ORACLE-K', e)]])
    print(f'{e[8:-3]:36s}{ngt:4d}{xb:9.4f}{ok:10.4f}{d:+9.4f}{nxb:6.0f}{nok:6.0f}')

ds = [d for *_, d in deltas]
w = sum(1 for x in ds if x < 0)
print(f'\nALL12 : paired median delta = {st.median(ds):+.4f} | wins {w}/{len(ds)} | '
      f'naive two-sided sign p = {sign_p(w, len(ds)):.4f}')
ds8 = [d for e, ngt, xb, ok, d in deltas if e in NEW8]
w8 = sum(1 for x in ds8 if x < 0)
print(f'NEW8  : paired median delta = {st.median(ds8):+.4f} | wins {w8}/8 | p = {sign_p(w8,8):.4f}')
ds4 = [d for e, ngt, xb, ok, d in deltas if e in CROWDED4]
w4 = sum(1 for x in ds4 if x < 0)
print(f'CROW4 : paired median delta = {st.median(ds4):+.4f} | wins {w4}/4 | p = {sign_p(w4,4):.4f}')

# ---- cluster-corrected: 5 phase-draw clusters = ncomp groups {1,2,5,15,30} ----
print('\n' + '='*100)
print('B. CLUSTER-CORRECTED (independence audit 2026-08-07): cluster = distinct phase/direction draw')
print('   ncomp group -> one independent random realisation (seed0 draw shared across BF)')
print('='*100)
clus = defaultdict(list)
for e, ngt, xb, ok, d in deltas:
    clus[ngt].append(d)
cl_meds = []
for ngt in sorted(clus):
    m = st.median(clus[ngt])
    cl_meds.append(m)
    print(f'  ncomp={ngt:2d}  n_eps={len(clus[ngt])}  cluster median delta = {m:+.4f}   '
          f'members {[f"{x:+.4f}" for x in clus[ngt]]}')
cw = sum(1 for x in cl_meds if x < 0)
print(f'\n  clusters C={len(cl_meds)} | negative {cw}/{len(cl_meds)} | '
      f'cluster sign-test p = {sign_p(cw, len(cl_meds)):.4f} (floor {2/2**len(cl_meds):.4f})')
print(f'  cluster-level median of cluster medians = {st.median(cl_meds):+.4f}')

# also cluster on distinct truth field (10) as the *loosest defensible* unit
print('\n  [looser unit] distinct truth fields = 10 (uniform/slope pairs share the field)')
fieldclus = defaultdict(list)
for e, ngt, xb, ok, d in deltas:
    fieldclus[e.replace('_slope', '_X').replace('_uniform', '_X')].append(d)
fm = [st.median(v) for v in fieldclus.values()]
fw = sum(1 for x in fm if x < 0)
print(f'  C={len(fm)} | negative {fw}/{len(fm)} | p = {sign_p(fw, len(fm)):.4f} | '
      f'median {st.median(fm):+.4f}')
