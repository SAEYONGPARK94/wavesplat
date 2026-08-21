"""NEUTRAL audit 06 -- can W-5e3 be said to REPRODUCE Rx-4's T-WONLY?
Rx-4 ran a DIFFERENT dataset (data_finemesh_2026-07-23, gate draw 4-8) and different
seeds (1524-1526).  PREREG sec.7 forbids POOLING.  Reproduction is a weaker,
directional claim -- so quantify how far it can honestly go.
NOTHING here is pooled; the two blocks are reported side by side only.
"""
import json, os, statistics as st
from math import comb

HERE = os.path.dirname(os.path.abspath(__file__))
RX4 = os.path.join(HERE, '..', 'Rx4_tracker_init_2026-08-06', 'results.jsonl')

def signtest(d):
    pos = sum(1 for x in d if x > 0); neg = sum(1 for x in d if x < 0); n = pos + neg
    if n == 0: return n, neg, pos, 1.0
    k = max(pos, neg)
    return n, neg, pos, min(1.0, 2.0 * sum(comb(n, i) for i in range(k, n + 1)) / 2 ** n)

def load(path, epfilter=None):
    recs = [json.loads(l) for l in open(path) if l.strip()]
    if epfilter: recs = [r for r in recs if epfilter(r['episode'])]
    eps = sorted({r['episode'] for r in recs}); cells = sorted({r['cell'] for r in recs})
    M = {c: {e: st.median([r['nRMSE'] for r in recs
                           if r['cell'] == c and r['episode'] == e]) for e in eps}
         for c in cells}
    return M, eps, sorted({r['seed'] for r in recs})

# Rx-4 CROWDED16 = the 16 BF4/BF5 15/30comp uniform episodes
crow = lambda e: ('BF4_15comp' in e or 'BF4_30comp' in e or
                  'BF5_15comp' in e or 'BF5_30comp' in e)
M4, E4, S4 = load(RX4, crow)
M5, E5, S5 = load(os.path.join(HERE, 'results.jsonl'))

print('=' * 88)
print('BLOCK PROVENANCE')
print('=' * 88)
print(f'  Rx-4 : data_finemesh_2026-07-23  episodes={len(E4)}  seeds={S4}')
print(f'         gate draw counts A=4 omega=4 phi=8 theta=8 truth=16 -> floor p=0.125')
print(f'  Rx-5 : data_indep2_2026-08-07    episodes={len(E5)}  seeds={S5}')
print(f'         gate draw counts all 5 axes = 16          -> floor p=3.05e-05')
print(f'  episode-name overlap = {len(set(E4) & set(E5))}  '
      f'(names differ by the _ra_rf_ds suffix; underlying draws are NOT the same)')

print('\n' + '=' * 88)
print('SIDE-BY-SIDE (NOT pooled): amplitude-only tracking at the shipped lr, vs AUTO')
print('=' * 88)
print(f"  {'block':10s} {'cell':12s} {'median delta vs AUTO':>21} {'wins/n':>9} {'p':>9}")
for name, M, E, cell in [('Rx-4', M4, E4, 'T-WONLY'), ('Rx-5', M5, E5, 'W-5e3')]:
    d = [M[cell][e] - M['AUTO'][e] for e in E]
    n, neg, pos, p = signtest(d)
    print(f'  {name:10s} {cell:12s} {st.median(d):+21.5f} {f"{max(pos,neg)}/{n}":>9} {p:9.5f}')
for name, M, E, cell in [('Rx-4', M4, E4, 'T-PHIONLY'), ('Rx-5', M5, E5, 'PHI-5e3')]:
    d = [M[cell][e] - M['AUTO'][e] for e in E]
    n, neg, pos, p = signtest(d)
    print(f'  {name:10s} {cell:12s} {st.median(d):+21.5f} {f"{max(pos,neg)}/{n}":>9} {p:9.5f}')
for name, M, E in [('Rx-4', M4, E4), ('Rx-5', M5, E5)]:
    d = [M['AUTO-NOTRACK'][e] - M['AUTO'][e] for e in E]
    n, neg, pos, p = signtest(d)
    print(f'  {name:10s} {"AUTO-NOTRACK":12s} {st.median(d):+21.5f} '
          f'{f"{max(pos,neg)}/{n}":>9} {p:9.5f}')

print('\n' + '=' * 88)
print('Rx-4 T-EARLY20  vs  Rx-5 E-20   (the one cell that carried over)')
print('=' * 88)
for name, M, E, cell in [('Rx-4', M4, E4, 'T-EARLY20'), ('Rx-5', M5, E5, 'E-20')]:
    d = [M[cell][e] - M['AUTO'][e] for e in E]
    n, neg, pos, p = signtest(d)
    print(f'  {name:6s} {cell:12s} median {st.median(d):+.5f}  wins {max(pos,neg)}/{n}  p={p:.5f}')

print('\n' + '=' * 88)
print('VERDICT ON THE WORD "REPRODUCE"')
print('=' * 88)
print("""  Shared: both blocks put amplitude-only tracking at lr=0.005 WORSE than AUTO,
          with the same sign and a comparable median.
  Not shared: different episode draws, different seeds, different gate grade.
  => Legitimate:  "the sign and rough magnitude of the amplitude-only penalty
                   recur on an independent, better-gated sample".
  => NOT legitimate: "W-5e3 reproduces T-WONLY" in the bit-identical / replication
                   sense the Rx-5 harness uses that word for AUTO-vs-B3.  There is
                   no shared (episode, seed) cell anywhere between the two blocks,
                   so no reproduction control exists.  PREREG sec.7 also forbids
                   pooling or summing the two.""")
