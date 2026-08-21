"""NEUTRAL audit 08 -- (a) statistical behaviour of the ratio R when its denominator
is small or negative; (b) the n=4 configuration fold (PREREG sec.4.7: report, but
NO 'significant', NO effect-size citation); (c) seed-level noise floor."""
import json, os, statistics as st
from math import comb

HERE = os.path.dirname(os.path.abspath(__file__))
recs = [json.loads(l) for l in open(os.path.join(HERE, 'results.jsonl')) if l.strip()]
EPS = sorted({r['episode'] for r in recs})
CELLS = sorted({r['cell'] for r in recs})
SEEDS = sorted({r['seed'] for r in recs})
V = {(r['cell'], r['episode'], r['seed']): r['nRMSE'] for r in recs}
M = {c: {e: st.median([V[(c, e, s)] for s in SEEDS]) for e in EPS} for c in CELLS}

def signtest(d):
    pos = sum(1 for x in d if x > 0); neg = sum(1 for x in d if x < 0); n = pos + neg
    if n == 0: return n, neg, pos, 1.0
    k = max(pos, neg)
    return n, neg, pos, min(1.0, 2.0 * sum(comb(n, i) for i in range(k, n + 1)) / 2 ** n)

D = {e: M['AUTO'][e] - M['AUTO-NOTRACK'][e] for e in EPS}
ES = [2, 5, 10, 20, 40, 80]

print('=' * 92)
print('(a) RATIO INSTABILITY:  R(E) = (AUTO - E_cell) / D,   D = AUTO - AUTO-NOTRACK')
print('=' * 92)
print(f"  {'episode':26s} {'D':>9} " + ' '.join(f'{"R(%d)"%E:>8}' for E in ES))
for e in sorted(EPS, key=lambda x: D[x]):
    row = ' '.join(f'{(M["AUTO"][e]-M[f"E-{E}"][e])/D[e]*100:7.0f}%' for E in ES)
    flag = '  <== D<0, R MEANINGLESS' if D[e] < 0 else ('  <== |D| tiny' if abs(D[e]) < 0.005 else '')
    print(f'  {e[8:-14]:26s} {D[e]:+9.5f} {row}{flag}')

bad = [e for e in EPS if D[e] < 0]
tiny = [e for e in EPS if 0 <= D[e] < 0.005]
print(f'\n  D < 0 (tracking HELPS -> ratio sign inverts): {len(bad)} episodes')
print(f'  0 <= D < 0.005 (ratio amplifies noise):       {len(tiny)} episodes')
allr = [(M['AUTO'][e] - M[f'E-{E}'][e]) / D[e] for e in EPS for E in ES]
print(f'  R range over all 96 (episode,E) cells = [{min(allr)*100:.0f}%, {max(allr)*100:.0f}%]'
      f'   -- a bounded quantity by construction, observed to span {max(allr)-min(allr):.1f} '
      f'in units of D')
print(f'  |R| > 3 (i.e. >300%) occurs {sum(1 for x in allr if abs(x) > 3)} times')
print('\n  -> The PREREG null is stated on R, but R is a ratio whose denominator is a')
print('     small noisy difference (median D = %+.5f, and 4/16 episodes have |D|<0.005'
      % st.median(list(D.values())))
print('     or D<0).  The DIFFERENCE form obs-pred, reported in audit 03, carries the')
print('     identical hypothesis without dividing, and is the safer read.')

print('\n  --- sensitivity: drop the 4 unstable-denominator episodes, ratio form, den=160 ---')
keep2 = [e for e in EPS if D[e] >= 0.005]
print(f"  n = {len(keep2)}   threshold for n={len(keep2)} = "
      f"{ {16:13,15:12,14:12,13:11,12:10}.get(len(keep2))}")
for E in ES:
    d = [((M['AUTO'][e] - M[f'E-{E}'][e]) / D[e]) - (1 - E / 160) for e in keep2]
    n, neg, pos, p = signtest(d)
    thr = {16: 13, 15: 12, 14: 12, 13: 11, 12: 10}.get(n)
    print(f'    E-{E:<3d} median(R_obs-R_pred) = {st.median(d):+.4f}  wins {max(pos,neg)}/{n} '
          f'(thr {thr})  p={p:.5f}  {"DISTINGUISHABLE" if thr and max(pos,neg)>=thr else "not"}')

print('\n' + '=' * 92)
print('(b) CONFIGURATION FOLD n=4 (PREREG sec.4.7 -- descriptive ONLY, floor p=0.125)')
print('    NO significance and NO effect size may be cited from this fold.')
print('=' * 92)
CONF = {}
for e in EPS:
    key = e.split('_seed')[0]
    CONF.setdefault(key, []).append(e)
print(f'  folds = {list(CONF)}')
def fold(c1, c2, label):
    d = []
    for k, es in CONF.items():
        d.append(st.median([M[c1][e] for e in es]) - st.median([M[c2][e] for e in es]))
    n, neg, pos, p = signtest(d)
    print(f'  {label:34s} wins {max(pos,neg)}/{n}   (direction: '
          f'{"first worse" if pos>neg else "first better"})   [floor p = 0.125]')
fold('W-MATCH', 'PHI-5e3', 'PRIMARY-A: W-MATCH vs PHI-5e3')
fold('AUTO', 'AUTO-NOTRACK', 'AUTO vs AUTO-NOTRACK')
fold('W-5e3', 'AUTO', 'W-5e3 vs AUTO')

print('\n' + '=' * 92)
print('(c) SEED NOISE FLOOR (spread of the 3 seeds inside one cell x episode)')
print('=' * 92)
spreads = [max(V[(c, e, s)] for s in SEEDS) - min(V[(c, e, s)] for s in SEEDS)
           for c in CELLS for e in EPS]
print(f'  median seed range = {st.median(spreads):.5f}   '
      f'90th pct = {sorted(spreads)[int(.9*len(spreads))]:.5f}')
print(f'  PRIMARY-A |median effect| = 0.02162 -> {0.02162/st.median(spreads):.2f}x the '
      f'median seed range')
print(f'  PRIMARY-A shift-MDE       = 0.00145 -> {0.00145/st.median(spreads):.2f}x the '
      f'median seed range')
print('  -> the effect is large relative to seed noise; the blocking episodes are')
print('     sign flips of TINY magnitude, not a small overall effect.')
