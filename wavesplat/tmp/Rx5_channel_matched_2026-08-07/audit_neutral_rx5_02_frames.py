"""NEUTRAL audit 02 -- how many tracked frames does each episode actually have?
PREREG sec.1 asserts 160 and uses it as THE denominator of the sec.5 null model
R(E) = 1 - E/160.  Check that assertion per episode."""
import json, os, collections

HERE = os.path.dirname(os.path.abspath(__file__))
recs = [json.loads(l) for l in open(os.path.join(HERE, 'results.jsonl')) if l.strip()]

byep = collections.defaultdict(set)
for r in recs:
    byep[r['episode']].add(len(r.get('vr_frames') or []))

print(f"{'episode':52s} {'n_frames'}")
n160 = n120 = 0
for e in sorted(byep):
    v = sorted(byep[e])
    print(f'  {e:50s} {v}')
    if v == [160]: n160 += 1
    elif v == [120]: n120 += 1
print(f'\nepisodes with 160 frames = {n160}   with 120 frames = {n120}')

print('\n--- consequence for the PRE-DECLARED null R(E)=1-E/160 ---')
print(f"{'E':>4} {'R_pred(160)':>12} {'R_pred(120)':>12} {'gap':>8}")
for E in (2, 5, 10, 20, 40, 80):
    a, b = 1 - E / 160, 1 - E / 120
    print(f'{E:4d} {a*100:11.1f}% {b*100:11.1f}% {(a-b)*100:7.1f}pp')
