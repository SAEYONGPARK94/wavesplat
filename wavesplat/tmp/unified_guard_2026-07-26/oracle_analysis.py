"""Oracle-analysis (no compute): SPARSE episodes where CURRENT learned already beats v2
(per-episode median-of-5), and what distinguishes them. Uses existing jsonls only."""
import json, re
import numpy as np

BATCHC = 'tmp/rerun_2026-07-25/batchC_ws_phasefix.jsonl'
S1_ARMS = 'tmp/wp_integration_2026-07-25/arm_runs.jsonl'
X_ARMS = 'tmp/unified_2026-07-26/arm_runs_extra.jsonl'
LANES = [(42, 0), (1, 1), (2, 2), (3, 3), (4, 4)]


def comp(ep): return int(re.search(r'_(\d+)comp', ep).group(1))
def med(xs):
    xs = [x for x in xs if x is not None and np.isfinite(x)]
    return float(np.median(xs)) if xs else float('nan')


v2 = {}; eps = set()
for l in open(BATCHC):
    d = json.loads(l); v2[(d['episode'], d['seed'])] = d['nRMSE']; eps.add(d['episode'])
learned = {}; rawinit = {}
for src in (S1_ARMS, X_ARMS):
    for l in open(src):
        d = json.loads(l)
        if d.get('arm') == 'learned' and d.get('err') is None and d.get('avg_nRMSE') is not None:
            learned[(d['episode'], d['est_seed'], d['diff_seed'])] = d['avg_nRMSE']
            if d.get('raw_init_nrmse') is not None:
                rawinit[(d['episode'], d['est_seed'], d['diff_seed'])] = d['raw_init_nrmse']

SPARSE = sorted([e for e in eps if comp(e) in (1, 2)])
rows = []
for ep in SPARSE:
    pv = med([v2.get((ep, E)) for (E, D) in LANES])
    pl = med([learned.get((ep, E, D)) for (E, D) in LANES])
    ri = med([rawinit.get((ep, E, D)) for (E, D) in LANES])
    rows.append(dict(ep=ep, comp=comp(ep), var=('slope' if 'slope' in ep else 'uniform'),
                     v2=pv, learned=pl, delta=pl - pv, raw_init=ri, learn_wins=pl < pv))

wins = [r for r in rows if r['learn_wins']]
print("="*94)
print(f"SPARSE episodes where CURRENT learned beats v2 (per-episode median-of-5): "
      f"{len(wins)}/{len(rows)}")
print("="*94)
print(f"{'episode':36s} {'comp':>4} {'var':>7} {'v2':>7} {'learned':>8} {'Δ(l-v2)':>8} {'raw_init':>8}")
for r in sorted(wins, key=lambda x: x['delta']):
    print(f"{r['ep']:36s} {r['comp']:>4} {r['var']:>7} {r['v2']:>7.4f} {r['learned']:>8.4f} "
          f"{r['delta']:>+8.4f} {r['raw_init']:>8.3f}")

print("\n--- distinguishing features: wins vs losses ---")
losses = [r for r in rows if not r['learn_wins']]
def summ(rs, name):
    if not rs: print(f"  {name}: none"); return
    nsl = sum(r['var']=='slope' for r in rs); nun = sum(r['var']=='uniform' for r in rs)
    c1 = sum(r['comp']==1 for r in rs); c2 = sum(r['comp']==2 for r in rs)
    ri = [r['raw_init'] for r in rs if np.isfinite(r['raw_init'])]
    print(f"  {name}: n={len(rs)}  comp1={c1} comp2={c2}  slope={nsl} uniform={nun}  "
          f"raw_init med={np.median(ri):.3f} [{np.min(ri):.3f},{np.max(ri):.3f}]  "
          f"mean|Δ|={np.mean([abs(r['delta']) for r in rs]):.4f}")
summ(wins, 'WINS (learned<v2) ')
summ(losses, 'LOSSES(learned>=v2)')

# net effect on sparse mean
print(f"\n  sparse means: v2={np.mean([r['v2'] for r in rows]):.4f} "
      f"learned={np.mean([r['learned'] for r in rows]):.4f}")
print(f"  win-set mean Δ={np.mean([r['delta'] for r in wins]):.4f} (learned better by this on wins)")
print(f"  loss-set mean Δ={np.mean([r['delta'] for r in losses]):+.4f}")
json.dump(rows, open('tmp/unified_guard_2026-07-26/oracle_analysis.json', 'w'), indent=0)
