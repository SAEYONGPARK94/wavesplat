"""e45 audit part 3 — e41 ladder, e43 dose, e39, e37 re-derivation."""
import json
import numpy as np
from scipy import stats

D = 'results/direction3/'


def paired(d):
    d = np.asarray(d, float)
    d = d[np.isfinite(d)]
    n = len(d)
    se = d.std(ddof=1) / np.sqrt(n)
    t, p = stats.ttest_1samp(d, 0)
    lo, hi = stats.t.interval(0.95, n - 1, loc=d.mean(), scale=se)
    return dict(mean=d.mean(), se=se, t=float(t), p=float(p), lo=lo, hi=hi,
                win=int((d > 0).sum()), n=n)


def fmt(s):
    return (f"{s['mean']:+.4f} +-{s['se']:.4f} CI-t[{s['lo']:+.4f},{s['hi']:+.4f}] "
            f"p{s['p']:.3g} win {s['win']}/{s['n']}")


print('=========== E-41 ladder (arena X) ===========')
e41 = json.load(open(D + 'e41_regime_prize_ladder.json'))
print('top keys:', list(e41.keys()))
X = e41['X_rows']
print('X summary keys:', list(e41['X'].keys()) if isinstance(e41['X'], dict) else type(e41['X']))
print('n rows X:', len(X), 'row keys:', sorted(X[0].keys()))
for p in ['d1', 'd2', 'ring', 'w']:
    sm = paired([r[f'M_{p}'] - r['m2'] for r in X])
    st_ = paired([r['oracle'] - r[f'T_{p}'] for r in X])
    print(f'  {p:5s} MAIN {fmt(sm)}')
    print(f'        TOTAL {fmt(st_)}')
so = paired([r['oracle'] - r['m2'] for r in X])
print('  oracle-m2:', fmt(so))
summ = sum(np.mean([r[f'M_{p}'] - r['m2'] for r in X]) for p in ['d1', 'd2', 'ring', 'w'])
print(f'  sum(MAIN) = {summ:+.4f}  vs oracle-m2 {so["mean"]:+.4f} '
      f'-> interaction share {(so["mean"]-summ)/so["mean"]*100:.0f}%')

print('\n=========== E-43 dose (from JSON) ===========')
e43 = json.load(open(D + 'e43_external_regime_dose.json'))
print('keys:', list(e43.keys()))
arms = e43.get('arms', e43)
for tag in ['D', 'R', 'W', 'C']:
    if tag in arms:
        print(f'-- arm {tag} --')
        for a in arms[tag]:
            print('  ', {k: (round(v, 4) if isinstance(v, float) else v)
                         for k, v in a.items() if k in
                         ('dose', 'name', 'sd', 'sT', 'sw', 'mean', 'se', 'lo',
                          'hi', 'win', 'n', 'gain')})
for k in ['D_cross', 'R_cross', 'W_cross']:
    if k in arms:
        print(k, arms[k])

print('\n=========== E-39 in-dist gap ===========')
e39 = json.load(open(D + 'e39_indist_gap.json'))
print('keys:', list(e39.keys()))
rows = e39['rows'] if 'rows' in e39 else None
if rows:
    print('n rows:', len(rows), 'row keys:', sorted(rows[0].keys()))
prov = json.load(open(D + 'e39_provenance.json'))
print('provenance:', json.dumps(prov)[:600])

print('\n=========== E-37 ladder (B_blindW etc.) ===========')
e37l = json.load(open(D + 'e37_bimodal_ladder.json'))
print('keys:', list(e37l.keys()))
r = e37l['rows'] if 'rows' in e37l else None
if r:
    print('n', len(r), 'row keys:', sorted(r[0].keys()))
