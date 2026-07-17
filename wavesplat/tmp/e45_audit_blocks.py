"""e45 audit part 2 — block heterogeneity + e43c/e43e/e42 blind stats."""
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
    return (f"{s['mean']:+.4f} +-{s['se']:.4f} CI[{s['lo']:+.4f},{s['hi']:+.4f}] "
            f"win {s['win']}/{s['n']}")


print('=========== per-block oracle - cur ===========')
ests = []

e37 = json.load(open(D + 'e37_crosstp_exploratory.json'))
r0 = e37['rows'][0]
print('e37 row keys:', sorted(r0.keys()))
d = [r['ens2_oracle'] - r['cur_grid'] for r in e37['rows']
     if 'ens2_oracle' in r]
if d:
    s = paired(d)
    print('block si=0..31   :', fmt(s))
    ests.append(('si0..31', s))

e42 = json.load(open(D + 'e42_blind_recovery.json'))
r0 = e42['rows'][0]
print('e42 row keys:', sorted(r0.keys()))

e43b = json.load(open(D + 'e43b_v3_pipeline.json'))
print('e43b row keys:', sorted(e43b['rows'][0].keys()))

e43d = json.load(open(D + 'e43d_gate.json'))
print('e43d keys:', list(e43d.keys()), 'row keys:', sorted(e43d['rows'][0].keys()))

e43e = json.load(open(D + 'e43e_fresh_battery.json'))
print('e43e row keys:', sorted(e43e['rows'][0].keys()))
