"""e45 audit part 4 — e39 gap, e37 contrasts, e42 U_BMA/HYBRID, e43c mech."""
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
            f"p{s['p']:.3g} n{s['n']}")


print('=========== E-39 in-dist gap by mode ===========')
e39 = json.load(open(D + 'e39_indist_gap.json'))
rows = e39['rows']
modes = sorted(set(r['mode'] for r in rows))
print('modes:', modes)
print('overall oracle-net:', fmt(paired([r['oracle'] - r['net'] for r in rows])))
for m in modes:
    rr = [r for r in rows if r['mode'] == m]
    print(f"  {m:12s} oracle-net: {fmt(paired([r['oracle']-r['net'] for r in rr]))}")
print('net - v7:', fmt(paired([r['net'] - r['v7'] for r in rows])))
print('stored summary:', json.dumps(e39.get('summary'))[:800])

print('\n=========== E-37 contrasts ===========')
e37l = json.load(open(D + 'e37_bimodal_ladder.json'))
print('contrasts:', json.dumps(e37l['contrasts'])[:1200])
print('w_blind:', e37l.get('w_blind'), 'w_oracle:', e37l.get('w_oracle'))

print('\n=========== E-42 U_BMA / HYBRID ===========')
e42 = json.load(open(D + 'e42_blind_recovery.json'))
R = e42['rows']
print('U_BMA - m2 :', fmt(paired([r['U_BMA'] - r['m2'] for r in R])))
print('fit_bmaF - U_BMA:', fmt(paired([r['fit_bmaF'] - r['U_BMA'] for r in R])))
fired = [r for r in R if r['fired']]
nf = [r for r in R if not r['fired']]
print(f'fired {len(fired)} / not {len(nf)}')
print('fired  fit_bmaF-cur:', fmt(paired([r['fit_bmaF'] - r['cur_grid'] for r in fired])))
print('notfired fit_bmaF-cur:', fmt(paired([r['fit_bmaF'] - r['cur_grid'] for r in nf])))
hyb = [(r['fit_bmaF'] if r['fired'] else r['cur_grid']) - r['cur_grid'] for r in R]
print('HYBRID - cur:', fmt(paired(hyb)))
print('uni do-no-harm keys:', sorted(e42['uni'][0].keys()) if e42.get('uni') else None)
if e42.get('uni'):
    u = e42['uni']
    if 'fit_bmaF' in u[0]:
        print('uni fit_bmaF-cur:', fmt(paired([r['fit_bmaF'] - r['cur_grid'] for r in u])))

print('\n=========== E-43c mechanism ===========')
e43c = json.load(open(D + 'e43c_true_node_rank.json'))
print('verdict:', e43c['verdict'][:200] if isinstance(e43c.get('verdict'), str) else e43c.get('verdict'))
print('pct_true_mean:', e43c.get('pct_true_mean'), 'pct_perm:', e43c.get('pct_perm_mean'),
      'paired_t:', e43c.get('paired_t'), 'p:', e43c.get('paired_p'),
      'ess_median:', e43c.get('ess_median'))
rc = [r for r in e43c['rows'] if not r.get('excluded')]
print('n scored:', len(rc), 'row keys:', sorted(rc[0].keys()))
print('FULL-cur re-derived:', fmt(paired([r['sc_full'] - r['sc_cur'] for r in rc])))
if 'sc_node_oracle' in rc[0]:
    print('node_oracle-cur:', fmt(paired([r['sc_node_oracle'] - r['sc_cur'] for r in rc])))
# best-node errors
for k in ['best_d1_err', 'best_d2_err', 'd1_err', 'd2_err']:
    if k in rc[0]:
        v = [r[k] for r in rc]
        print(k, 'median', np.median(v).round(2))
