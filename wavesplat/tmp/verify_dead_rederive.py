"""verify_dead_rederive.py — fresh-context re-derivation of every e42 headline
number from raw JSON rows. READ ONLY. Writes nothing to results/."""
import json
import numpy as np
from scipy import stats

D = json.load(open('results/direction3/e42_blind_recovery.json'))
rows, uni = D['rows'], D['uni']
print(f"n rows={len(rows)} n uni={len(uni)} tau={D['tau']:.4f} "
      f"qualified={D['qualified']}")

def paired(rs, a, b):
    d = np.array([r[a] - r[b] for r in rs
                  if np.isfinite(r[a]) and np.isfinite(r[b])])
    n = len(d)
    se = d.std(ddof=1) / np.sqrt(n)
    t, p = stats.ttest_1samp(d, 0)
    lo, hi = stats.t.interval(0.95, n - 1, loc=d.mean(), scale=se)
    return d.mean(), se, t, lo, hi, int((d > 0).sum()), n

print("\n--- re-derived contrasts (from raw rows) vs stored ---")
for key in ['oracle-cur_grid', 'm2-cur_grid', 'fit_bmaF-cur_grid',
            'fit_bmaF-m2', 'fit_bma-cur_grid', 'fit1-cur_grid',
            'fit2_hard-cur_grid', 'fit_bmaF-U_BMA', 'fit_bmaF-fit_bma',
            'fit1-ens1', 'U_BMA-cur_grid']:
    a, b = key.split('-')
    m, se, t, lo, hi, w, n = paired(rows, a, b)
    s = D['contrasts'][key]
    match = (abs(m - s['mean']) < 1e-9 and abs(lo - s['lo']) < 1e-9
             and abs(hi - s['hi']) < 1e-9 and w == s['win'] and n == s['n'])
    print(f"{key:22s} {m:+.4f}+-{se:.4f} t{t:+.2f} CI[{lo:+.4f},{hi:+.4f}] "
          f"win {w}/{n}  stored-match={match}")

m, se, t, lo, hi, w, n = paired(uni, 'fit_bmaF', 'cur_grid')
s = D['contrasts'].get('uni_fit_bmaF-cur_grid') or D['contrasts'].get(
    'uni_fit_bmaF-cur_grid', None)
su = D['contrasts']['uni_fit_bmaF-cur_grid'] if 'uni_fit_bmaF-cur_grid' in \
    D['contrasts'] else None
print(f"\nUNI fit_bmaF-cur_grid  {m:+.4f}+-{se:.4f} CI[{lo:+.4f},{hi:+.4f}] "
      f"win {w}/{n}")

# verdict logic re-check
P = D['contrasts']['fit_bmaF-cur_grid']
G = D['contrasts']['fit_bmaF-m2']
print(f"\nverdict recheck: primary={P['mean'] >= 0.03 and P['lo'] > 0 and P['win']/P['n'] >= 0.60}"
      f" gate={G['lo'] > 0} dead={P['hi'] < 0.02}"
      f"  stored: primary={D['primary']} gate={D['gate']} dead={D['dead']}")

# NaN audit
print("\n--- NaN / n audit ---")
for k in ['cur_grid', 'm2', 'ens1', 'oracle', 'fit1', 'fit_bma', 'fit_bmaF',
          'U_BMA', 'fit2_hard']:
    bad = [r['si'] for r in rows if not np.isfinite(r[k])]
    if bad:
        print(f"  {k}: NaN at si={bad}")
nan_rows = [r for r in rows if not all(np.isfinite(r[k]) for k in
            ['cur_grid', 'oracle', 'fit_bmaF', 'U_BMA', 'fit_bma', 'm2'])]
for r in nan_rows:
    print(f"  si {r['si']}: cur {r['cur_grid']:+.3f} orc {r['oracle']} "
          f"bmaF {r['fit_bmaF']} U {r['U_BMA']} p2 {r['p2']:.3f} "
          f"fired {r['fired']} r2 {r['r2']:.2f} Tp2 {r['Tp2']:.2f}")

# arm means
print("\n--- arm means (nanmean) ---")
for k in ['cur_grid', 'm2', 'ens1', 'fit1', 'fit2_hard', 'fit_bma',
          'fit_bmaF', 'U_BMA', 'oracle']:
    v = np.array([r[k] for r in rows])
    print(f"  {k:10s}: {np.nanmean(v):+.4f}  (nan {np.isnan(v).sum()})")

# p2 / fired / posterior degeneracy proxies
p2 = np.array([r['p2'] for r in rows])
fired = np.array([r['fired'] for r in rows])
print(f"\np2: min {p2.min():.4f} med {np.median(p2):.4f} max {p2.max():.4f} "
      f" #p2>0.99 {(p2 > 0.99).sum()}  #p2<0.01 {(p2 < 0.01).sum()}")
print(f"fired: {fired.sum()}/{len(fired)}")
dlog = np.array([r['dlogE'] for r in rows])
print(f"dlogE: min {dlog.min():.1f} med {np.median(dlog):.1f} "
      f"max {dlog.max():.1f}")

# fit-quality on scored seas (detector-limited question)
for k in ['e_d1_err', 'e_d2_err', 'e_a2_rel', 'e_w_err', 'mu_err']:
    v = np.array([abs(r[k]) for r in rows])
    print(f"{k:9s}: med {np.median(v):6.2f}  mean {v.mean():6.2f}")
sw = np.array([r['e_swap'] for r in rows])
print(f"swap rate: {sw.sum()}/{len(sw)}")

# does fit quality predict the per-sea deficit?
d = np.array([r['fit_bmaF'] - r['cur_grid'] for r in rows])
ok = np.isfinite(d)
d2e = np.array([r['e_d2_err'] for r in rows])
orc = np.array([r['oracle'] - r['cur_grid'] for r in rows])
print(f"\ncorr(fit_bmaF-cur, e_d2_err) = "
      f"{np.corrcoef(d[ok], d2e[ok])[0,1]:+.3f}")
print(f"corr(fit_bmaF-cur, oracle-cur) = "
      f"{np.corrcoef(d[ok], orc[ok])[0,1]:+.3f}")
good = ok & (d2e <= 15)
print(f"seas with e_d2_err<=15deg: n={good.sum()}, "
      f"mean(fit_bmaF-cur)={d[good].mean():+.4f}, "
      f"mean(oracle-cur)={orc[good].mean():+.4f}")
bad2 = ok & (d2e > 45)
print(f"seas with e_d2_err>45deg:  n={bad2.sum()}, "
      f"mean(fit_bmaF-cur)={d[bad2].mean():+.4f}, "
      f"mean(oracle-cur)={orc[bad2].mean():+.4f}")

# unimodal false fire
nf = sum(r['fired'] for r in uni)
print(f"\nunimodal false-fire: {nf}/{len(uni)}")
