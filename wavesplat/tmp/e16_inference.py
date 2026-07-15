"""Standalone CORRECT inference for E-16, recomputed from saved rows.

The e16_physics_operator.py run in progress loaded a wild_sea() with a bug
(unrestricted DGP + non-recentered statistic -> p~0.5 always). The per-sea rows
it saves are correct; only the final p_wild is wrong. This recomputes the
decisive inference from those rows with a NULL-IMPOSED Webb wild bootstrap-t,
plus the (already-valid) sea-level t-test. Read the json, do not recompute seas.
"""
import json
import sys
import numpy as np
from scipy import stats

J = sys.argv[1] if len(sys.argv) > 1 else \
    'results/direction3/e16_physics_operator.json'
d = json.load(open(J))
rows = d['rows'] if isinstance(d, dict) and 'rows' in d else d

e15 = json.load(open('results/direction3/e15_hos_multidraw.json'))
m3_stored = e15['summary']['sea_means']

seas = sorted(set(m3_stored) & {r['sea'] for r in rows if r['order'] == 1})
k = len(seas)
dM3 = np.array([m3_stored[s] for s in seas])
dM1 = np.array([np.mean([r['delta'] for r in rows
                         if r['sea'] == s and r['order'] == 1]) for s in seas])
dd = dM3 - dM1
steep = np.array([[r['steepness'] for r in rows if r['sea'] == s][0]
                  for s in seas])
lamv = np.array([[r['lam_p'] for r in rows if r['sea'] == s][0] for s in seas])

WEBB = np.array([-np.sqrt(1.5), -1., -np.sqrt(.5), np.sqrt(.5), 1., np.sqrt(1.5)])


def wild_t(y, nboot=20000, seed=0):
    """Null-imposed Webb wild bootstrap-t p-value for H0: mean(y)=0."""
    rng = np.random.default_rng(seed)
    t_o = y.mean() / (y.std(ddof=1) / np.sqrt(len(y)))
    b = []
    for _ in range(nboot):
        yb = y * rng.choice(WEBB, len(y))
        se = yb.std(ddof=1) / np.sqrt(len(y))
        b.append(yb.mean() / se if se > 0 else 0.0)
    b = np.array(b)
    return float(t_o), float((np.abs(b) >= abs(t_o)).mean())


t3, p3 = stats.ttest_1samp(dM3, 0.0)
t1, p1 = stats.ttest_1samp(dM1, 0.0)
tdd, pdd = stats.ttest_1samp(dd, 0.0)
tdd_w, pdd_w = wild_t(dd)
g1 = bool(dd.mean() > 0 and pdd < 0.05 and pdd_w < 0.05)

rho_dd, p_dd_s = stats.spearmanr(steep, dd)
rho_lam_dd, _ = stats.spearmanr(lamv, dd)
rho_lam_m1, p_lam_m1 = stats.spearmanr(lamv, dM1)
rho_lam_m3, _ = stats.spearmanr(lamv, dM3)

n_m1_draws = len([r for r in rows if r['order'] == 1]) // max(k, 1)
print(f"{'='*74}\nE-16 CORRECTED inference  ({k} seas, M1 fresh vs M3 reused)\n{'='*74}")
print(f"  Delta(M3) mean {dM3.mean():+.4f}  t({k-1}) {t3:+.2f} p {p3:.4g}  "
      f"sign {(dM3>0).sum()}/{k}")
print(f"  Delta(M1) mean {dM1.mean():+.4f}  t({k-1}) {t1:+.2f} p {p1:.4g}  "
      f"sign {(dM1>0).sum()}/{k}   [learning on CORRECT physics]")
print(f"  paired dd mean {dd.mean():+.4f}  t({k-1}) {tdd:+.2f} p {pdd:.4g}  "
      f"wildboot t {tdd_w:+.2f} p {pdd_w:.4g}  sign {(dd>0).sum()}/{k}")
print(f"  *** G1 decisive (dd>0, both p<0.05): {'PASS' if g1 else 'FAIL'}")
print(f"\n  G3 dd vs steepness (confound-free) Spearman {rho_dd:+.3f} p {p_dd_s:.4f}")
print(f"     dd vs lam {rho_lam_dd:+.3f} | M1 vs lam {rho_lam_m1:+.3f} p "
      f"{p_lam_m1:.4f} | M3 vs lam {rho_lam_m3:+.3f}")
print(f"\n{'sea':<13}{'steep':>8}{'lam':>7}{'d(M3)':>9}{'d(M1)':>9}{'dd':>9}")
for i in np.argsort(steep):
    print(f"{seas[i]:<13}{steep[i]:8.4f}{lamv[i]:7.1f}{dM3[i]:+9.4f}"
          f"{dM1[i]:+9.4f}{dd[i]:+9.4f}")

out = J.replace('.json', '_corrected.json')
json.dump(dict(k_seas=k, m1_draws=n_m1_draws,
               delta_M3=float(dM3.mean()), delta_M1=float(dM1.mean()),
               paired_dd=float(dd.mean()), t_dd=float(tdd), p_dd=float(pdd),
               t_dd_wild=float(tdd_w), p_dd_wild=float(pdd_w), G1_pass=g1,
               sign_dd=int((dd > 0).sum()), sign_M1=int((dM1 > 0).sum()),
               sign_M3=int((dM3 > 0).sum()),
               rho_dd_steep=float(rho_dd), p_dd_steep=float(p_dd_s),
               rho_lam_M1=float(rho_lam_m1), rho_lam_M3=float(rho_lam_m3),
               sea_dd={s: float(v) for s, v in zip(seas, dd)},
               sea_M1={s: float(v) for s, v in zip(seas, dM1)}),
          open(out, 'w'), indent=2)
print(f"\n-> {out}")
