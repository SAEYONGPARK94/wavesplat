"""E-12e — attack E-12d's PASS. Four holes I suspect in my own result.

H1  SMALL-CLUSTER INFERENCE. d3_cluster_se.py builds its 95% CI with 1.96 — the
    NORMAL critical value — on G=5 clusters. With 5 clusters the reference
    distribution is t(G-1)=t(4), whose 95% critical value is 2.776, not 1.96.
    This is exactly the defect that made E-10 fail. Does E-12d survive the
    correct reference distribution? Also do the most conservative test possible:
    a plain t-test on the FIVE record-level means (n=5).

H2  LEAVE-ONE-RECORD-OUT. LJ18g contributes +0.105 vs a pooled +0.060. Is the
    result driven by one record?

H3  A SECOND ORACLE INPUT. sigma0 = 0.05 * Hs, and Hs = 4*std(Zmem) — the DENSE
    GT buffer (d3_gate_promotion_eval.py:61). So E-12d removed the dense kp but
    left a dense-derived noise scale. It is PAIRED (both arms get it), so Delta
    stays valid, but "sensor-realizable" is then overstated. How big is the gap
    between the oracle Hs and the Hs a sensor would compute from its own 600
    points (4*std(z))? This project already knows the noise floor is a sensitive
    knob (2026-07-11: "do not pick the noise floor from in-window residuals").

H4  FORKING PATHS. I ran E-12b (PASS), rejected it, redesigned, ran E-12d (PASS).
    Did I just fish until something passed? Check: was the direction the same
    every time, and was the baseline already significant?
"""
import sys
import os
import json
import numpy as np
from scipy import stats

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
os.chdir(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, 'scripts')

from d3_cluster_se import cluster_se            # noqa: E402
from d3_falcon_real_operator import RECORDS     # noqa: E402
from d3_sa_multirecord import MRecord           # noqa: E402
from d3_external_figs import Window             # noqa: E402

RECS = ['YS17', 'AA15a', 'AA14', 'AA15b', 'LJ18g']
d = json.load(open('results/direction3/e12d_sparse_hint.json'))
rows = d['rows']
g = np.array([r['record'] for r in rows])

print('=' * 78)
print('H1 — small-cluster inference: is 1.96 the right critical value on G=5?')
print('=' * 78)
for mode in ('spatial', 'sparse'):
    dd = np.array([r[f'd_{mode}'] for r in rows])
    se, G = cluster_se(dd, g)
    mu = dd.mean()
    tcrit_n = 1.96
    tcrit_t = stats.t.ppf(0.975, G - 1)          # correct: t(G-1)
    ci_n = (mu - tcrit_n * se, mu + tcrit_n * se)
    ci_t = (mu - tcrit_t * se, mu + tcrit_t * se)
    p_t = 2 * (1 - stats.t.cdf(abs(mu / se), G - 1))
    print(f"\n{mode:>8}: delta {mu:+.4f}, CR-SE {se:.4f}, t {mu/se:.2f}, G={G}")
    print(f"          repo CI (1.96, WRONG on G=5) : [{ci_n[0]:+.4f}, {ci_n[1]:+.4f}]")
    print(f"          correct CI (t(4)={tcrit_t:.3f})   : [{ci_t[0]:+.4f}, {ci_t[1]:+.4f}]"
          f"   -> p = {p_t:.4f}  {'EXCLUDES 0' if ci_t[0] > 0 else '*** INCLUDES 0 ***'}")

print('\n  most conservative test — t-test on the FIVE record means (n=5):')
for mode in ('spatial', 'sparse'):
    m = np.array([np.mean([r[f'd_{mode}'] for r in rows if r['record'] == nm])
                  for nm in RECS])
    t, p = stats.ttest_1samp(m, 0.0)
    print(f"    {mode:>8}: mean {m.mean():+.4f}  t(4) {t:+.2f}  p {p:.4f}  "
          f"{'PASS' if p < 0.05 else 'FAIL'}   per-record {np.round(m, 3)}")
# sign test
print(f"\n    sign test (5/5 records positive): two-sided p = {2*0.5**5:.4f}")

print('\n' + '=' * 78)
print('H2 — leave-one-record-out (is one record driving it?)')
print('=' * 78)
print(f"{'dropped':<10}{'delta':>10}{'CR-SE':>9}{'t(3)':>8}{'p':>9}{'verdict':>10}")
for drop in RECS:
    m = g != drop
    dd = np.array([r['d_sparse'] for r in rows])[m]
    se, G = cluster_se(dd, g[m])
    mu = dd.mean()
    p = 2 * (1 - stats.t.cdf(abs(mu / se), G - 1))
    print(f"{drop:<10}{mu:+10.4f}{se:9.4f}{mu/se:8.2f}{p:9.4f}"
          f"{'PASS' if p < 0.05 and mu >= 0.020 else 'fail':>10}")

print('\n' + '=' * 78)
print('H3 — the second oracle input: sigma0 = 0.05 * Hs, Hs = 4*std(DENSE Zmem)')
print('=' * 78)
print('   how far is the oracle Hs from the Hs a sensor gets from its own 600 pts?')
print(f"\n{'record':<8}{'Hs_dense':>10}{'Hs_sparse':>11}{'ratio':>8}"
      f"{'-> sigma0 ratio':>17}")
gaps = []
for name in RECS:
    cfg = RECORDS[name]
    rec = MRecord(cfg['path'], stride=cfg['stride'])
    rr = []
    for wi in range(12):
        w = Window(name, wi, rec=rec, net=False or None, head=None) \
            if False else None
        break
    rec.f.close()
# Window() would load the net; rebuild the two Hs directly instead
from d3_gate_promotion_eval import window_data  # noqa: E402
for name in RECS:
    cfg = RECORDS[name]
    rec = MRecord(cfg['path'], stride=cfg['stride'])
    nfw = int(round(40.0 / rec.dt))
    n_lead = int(round(cfg['lead_max'] / rec.dt))
    i0s = np.linspace(nfw + 1, rec.T_all - n_lead - 2, 12).astype(int)
    rs = []
    for wi, i0 in enumerate(i0s):
        P, z, Pq, gt_q, leads, Hs, kp, _ = window_data(
            rec, cfg, int(i0), nfw, n_lead, wi)
        Hs_sparse = 4.0 * float(np.std(z))
        rs.append(Hs / Hs_sparse)
        gaps.append(Hs / Hs_sparse)
    rec.f.close()
    print(f"{name:<8}{'':>10}{'':>11}{np.median(rs):8.3f}"
          f"{np.median(rs):17.3f}")
gaps = np.array(gaps)
print(f"\n   pooled: median {np.median(gaps):.3f}x, range "
      f"[{gaps.min():.3f}, {gaps.max():.3f}]")
print('   (sigma0 scales linearly with Hs, so this IS the sigma0 error factor)')

print('\n' + '=' * 78)
print('H4 — forking paths')
print('=' * 78)
print('   frozen baseline (before any of this): +0.0334, CR-t 3.70  -> ALREADY significant')
print('   E-12b causal arm  : +0.0703  (same direction, larger)')
print('   E-12d sparse arm  : +0.0602  (same direction, larger)')
print('   -> every hint variant moves the SAME way. The finding is not a')
print('      significance that appeared only after searching; the baseline was')
print('      already significant and every re-specification enlarges it.')
