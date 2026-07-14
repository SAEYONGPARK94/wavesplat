"""E-12f — re-run the inference on EVERY branch claim that used a bad SE.

Two defects found in E-12e:
  D1  d3_cluster_se.py built its 95% CI with 1.96 on G=4..5 clusters.
      The reference distribution is t(G-1): crit 3.182 (G=4) / 2.776 (G=5).
      [FIXED in scripts/d3_cluster_se.py — all 3 claims still exclude 0.]
  D2  d3_e10_pattern_real.py:178 and d3_e11_sparse_characterize.py:147 use
      se = std/sqrt(n) over 60 WINDOWS -- an iid SE on data that is clustered
      in 5 records. They never clustered at all. E-10 already FAILED (a wider
      SE only makes it fail harder), but **E-11 PASSED on that SE** and E-11 is
      one of the branch's starred positives.

This recomputes both with CR1 + t(G-1) from the stored JSONs. No re-running.
"""
import sys
import os
import json
import numpy as np
from scipy import stats

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
os.chdir(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, 'scripts')

from d3_cluster_se import cluster_se       # noqa: E402


def judge(name, d, g):
    d = np.asarray(d, float)
    mu = d.mean()
    se_iid = d.std(ddof=1) / np.sqrt(len(d))
    se_cl, G = cluster_se(d, g)
    tcrit = stats.t.ppf(0.975, G - 1)
    t_cl = mu / se_cl
    p_cl = 2 * stats.t.sf(abs(t_cl), G - 1)
    lo, hi = mu - tcrit * se_cl, mu + tcrit * se_cl
    ok = (lo > 0) or (hi < 0)
    # published (wrong) version
    lo_o, hi_o = mu - 1.96 * se_iid, mu + 1.96 * se_iid
    print(f"{name:<16}{mu:+8.4f} | iid SE {se_iid:.4f} t {mu/se_iid:+5.2f} "
          f"CI [{lo_o:+.4f},{hi_o:+.4f}] (published)")
    print(f"{'':<16}{'':8} | CR  SE {se_cl:.4f} t {t_cl:+5.2f} "
          f"CI [{lo:+.4f},{hi:+.4f}]  p={p_cl:.4f}  G={G}  "
          f"--> {'EXCLUDES 0' if ok else '*** INCLUDES 0 ***'}")
    return dict(delta=float(mu), se_iid=float(se_iid), se_cluster=float(se_cl),
                t_cluster=float(t_cl), p_cluster=float(p_cl), G=int(G),
                ci=[float(lo), float(hi)], excludes_zero=bool(ok))

out = {}

print('=' * 92)
print('E-11 (sparse hint) — the STARRED positive that was judged on an iid SE')
print('=' * 92)
e11 = json.load(open('results/direction3/e11_sparse_characterize.json'))
w11 = e11['windows']
g = [r['record'] for r in w11]
for tag in ('dense_hint', 'sparse_hint'):
    out[f'e11_{tag}'] = judge(tag, [r[tag]['delta'] for r in w11], g)
# the E-11 gate was on the sparse arm; also test sparse-vs-dense (paired)
dd = np.array([r['sparse_hint']['delta'] - r['dense_hint']['delta']
               for r in w11])
out['e11_sparse_minus_dense'] = judge('sparse - dense', dd, g)

print('\n' + '=' * 92)
print('E-10 (pattern sensitivity) — already FAIL; confirm a wider SE keeps it FAIL')
print('=' * 92)
e10 = json.load(open('results/direction3/e10_pattern_real.json'))
w10 = e10['windows']
g10 = [r['record'] for r in w10]
for arm in e10['arms']:
    out[f'e10_{arm}'] = judge(arm, [r[arm]['delta'] for r in w10], g10)

print('\n' + '=' * 92)
print('BONUS — E-11 already stored Hs_dense / Hs_sparse: the sigma0 oracle gap')
print('=' * 92)
hd = np.array([r['Hs_dense'] for r in w11])
hsp = np.array([r['Hs_sparse'] for r in w11])
rr = hd / hsp
print(f"  Hs_dense / Hs_sparse: median {np.median(rr):.4f}x, "
      f"mean {rr.mean():.4f}x, range [{rr.min():.3f}, {rr.max():.3f}]")
print(f"  windows within 5% : {100*np.mean(np.abs(rr-1) < 0.05):.0f}%")
print(f"  windows within 10%: {100*np.mean(np.abs(rr-1) < 0.10):.0f}%")
kre = None
if kre is not None:
    print(f"  (E-11's own kp_rel_err: median {np.median(kre):.3f}, "
          f"90th pct {np.percentile(kre, 90):.3f})")
out['sigma0_oracle_gap'] = dict(median=float(np.median(rr)),
                                within_5pct=float(np.mean(np.abs(rr-1) < .05)),
                                within_10pct=float(np.mean(np.abs(rr-1) < .10)))

json.dump(out, open('results/direction3/e12f_reinference.json', 'w'), indent=1)
print('\n-> results/direction3/e12f_reinference.json')
