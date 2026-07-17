"""verify_e43c_json — fresh-context audit, JSON-only re-derivation.

Attacks 2 (recompute FULL-cur, t-CI), 4 (seed hygiene: drop si=300..303),
5 (node-oracle CI vs prize elsewhere), 6 (ESS distribution), 7 (quantization
stats) from results/direction3/e43c_true_node_rank.json. No GPU needed.
"""
import json
import numpy as np
from scipy import stats

J = json.load(open('results/direction3/e43c_true_node_rank.json'))
rows = J['rows']
print(f"rows: {len(rows)}  excluded: {sum(r.get('excluded', False) for r in rows)}")
R = [r for r in rows if not r.get('excluded', False)]
si = np.array([r['si'] for r in R])
gf = np.array([r['sc_full'] - r['sc_cur'] for r in R])
go = np.array([r['sc_node_oracle'] - r['sc_cur'] for r in R])
assert not np.isnan(gf).any() and not np.isnan(go).any(), 'NaN scores present'
print(f"si range {si.min()}..{si.max()}  unique {len(set(si.tolist()))}")


def report(name, g):
    n = len(g)
    m = g.mean()
    se = g.std(ddof=1) / np.sqrt(n)
    lo_t, hi_t = stats.t.interval(0.95, n - 1, loc=m, scale=se)
    lo_z, hi_z = m - 1.96 * se, m + 1.96 * se
    t, p = stats.ttest_1samp(g, 0)
    win = int((g > 0).sum())
    print(f"{name}: n={n} mean {m:+.4f} +- {se:.4f}  "
          f"tCI [{lo_t:+.4f},{hi_t:+.4f}]  zCI [{lo_z:+.4f},{hi_z:+.4f}]  "
          f"win {win}/{n}  t {t:+.2f} p {p:.3g}")
    return m, se, lo_t, hi_t, win


print("\n--- attack 2: recompute headline ---")
m, se, lo, hi, win = report('FULL-cur (all 48)   ', gf)
rep = (0.0671, 0.0172, 0.0325, 0.1016, 32)
ok = (abs(m - rep[0]) < 5e-4 and abs(se - rep[1]) < 5e-4
      and abs(lo - rep[2]) < 5e-4 and abs(hi - rep[3]) < 5e-4 and win == rep[4])
print(f"matches reported +0.0671 +-0.0172 CI[+0.0325,+0.1016] win 32/48: {ok}")
print(f"reopen bars: mean>+0.03 {m > 0.03} | CI excl 0 {lo > 0} | "
      f"win>=60% {win / len(gf) >= 0.6}")

print("\n--- attack 4: drop smoke seas si=300..303 ---")
mask = si >= 304
report('FULL-cur (304..347) ', gf[mask])
print(f"smoke-4 mean: {gf[~mask].mean():+.4f} (si 300..303)")
# leave-one-out worst case
loo = [(np.delete(gf, i).mean(), si[i]) for i in range(len(gf))]
loo.sort()
print(f"leave-1-out mean range: [{loo[0][0]:+.4f} (drop si{loo[0][1]}), "
      f"{loo[-1][0]:+.4f} (drop si{loo[-1][1]})]")

print("\n--- attack 5: node-oracle on this block ---")
report('node_orc-cur (48)   ', go)
report('node_orc-cur (304+) ', go[mask])
print(f"recovery fraction FULL/oracle: {gf.mean() / go.mean():.3f}")
print("elsewhere: +0.0715 +-0.0141 (si=100..163), +0.0632 (si=200..263), "
      "+0.0982 (si=0..31 discovery)")

print("\n--- attack 6: ESS ---")
ess = np.array([r['ess'] for r in R])
print(f"ESS median {np.median(ess):.1f}  min {ess.min():.1f}  "
      f"max {ess.max():.1f}  frac<10 {np.mean(ess < 10):.3f}  "
      f"frac<48 {np.mean(ess < 48):.3f}")
tw = np.array([r['true_w_post'] for r in R])
print(f"true-node posterior mass: median {np.median(tw):.2e} max {tw.max():.2e}")

print("\n--- attack 7: quantization / errors ---")
d1e = np.array([r['best_d1_err'] for r in R])
d2e = np.array([r['best_d2_err'] for r in R])
bmt = np.array([r['dlogE_best_minus_true'] for r in R])
print(f"d1 err: median {np.median(d1e):.1f} | <=5deg {np.mean(d1e <= 5):.2f} "
      f"| <=15deg {np.mean(d1e <= 15):.2f} | >45deg {np.mean(d1e > 45):.2f}")
print(f"d2 err: median {np.median(d2e):.1f} | >45deg {np.mean(d2e > 45):.2f}")
big = (d1e > 45) & (d2e > 45)
print(f"both-dirs catastrophic (likely lobe SWAP, err not swap-corrected): "
      f"{big.sum()}/{len(R)}; their FULL-cur mean {gf[big].mean():+.4f}")
print(f"argmax==true-node (bmt==0): {int((bmt == 0).sum())}/{len(R)}; "
      f"best-true nats median {np.median(bmt):.1f}")
r_, p_ = stats.pearsonr(np.minimum(d1e, 60), gf)
print(f"corr(min(d1e,60), FULL-cur gain): {r_:+.3f} (p {p_:.3f})")

print("\n--- reported summary fields cross-check ---")
pt = np.array([r['pct_true'] for r in R])
pp = np.array([r['pct_perm'] for r in R])
tmp_ = np.array([r['dlogE_true_minus_perm'] for r in R])
t2, p2 = stats.ttest_rel(pt, pp)
print(f"pct_true mean {pt.mean():.3f} (rep 0.996) | pct_perm {pp.mean():.3f} "
      f"(rep 0.649) | t {t2:+.2f} (rep +9.71) p {p2:.3g}")
print(f"true-perm nats mean {tmp_.mean():+.1f} (rep +36.0), "
      f">0 in {int((tmp_ > 0).sum())}/{len(R)} (rep 48/48)")
print(f"JSON stored: pct_true_mean {J['pct_true_mean']:.4f} paired_t "
      f"{J['paired_t']:.2f} ess_median {J['ess_median']:.1f}")
