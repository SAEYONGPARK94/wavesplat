"""D-20 (5-record) CORRECT cluster-robust inference, recomputed from saved rows.

The gate in d3_d20_replay_ctrl.py reports a WINDOW-POOLED iid t (treats all
windows as independent -> the flaw the audit condemned, §10.3). The valid
inference clusters on the RECORD: k=5 record-level paired means (learned-hand),
sea-mean t(k-1) + Webb wild cluster bootstrap + sign test. Read the json, do
not re-run the replay.
"""
import json
import sys
import numpy as np
from scipy import stats

J = sys.argv[1] if len(sys.argv) > 1 else \
    'results/direction3/d20_replay_eval.json'
d = json.load(open(J))
rows = d['rows']
RECS = d['config']['records']
TAUS = d['config']['taus']

WEBB = np.array([-np.sqrt(1.5), -1., -np.sqrt(.5), np.sqrt(.5), 1., np.sqrt(1.5)])


def wild_t(y, nboot=20000, seed=0):
    rng = np.random.default_rng(seed)
    t_o = y.mean() / (y.std(ddof=1) / np.sqrt(len(y)))
    b = []
    for _ in range(nboot):
        yb = y * rng.choice(WEBB, len(y))
        se = yb.std(ddof=1) / np.sqrt(len(y))
        b.append(yb.mean() / se if se > 0 else 0.0)
    b = np.array(b)
    return float(t_o), float((np.abs(b) >= abs(t_o)).mean())


print(f"{'='*70}\nD-20 cluster-robust re-adjudication ({len(RECS)} records)\n{'='*70}")

# ---- G-D20-2 headline: learned-FF residual <= hand-FF, clustered on record ----
rec_d = {}
for name in RECS:
    dv = [r['pct_learned'] - r['pct_hand'] for r in rows if r['record'] == name]
    rec_d[name] = float(np.mean(dv)) if dv else float('nan')
d_rec = np.array([rec_d[n] for n in RECS])
k = len(d_rec)
t_c, p_c = stats.ttest_1samp(d_rec, 0.0)
t_w, p_w = wild_t(d_rec)
n_better = int((d_rec < 0).sum())
# sign test (two-sided) on k records
p_sign = float(stats.binomtest(n_better, k, 0.5).pvalue)

print(f"\nG-D20-2  learned-FF residual - hand-FF  (negative = learning helps)")
print(f"{'record':>8}  paired mean(learn-hand)")
for n in RECS:
    print(f"{n:>8}  {rec_d[n]:+.4f}  {'LEARN' if rec_d[n] < 0 else 'hand'}")
print(f"\n  record-mean          {d_rec.mean():+.4f}")
print(f"  cluster t({k-1})       {t_c:+.3f}  p {p_c:.4f}")
print(f"  Webb wild boot-t     {t_w:+.3f}  p {p_w:.4f}")
print(f"  sign                 {n_better}/{k} learned-better  (sign-test p {p_sign:.4f})")
headline = bool(d_rec.mean() < 0 and p_c < 0.05 and p_w < 0.05)
print(f"  *** cluster-robust HEADLINE: {'PASS' if headline else 'not significant (k=%d wall)' % k}")

# ---- G-D20-1: hand-FF < persistence, clustered on record ----
print(f"\nG-D20-1  hand-FF < persistence (forecast value reproduced)")
for tau in TAUS:
    hp = []
    for name in RECS:
        s = [r for r in rows if r['record'] == name and r['tau'] == tau]
        if s:
            hp.append(np.mean([r['pct_hand'] for r in s])
                      < np.mean([r['pct_persist'] for r in s]))
    print(f"  tau {tau}: {sum(hp)}/{len(hp)} records hand<persist")

out = J.replace('.json', '_clusterrobust.json')
json.dump(dict(records=RECS, rec_paired_mean=rec_d,
               record_mean=float(d_rec.mean()),
               cluster_t=float(t_c), cluster_p=float(p_c),
               wild_t=float(t_w), wild_p=float(p_w),
               n_learned_better=n_better, k=k, sign_p=p_sign,
               headline_pass=headline), open(out, 'w'), indent=2)
print(f"\n-> {out}")
