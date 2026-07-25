"""STAGE-3b analysis — pre-registered primary + secondaries. Reads wp_real.json."""
import json
import numpy as np
from scipy import stats as ss

d = json.load(open('tmp/wp_real_2026-07-25/wp_real.json'))
R = d['results']
COLLAPSE = {tuple(c) for c in d['collapse']}
RECS = ['YS17', 'AA15a', 'AA14', 'AA15b', 'LJ18g']


def rank_biserial_signed(delta):
    """matched-pairs rank-biserial: (R+ - R-)/(R+ + R-); + => learned better."""
    dd = np.array([x for x in delta if x != 0.0])
    if len(dd) == 0:
        return 0.0
    ranks = ss.rankdata(np.abs(dd))
    rp = ranks[dd > 0].sum(); rm = ranks[dd < 0].sum()
    return float((rp - rm) / (rp + rm)) if (rp + rm) > 0 else 0.0


normal = [r for r in R if not r['collapse']]
coll = [r for r in R if r['collapse']]
delN = np.array([r['delta_now'] for r in normal])          # hand - learned; + => learned better

print(f"=== PRIMARY: normal-{len(normal)} windows, nowcast map nRMSE, paired ===")
print(f"  hand mean  {np.mean([r['hand_now'] for r in normal]):.4f}   "
      f"learned mean {np.mean([r['learned_now'] for r in normal]):.4f}")
print(f"  mean Delta (hand-learned) {delN.mean():+.4f}   median Delta {np.median(delN):+.4f}")
print(f"  learned-better wins {int((delN>0).sum())}/{len(delN)}")
W, p = ss.wilcoxon(delN, alternative='two-sided', zero_method='wilcox')
print(f"  two-sided Wilcoxon W={W:.1f}  p={p:.4g}   rank-biserial={rank_biserial_signed(delN):+.3f}")
med = np.median(delN)
if p < 0.05 and med > 0:
    verdict = "WIN (niche EXTENDS to real data)"
elif p < 0.05 and med < 0:
    verdict = "LOSS-significant (learned worse on real; niche synthetic-crowded only)"
elif med > 0:
    verdict = "BOUND (learned better in point est. but p>=0.05, underpowered)"
else:
    verdict = "LOSS (median Delta<=0, learned not better; niche synthetic-crowded only)"
print(f"  >>> PRIMARY VERDICT: {verdict}")

print(f"\n=== ALL-30 (secondary) ===")
del30 = np.array([r['delta_now'] for r in R])
W3, p3 = ss.wilcoxon(del30, alternative='two-sided', zero_method='wilcox')
print(f"  mean {del30.mean():+.4f} median {np.median(del30):+.4f} wins {int((del30>0).sum())}/30 "
      f"W={W3:.1f} p={p3:.4g}")

print(f"\n=== (a) COLLAPSE windows (hand>=1.0), 'fixed' = learned<1.0 ===")
for r in sorted(coll, key=lambda x: (x['record'], x['win'])):
    fixed = 'FIXED' if r['learned_now'] < 1.0 else 'not fixed'
    print(f"  {r['record']} w{r['win']}: hand {r['hand_now']:.3f} -> learned {r['learned_now']:.3f}"
          f" (d {r['delta_now']:+.3f})  [{fixed}]")

print(f"\n=== (b) DOPPLER 2x2 on YS17 (gated record), mean nowcast over 6 windows ===")
ys = [r for r in R if r['record'] == 'YS17']
hu0 = np.mean([r['hand_now'] for r in ys]); hum = np.mean([r['hand_now_umed'] for r in ys])
lu0 = np.mean([r['learned_now_u0_helper'] for r in ys]); lum = np.mean([r['learned_now_umed'] for r in ys])
print(f"  U_med applied = {d['ys17_u_med']}")
print(f"                 U=0        U=U_med     delta(U0-Umed)")
print(f"    hand      {hu0:8.3f}   {hum:8.3f}   {hu0-hum:+.3f}")
print(f"    learned   {lu0:8.3f}   {lum:8.3f}   {lu0-lum:+.3f}")
print(f"  (learned U=0 helper vs a2f primary mean {np.mean([r['learned_now'] for r in ys]):.3f})")
dhand = np.array([r['hand_now']-r['hand_now_umed'] for r in ys])
dlearn = np.array([r['learned_now_u0_helper']-r['learned_now_umed'] for r in ys])
print(f"  per-window Doppler gain (U0-Umed): hand mean {dhand.mean():+.3f} ({int((dhand>0).sum())}/6),"
      f" learned mean {dlearn.mean():+.3f} ({int((dlearn>0).sum())}/6)")

print(f"\n=== (c) PER-RECORD splits + proposal sparsity (n_eff) ===")
print(f"  {'rec':6s} {'n':>2s} {'hand':>6s} {'learn':>6s} {'medD':>7s} {'wins':>5s} {'n_eff':>6s}")
for rec in RECS:
    rr = [r for r in R if r['record'] == rec]
    dn = np.array([r['delta_now'] for r in rr])
    print(f"  {rec:6s} {len(rr):2d} {np.mean([r['hand_now'] for r in rr]):6.3f} "
          f"{np.mean([r['learned_now'] for r in rr]):6.3f} {np.median(dn):+7.3f} "
          f"{int((dn>0).sum()):2d}/{len(rr)} {np.mean([r['n_eff'] for r in rr]):6.1f}")
print(f"  pooled n_eff (all 30) = {np.mean([r['n_eff'] for r in R]):.1f} "
      f"(normal {np.mean([r['n_eff'] for r in normal]):.1f}); crowded-synth proposals were dense (~15-30)")

print(f"\n=== lead-averaged (context), normal-{len(normal)} ===")
lh = np.array([r['hand_lead'] for r in normal]); ll = np.array([r['learned_lead'] for r in normal])
dl = lh - ll
print(f"  hand {lh.mean():.3f} learned {ll.mean():.3f} meanD {dl.mean():+.3f} "
      f"wins {int((dl>0).sum())}/{len(dl)} p={ss.wilcoxon(dl, alternative='two-sided')[1]:.4g}")

print(f"\n=== beamform-map validity (mission check) ===")
edge = [r for r in R if r['bf_peak_edge']]
print(f"  peak-at-k-edge windows: {len(edge)}/30 "
      f"(edge => real obs out of the net's k-range [-0.6,0.6])")
peaks = np.array([r['bf_peak'] for r in R])
print(f"  |peak k| range {np.hypot(peaks[:,0],peaks[:,1]).min():.3f}-{np.hypot(peaks[:,0],peaks[:,1]).max():.3f}")

print(f"\n=== FULL PAIRED TABLE (30 windows) ===")
print(f"  {'rec':6s} {'w':>1s} {'C':>1s} {'hand':>6s} {'learn':>6s}(sd)   {'d':>7s} {'n_eff':>5s}")
for r in sorted(R, key=lambda x: (RECS.index(x['record']), x['win'])):
    c = '*' if r['collapse'] else ' '
    print(f"  {r['record']:6s} {r['win']:1d} {c} {r['hand_now']:6.3f} {r['learned_now']:6.3f}"
          f"({r['learned_now_sd']:.3f}) {r['delta_now']:+7.3f} {r['n_eff']:5.1f}")
