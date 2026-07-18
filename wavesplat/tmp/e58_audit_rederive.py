"""E-58 §8 independent audit — STEP 1: raw-data re-derivation of existing JSON.
No new compute. Verifies summary, tests heterogeneity, paired CRB-vs-emp/dist.
Follows [[feedback_dual_verification_default]] (raw re-derivation arm)."""
import json, numpy as np
from scipy import stats

J = json.load(open('wavesplat/results/direction3/e58_envelope_validation.json'))
eps = J['episodes']
def col(k): return np.array([e[k] for e in eps])
ca  = col('spear_crb_act'); ea = col('spear_emp_act')
da  = col('spear_dist_act'); cd = col('spear_crb_dist')
pcd = col('partial_crb_act_dist'); kp = col('kp'); names=[e['ep'] for e in eps]

print("=== 1. VERIFY SUMMARY (raw re-derivation) ===")
for lab,v in [('CRB-act',ca),('partial|dist',pcd),('emp-act',ea),('dist-act',da)]:
    print(f"  {lab:14s} median={np.median(v):+.3f}  n={len(v)}  min={v.min():+.3f} max={v.max():+.3f}")

# regime partition by kp (comp count proxy)
print("\n=== 2. HETEROGENEITY: partition by kp ===")
for u in sorted(set(np.round(kp,4))):
    m = np.abs(kp-u)<1e-4
    print(f"  kp={u:.4f}  n={m.sum():2d}  CRB-act med={np.median(ca[m]):+.3f}  "
          f"emp med={np.median(ea[m]):+.3f}  dist med={np.median(da[m]):+.3f}  "
          f"partial med={np.median(pcd[m]):+.3f}")

# comp count from name
comp = np.array([int(n.split('comp')[0].split('_')[-1]) for n in names])
print("\n  by component count:")
for c in sorted(set(comp)):
    m=comp==c
    print(f"  {c:2d}comp n={m.sum():2d}  CRB-act med={np.median(ca[m]):+.3f}  partial={np.median(pcd[m]):+.3f}")

# does rho scale with under-determination (lower kp = more under-determined)?
rho_kp, p_kp = stats.spearmanr(kp, ca)
rho_cp, p_cp = stats.spearmanr(comp, ca)
print(f"\n  Spearman(kp, CRB-act) = {rho_kp:+.3f} p={p_kp:.4f}  (neg => lower kp/more underdet -> stronger envelope)")
print(f"  Spearman(comp, CRB-act) = {rho_cp:+.3f} p={p_cp:.4f}  (pos => more comps -> stronger envelope)")

print("\n=== 3. PAIRED: CRB vs emp-UQ (per-episode) ===")
d = ca - ea
w = stats.wilcoxon(ca, ea) if len(ca)>5 else None
print(f"  CRB-emp per-ep diff: median={np.median(d):+.3f} mean={d.mean():+.3f}  win(CRB>emp)={int((d>0).sum())}/{len(d)}")
print(f"  Wilcoxon CRB vs emp: stat={w.statistic:.1f} p={w.pvalue:.4f}" if w else "")
for c in sorted(set(comp)):
    m=comp==c; dm=ca[m]-ea[m]
    print(f"    {c:2d}comp: CRB-emp med={np.median(dm):+.3f} win={int((dm>0).sum())}/{m.sum()}")

print("\n=== 4. PAIRED: CRB vs dist (does CRB beat pure geometry?) ===")
d2 = ca - da
print(f"  CRB-dist per-ep diff: median={np.median(d2):+.3f} win={int((d2>0).sum())}/{len(d2)}")
for c in sorted(set(comp)):
    m=comp==c; dm=ca[m]-da[m]
    print(f"    {c:2d}comp: CRB-dist med={np.median(dm):+.3f} win={int((dm>0).sum())}/{m.sum()}")

print("\n=== 5. partial(CRB,act|dist) survival by regime ===")
for c in sorted(set(comp)):
    m=comp==c
    print(f"    {c:2d}comp: partial med={np.median(pcd[m]):+.3f}  min={pcd[m].min():+.3f}  frac>0={ (pcd[m]>0).mean():.2f}")
