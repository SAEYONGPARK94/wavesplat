"""Agent G — ROUTE 2 independent re-derivation (dual verification, discipline
[[feedback_dual_verification_default]]). Deliberately does NOT import g_uq_analyze.
Different resampling (repeated random unit split-half, 200 reps) instead of LOSO;
NumPy only; independent coverage/NLL code; plus a direct per-unit alpha census
and a "shape-only" check that isolates the CRB spatial shape from its level.

Usage: python3 tmp/g_uq_verify2.py results/direction3/g_uq_raw_S.npz
"""
import sys
import numpy as np

p = sys.argv[1]
z = np.load(p, allow_pickle=True)
e = z['err'].astype(float)
u = z['unit']
units = np.array(sorted(set(u.tolist())))
crb = np.maximum(z['crb'].astype(float), 1e-12)
emp = np.maximum(z['emp'].astype(float), 1e-12)
res = np.maximum(z['resid'].astype(float), 1e-12) if 'resid' in z.files else np.ones_like(e)
print(f"ROUTE2 {p}: rows={len(e)} units={len(units)}")

# --- per-unit alpha census (does ONE global inefficiency factor transfer?) -----
print("\n[R2-A] per-unit alpha = sqrt(mean(e^2/anchor^2))  (ideal: constant across units)")
for nm, anc in (('crb', crb), ('emp', emp), ('const', np.ones_like(e))):
    al = np.array([np.sqrt(np.mean((e[u == uu] / anc[u == uu]) ** 2)) for uu in units])
    print(f"   {nm:6s}: med {np.median(al):9.3f}  min {al.min():9.3f}  max {al.max():9.3f}  "
          f"max/min {al.max()/al.min():6.2f}x  CV {al.std()/al.mean():.2f}")

# --- repeated split-half over units ------------------------------------------
rng = np.random.default_rng(20260723)
NB = 200
names = ['const', 'crb', 'emp', 'crb_shape']
# crb_shape: within-unit-normalised CRB * runtime residual level
crb_n = np.empty_like(crb)
for uu in units:
    m = u == uu
    crb_n[m] = crb[m] / crb[m].mean()
anch = dict(const=np.ones_like(e), crb=crb, emp=emp, crb_shape=res * crb_n)

acc = {n: [] for n in names}
cov = {n: [] for n in names}
for b in range(NB):
    perm = rng.permutation(len(units))
    tru = set(units[perm[:len(units) // 2]].tolist())
    tr = np.isin(u, list(tru)); te = ~tr
    for n in names:
        a = anch[n]
        al = np.sqrt(np.mean((e[tr] / a[tr]) ** 2))
        s = al * a[te]
        nll = np.mean(0.5 * np.log(2 * np.pi * s * s) + 0.5 * (e[te] / s) ** 2)
        acc[n].append(nll)
        cov[n].append((np.mean(np.abs(e[te] / s) < 1.0), np.mean(np.abs(e[te] / s) < 1.959964)))

print(f"\n[R2-B] repeated unit split-half ({NB} reps): pooled test NLL and coverage")
for n in names:
    v = np.array(acc[n]); c = np.array(cov[n])
    print(f"   {n:10s} NLL {v.mean():+.4f} +- {v.std():.4f}   "
          f"cov68 {c[:,0].mean():.3f}  cov95 {c[:,1].mean():.3f}")
for n in ('const', 'emp', 'crb_shape'):
    d = np.array(acc[n]) - np.array(acc['crb'])
    print(f"   dNLL({n:10s} - crb) = {d.mean():+.4f}  [{np.percentile(d,2.5):+.4f},"
          f"{np.percentile(d,97.5):+.4f}]  frac>0 {np.mean(d>0):.2f}")
d = np.array(acc['const']) - np.array(acc['crb_shape'])
print(f"   dNLL(const      - crb_shape) = {d.mean():+.4f}  [{np.percentile(d,2.5):+.4f},"
      f"{np.percentile(d,97.5):+.4f}]  frac>0 {np.mean(d>0):.2f}")

# --- dynamic-range diagnostic: does sigma_CRB over-disperse? ------------------
print("\n[R2-C] dynamic-range check, pooled over units (log-log regression of |e| RMS on anchor)")
for nm, anc in (('crb', crb), ('emp', emp)):
    q = np.quantile(anc, np.linspace(0, 1, 21))
    b = np.clip(np.digitize(anc, q[1:-1]), 0, 19)
    ms, mr = [], []
    for j in range(20):
        m = b == j
        if m.sum() > 20:
            ms.append(anc[m].mean()); mr.append(np.sqrt(np.mean(e[m] ** 2)))
    ms, mr = np.log(np.array(ms)), np.log(np.array(mr))
    sl, ic = np.polyfit(ms, mr, 1)
    print(f"   {nm:6s}: slope {sl:+.3f} (ideal 1.0)  anchor range {np.exp(ms.max()-ms.min()):8.1f}x  "
          f"actual-error range {np.exp(mr.max()-mr.min()):6.1f}x")
