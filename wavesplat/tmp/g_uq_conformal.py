"""Agent G — threat-T2 rebuttal: is the NEGATIVE an artefact of Gaussian NLL punishing
over-dispersion? Re-run the same comparison with NORMALIZED SPLIT-CONFORMAL prediction
intervals (Papadopoulos-style), whose coverage is valid BY CONSTRUCTION, and score the
bars on the metric conformal actually competes on: MEAN INTERVAL WIDTH at matched coverage.

Split by unit (train/calibration units vs held-out units), nominal 90%.
Usage: python3 tmp/g_uq_conformal.py results/direction3/g_uq_raw_S.npz
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
crb_n = np.empty_like(crb)
for uu in units:
    m = u == uu
    crb_n[m] = crb[m] / crb[m].mean()
anch = dict(const=np.ones_like(e), crb=crb, emp=emp, crb_shape=res * crb_n)
NOM = 0.90
rng = np.random.default_rng(7)
NB = 200
cov = {k: [] for k in anch}
wid = {k: [] for k in anch}
for b in range(NB):
    perm = rng.permutation(len(units))
    tru = set(units[perm[:len(units) // 2]].tolist())
    tr = np.isin(u, list(tru)); te = ~tr
    for k, a in anch.items():
        s = np.quantile(np.abs(e[tr]) / a[tr], NOM)     # conformal quantile of normalized score
        hw = s * a[te]
        cov[k].append(float(np.mean(np.abs(e[te]) <= hw)))
        wid[k].append(float(np.mean(hw)))
print(f"NORMALIZED SPLIT-CONFORMAL, nominal {NOM:.0%}, {NB} unit split-halves  ({p})")
print(f"{'anchor':<12s}{'coverage':>10s}{'mean width':>12s}{'median width':>14s}")
for k in anch:
    c, w = np.array(cov[k]), np.array(wid[k])
    print(f"{k:<12s}{c.mean():>10.3f}{w.mean():>12.4f}{np.median(w):>14.4f}")
print("\n  (lower width at equal/greater coverage = better. If const is narrowest, the")
print("   CRB anchor buys nothing even under a metric that guarantees coverage.)")
