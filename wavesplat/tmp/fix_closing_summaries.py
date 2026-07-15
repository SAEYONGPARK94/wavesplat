"""Recompute correct (nan-filtered, correctly-oriented) summaries for E-18/E-19
from the saved rows and overwrite the committed JSON summaries, so the artifacts
match the text (examiner found the shipped E-18 summary all-NaN and the E-19 AUC
sign-flipped). Reads rows only; no rebuild.
"""
import json
import numpy as np
from scipy import stats

R = 'results/direction3'


def ols(y, X):
    X = np.column_stack([np.ones(len(y))] + X)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    dof = len(y) - X.shape[1]
    cov = (resid @ resid) / dof * np.linalg.inv(X.T @ X)
    se = np.sqrt(np.diag(cov))
    t = beta / se
    return beta, t, 2 * stats.t.sf(np.abs(t), dof)


# -------- E-18 --------
d = json.load(open(f'{R}/e18_decorrelated_2factor.json'))
rows = d['rows']
n_all = len(rows)
fr = [r for r in rows if np.isfinite(r['dd']) and np.isfinite(r['d_lin'])]
dd = np.array([r['dd'] for r in fr]); dlin = np.array([r['d_lin'] for r in fr])
ar = np.array([r['aperture_ratio'] for r in fr]); st = np.array([r['steepness'] for r in fr])
zar, zst = stats.zscore(ar), stats.zscore(st)
b_dd, t_dd, p_dd = ols(dd, [zar, zst, zar * zst])
b_dl, t_dl, p_dl = ols(dlin, [zar, zst])
# sea-level (cluster on sea) — the honest, non-iid version
seas = sorted({r['sea'] for r in fr})
sm_dl = np.array([np.mean([r['d_lin'] for r in fr if r['sea'] == s]) for s in seas])
sm_ar = np.array([[r['aperture_ratio'] for r in fr if r['sea'] == s][0] for s in seas])
rho_sea, p_sea = stats.spearmanr(sm_ar, sm_dl)
rho_ap_dd, p_ap_dd = stats.spearmanr(ar, dd)
rho_st_dd, p_st_dd = stats.spearmanr(st, dd)
hi = ar >= np.median(ar)
d['summary'] = dict(
    n_cells_total=n_all, n_cells_finite=len(fr),
    n_dropped_nonfinite=n_all - len(fr),
    confound_corr=float(np.corrcoef(ar, st)[0, 1]),
    dd_aperture_beta=float(b_dd[1]), dd_aperture_p=float(p_dd[1]),
    dd_steep_beta=float(b_dd[2]), dd_steep_p=float(p_dd[2]),
    dd_interaction_beta=float(b_dd[3]), dd_interaction_p=float(p_dd[3]),
    dlin_aperture_beta=float(b_dl[1]), dlin_aperture_t_iid=float(t_dl[1]),
    dlin_aperture_p_iid=float(p_dl[1]),
    dlin_steep_beta=float(b_dl[2]), dlin_steep_p_iid=float(p_dl[2]),
    dlin_aperture_spearman_sealevel=float(rho_sea),
    dlin_aperture_p_sealevel=float(p_sea), n_seas=len(seas),
    rho_ap_dd=float(rho_ap_dd), p_ap_dd=float(p_ap_dd),
    rho_st_dd=float(rho_st_dd), p_st_dd=float(p_st_dd),
    sign_flip_high_aperture=int(sum(r['flip'] for r, h in zip(fr, hi) if h)),
    n_high_aperture=int(hi.sum()),
    dd_hi=float(dd[hi].mean()), dd_lo=float(dd[~hi].mean()),
    dlin_hi=float(dlin[hi].mean()), dlin_lo=float(dlin[~hi].mean()),
    note='nan-filtered; iid t on cells overstates precision, sea-level Spearman '
         '(k=18) is the honest number')
json.dump(d, open(f'{R}/e18_decorrelated_2factor.json', 'w'), indent=2)
print(f"E-18 summary fixed: confound {d['summary']['confound_corr']:+.3f} | "
      f"dlin~aperture sea-level rho {rho_sea:+.3f} p {p_sea:.4f} | "
      f"dlin~steep p {p_dl[2]:.3f} | dd~aperture rho {rho_ap_dd:+.3f} p {p_ap_dd:.4f}"
      f" | flip {d['summary']['sign_flip_high_aperture']}/{hi.sum()} "
      f"(dropped {n_all-len(fr)})")

# -------- E-19 --------
e = json.load(open(f'{R}/e19_crb_binding.json'))
crb_rows = e['crb_rows']; crbmap = {r['sea']: r['crb_nrmse'] for r in crb_rows}
armap = {r['sea']: r['aperture_ratio'] for r in crb_rows}
e18 = json.load(open(f'{R}/e18_decorrelated_2factor.json'))['rows']
per = {}
for r in e18:
    per.setdefault(r['sea'], []).append(r)
C, DL, DD, AR = [], [], [], []
for s, rs in per.items():
    if s not in crbmap:
        continue
    dl = np.nanmean([x['d_lin'] for x in rs])
    if not np.isfinite(dl):
        continue
    C.append(crbmap[s]); DL.append(dl); DD.append(np.nanmean([x['dd'] for x in rs]))
    AR.append(armap[s])
C, DL, DD, AR = np.array(C), np.array(DL), np.array(DD), np.array(AR)
r_cd, p_cd = stats.spearmanr(C, DL)
fin = np.isfinite(DD)
r_cdd, p_cdd = stats.spearmanr(C[fin], DD[fin])
r_ar, p_ar = stats.spearmanr(AR, DL)              # the raw ratio, for collinearity
r_car, _ = stats.spearmanr(C, AR)
neg = DL < 0
auc = np.mean([1.0 if a > b else (0.5 if a == b else 0.0)
               for a in C[~neg] for b in C[neg]]) if neg.any() and (~neg).any() else np.nan
best = max((((C[neg] < thr).mean() + (C[~neg] >= thr).mean()) / 2, float(thr))
          for thr in np.unique(C))
e['summary'] = dict(
    n_seas=len(C), spearman_crb_dlin=[float(r_cd), float(p_cd)],
    spearman_crb_dd=[float(r_cdd), float(p_cdd)], auc=float(auc),
    best_threshold=best[1], best_balanced_acc=float(best[0]),
    crb_median_welldet=float(np.median(C[neg])),
    crb_median_underdet=float(np.median(C[~neg])),
    collinearity=dict(spearman_aperture_dlin=[float(r_ar), float(p_ar)],
                      spearman_crb_aperture=float(r_car),
                      note='CRB is collinear with the raw aperture ratio and '
                           'predicts Delta_lin no better (ratio rho stronger); '
                           'CRB supplies information-theoretic vocabulary, not an '
                           'independent falsifiable prediction'))
json.dump(e, open(f'{R}/e19_crb_binding.json', 'w'), indent=2)
print(f"E-19 summary fixed: Spearman(CRB,dlin) {r_cd:+.3f} p {p_cd:.4f} | "
      f"AUC {auc:.3f} | best thr {best[1]:.3f} bal-acc {best[0]:.2f} | "
      f"collinearity: aperture rho {r_ar:+.3f} (vs CRB {r_cd:+.3f}), "
      f"CRB~aperture {r_car:+.3f}")
