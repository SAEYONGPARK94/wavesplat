"""Independent re-derivation of the routability headline (dual-verification).

Different implementation choices from compute_routability.py on purpose:
  - RandomForest (bagging) instead of GradientBoosting (boosting)
  - different CV seed and a different split scheme (repeated stratified 10-fold)
  - out-of-fold pooled AUC instead of per-fold averaging
  - paired bootstrap over episodes (same resample applied to both columns)
Adds the stratified check the first agent named as the open vulnerability:
does the OBS>NCOMP ordering survive inside the crowded-only / sparse-only strata?
"""
import json, os, sys
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

SEED = 77_2026                      # deliberately NOT the first agent's 20260801
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
sys.path.insert(0, HERE)
from compute_routability import load_synthetic, OBS_FEATS   # data assembly only


def oof_auc(X, y, model, n_splits=10, n_repeats=10, seed=SEED):
    """Pooled out-of-fold AUC, averaged over repeats."""
    cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats,
                                 random_state=seed)
    per_repeat, buf_p, buf_y = [], [], []
    for i, (tr, te) in enumerate(cv.split(X, y)):
        m = model()
        m.fit(X[tr], y[tr])
        buf_p.append(m.predict_proba(X[te])[:, 1]); buf_y.append(y[te])
        if (i + 1) % n_splits == 0:
            per_repeat.append(roc_auc_score(np.concatenate(buf_y),
                                            np.concatenate(buf_p)))
            buf_p, buf_y = [], []
    return float(np.mean(per_repeat)), np.asarray(per_repeat)


def paired_boot(a_scores, b_scores, n=10_000, seed=SEED):
    """Bootstrap CI of the (a - b) gap over repeat-level estimates."""
    rng = np.random.default_rng(seed)
    d = a_scores - b_scores
    bs = np.array([rng.choice(d, d.size, replace=True).mean() for _ in range(n)])
    return float(d.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def run(dz, rows, tag, strat=None):
    keep = [r for r in rows if abs(r['delta']) > dz]
    if strat:
        keep = [r for r in keep if r['regime'] == strat]
    if len(keep) < 12:
        return dict(tag=tag, dz=dz, n=len(keep), skipped='too few')
    y = np.array([1 if r['delta'] > 0 else 0 for r in keep])
    if len(set(y)) < 2 or min(np.bincount(y)) < 4:
        return dict(tag=tag, dz=dz, n=len(keep), pos=int(y.sum()),
                    skipped='degenerate label')
    Xo = np.array([[r[f] for f in OBS_FEATS] for r in keep], float)
    Xk = np.array([[r['n_components']] for r in keep], float)
    lin = lambda: Pipeline([('s', StandardScaler()),
                            ('m', LogisticRegression(max_iter=5000))])
    rf = lambda: RandomForestClassifier(n_estimators=400, min_samples_leaf=2,
                                        random_state=SEED, n_jobs=-1)
    o_lin, o_lin_r = oof_auc(Xo, y, lin)
    o_rf,  o_rf_r  = oof_auc(Xo, y, rf)
    k_lin, k_lin_r = oof_auc(Xk, y, lin)
    k_rf,  k_rf_r  = oof_auc(Xk, y, rf)
    k_best_r = k_rf_r if k_rf >= k_lin else k_lin_r
    gap, lo, hi = paired_boot(o_rf_r, k_best_r)
    return dict(tag=tag, dz=dz, n=len(keep), pos=int(y.sum()),
                obs_lin=round(o_lin, 4), obs_rf=round(o_rf, 4),
                knob_lin=round(k_lin, 4), knob_rf=round(k_rf, 4),
                gap_obsrf_minus_knobbest=round(gap, 4),
                gap_ci=[round(lo, 4), round(hi, 4)],
                excludes_zero=bool(lo > 0 or hi < 0))


if __name__ == '__main__':
    rows = load_synthetic()
    out = []
    for dz in (0.00, 0.02, 0.05):
        out.append(run(dz, rows, f'all_dz{dz:.2f}'))
    for strat in ('crowded', 'sparse', 'moderate'):
        out.append(run(0.00, rows, f'{strat}_dz0.00', strat=strat))
    for r in out:
        print(json.dumps(r))
    json.dump(out, open(os.path.join(HERE, 'verify_independent.json'), 'w'),
              indent=1)
