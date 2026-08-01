"""Routability lower-bound audit (CRUX-A re-instrumentation), 2026-08-01.

Restates the identifiability claim as: what is the best achievable ROC-AUC of ANY classifier,
over a DECLARED runtime-observable feature class, at predicting which arm wins per episode?
Compared against the same classifiers given the simulator knob n_components (the answer key).

Protocol frozen in PRESTATE.md (sha256 9bff2c169b6eff35051281aeaa2302fffd9c000dcfd60cc985fbf458d02d8234)
BEFORE any number here was produced. Zero new estimator compute: every input is JSON on disk.
Modifies no repo code, no defaults. Re-runnable and deterministic.

    python3 tmp/routability_2026-08-01/compute_routability.py     # from package root
"""
import json
import os
import re
import sys

import numpy as np
from scipy.stats import rankdata, mannwhitneyu
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

SEED = 20260801
N_BOOT = 10_000
N_SPLITS, N_REPEATS = 5, 20
DEAD_ZONES = [0.00, 0.02, 0.05]   # 0.02 = headline (PRESTATE §1)
PRIMARY_DZ = 0.02

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
OBSAXIS = os.path.join(ROOT, 'tmp/obsaxis_2026-07-26/obsaxis_stats.json')
BEAMFORM = os.path.join(ROOT, 'tmp/unified_guard_2026-07-26/explore_beamform.json')
SIGNALS = os.path.join(ROOT, 'tmp/unified_guard_2026-07-26/explore_signals.json')
BF_REAL = os.path.join(ROOT, 'tmp/figs_opus_real_2026-07-26/bf_ratio_real.json')
BLIND_REAL = os.path.join(ROOT, 'tmp/blind_auto_real_2026-07-26/blind_auto_real.json')
OUT = os.path.join(HERE, 'routability.json')

DIFF_SEEDS = (0, 1, 2, 3, 4)      # the 5 lanes the label uses

# ---------------------------------------------------------------- feature class (PRESTATE §2)
BF_FEATS = ['bf_p2_p1', 'bf_p3_p1', 'bf_p8_p1',
            'npk_10', 'npk_20', 'npk_30', 'npk_50', 'n_peaks_found']
PROP_FEATS = ['n_sel', 'kp_sel', 'dom_frac', 'top2_frac', 'top3_frac',
              'nmerge_abs015', 'nmerge_abs0225',
              'nmerge_kp15', 'nmerge_kp25', 'nmerge_kp40', 'n_sel_range']
OBS_FEATS = BF_FEATS + PROP_FEATS


# ------------------------------------------------------------------------------- data assembly
def load_synthetic():
    """100 episodes: frozen label delta + the declared GT-free observable vector."""
    ob = json.load(open(OBSAXIS))['synthetic']
    bm = {r['ep']: r for r in json.load(open(BEAMFORM))}
    sig = {}
    for r in json.load(open(SIGNALS)):
        sig.setdefault(r['ep'], {})[r['ds']] = r

    rows = []
    for r in ob:
        ep = r['episode']
        b, s = bm[ep], sig[ep]
        assert abs(b['p2_over_p1'] - r['bfr']) < 1e-12, 'bf_peak_ratio source mismatch'
        assert set(DIFF_SEEDS) <= set(s), f'missing diffusion seed for {ep}'
        ns = [s[d]['n_sel'] for d in DIFF_SEEDS]

        def m(k):
            return float(np.median([s[d][k] for d in DIFF_SEEDS]))

        rows.append(dict(
            episode=ep, delta=r['delta'], n_components=int(r['comp']), regime=r['regime'],
            # --- beamform block (observation-only)
            bf_p2_p1=b['p2_over_p1'], bf_p3_p1=b['p3_over_p1'], bf_p8_p1=b['p8_over_p1'],
            npk_10=float(b['npk_10']), npk_20=float(b['npk_20']),
            npk_30=float(b['npk_30']), npk_50=float(b['npk_50']),
            n_peaks_found=float(b['n_peaks_found']),
            # --- proposal block (observation-only, median over the 5 diffusion seeds)
            n_sel=m('n_sel'), kp_sel=m('kp'), dom_frac=m('dom_frac'),
            top2_frac=m('top2_frac'), top3_frac=m('top3_frac'),
            nmerge_abs015=m('nmerge_abs015'), nmerge_abs0225=m('nmerge_abs0225'),
            nmerge_kp15=m('nmerge_kp15'), nmerge_kp25=m('nmerge_kp25'),
            nmerge_kp40=m('nmerge_kp40'),
            n_sel_range=float(max(ns) - min(ns)),
        ))
    assert len(rows) == 100
    return rows


def load_real():
    """30 real windows on the 2-D subspace that exists on both sides (PRESTATE §7)."""
    bf = {(r['record'], r['win']): r['bf_ratio'] for r in json.load(open(BF_REAL))['rows']}
    ba = json.load(open(BLIND_REAL))['rows']
    out = []
    for r in ba:
        k = (r['record'], r['win'])
        assert abs(bf[k] - r['bf_peak_ratio']) < 1e-6, f'real bf_ratio mismatch at {k}'
        ns = [r['per_seed'][str(d)]['n_selected'] for d in DIFF_SEEDS]
        out.append(dict(record=k[0], win=k[1], bf_p2_p1=float(bf[k]),
                        n_sel=float(np.median(ns)), n_sel_seeds=ns,
                        collapse=bool(r['collapse']), route=r['route'],
                        guard_fired=bool(r['guard_fired'])))
    assert len(out) == 30
    return out


# ------------------------------------------------------------------------------------ AUC math
def auc_from_scores(score, y):
    """Mann-Whitney AUC with tie-correct average ranks. y in {0,1}, positive = learned wins."""
    r = rankdata(score)
    n1 = float(y.sum())
    n0 = float(len(y) - n1)
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def boot_indices(y, n_boot=N_BOOT, seed=SEED):
    """Episode-level resamples with replacement; single-class draws discarded and redrawn."""
    rng = np.random.default_rng(seed)
    n = len(y)
    keep, discarded = [], 0
    while len(keep) < n_boot:
        idx = rng.integers(0, n, size=n)
        yy = y[idx]
        if yy.sum() == 0 or yy.sum() == n:
            discarded += 1
            continue
        keep.append(idx)
    return np.asarray(keep), discarded


def boot_auc(score, y, idx):
    """Vectorised AUC over B resamples sharing the SAME idx matrix (=> paired gaps)."""
    S = score[idx]                       # B x n
    Y = y[idx].astype(float)
    R = rankdata(S, axis=1)
    n1 = Y.sum(axis=1)
    n0 = S.shape[1] - n1
    rsum = (R * Y).sum(axis=1)
    return (rsum - n1 * (n1 + 1) / 2.0) / (n1 * n0)


def ci(v, lo=2.5, hi=97.5):
    return [float(np.percentile(v, lo)), float(np.percentile(v, hi))]


# ------------------------------------------------------------------------------------ fitting
def make_clf(kind):
    if kind == 'logreg':
        return Pipeline([('sc', StandardScaler()),
                         ('lr', LogisticRegression(C=1.0, penalty='l2', solver='lbfgs',
                                                   max_iter=5000))])
    if kind == 'gboost':
        return GradientBoostingClassifier(n_estimators=200, max_depth=2, learning_rate=0.05,
                                          random_state=SEED)
    raise ValueError(kind)


def oof_scores(X, y, kind):
    """Repeated stratified 5-fold; per-episode score = mean OOF prob over the 20 repeats."""
    cv = RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=SEED)
    acc = np.zeros(len(y))
    per_repeat = np.zeros((N_REPEATS, len(y)))
    for f, (tr, te) in enumerate(cv.split(X, y)):
        clf = make_clf(kind)
        clf.fit(X[tr], y[tr])
        p = clf.predict_proba(X[te])[:, 1]
        acc[te] += p
        per_repeat[f // N_SPLITS, te] = p
    acc /= N_REPEATS
    rep_auc = np.array([auc_from_scores(per_repeat[i], y) for i in range(N_REPEATS)])
    return acc, rep_auc


# ------------------------------------------------------------------------------------ analysis
def analyse(rows, dz, verbose=True):
    kept = [r for r in rows if abs(r['delta']) > dz]
    y = np.array([1 if r['delta'] < 0 else 0 for r in kept], dtype=int)
    n = len(kept)

    # drop zero-variance observables (PRESTATE §2a); report the drop
    raw = {f: np.array([r[f] for r in kept], float) for f in OBS_FEATS}
    dropped = sorted(f for f in OBS_FEATS if np.ptp(raw[f]) == 0)
    used = [f for f in OBS_FEATS if f not in dropped]

    def mat(feats):
        return np.column_stack([np.array([r[f] for r in kept], float) for f in feats])

    sets = {
        'OBS': used,
        'NCOMP': ['n_components'],
        'OBS+NCOMP': used + ['n_components'],
        'bf_p2_p1_only': ['bf_p2_p1'],
        'n_sel_only': ['n_sel'],
    }

    idx, n_discard = boot_indices(y)
    res, boots = {}, {}
    for sname, feats in sets.items():
        X = mat(feats)
        for kind in ('logreg', 'gboost'):
            s, rep = oof_scores(X, y, kind)
            a = auc_from_scores(s, y)
            b = boot_auc(s, y, idx)
            boots[(sname, kind)] = b
            res[f'{sname}|{kind}'] = dict(
                feature_set=sname, n_features=len(feats), classifier=kind,
                auc=a, ci95=ci(b), boot_sd=float(b.std()),
                repeat_auc_mean=float(rep.mean()), repeat_auc_sd=float(rep.std()),
                p_auc_gt_half=float((b <= 0.5).mean()))

    # best observable reader (pre-stated: "best of linear/nonlinear")
    best_kind = max(('logreg', 'gboost'), key=lambda k: res[f'OBS|{k}']['auc'])
    gaps = {}
    for kind in ('logreg', 'gboost'):
        g = boots[('NCOMP', kind)] - boots[('OBS', kind)]
        gaps[f'ncomp_minus_obs|{kind}'] = dict(
            point=res[f'NCOMP|{kind}']['auc'] - res[f'OBS|{kind}']['auc'],
            ci95=ci(g), p_gap_le_0=float((g <= 0).mean()))
    # headline gap: best NCOMP reader vs best OBS reader
    best_nc = max(('logreg', 'gboost'), key=lambda k: res[f'NCOMP|{k}']['auc'])
    g_best = boots[('NCOMP', best_nc)] - boots[('OBS', best_kind)]
    gaps['headline_bestNCOMP_minus_bestOBS'] = dict(
        ncomp_classifier=best_nc, obs_classifier=best_kind,
        point=res[f'NCOMP|{best_nc}']['auc'] - res[f'OBS|{best_kind}']['auc'],
        ci95=ci(g_best), p_gap_le_0=float((g_best <= 0).mean()))
    # nonlinear-minus-linear on OBS (P4's "|rho| is blind to nonlinearity" branch)
    g_nl = boots[('OBS', 'gboost')] - boots[('OBS', 'logreg')]
    gaps['obs_gboost_minus_logreg'] = dict(
        point=res['OBS|gboost']['auc'] - res['OBS|logreg']['auc'],
        ci95=ci(g_nl), p_le_0=float((g_nl <= 0).mean()))

    # ---- per-feature marginal AUCs (no fitting; PRESTATE §4)
    marg = {}
    pvals = {}
    for f in used + ['n_components']:
        v = np.array([r[f] for r in kept], float)
        a = auc_from_scores(v, y)
        b = boot_auc(v, y, idx)
        try:
            p = float(mannwhitneyu(v[y == 1], v[y == 0], alternative='two-sided').pvalue)
        except ValueError:
            p = 1.0
        pvals[f] = p
        marg[f] = dict(auc=a, auc_oriented=max(a, 1 - a), ci95=ci(b), p_raw=p)
    # Holm-Bonferroni over the observables only (n_components is the reference, not a candidate)
    obs_p = sorted(((pvals[f], f) for f in used))
    m = len(obs_p)
    prev = 0.0
    for i, (p, f) in enumerate(obs_p):
        adj = min(1.0, max(prev, (m - i) * p))
        prev = adj
        marg[f]['p_holm'] = adj
        marg[f]['holm_sig_0.05'] = bool(adj < 0.05)
    marg['n_components']['p_holm'] = None
    marg['n_components']['holm_sig_0.05'] = None

    out = dict(
        dead_zone=dz, n_total=len(rows), n_after_exclusion=n,
        n_excluded=len(rows) - n,
        n_pos_learned_wins=int(y.sum()), n_neg_v2_wins=int(n - y.sum()),
        class_balance=float(y.mean()),
        dropped_zero_variance=dropped, n_features_used=len(used), features_used=used,
        bootstrap=dict(n_resamples=N_BOOT, seed=SEED, n_discarded_single_class=n_discard),
        auc=res, gaps=gaps, marginal_auc=marg,
        best_obs_classifier=best_kind,
    )
    if verbose:
        print(f'\n{"="*86}\nDEAD ZONE |delta| <= {dz:.2f}   N={n}  '
              f'(learned wins {int(y.sum())}, v2 wins {int(n-y.sum())})')
        if dropped:
            print(f'  dropped zero-variance observables: {dropped}')
        for k, v in res.items():
            print(f'  {k:26s} AUC {v["auc"]:.4f}  CI [{v["ci95"][0]:.4f}, {v["ci95"][1]:.4f}]'
                  f'  (repeat sd {v["repeat_auc_sd"]:.4f})')
        h = gaps['headline_bestNCOMP_minus_bestOBS']
        print(f'  GAP ncomp({h["ncomp_classifier"]}) - obs({h["obs_classifier"]}) = '
              f'{h["point"]:+.4f}  CI [{h["ci95"][0]:+.4f}, {h["ci95"][1]:+.4f}]')
    return out, kept


# ------------------------------------------------------------------------- support mismatch (§7)
def support_check(kept, real):
    syn_bf = np.array([r['bf_p2_p1'] for r in kept])
    syn_ns = np.array([r['n_sel'] for r in kept])
    mu = np.array([syn_bf.mean(), syn_ns.mean()])
    sd = np.array([syn_bf.std(ddof=1), syn_ns.std(ddof=1)])
    Z = np.column_stack([(syn_bf - mu[0]) / sd[0], (syn_ns - mu[1]) / sd[1]])
    N = len(kept)

    per_win, n1_r, n2_r = [], [0.05, 0.10, 0.20], [0.5, 1.0, 2.0]
    for w in real:
        z = np.array([(w['bf_p2_p1'] - mu[0]) / sd[0], (w['n_sel'] - mu[1]) / sd[1]])
        d2 = np.sqrt(((Z - z) ** 2).sum(axis=1))
        rec = dict(record=w['record'], win=w['win'], bf_p2_p1=w['bf_p2_p1'], n_sel=w['n_sel'],
                   collapse=w['collapse'])
        for r in n1_r:
            c = int((np.abs(syn_bf - w['bf_p2_p1']) <= r).sum())
            rec[f'N1_{r:.2f}_count'] = c
            rec[f'N1_{r:.2f}_frac'] = c / N
        for r in n2_r:
            c = int((d2 <= r).sum())
            rec[f'N2_{r:.1f}_count'] = c
            rec[f'N2_{r:.1f}_frac'] = c / N
        rec['N2_nearest_dist'] = float(d2.min())
        per_win.append(rec)

    def summ(key):
        v = np.array([w[key] for w in per_win], float)
        return dict(median=float(np.median(v)), mean=float(v.mean()),
                    min=float(v.min()), max=float(v.max()),
                    n_windows_zero=int((v == 0).sum()))

    q1, q3 = np.percentile([w['bf_p2_p1'] for w in real], [25, 75])
    return dict(
        n_syn_reference=N, syn_bf_mean=float(mu[0]), syn_bf_sd=float(sd[0]),
        syn_nsel_mean=float(mu[1]), syn_nsel_sd=float(sd[1]),
        real_bf_iqr=[float(q1), float(q3)],
        n_syn_in_real_bf_iqr=int(((syn_bf >= q1) & (syn_bf <= q3)).sum()),
        frac_syn_in_real_bf_iqr=float(((syn_bf >= q1) & (syn_bf <= q3)).mean()),
        real_bf_range=[float(min(w['bf_p2_p1'] for w in real)),
                       float(max(w['bf_p2_p1'] for w in real))],
        syn_bf_range=[float(syn_bf.min()), float(syn_bf.max())],
        real_nsel_range=[float(min(w['n_sel'] for w in real)),
                         float(max(w['n_sel'] for w in real))],
        syn_nsel_range=[float(syn_ns.min()), float(syn_ns.max())],
        real_nsel_at_cap30=int(sum(w['n_sel'] >= 30 for w in real)),
        syn_nsel_at_cap30=int((syn_ns >= 30).sum()),
        summary={f'N1_{r:.2f}_frac': summ(f'N1_{r:.2f}_frac') for r in n1_r}
                | {f'N2_{r:.1f}_frac': summ(f'N2_{r:.1f}_frac') for r in n2_r},
        per_window=per_win,
    )


# ------------------------------------------------ POST-HOC validity diagnostics (RESULT.md §D)
# NOT in PRESTATE.md. Added after the first numbers, and labelled POST-HOC everywhere, because
# an AUC of 0.96 from 18 features at n=60 obliges a demonstration that the CV+scoring pipeline
# does not leak. These are validity checks on the instrument, not tests of any hypothesis.
N_PERM = 30


def permutation_null(kept, n_perm=N_PERM):
    y = np.array([1 if r['delta'] < 0 else 0 for r in kept], int)
    used = [f for f in OBS_FEATS if np.ptp(np.array([r[f] for r in kept], float)) > 0]
    X = np.column_stack([np.array([r[f] for r in kept], float) for f in used])
    Xn = np.array([[r['n_components']] for r in kept], float)
    rng = np.random.default_rng(SEED)
    out = {}
    for name, M in (('OBS', X), ('NCOMP', Xn)):
        for kind in ('logreg', 'gboost'):
            a = []
            for _ in range(n_perm):
                yp = rng.permutation(y)
                s, _r = oof_scores(M, yp, kind)
                a.append(auc_from_scores(s, yp))
            a = np.array(a)
            out[f'{name}|{kind}'] = dict(n_perm=n_perm, mean=float(a.mean()), sd=float(a.std()),
                                         p95=float(np.percentile(a, 95)), max=float(a.max()))
            print(f'  [POST-HOC perm null] {name}|{kind}: mean {a.mean():.4f} sd {a.std():.4f} '
                  f'max {a.max():.4f}')
    return out


def evaluability_on_deployment_support(kept, real, sup):
    """Can the fitted classifier even be *tested* where the real windows live? (POST-HOC)"""
    lo, hi = sup['real_bf_iqr']
    inside = [r for r in kept if lo <= r['bf_p2_p1'] <= hi]
    y_in = [1 if r['delta'] < 0 else 0 for r in inside]
    return dict(
        note='POST-HOC. Counts the post-dead-zone synthetic episodes that lie inside the real '
             'bf_peak_ratio IQR, i.e. the subset on which a deployment-relevant AUC could be '
             'estimated at all.',
        real_bf_iqr=[lo, hi], n_inside=len(inside), n_reference=len(kept),
        labels_inside=dict(learned_wins=int(sum(y_in)), v2_wins=int(len(y_in) - sum(y_in))),
        cv_possible=bool(len(inside) >= 10 and 0 < sum(y_in) < len(y_in)),
        episodes_inside=[r['episode'] for r in inside])


def shipped_rule_skill(kept):
    """POST-HOC descriptive: how much of the achievable routability does the SHIPPED rule take?

    Shipped route (tmp/figs_opus_diag_2026-07-26/common.py:127-131, faithful to
    wavesplat/optimizer/sequential_update.py): route 'learned' iff n_selected >= 8, then veto
    back to 'v2' iff bf_peak_ratio < 0.05. Evaluated here at episode level on the median-of-5
    n_sel, which is the aggregation the label uses. A binary rule has no ROC curve, so its
    single operating point is reported as sensitivity/specificity/balanced accuracy; the
    balanced accuracy of a binary rule equals the AUC of that rule used as a 0/1 score.
    """
    y = np.array([1 if r['delta'] < 0 else 0 for r in kept], int)
    ns = np.array([r['n_sel'] for r in kept])
    bf = np.array([r['bf_p2_p1'] for r in kept])
    out = {}
    for name, pred in (('router_only_nsel_ge_8', (ns >= 8).astype(int)),
                       ('router_plus_guard', ((ns >= 8) & (bf >= 0.05)).astype(int))):
        tp = int(((pred == 1) & (y == 1)).sum()); fn = int(((pred == 0) & (y == 1)).sum())
        tn = int(((pred == 0) & (y == 0)).sum()); fp = int(((pred == 1) & (y == 0)).sum())
        sens = tp / max(tp + fn, 1); spec = tn / max(tn + fp, 1)
        out[name] = dict(tp=tp, fp=fp, tn=tn, fn=fn, sensitivity=sens, specificity=spec,
                         balanced_accuracy=0.5 * (sens + spec),
                         auc_as_binary_score=auc_from_scores(pred.astype(float), y),
                         n_routed_learned=int(pred.sum()))
    out['note'] = ('POST-HOC. In-sample: both thresholds (8, 0.05) were chosen on these same '
                   'episodes. Compare against the CV AUC of the fitted classifiers, which are '
                   'out-of-fold; the comparison is therefore biased in the shipped rule\'s '
                   'favour and is still reported below it.')
    return out


def main():
    syn, real = load_synthetic(), load_real()
    results, kept_primary = {}, None
    for dz in DEAD_ZONES:
        r, kept = analyse(syn, dz)
        results[f'{dz:.2f}'] = r
        if dz == PRIMARY_DZ:
            kept_primary = kept
    sup = support_check(kept_primary, real)
    print(f'\n{"="*86}\nPOST-HOC VALIDITY DIAGNOSTICS (not in PRESTATE.md)')
    perm = permutation_null(kept_primary)
    evalb = evaluability_on_deployment_support(kept_primary, real, sup)
    print(f"  [POST-HOC] deployment-support evaluability: {evalb['n_inside']}/{evalb['n_reference']} "
          f"episodes inside the real bf IQR, labels {evalb['labels_inside']}, "
          f"CV possible = {evalb['cv_possible']}")
    ship = {f'{dz:.2f}': shipped_rule_skill([r for r in syn if abs(r['delta']) > dz])
            for dz in DEAD_ZONES}
    for k, v in ship[f'{PRIMARY_DZ:.2f}'].items():
        if isinstance(v, dict):
            print(f"  [POST-HOC] shipped {k:22s} bal.acc {v['balanced_accuracy']:.4f} "
                  f"(sens {v['sensitivity']:.3f} spec {v['specificity']:.3f})")

    print(f'\n{"="*86}\nSUPPORT MISMATCH (synthetic reference n={sup["n_syn_reference"]}, '
          f'post-dead-zone {PRIMARY_DZ})')
    print(f"  real bf IQR {sup['real_bf_iqr'][0]:.3f}-{sup['real_bf_iqr'][1]:.3f}: "
          f"{sup['n_syn_in_real_bf_iqr']}/{sup['n_syn_reference']} synthetic inside")
    for k, v in sup['summary'].items():
        print(f"  {k:16s} median {v['median']*100:5.1f}%  min {v['min']*100:5.1f}%  "
              f"max {v['max']*100:5.1f}%  zero-neighbour windows {v['n_windows_zero']}/30")
    print(f"  n_sel: real range {sup['real_nsel_range']}, {sup['real_nsel_at_cap30']}/30 at cap 30; "
          f"synthetic {sup['syn_nsel_at_cap30']}/{sup['n_syn_reference']} at cap")

    json.dump(dict(
        meta=dict(script=os.path.relpath(__file__, ROOT), seed=SEED, n_boot=N_BOOT,
                  cv=f'RepeatedStratifiedKFold({N_SPLITS} splits x {N_REPEATS} repeats)',
                  prestate_sha256='9bff2c169b6eff35051281aeaa2302fffd9c000dcfd60cc985fbf458d02d8234',
                  primary_dead_zone=PRIMARY_DZ, sources=dict(
                      label=os.path.relpath(OBSAXIS, ROOT),
                      beamform=os.path.relpath(BEAMFORM, ROOT),
                      signals=os.path.relpath(SIGNALS, ROOT),
                      real_bf=os.path.relpath(BF_REAL, ROOT),
                      real_blind=os.path.relpath(BLIND_REAL, ROOT))),
        by_dead_zone=results, support_mismatch=sup,
        posthoc_permutation_null=perm, posthoc_deployment_evaluability=evalb,
        posthoc_shipped_rule=ship,
        synthetic_features=syn, real_features=real,
    ), open(OUT, 'w'), indent=1)
    print(f'\nwrote {OUT}')


if __name__ == '__main__':
    main()
