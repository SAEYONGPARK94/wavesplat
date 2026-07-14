"""Record-cluster bootstrap for every window-level headline claim.

Persona-debate action item #1 (docs/PERSONA_DEBATE_phd_value_2026-07-12.md):
window-level t statistics treat windows as independent although they are
clustered by record. This re-analysis resamples RECORDS with replacement
(B=10,000) and reports percentile CIs + bootstrap p, alongside the naive
window t, the record-mean t (df = n_rec - 1), and the record sign count.
Read-only: consumes committed JSONs, no retraining.

Contrasts (file + arms disclosed per row):
  A  gamma-prior net vs hand   sa_multirecord_mmse{,_net}, cond A,
                               common_unobs corr_bl, window-paired
  B  bw_s0_head - hand         head_anatomy.json (60 win), all / bad
  C  fix3 - hand (floor only)  head_anatomy.json, all / bad
  D  nrmse learned - hand      softblend_rmse_eval.json (RMSE audit)
"""
import json
import os
import sys

import numpy as np
from scipy import stats

os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

B = 10000
RNG = np.random.default_rng(0)
OUT = 'results/direction3/cluster_bootstrap_2026-07-12.json'


def analyze(name, recs, deltas):
    """recs: record label per window; deltas: per-window paired delta."""
    recs = np.asarray(recs)
    d = np.asarray(deltas, float)
    names = sorted(set(recs))
    k = len(names)
    groups = {nm: d[recs == nm] for nm in names}
    rec_means = np.array([groups[nm].mean() for nm in names])
    # naive window-level t
    se_w = d.std(ddof=1) / np.sqrt(len(d))
    t_w = d.mean() / se_w
    # record-mean t (df = k-1)
    se_r = rec_means.std(ddof=1) / np.sqrt(k)
    t_r = rec_means.mean() / se_r if se_r > 0 else np.inf
    # ---- pairs cluster bootstrap (what this script used to decide on) -------
    # E-12e (2026-07-14) — THIS DECISION RULE IS INVALID AT SMALL k AND MUST NOT
    # BE USED ALONE. Resampling k clusters with replacement and taking the 2.5/
    # 97.5 percentiles of the resampled MEAN makes the CI collapse onto the RANGE
    # of the k cluster means: at k=3 the interval came out as exactly
    # [min(rec_means), max(rec_means)]. It then "excludes 0" whenever all k
    # cluster means share a sign -- i.e. it is a k-point SIGN TEST wearing a
    # bootstrap costume, and it prints p = 0.0000 while doing it. It caught D-20
    # (k=3) as "SURVIVES" although the record-mean t is only -1.59 (p 0.25).
    boots = np.empty(B)
    for b in range(B):
        pick = RNG.integers(0, k, k)
        boots[b] = np.concatenate([groups[names[i]] for i in pick]).mean()
    lo, hi = np.percentile(boots, [2.5, 97.5])
    p_pairs = float(min((boots <= 0).mean(), (boots >= 0).mean()) * 2)

    # ---- wild cluster bootstrap-t, Rademacher (Cameron-Gelbach-Miller 2008) --
    # The standard small-G tool. Impose H0 (mu = 0), flip the SIGN of each
    # cluster's residuals, and rebuild the t statistic each time.
    def cr_t(y):
        mu_ = y.mean()
        acc = 0.0
        for nm in names:
            yi = y[recs == nm]
            acc += (yi.sum() - len(yi) * mu_) ** 2
        c = (k / (k - 1)) * ((len(y) - 1) / (len(y) - k)) if k > 1 else 1.0
        se_ = np.sqrt(c * acc / len(y) ** 2)
        return mu_ / se_ if se_ > 0 else np.inf
    # Rademacher (+-1) admits only 2^k patterns, so its p-value is quantised in
    # steps of 2^-k and is far too coarse below ~12 clusters. Webb (2014) 6-point
    # weights are the standard remedy at small G: 6^k patterns.
    idx = {nm: i for i, nm in enumerate(names)}
    ci = np.array([idx[r] for r in recs])
    WEBB = np.array([-np.sqrt(1.5), -1.0, -np.sqrt(0.5),
                     np.sqrt(0.5), 1.0, np.sqrt(1.5)])
    use_webb = k < 12
    t_obs = cr_t(d)                              # H0-imposed residuals are d
    wb = np.empty(B)
    for b in range(B):
        w = (RNG.choice(WEBB, k) if use_webb else RNG.choice([-1.0, 1.0], k))
        wb[b] = cr_t(d * w[ci])
    p_wild = float((np.abs(wb) >= abs(t_obs)).mean())
    # smallest attainable p: 1/(#patterns) -- Rademacher 2^-k, Webb 6^-k. Under
    # Rademacher a k<=4 claim can NEVER reach .05 (2^-4 = .0625) no matter how
    # much data sits inside the clusters; Webb removes that floor but not the
    # underlying fact that k clusters carry only k independent pieces of
    # evidence about between-cluster variation.
    p_min = (6.0 ** (-k)) if use_webb else (2.0 ** (-k))

    # record-mean t against its own t(k-1) reference (the honest simple test)
    tcrit_r = float(stats.t.ppf(0.975, k - 1)) if k > 1 else np.inf
    p_rec = float(2 * stats.t.sf(abs(t_r), k - 1)) if k > 1 else np.nan

    survives = bool(p_wild < 0.05 and p_rec < 0.05)
    res = {'contrast': name, 'n_windows': int(len(d)), 'n_records': k,
           'mean': float(d.mean()), 't_window': float(t_w),
           'rec_means': {nm: float(groups[nm].mean()) for nm in names},
           't_record': float(t_r), 't_record_crit': tcrit_r,
           'p_record': p_rec,
           'sign': f"{int((rec_means > 0).sum())}/{k}",
           'pairs_boot_ci_INVALID_small_k': [float(lo), float(hi)],
           'p_pairs_boot_INVALID_small_k': p_pairs,
           't_cluster_robust': float(t_obs),
           'p_wild_cluster': p_wild, 'p_min_attainable': p_min,
           'underpowered_by_design': bool(p_min >= 0.05),
           'survives': survives}
    flag = ('SURVIVES' if survives else
            'NOT SIG (cluster)')
    if p_min >= 0.05:
        flag += f'  [k={k}: min attainable p = {p_min:.3f} — CANNOT reach .05]'
    print(f"{name:34s} mean {d.mean():+.3f}  t_win {t_w:+.2f}  "
          f"t_rec {t_r:+.2f} (p {p_rec:.3f})  sign {res['sign']}  "
          f"p_wild {p_wild:.4f}  {flag}")
    return res


def main():
    out = []
    # A. gamma-prior, 8 records, window-paired
    h8 = json.load(open('results/direction3/sa_multirecord_mmse.json'))
    n8 = json.load(open('results/direction3/sa_multirecord_mmse_net.json'))
    recs, dd = [], []
    for rec in h8:
        hw = [w['conditions']['A']['mean']['common_unobs']['corr_bl']
              for w in h8[rec]['windows']]
        nw = [w['conditions']['A']['mean']['common_unobs']['corr_bl']
              for w in n8[rec]['windows']]
        for a, b in zip(hw, nw):
            recs.append(rec)
            dd.append(b - a)
    out.append(analyze('A gamma-net - hand (8rec nowcast)', recs, dd))

    # B, C from head_anatomy
    ha = json.load(open('results/direction3/head_anatomy.json'))
    hand = np.array([r['hand'] for r in ha])
    rec60 = [r['record'] for r in ha]
    bad = hand < 0.5
    for label, key in (('B head(bw_s0_head) - hand', 'bw_s0_head'),
                       ('C floor-only(fix3) - hand', 'fix3')):
        d60 = np.array([r[key] for r in ha]) - hand
        out.append(analyze(f'{label} [all60]', rec60, d60))
        out.append(analyze(f'{label} [bad{int(bad.sum())}]',
                           [r for r, b in zip(rec60, bad) if b], d60[bad]))

    # D. RMSE audit
    sb = json.load(open('results/direction3/softblend_rmse_eval.json'))
    out.append(analyze('D nrmse learned - hand (RMSE)',
                       [r['record'] for r in sb],
                       [r['nrmse_learned'] - r['nrmse_hand'] for r in sb]))

    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"\nwrote {OUT}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
