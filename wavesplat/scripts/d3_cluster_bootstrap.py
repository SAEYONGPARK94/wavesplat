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
    # cluster bootstrap: resample records with replacement, pool windows
    boots = np.empty(B)
    for b in range(B):
        pick = RNG.integers(0, k, k)
        boots[b] = np.concatenate([groups[names[i]] for i in pick]).mean()
    lo, hi = np.percentile(boots, [2.5, 97.5])
    p_boot = float(min((boots <= 0).mean(), (boots >= 0).mean()) * 2)
    res = {'contrast': name, 'n_windows': int(len(d)), 'n_records': k,
           'mean': float(d.mean()), 't_window': float(t_w),
           'rec_means': {nm: float(groups[nm].mean()) for nm in names},
           't_record': float(t_r),
           'sign': f"{int((rec_means > 0).sum())}/{k}",
           'boot_ci': [float(lo), float(hi)], 'p_boot': p_boot,
           'survives': bool(lo > 0 or hi < 0)}
    print(f"{name:34s} mean {d.mean():+.3f}  t_win {t_w:+.2f}  "
          f"t_rec {t_r:+.2f}  sign {res['sign']}  "
          f"CI [{lo:+.3f},{hi:+.3f}]  p {p_boot:.4f}  "
          f"{'SURVIVES' if res['survives'] else 'NOT SIG (cluster)'}")
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
