"""G1 power check — WaveSplat (analytic prior) vs WavePrior (learned prior).

The COMPARE doc's headline (HOS 3/3, real 4/6, linear 1/5) rests on n=3..6 with
overlapping std. This driver scales the seed/window count and runs *paired*
significance tests the built-in `stats()` never computed:

  - paired t-test, Wilcoxon signed-rank, exact sign test (binomial)
  - paired Cohen's d, bootstrap 95% CI of the mean paired delta

Each rep is a fresh sea realization; both estimators are deterministic
closed-form MMSE (no random restart), so one seed == one clean paired sample.
Positive delta = WavePrior better (lower nRMSE).

Usage:
  python3 tmp/G1_waveprior_power/run_power.py --cases hos --reps 24
  python3 tmp/G1_waveprior_power/run_power.py --cases hos,linear,real --reps 24
"""
import argparse
import json
import os
import time

import numpy as np

# importing the module chdir's to the repo root and fixes sys.path
import d3_compare_gifs as cg  # noqa: E402  (import side effects intended)

OUT_DIR = 'tmp/G1_waveprior_power'
HEADLINE_KEYS = ('map_nrmse_nowcast', 'map_nrmse_mean_over_leads',
                 'map_corr_nowcast', 'map_corr_mean_over_leads')


def paired_tests(d):
    """d = per-rep paired delta (WS - WP). Positive => WP better on nRMSE."""
    from scipy import stats as ss
    d = np.asarray([x for x in d if x is not None], float)
    n = len(d)
    out = {'n': int(n), 'mean': float(np.mean(d)), 'std': float(np.std(d, ddof=1)),
           'per_rep': [float(x) for x in d]}
    # paired t-test (H0: mean delta = 0)
    t, p_t = ss.ttest_1samp(d, 0.0)
    out['ttest'] = {'t': float(t), 'p': float(p_t)}
    # Wilcoxon signed-rank (nonparametric)
    try:
        w, p_w = ss.wilcoxon(d)
        out['wilcoxon'] = {'stat': float(w), 'p': float(p_w)}
    except ValueError as e:
        out['wilcoxon'] = {'error': str(e)}
    # exact sign test: how many reps favour WP (d>0), binomial vs 0.5
    n_pos = int(np.sum(d > 0))
    n_nonzero = int(np.sum(d != 0))
    p_sign = float(ss.binomtest(n_pos, n_nonzero, 0.5).pvalue) if n_nonzero else 1.0
    out['sign_test'] = {'wp_wins': n_pos, 'n_nonzero': n_nonzero, 'p': p_sign}
    # paired Cohen's d
    sd = np.std(d, ddof=1)
    out['cohens_d'] = float(np.mean(d) / sd) if sd > 0 else None
    # bootstrap 95% CI of the mean delta
    rng = np.random.default_rng(0)
    boot = [np.mean(rng.choice(d, n, replace=True)) for _ in range(10000)]
    out['boot95'] = [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))]
    return out


def run_case(case, reps, record='YS17'):
    rows, t_seed = [], []
    for r in range(reps):
        t0 = time.time()
        sc = cg.scene_sim(case, seed=r) if case in ('linear', 'hos') \
            else cg.scene_real(record, win=r)
        sc['n_obs'] = len(sc['z'])
        met = cg.metrics(sc, cg.solve(sc))
        rows.append(met)
        t_seed.append(time.time() - t0)
        ws = met['WaveSplat']['map_nrmse_nowcast']
        wp = met['WavePrior']['map_nrmse_nowcast']
        print(f"  [{case} {r:2d}/{reps}] WS {ws:.3f} / WP {wp:.3f} "
              f"| d_now {ws - wp:+.3f} | {t_seed[-1]:.1f}s", flush=True)

    res = {'case': case, 'reps': reps, 'sec_per_seed': float(np.mean(t_seed))}
    for key in HEADLINE_KEYS:
        d = [m['WaveSplat'][key] - m['WavePrior'][key] for m in rows
             if m['WaveSplat'][key] is not None and m['WavePrior'][key] is not None]
        sign = 1.0 if 'nrmse' in key else -1.0  # corr: positive delta = WS better; flip
        res[key] = paired_tests([sign * x for x in d])
        res[key]['note'] = ('positive delta favours WavePrior '
                            + ('(lower nRMSE)' if 'nrmse' in key else '(higher corr)'))
    # also record raw per-arm means for the table
    for name in ('WaveSplat', 'WavePrior'):
        res[name] = {k: float(np.mean([m[name][k] for m in rows
                                        if m[name][k] is not None]))
                     for k in HEADLINE_KEYS}
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cases', default='hos')
    ap.add_argument('--reps', type=int, default=24)
    ap.add_argument('--record', default='YS17')
    ap.add_argument('--tag', default='')
    a = ap.parse_args()
    cases = a.cases.split(',')

    allm = {}
    for case in cases:
        print(f"\n=== {case}  (reps={a.reps}) ===", flush=True)
        allm[case] = run_case(case, a.reps, a.record)

    tag = ('_' + a.tag) if a.tag else ''
    path = os.path.join(OUT_DIR, f"power{tag}.json")
    with open(path, 'w') as f:
        json.dump(allm, f, indent=1)
    print(f"\n-> {path}")

    # compact verdict per case on the two headline metrics
    print("\n=== VERDICT (headline: map_nrmse_nowcast) ===")
    for case in cases:
        r = allm[case]['map_nrmse_nowcast']
        print(f"  {case:7s} n={r['n']:2d}  meanD {r['mean']:+.4f} "
              f"[{r['boot95'][0]:+.4f},{r['boot95'][1]:+.4f}]  "
              f"t p={r['ttest']['p']:.3f}  sign {r['sign_test']['wp_wins']}/"
              f"{r['sign_test']['n_nonzero']} p={r['sign_test']['p']:.3f}  "
              f"d={r['cohens_d']:.2f}")


if __name__ == '__main__':
    main()
